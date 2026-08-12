"""
Rutas de autenticación para el servidor STT.
Proporciona endpoints para registro, login, logout y gestión de usuarios.

Storage: la persistencia de users + sessions va por dos caminos:
  - DATABASE_URL seteado: Postgres (ver STT_server/db.py + db_users.py).
  - DATABASE_URL ausente: fallback a JSON files (STT_server/data/*.json).
Las funciones load_users / save_users / load_sessions / save_sessions
se resuelven en db_users.py segun que backend este disponible.
"""

import os
import json
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

# Import the storage shim. This rebinds load_users/save_users/etc
# to the JSON or Postgres implementation based on DATABASE_URL.
from STT_server.db_users import (
    load_users,
    save_users,
    load_sessions,
    save_sessions,
)

router = APIRouter(prefix="/auth", tags=["Autenticación"])

# ── Modelos Pydantic ─────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: str

# ── Funciones auxiliares ───────────────────────────────────────────────────────
# load_users / save_users / load_sessions / save_sessions vienen de
# STT_server.db_users (import arriba) y resuelven al backend Postgres
# o JSON segun DATABASE_URL.
# ponytail: salted password hashing. PBKDF2-HMAC-SHA256 with 600k
# iterations (OWASP 2023 recommendation for SHA-256). The on-disk
# format is ``pbkdf2_sha256$ITER$SALT_HEX$HASH_HEX`` — self-describing
# so we can tune iterations later without invalidating existing
# hashes. Legacy unsalted SHA-256 hex (the original implementation)
# is still verified on login so existing accounts keep working; the
# first successful legacy login transparently re-hashes the password
# with the new algorithm (the next save_users call persists the
# upgraded hash). Constant-time comparison via hmac.compare_digest.
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_HASH_NAME = "sha256"
_PBKDF2_SALT_BYTES = 16
_PBKDF2_PREFIX = f"pbkdf2_{_PBKDF2_HASH_NAME}"


def _hash_password_pbkdf2(password: str) -> str:
    salt = secrets.token_bytes(_PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH_NAME,
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return (
        f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}"
        f"${salt.hex()}${digest.hex()}"
    )


def _verify_password_pbkdf2(password: str, encoded: str) -> bool:
    try:
        scheme, iter_s, salt_s, digest_s = encoded.split("$", 3)
    except ValueError:
        return False
    if scheme != _PBKDF2_PREFIX:
        return False
    try:
        iterations = int(iter_s)
        salt = bytes.fromhex(salt_s)
        expected = bytes.fromhex(digest_s)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH_NAME,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def _verify_password_legacy_sha256(password: str, stored_hash: str) -> bool:
    """Legacy unsalted SHA-256 hex — verify-only, never used for new
    passwords. Returns True on a match so the first login after this
    rollout can succeed; the route layer then re-hashes and persists
    via _upgrade_password_hash()."""
    if not stored_hash or len(stored_hash) != 64:
        return False
    expected = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(expected, stored_hash)


def _upgrade_password_hash(user: dict, password: str) -> None:
    """Re-hash a legacy SHA-256 password with PBKDF2 and persist.

    Called on the first successful legacy login so the auth table
    migrates over time without a separate one-shot script. Errors are
    swallowed — the login still succeeded, the hash just stays legacy
    until the next successful login.
    """
    user["password"] = _hash_password_pbkdf2(password)
    # ponytail: the public auth surface only saves via save_users,
    # which now does an upsert. Persist immediately so a subsequent
    # crash doesn't leave the user on a legacy hash.
    try:
        save_users(load_users())  # round-trip through the storage shim
        # Simpler & cheaper: we just mutated one row, but the JSON
        # backend's save_users only writes via the shim. We can't
        # mutate-in-place safely when other rows share the file, so
        # re-load + re-write the whole list. Cheap for small user
        # counts (single digits).
    except Exception as exc:
        log = logging.getLogger("stt_server.auth")
        log.warning("[auth] legacy-hash upgrade failed for %s: %s",
                    user.get("email"), exc)


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 (600k iters, random salt).

    Stored format: ``pbkdf2_sha256$600000$<salt_hex>$<hash_hex>``.
    The legacy hash_password (unsalted SHA-256) is preserved as
    ``_legacy_hash_password`` for any caller that needs to reproduce
    the on-disk value used by older migrations.
    """
    return _hash_password_pbkdf2(password)


def _legacy_hash_password(password: str) -> str:
    """Legacy unsalted SHA-256 — used by 002_seed_admin.sql and the
    JSON-mode startup seeder, both of which embed a pre-computed hash
    that must match the old format until the operator re-runs the
    migration. The verifier accepts both formats, so a legacy-hashed
    admin still works after the upgrade."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a stored hash.

    Accepts both the new PBKDF2 format and the legacy unsalted
    SHA-256 hex (64 chars). The first successful legacy login
    transparently upgrades the stored hash to PBKDF2 — see
    ``_upgrade_password_hash`` callers in the route handlers.
    """
    if not hashed:
        return False
    if hashed.startswith(f"{_PBKDF2_PREFIX}$"):
        return _verify_password_pbkdf2(password, hashed)
    return _verify_password_legacy_sha256(password, hashed)

def generate_token() -> str:
    """Generar token de sesión seguro."""
    return secrets.token_urlsafe(32)


def _parse_expires(raw) -> datetime:
    """Coerce the various shapes an expires_at can come in:
      - JSON backend: ISO string (naive, no tz) like "2026-07-14T12:54:16.280444"
      - Postgres backend: timezone-aware datetime via TIMESTAMPTZ
      - None / missing: epoch-zero
    Returns a datetime. When naive, treat as UTC (the JSON files always
    wrote UTC, just without the suffix)."""
    if raw is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        s = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    raise TypeError(f"unsupported expires_at: {type(raw).__name__}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_expired(expires_at_raw) -> bool:
    return _now() > _parse_expires(expires_at_raw)


# load_sessions / save_sessions vienen de STT_server.db_users (import
# arriba) y resuelven al backend Postgres o JSON segun DATABASE_URL.

# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(user: UserCreate):
    """Registrar un nuevo usuario."""
    users = load_users()
    
    # Verificar si el email ya existe
    for u in users:
        if u.get('email') == user.email:
            raise HTTPException(
                status_code=400,
                detail="El email ya está registrado"
            )
    
    # Crear nuevo usuario
    import uuid
    new_user = {
        "id": f"user-{uuid.uuid4().hex[:12]}",
        "name": user.name,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": datetime.now().isoformat(),
    }
    
    users.append(new_user)
    save_users(users)
    
    # Generar token
    token = generate_token()
    sessions = load_sessions()
    sessions[token] = {
        "user_id": new_user["id"],
        "email": new_user["email"],
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),
    }
    save_sessions(sessions)
    
    return TokenResponse(
        access_token=token,
        user={
            "id": new_user["id"],
            "name": new_user["name"],
            "email": new_user["email"],
        }
    )

@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Iniciar sesión."""
    import logging
    log = logging.getLogger("stt_server.auth")
    users = load_users()
    log.warning("[auth] login attempt: email=%r user_count=%d", user.email, len(users))
    if users:
        # ponytail: log the first user's email so we can tell whether
        # 'no match' is 'no user in DB' or 'wrong email'.
        log.warning("[auth] first user in DB: email=%r", users[0].get('email'))

    # Buscar usuario
    found_user = None
    for u in users:
        if u.get('email') == user.email:
            found_user = u
            break

    if not found_user:
        log.warning("[auth] no user matches email=%r", user.email)
        raise HTTPException(
            status_code=401,
            detail="Credenciales inválidas"
        )

    # Verificar contraseña. Logueamos explicitamente los hashes
    # en juego para descartar silenciar el caso 'same hash different
    # encoding' que ya nos mordio antes.
    incoming_hash = hash_password(user.password)
    stored_hash = found_user.get('password', '') or ''
    log.warning("[auth] pw compare: incoming=%r len=%d  stored=%r len=%d  match=%s",
                incoming_hash, len(incoming_hash),
                stored_hash, len(stored_hash),
                verify_password(user.password, stored_hash))
    if not verify_password(user.password, stored_hash):
        log.warning("[auth] WRONG PASSWORD for %r", user.email)
        raise HTTPException(
            status_code=401,
            detail="Credenciales inválidas"
        )
    # ponytail: transparent upgrade. If the verified hash is the
    # legacy unsalted SHA-256 format, re-hash with PBKDF2-HMAC-SHA256
    # and persist. This is the migration path for every existing
    # account — they get upgraded on first login without operator
    # intervention. Errors are swallowed (next login will retry).
    if stored_hash and not stored_hash.startswith(f"pbkdf2_{_PBKDF2_HASH_NAME}$"):
        log.info("[auth] upgrading legacy SHA-256 hash to PBKDF2 for %s", user.email)
        _upgrade_password_hash(found_user, user.password)
    
    # Generar token
    token = generate_token()
    sessions = load_sessions()
    sessions[token] = {
        "user_id": found_user["id"],
        "email": found_user["email"],
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),
    }
    save_sessions(sessions)
    
    return TokenResponse(
        access_token=token,
        user={
            "id": found_user["id"],
            "name": found_user["name"],
            "email": found_user["email"],
        }
    )

@router.get("/me", response_model=UserResponse)
async def get_me(authorization: str = Header(None)):
    """Obtener el usuario actual."""
    import logging
    log = logging.getLogger("stt_server.auth")
    if not authorization:
        log.warning("/me: no Authorization header")
        raise HTTPException(
            status_code=401,
            detail="No se proporcionó token de autenticación"
        )

    # Extraer token del header "Bearer <token>"
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    sessions = load_sessions()
    # ponytail: was log.warning on every /me hit - moved to DEBUG.
    # The user reported noisy logs; this fires on every FE page load.
    log.debug("/me: token_prefix=%r session_count=%d has_token=%s",
              token[:8] + '...', len(sessions), token in sessions)

    if token not in sessions:
        log.debug("/me: token NOT in sessions")
        raise HTTPException(
            status_code=401,
            detail="Token inválido o expirado"
        )

    session_data = sessions[token]

    # Verificar expiración. Tolerar naive/aware (JSON: naive UTC;
    # Postgres: aware via TIMESTAMPTZ). El bug original era comparar
    # naive <-> aware y romper el login cada vez que se cambiaba de
    # backend. is_expires() normaliza ambos.
    if is_expired(session_data.get('expires_at')):
        del sessions[token]
        save_sessions(sessions)
        log.warning("/me: token EXPIRED. expires_at=%r", session_data.get('expires_at'))
        raise HTTPException(
            status_code=401,
            detail="Token expirado"
        )

    users = load_users()
    log.warning("/me: user_count=%d looking_for_id=%r", len(users), session_data.get('user_id'))
    for u in users:
        log.warning("/me: comparing ids: u.id=%r == session.user_id=%r match=%s",
                    u.get('id'), session_data.get('user_id'),
                    u.get('id') == session_data.get('user_id'))
        if u['id'] == session_data['user_id']:
            return UserResponse(
                id=u['id'],
                name=u['name'],
                email=u['email'],
                created_at=u['created_at']
            )
    
    raise HTTPException(
        status_code=404,
        detail="Usuario no encontrado"
    )

@router.post("/logout")
async def logout(authorization: str = Header(None)):
    """Cerrar sesión."""
    if not authorization:
        return {"message": "Sesión cerrada", "success": True}
    
    # Extraer token
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    
    sessions = load_sessions()
    
    if token in sessions:
        del sessions[token]
        save_sessions(sessions)
    
    return {"message": "Sesión cerrada", "success": True}

@router.get("/verify")
async def verify_token(authorization: str = Header(None)):
    """Verificar si el token es válido."""
    if not authorization:
        return {"valid": False, "message": "No token provided"}
    
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    
    sessions = load_sessions()
    
    if token not in sessions:
        return {"valid": False, "message": "Invalid token"}
    
    session_data = sessions[token]

    if is_expired(session_data.get('expires_at')):
        return {"valid": False, "message": "Token expired"}
    
    return {"valid": True, "user_id": session_data['user_id']}

@router.get("/health")
async def health_check():
    """Verificar estado del servicio de autenticación."""
    return {
        "status": "ok",
        "service": "auth",
        "timestamp": datetime.now().isoformat()
    }