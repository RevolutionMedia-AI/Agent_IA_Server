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
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Body
from pydantic import BaseModel
from fastapi.responses import JSONResponse

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

def hash_password(password: str) -> str:
    """Hashear contraseña con SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verificar contraseña."""
    return hash_password(password) == hashed

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
    if not authorization:
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
    
    if token not in sessions:
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
        raise HTTPException(
            status_code=401,
            detail="Token expirado"
        )
    
    users = load_users()
    for u in users:
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


@router.get("/debug/auth-diag")
async def debug_auth_diag():
    """ponytail: one-shot diagnostic for the 401 the user is hitting.
    Reads the admin row from whatever backend is active, prints the
    stored hash, the hash the BE would compute, and whether they
    match. Plus the users table columns. Wire to a curl so the user
    can see the diagnostic without SSH access into the container.

    Remove this endpoint once the auth issue is resolved.
    """
    import logging
    log = logging.getLogger("stt_server.diag")
    out = {"stage": "ok"}
    try:
        from STT_server.db_users import load_users
        users = load_users()
        out["user_count"] = len(users)
        if users:
            out["first_user"] = {
                "id": users[0].get("id"),
                "email": users[0].get("email"),
                "role": users[0].get("role"),
                "password_hash_len": len(users[0].get("password", "") or ""),
            }
        target_user = next(
            (u for u in users if (u.get("email") or "").lower() == "admin@revolutionmedia.ai"),
            None,
        )
        if target_user is None:
            out["admin_row"] = "NOT_FOUND"
        else:
            stored = target_user.get("password") or ""
            out["admin_row"] = {
                "id": target_user.get("id"),
                "email": target_user.get("email"),
                "role": target_user.get("role"),
                "stored_hash_first_8": stored[:8] if stored else "(empty)",
                "stored_hash_len": len(stored),
            }
        from STT_server.db import is_postgres
        if is_postgres():
            from STT_server.db import get_conn
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_name = 'users' ORDER BY ordinal_position"
                    )
                    out["users_columns"] = [
                        {"name": r["column_name"], "type": r["data_type"]}
                        for r in cur.fetchall()
                    ]
        import hashlib
        out["sha256_of_Adminrevolutionmedia@109"] = hashlib.sha256(b"Adminrevolutionmedia@109").hexdigest()
        if target_user:
            out["hash_matches"] = (target_user.get("password") or "") == out["sha256_of_Adminrevolutionmedia@109"]
    except Exception as exc:
        out["stage"] = "error"
        out["error"] = repr(exc)
    return out