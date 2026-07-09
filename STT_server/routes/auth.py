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
    users = load_users()
    
    # Buscar usuario
    found_user = None
    for u in users:
        if u.get('email') == user.email:
            found_user = u
            break
    
    if not found_user:
        raise HTTPException(
            status_code=401,
            detail="Credenciales inválidas"
        )
    
    # Verificar contraseña
    if not verify_password(user.password, found_user.get('password', '')):
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