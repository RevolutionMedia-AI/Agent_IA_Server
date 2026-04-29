"""
Rutas de autenticación para el servidor STT.
Proporciona endpoints para registro, login, logout y gestión de usuarios.
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Body
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/auth", tags=["Autenticación"])

# ── Rutas de almacenamiento ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

# ── Modelos Pydantic ─────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
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
def ensure_data_dir():
    """Crear directorio de datos si no existe."""
    os.makedirs(DATA_DIR, exist_ok=True)

def load_users() -> list:
    """Cargar usuarios desde el archivo JSON."""
    ensure_data_dir()
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_users(users: list):
    """Guardar usuarios en el archivo JSON."""
    ensure_data_dir()
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    """Hashear contraseña con SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verificar contraseña."""
    return hash_password(password) == hashed

def generate_token() -> str:
    """Generar token de sesión seguro."""
    return secrets.token_urlsafe(32)

def load_sessions() -> dict:
    """Cargar sesiones desde el archivo JSON."""
    ensure_data_dir()
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_sessions(sessions: dict):
    """Guardar sesiones en el archivo JSON."""
    ensure_data_dir()
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

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
    
    # Verificar expiración
    expires_at = datetime.fromisoformat(session_data['expires_at'])
    if datetime.now() > expires_at:
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
    expires_at = datetime.fromisoformat(session_data['expires_at'])
    
    if datetime.now() > expires_at:
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