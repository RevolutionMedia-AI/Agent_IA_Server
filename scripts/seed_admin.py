"""
Seed admin: crea (o actualiza) el usuario administrador inicial en
STT_server/data/users.json.

Reutiliza la misma logica de hashing que STT_server.routes.auth
(SHA-256) para que /auth/login acepte las credenciales resultantes.

Uso:
    .venv\\Scripts\\python.exe scripts\\seed_admin.py
"""
import os
import sys
import json
import hashlib
import uuid
from datetime import datetime

# Raiz del proyecto (un nivel arriba de scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "STT_server", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

ADMIN_EMAIL = "admin@revolutionmedia.ai"
ADMIN_PASSWORD = "Adminrevolutionmedia@109"
ADMIN_NAME = "Revolution Media Admin"
ADMIN_ROLE = "admin"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except (json.JSONDecodeError, IOError):
            users = []
    else:
        users = []

    existing = next(
        (u for u in users if u.get("email", "").lower() == ADMIN_EMAIL.lower()),
        None,
    )

    if existing:
        existing["name"] = ADMIN_NAME
        existing["password"] = hash_password(ADMIN_PASSWORD)
        existing["role"] = ADMIN_ROLE
        existing["updated_at"] = datetime.now().isoformat()
        action = "updated"
        user = existing
    else:
        user = {
            "id": f"user-{uuid.uuid4().hex[:12]}",
            "name": ADMIN_NAME,
            "email": ADMIN_EMAIL,
            "password": hash_password(ADMIN_PASSWORD),
            "role": ADMIN_ROLE,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        users.append(user)
        action = "created"

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    print(f"[seed_admin] Admin user {action}:")
    print(f"  id    : {user['id']}")
    print(f"  email : {user['email']}")
    print(f"  role  : {user.get('role')}")
    print(f"  file  : {USERS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
