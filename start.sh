#!/bin/sh
set -e
# Default to 8080 when PORT is not set
PORT="${PORT:-8080}"
echo "Starting app on port $PORT"

# ponytail: self-heal users.json. The .dockerignore once excluded
# STT_server/data/ from the build context, which meant a fresh deploy
# shipped a container with no users.json at all -> load_users() returned
# [] -> /auth/login returned 401 for everyone. Even after the
# .dockerignore was fixed, Railway can cache the build context and
# keep shipping the broken image. So the runtime defends itself:
# if users.json is missing OR has no admin, seed the default.
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('/app/STT_server/data/users.json')
existing = []
if p.exists():
    try:
        existing = json.loads(p.read_text() or '[]')
    except Exception:
        existing = []
needs_seed = not existing or not any(
    (u.get('email', '').lower() == 'admin@revolutionmedia.ai') for u in existing
)
if needs_seed:
    print('[startup] users.json missing or has no admin -> seeding default admin')
    p.parent.mkdir(parents=True, exist_ok=True)
    admin = {
        'id': 'user-admin-001',
        'name': 'Revolution Media Admin',
        'email': 'admin@revolutionmedia.ai',
        'password': hashlib.sha256(b'Adminrevolutionmedia@109').hexdigest(),
        'role': 'admin',
        'created_at': '2026-07-09T00:00:00Z',
        'updated_at': '2026-07-09T00:00:00Z',
    }
    p.write_text(json.dumps([admin], indent=2))
    print('[startup] admin seeded:', admin['email'])
else:
    print(f"[startup] users.json OK ({len(existing)} user(s), admin present)")
PY

exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
