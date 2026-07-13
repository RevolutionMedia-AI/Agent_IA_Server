#!/bin/sh
set -e
# Default to 8080 when PORT is not set
PORT="${PORT:-8080}"
echo "Starting app on port $PORT"

# ponytail: self-heal the active auth backend. The .dockerignore once
# excluded STT_server/data/ from the build context, which meant a
# fresh deploy shipped a container with no users.json at all ->
# load_users() returned [] -> /auth/login returned 401 for everyone.
# Even after the .dockerignore was fixed, Railway can cache the build
# context and keep shipping the broken image.
#
# This runtime now seeds WHATEVER backend the BE is about to use:
#   - Postgres (DATABASE_URL set): INSERT ... ON CONFLICT DO NOTHING
#   - JSON (no DATABASE_URL): write users.json with the default admin
#
# Both branches are idempotent — running on every restart doesn't
# duplicate or break data.
python - <<'PY'
import hashlib, json, os
from pathlib import Path

ADMIN_ID    = 'user-admin-001'
ADMIN_EMAIL = 'admin@revolutionmedia.ai'
ADMIN_NAME  = 'Revolution Media Admin'
ADMIN_ROLE  = 'admin'
ADMIN_HASH  = hashlib.sha256(b'Adminrevolutionmedia@109').hexdigest()

pg_url = os.environ.get('DATABASE_URL') or ''
# Railway also splits the URL across PG* vars. Fall back if so.
if not pg_url:
    parts = {k: os.environ.get(k) for k in ('PGHOST','PGPORT','PGUSER','PGPASSWORD','PGDATABASE')}
    if all(parts.values()):
        pg_url = f"postgresql://{parts['PGUSER']}:{parts['PGPASSWORD']}@{parts['PGHOST']}:{parts.get('PGPORT','5432')}/{parts['PGDATABASE']}"

if pg_url:
    # Postgres backend. Insert the admin if missing. ON CONFLICT
    # DO NOTHING means re-running on every container start is safe.
    try:
        import psycopg2
        conn = psycopg2.connect(pg_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (id, name, email, password, role) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (ADMIN_ID, ADMIN_NAME, ADMIN_EMAIL, ADMIN_HASH, ADMIN_ROLE),
                )
                conn.commit()
            print(f"[startup] admin ensured in Postgres ({ADMIN_EMAIL})")
        finally:
            conn.close()
    except Exception as exc:
        # Don't fail the container if the DB is down — the user
        # can run the migration manually. Log clearly so the cause
        # is visible in Railway logs.
        print(f"[startup] WARN: could not seed admin in Postgres: {exc}")

    # ponytail: auto-apply the schema extensions on every start.
    # 001_schema.sql + 004_extend_business_tables.sql are idempotent
    # (CREATE TABLE IF NOT EXISTS / ALTER TABLE ADD COLUMN IF NOT
    # EXISTS) so re-running is safe. Without this the FE would write
    # stt_provider/tts_model/welcome_message into columns that don't
    # exist yet, and Postgres would reject the INSERT.
    if [ -d /app/db/migrations ]; then
        for m in /app/db/migrations/*.sql; do
            echo "[startup] applying migration $(basename $m)"
            python -c "
import sys
with open('$m', 'r') as f:
    sql = f.read()
try:
    import psycopg2
    c = psycopg2.connect('$pg_url')
    with c.cursor() as cur:
        cur.execute(sql)
    c.commit()
    c.close()
    print('  OK')
except Exception as e:
    print(f'  WARN: {e}')
" || true
        done
    fi

    # ponytail: backfill JSON -> Postgres on first boot so existing
    # local-dev data (agents, numbers, API keys) survives the first
    # Postgres-backed deploy. Idempotent - skips rows already present.
    python -c "
import sys
sys.path.insert(0, '/app')
try:
    from STT_server.db_agents import backfill_from_json as a_backfill
    from STT_server.db_phone_numbers import backfill_from_json as n_backfill
    from STT_server.db_tools import backfill_from_json as t_backfill
    a = a_backfill(); n = n_backfill(); t = t_backfill()
    if a or n or t:
        print(f'[startup] backfilled from JSON -> Postgres: agents={a} numbers={n} tools={t}')
    else:
        print('[startup] JSON backfill: nothing to copy (empty or already migrated)')
except Exception as e:
    print(f'[startup] WARN: JSON backfill failed: {e}')
" || true
else:
    # JSON backend. Same self-heal as before.
    p = Path('/app/STT_server/data/users.json')
    existing = []
    if p.exists():
        try:
            existing = json.loads(p.read_text() or '[]')
        except Exception:
            existing = []
    needs_seed = not existing or not any(
        (u.get('email', '').lower() == ADMIN_EMAIL) for u in existing
    )
    if needs_seed:
        print(f'[startup] users.json missing or has no admin -> seeding default admin')
        p.parent.mkdir(parents=True, exist_ok=True)
        admin = {
            'id': ADMIN_ID,
            'name': ADMIN_NAME,
            'email': ADMIN_EMAIL,
            'password': ADMIN_HASH,
            'role': ADMIN_ROLE,
            'created_at': '2026-07-09T00:00:00Z',
            'updated_at': '2026-07-09T00:00:00Z',
        }
        p.write_text(json.dumps([admin], indent=2))
        print(f"[startup] admin seeded: {ADMIN_EMAIL}")
    else:
        print(f"[startup] users.json OK ({len(existing)} user(s), admin present)")
PY

exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
