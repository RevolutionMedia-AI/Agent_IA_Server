#!/bin/sh
# Default to 8080 when PORT is not set
PORT="${PORT:-8080}"
echo "Starting app on port $PORT"

# ponytail: self-heal the active auth backend + auto-apply migrations
# + backfill JSON to Postgres. All inline as a single heredoc so we
# don't depend on quoting tricks that can break under different shells
# (the previous version used `python -c "..."` with f-strings and
# produced a SyntaxError on every restart).
#
# Why heredoc and not python -c: the previous version embedded
# `$m` (the migration filename) and `$pg_url` (the DATABASE_URL)
# inside a double-quoted python -c string. The shell expanded them
# before handing the string to python, but inside the string we also
# had f-strings and single-quoted literals that confused the parser
# depending on whether $m contained special characters. The heredoc
# sidesteps all of that.
ADMIN_ID='user-admin-001'
ADMIN_EMAIL='admin@revolutionmedia.ai'
ADMIN_NAME='Revolution Media Admin'
ADMIN_ROLE='admin'
ADMIN_HASH=$(printf '%s' 'Adminrevolutionmedia@109' | python -c 'import sys, hashlib; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')

# Resolve DATABASE_URL (Railway sometimes splits across PG* vars).
if [ -z "$DATABASE_URL" ] && [ -n "$PGHOST" ] && [ -n "$PGUSER" ] && [ -n "$PGPASSWORD" ] && [ -n "$PGDATABASE" ]; then
    export DATABASE_URL="postgresql://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT:-5432}/${PGDATABASE}"
fi

# --- DB init script. Runs once at container start. ---
# We branch on DATABASE_URL: with one, we seed the admin in Postgres
# and apply migrations; without one, we fall back to writing
# users.json. Both branches are idempotent.
export ADMIN_ID ADMIN_EMAIL ADMIN_NAME ADMIN_ROLE ADMIN_HASH
python <<'PYEOF'
import os
ADMIN_ID    = os.environ['ADMIN_ID']
ADMIN_EMAIL = os.environ['ADMIN_EMAIL']
ADMIN_NAME  = os.environ['ADMIN_NAME']
ADMIN_ROLE  = os.environ['ADMIN_ROLE']
ADMIN_HASH  = os.environ['ADMIN_HASH']
PG_URL      = os.environ.get('DATABASE_URL', '')

if PG_URL:
    import psycopg2
    # 1. Seed the admin user (idempotent).
    try:
        conn = psycopg2.connect(PG_URL)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (id, name, email, password, role) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (ADMIN_ID, ADMIN_NAME, ADMIN_EMAIL, ADMIN_HASH, ADMIN_ROLE),
            )
            conn.commit()
        conn.close()
        print(f"[startup] admin ensured in Postgres ({ADMIN_EMAIL})")
    except Exception as exc:
        print(f"[startup] WARN: could not seed admin in Postgres: {exc}")

    # 2. Apply migrations from /app/db/migrations/*.sql in order.
    #    Each file is idempotent (CREATE TABLE IF NOT EXISTS /
    #    ALTER TABLE ADD COLUMN IF NOT EXISTS) so re-running is safe.
    #
    #    ponytail: C1 from the call-flow audit. psycopg2's
    #    cursor.execute() is single-statement by design (SQL
    #    injection mitigation), so naively running the whole
    #    migration file would only execute the FIRST statement and
    #    leave the rest of the schema uncreated on a fresh deploy.
    #    Split the file into statements and execute each one.
    def _split_sql_statements(sql_text):
        # ponytail: the C1 splitter cut on `;` at the top level, which
        # broke any DO $$ ... END $$; block (it split inside the
        # $$ ... $$ string and Postgres saw an unterminated
        # dollar-quoted string). Track dollar-quoted regions so a
        # `;` inside `$$ ... $$` doesn't end a statement. Also handles
        # `$_tag_$ ... $_tag_$` with custom tags. The closing tag must
        # match the opening tag (Postgres rule); the inner `;` is
        # just data.
        statements = []
        current = []
        # states: normal | line_comment | block_comment | sq | dq | dollar
        state = "normal"
        dollar_tag = None  # the active dollar-quote tag (None when not in one)
        i = 0
        while i < len(sql_text):
            c = sql_text[i]
            nxt = sql_text[i + 1] if i + 1 < len(sql_text) else ""
            if state == "line_comment":
                current.append(c)
                if c == "\n":
                    state = "normal"
            elif state == "block_comment":
                if c == "*" and nxt == "/":
                    current.append("*/")
                    i += 1
                    state = "normal"
                elif c != "\n":
                    current.append(c)
            elif state == "sq":
                if c == "'" and nxt == "'":
                    current.append("''")
                    i += 1
                elif c == "'":
                    state = "normal"
                    current.append(c)
                else:
                    current.append(c)
            elif state == "dq":
                if c == '"' and nxt == '"':
                    current.append('""')
                    i += 1
                elif c == '"':
                    state = "normal"
                    current.append(c)
                else:
                    current.append(c)
            elif state == "dollar":
                # Inside a $$ or $tag$ block. Postgres allows the
                # closing tag anywhere, but the only way the splitter
                # can know is to scan ahead for the same tag.
                if c == "$":
                    # Look ahead for the matching tag.
                    j = i + 1
                    while j < len(sql_text) and (sql_text[j].isalnum() or sql_text[j] == "_"):
                        j += 1
                    candidate = sql_text[i:j + 1]  # includes the trailing $
                    if candidate == dollar_tag:
                        current.append(candidate)
                        i = j
                        state = "normal"
                        dollar_tag = None
                        continue
                    # Not the closing tag: it's just data.
                    current.append(c)
                else:
                    current.append(c)
            else:  # normal
                if c == "-" and nxt == "-":
                    state = "line_comment"
                    current.append("--")
                    i += 1
                elif c == "/" and nxt == "*":
                    state = "block_comment"
                    current.append("/*")
                    i += 1
                elif c == "'":
                    state = "sq"
                    current.append(c)
                elif c == '"':
                    state = "dq"
                    current.append(c)
                elif c == "$" and nxt == "$":
                    # $$ starts a dollar-quoted block. No custom tag
                    # here, just two consecutive $s.
                    current.append("$$")
                    i += 1
                    state = "dollar"
                    dollar_tag = "$$"
                elif c == "$":
                    # $tag$ style dollar-quote.
                    j = i + 1
                    while j < len(sql_text) and (sql_text[j].isalnum() or sql_text[j] == "_"):
                        j += 1
                    if j < len(sql_text) and sql_text[j] == "$":
                        tag = sql_text[i:j + 1]
                        current.append(tag)
                        i = j
                        state = "dollar"
                        dollar_tag = tag
                    else:
                        # Lone $ — not a dollar-quote, just data.
                        current.append(c)
                elif c == ";":
                    stmt = "".join(current).strip()
                    if stmt:
                        statements.append(stmt)
                    current = []
                else:
                    current.append(c)
            i += 1
        last = "".join(current).strip()
        if last:
            statements.append(last)
        return statements

    import glob
    for m in sorted(glob.glob('/app/db/migrations/*.sql')):
        name = os.path.basename(m)
        print(f"[startup] applying migration {name}")
        try:
            with open(m, 'r') as f:
                sql = f.read()
            conn = psycopg2.connect(PG_URL)
            ok = 0
            errors = []
            for stmt in _split_sql_statements(sql):
                try:
                    with conn.cursor() as cur:
                        cur.execute(stmt)
                    ok += 1
                except Exception as stmt_exc:
                    errors.append(f"{type(stmt_exc).__name__}: {stmt_exc}")
            conn.commit()
            conn.close()
            if errors:
                print(f"  applied {ok} statement(s), {len(errors)} error(s):")
                for e in errors:
                    print(f"    WARN: {e}")
            else:
                print(f"  OK ({ok} statements)")
        except Exception as exc:
            print(f"  WARN: {exc}")

    # 3. Backfill JSON -> Postgres so existing local-dev data
    #    (agents, phone numbers, API keys, call usage) survives the
    #    first Postgres-backed deploy. Idempotent — skips rows that
    #    already exist.
    sys_path = '/app'
    if sys_path not in __import__('sys').path:
        __import__('sys').path.insert(0, sys_path)
    try:
        from STT_server.db_agents import backfill_from_json as a_backfill
        from STT_server.db_phone_numbers import backfill_from_json as n_backfill
        from STT_server.db_tools import backfill_from_json as t_backfill
        from STT_server.db_call_usage import backfill_from_json as u_backfill
        a = a_backfill(); n = n_backfill(); t = t_backfill(); u = u_backfill()
        if a or n or t or u:
            print(f"[startup] backfilled from JSON -> Postgres: agents={a} numbers={n} tools={t} usage={u}")
        else:
            print("[startup] JSON backfill: nothing to copy (empty or already migrated)")
    except Exception as exc:
        print(f"[startup] WARN: JSON backfill failed: {exc}")
else:
    # JSON backend. Self-heal users.json if missing.
    import json
    from pathlib import Path
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
        print('[startup] users.json missing or has no admin -> seeding default admin')
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
PYEOF

exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
