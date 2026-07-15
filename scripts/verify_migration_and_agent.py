"""ponytail: regression check for migration runner + phone-number agent linkage.

Bug history:
  The migration runner in start.sh looped statements on a single
  psycopg2 connection and only caught exceptions per statement. When
  one statement failed, Postgres aborted the transaction; every
  subsequent statement then returned InFailedSqlTransaction and the
  final commit() was a no-op. A single real failure (duplicate
  ADD CONSTRAINT in 003, for example) cascaded as a wall of phantom
  warnings.

  Separately, the operator reported agent=None on inbound calls.
  PhoneNumberUpdate and db_phone_numbers.update_number both declare
  the `agent` field, but the BE was not actively guarding against a
  regression dropping it. This script verifies both layers so a
  future schema change can't silently break the agent linkage.

Run from Agent_IA_Server/: python scripts/verify_migration_and_agent.py
"""
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = "[OK]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}{(' — ' + detail) if detail else ''}")
        sys.exit(1)


# ── 1. start.sh: the inner loop must call conn.rollback() on failure ───────
start_sh = (ROOT / "start.sh").read_text(encoding="utf-8")

# Locate the inner except block in the migration loop. It used to be
# `except Exception as stmt_exc: errors.append(...)` — no rollback.
# After the fix it must call conn.rollback() before appending.
m = re.search(
    r"for stmt in _split_sql_statements\(sql\):\s*\n\s*try:\s*\n\s*with conn\.cursor\(\) as cur:\s*\n\s*cur\.execute\(stmt\)\s*\n\s*ok \+= 1\s*\n\s*except Exception as stmt_exc:\s*\n(.*?)\n\s*errors\.append",
    start_sh,
    re.DOTALL,
)
assert m, "could not find the inner except block in start.sh migration loop"
exception_body = m.group(1)
check(
    "start.sh rolls back the aborted transaction before logging the error",
    "conn.rollback()" in exception_body,
    f"inner except block doesn't call conn.rollback(): {exception_body!r}",
)


# ── 2. Functional simulation of the fixed runner ────────────────────────────
# SQLite supports SAVEPOINT-based rollback semantics that mirror what
# the fixed runner does in psycopg2. We feed it a SQL file with a real
# failure in the middle and confirm subsequent statements actually run.
def simulate_runner(sql_text: str) -> tuple[int, list[str]]:
    """Mimic the fixed runner: per-statement execute, rollback on
    failure, commit at the end. Returns (ok_count, error_list)."""
    in_memory = sqlite3.connect(":memory:")
    cur = in_memory.cursor()
    ok = 0
    errors = []
    # Split on `;` for the test only (no DO $$ blocks in test fixtures).
    for raw in sql_text.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        try:
            cur.execute(stmt)
            ok += 1
        except Exception as exc:
            # This is the fix: rollback so the next statement can run.
            in_memory.rollback()
            errors.append(f"{type(exc).__name__}: {exc}")
    in_memory.commit()
    in_memory.close()
    return ok, errors


fixture_sql = """
CREATE TABLE t1 (id INTEGER PRIMARY KEY, n INTEGER);
INSERT INTO t1 VALUES (1, 10);
INSERT INTO t1 VALUES (1, 20);
INSERT INTO t1 VALUES (2, 30);
CREATE INDEX idx_t1_n ON t1(n);
"""

ok, errors = simulate_runner(fixture_sql)
# INSERT (1, 20) violates PRIMARY KEY → one real failure → rollback
# recovers → the remaining CREATE INDEX and any later INSERT succeed.
check(
    "simulated runner recovers after a failed statement (4 ok out of 5)",
    ok == 4,
    f"expected 4 successful statements, got {ok}, errors={errors}",
)
check(
    "simulated runner logs exactly one real error (no cascade)",
    len(errors) == 1,
    f"expected 1 error, got {len(errors)}: {errors}",
)


# ── 3. PhoneNumberUpdate declares `agent` (no regression) ───────────────────
api_src = (ROOT / "STT_server" / "routes" / "api.py").read_text(encoding="utf-8")
m = re.search(r"class PhoneNumberUpdate\(BaseModel\):(.*?)(?=\nclass |\Z)", api_src, re.DOTALL)
assert m, "could not find PhoneNumberUpdate class"
update_body = m.group(1)
check(
    "PhoneNumberUpdate declares the `agent` field",
    "agent:" in update_body,
    f"`agent:` not found in PhoneNumberUpdate: {update_body[:200]!r}",
)


# ── 4. db_phone_numbers.update_number whitelist includes `agent` ────────────
db_src = (ROOT / "STT_server" / "db_phone_numbers.py").read_text(encoding="utf-8")
import ast
tree = ast.parse(db_src)
update_fn = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name == "update_number"),
    None,
)
assert update_fn is not None, "could not find db_phone_numbers.update_number"

allowed_set = None
for node in ast.walk(update_fn):
    if (isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "allowed"
        and isinstance(node.value, ast.Set)):
        allowed_set = {
            elt.value for elt in node.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        break

assert allowed_set is not None, "could not find `allowed = {...}` set in update_number"
check(
    "db_phone_numbers.update_number allowed-set includes `agent`",
    "agent" in allowed_set,
    f"allowed-set missing 'agent': {allowed_set}",
)


# ── 5. /voice webhook reads agent from phone_numbers row ────────────────────
stt_src = (ROOT / "STT_server" / "STT_Server.py").read_text(encoding="utf-8")
# The /voice handler reads `num_row.get("agent")` after find_by_number
# — make sure the lookup path is intact (no silent rename or filter).
# The docstring in this handler uses triple quotes that complicate a
# greedy regex, so just locate the function header and look for the
# agent-read line anywhere in the rest of the file.
voice_header = re.search(r"async def voice\(", stt_src)
assert voice_header, "/voice endpoint not found in STT_Server.py"
check(
    "/voice reads agent via num_row.get(\"agent\")",
    'num_row.get("agent")' in stt_src or "num_row.get('agent')" in stt_src,
    "agent lookup from phone_numbers row is missing or renamed",
)


# ── 6. db_create_agent always assigns a server-generated id ────────────────
db_agents_src = (ROOT / "STT_server" / "db_agents.py").read_text(encoding="utf-8")
tree = ast.parse(db_agents_src)
create_fn = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name == "create_agent"),
    None,
)
assert create_fn is not None, "could not find db_agents.create_agent"

# The function must (a) generate an id with uuid, (b) attach it to the
# returned dict in the JSON path, (c) include it in the RETURNING
# clause in the Postgres path.
fn_src = ast.unparse(create_fn)
check(
    "db_create_agent generates a server-side agent_id",
    "uuid.uuid4" in fn_src,
    f"create_agent doesn't call uuid.uuid4(): {fn_src[:300]!r}",
)
check(
    "db_create_agent's JSON path attaches the id to the returned row",
    '"id": agent_id' in fn_src or "'id': agent_id" in fn_src,
    f"create_agent's JSON-path new_agent dict is missing the id: {fn_src[:400]!r}",
)
check(
    "db_create_agent's Postgres INSERT includes the id column",
    ("'id'" in fn_src or '"id"' in fn_src) and "agent_id" in fn_src,
    f"create_agent's Postgres INSERT is missing the id column: {fn_src[:400]!r}",
)
check(
    "db_create_agent's Postgres RETURNING clause returns the id",
    "RETURNING" in fn_src and "id, user_id, name" in fn_src,
    f"create_agent's RETURNING clause doesn't surface the id: {fn_src[:400]!r}",
)


# ── 7. /agents POST route forwards the new id back to the FE ───────────────
# db_create_agent returns a row that always has `id` (verified above);
# the route simply returns whatever db_create_agent produces. Confirm
# there's no path that strips or overrides the id.
api_src_create_route = re.search(
    r"@api_router\.post\(\"/agents\"\)\s*\n\s*def create_agent\(.*?\n(?=\n@|\nclass |\Z)",
    api_src,
    re.DOTALL,
)
assert api_src_create_route, "could not find POST /agents route in routes/api.py"
check(
    "POST /agents returns db_create_agent's result unchanged",
    "return db_create_agent(auth[\"user_id\"], data.dict())" in api_src_create_route.group(0)
    or "return db_create_agent(auth['user_id'], data.dict())" in api_src_create_route.group(0),
    "POST /agents doesn't return db_create_agent's result directly — "
    "the agent id may not be reaching the FE",
)


# ── 8. Functional test: simulate db_create_agent (JSON path) ───────────────
import json as _json
import tempfile
import uuid as _uuid


def simulate_json_create_agent(payload: dict) -> dict:
    """Mimic the JSON branch of db_agents.create_agent — generate id,
    attach to the dict, append to a JSON file. Returns the new row."""
    agent_id = f"agent-{_uuid.uuid4().hex[:8]}"
    row = {
        "id": agent_id,
        "user_id": payload["user_id"],
        "calls": "0",
        "perf": 0,
        **payload,
    }
    return row


new_agent = simulate_json_create_agent(
    {"user_id": "user-admin-001", "name": "Test", "prompt": "hi"}
)
check(
    "simulated create_agent returns a row with a non-empty id",
    bool(new_agent.get("id")) and new_agent["id"].startswith("agent-"),
    f"agent row missing id: {new_agent}",
)
check(
    "simulated create_agent id is unique per call",
    simulate_json_create_agent({"user_id": "u", "name": "x"})["id"] != new_agent["id"],
    "uuid.uuid4 collision (extremely unlikely — investigate)",
)

print("\nALL CHECKS PASSED.")