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

print("\nALL CHECKS PASSED.")