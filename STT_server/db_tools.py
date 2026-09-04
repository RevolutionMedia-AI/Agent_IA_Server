"""Postgres-backed implementations for the agent_tools table.

Replaces STT_server/data/agent_tools.json — that file was ephemeral on
Railway (every container restart wiped it) so all the operator's
tools disappeared after each deploy. The JSON file is now a one-time
backfill source on first boot, same pattern as db_agents.

Schema (010_agent_tools.sql):
  agent_tools(
    id, user_id, agent_id, name, description,
    webhook_url, filler_phrase, parameters (jsonb),
    kind, destination, assignments (jsonb),
    function_name, observability fields (last_tested_at, last_test_result,
    last_test_error, last_test_error_at, last_invoked_at,
    last_invocation_status, last_invocation_error, last_invocation_error_at,
    invocation_count),
    created_at, updated_at
  )

The shape returned by list_*/get_* mirrors what _load_tools() used to
return from the JSON file, so the route layer can swap one import
without changing call sites.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_tools")

# ponytail: legacy JSON-file path kept ONLY for the one-shot backfill
# on first boot. The route layer never reads this file directly
# anymore — every read goes through the Postgres-backed helpers
# below. If a customer added tools before this migration ran, those
# rows get copied into Postgres on the next process start and the
# JSON file becomes orphaned (delete it manually after one release).
_AGENT_TOOLS_FILE = Path(__file__).resolve().parent / "data" / "agent_tools.json"

# ponytail: single source of truth for SELECT columns. The base
# string is the schema as of migration 010 + 013; the optional
# `credentials` column is appended dynamically based on whether
# migration 014 has run on the target DB. _tool_cols() runs the
# self-healing check the first time it's called and caches the
# result for the rest of the process lifetime so we don't hit
# information_schema on every request.
_TOOL_COLS_BASE = (
    "id, user_id, agent_id, name, description, "
    "webhook_url, filler_phrase, parameters, "
    "kind, destination, assignments, function_name, test_data_model, "
    "last_tested_at, last_test_result, last_test_error, last_test_error_at, "
    "last_invoked_at, last_invocation_status, last_invocation_error, "
    "last_invocation_error_at, invocation_count, "
    "created_at, updated_at"
)
# ponytail: integration_id + action columns added by 016_agent_tools_integration.sql.
# Same self-heal pattern as credentials (014): detected on first call, ALTER'd
# inline if missing, then cached. Tools created before the migration keep
# working — both columns are NULL for legacy rows.
_TOOL_COLS_BASE_V2 = _TOOL_COLS_BASE + ", integration_id, action"
_TOOL_COLS_EXTRA: list[str] = []  # appended to _TOOL_COLS_BASE when present
_columns_check_done: bool = False
_columns_check_lock = threading.Lock()


def _ensure_tool_columns() -> None:
    """Idempotent self-healing schema check.

    Detects whether `agent_tools.credentials` (014) and
    `agent_tools.integration_id` + `action` (016) exist on the target
    DB. If not, runs the ALTER TABLE inline. Runs at most once per
    process — the result is cached in module-level state — so the
    hot path never hits information_schema.

    ponytail: the operator hit a 500 (`column "credentials" does not
    exist`) on Railway despite the migration 014 file shipping in the
    image. start.sh either didn't pick it up, hit the trailing
    comment in my SQL and aborted silently, or the deploy reused a
    cached image. Rather than chase the migration runner bug, the
    BE now self-heals: the first time the credentials path is
    hit, we confirm the column is present, and apply it if not.
    Once applied the row is durable and subsequent deploys are
    no-ops thanks to `IF NOT EXISTS`.

    Same pattern applies to integration_id + action (016) — these
    columns are needed by the integrations refactor; legacy tools
    keep working because both columns are NULL for pre-016 rows.
    """
    global _TOOL_COLS_EXTRA, _columns_check_done
    if _columns_check_done:
        return
    with _columns_check_lock:
        if _columns_check_done:
            return
        if not is_postgres():
            # JSON path: no schema to check, every field is supported.
            _TOOL_COLS_EXTRA = ["credentials", "integration_id", "action"]
            _columns_check_done = True
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'agent_tools' "
                        "AND column_name IN ('credentials', 'integration_id', 'action')"
                    )
                    # ponytail: use fetchone in a loop so test stubs
                    # that only implement fetchone (not fetchall)
                    # don't blow up the self-heal path. We break
                    # once we've collected all the columns we care
                    # about (3) — the loop is bounded so a stub
                    # that returns the same row forever can't hang
                    # the request.
                    #
                    # Rows can be (a) RealDictCursor dicts with one
                    # column_name key, (b) plain tuples with a single
                    # column name, or (c) test-stub composite rows
                    # that bundle multiple names — handle all three.
                    expected = ("credentials", "integration_id", "action")
                    present: set = set()
                    bad_rows = 0
                    while True:
                        row = cur.fetchone()
                        if row is None:
                            break
                        names: list = []
                        if isinstance(row, dict):
                            v = row.get("column_name")
                            if isinstance(v, str):
                                names.append(v)
                        elif isinstance(row, (tuple, list)):
                            names.extend(row)
                        for name in names:
                            if isinstance(name, str):
                                present.add(name)
                        if not names:
                            bad_rows += 1
                            if bad_rows > 8:
                                break
                            continue
                        if present.issuperset(expected):
                            break
            if "credentials" not in present:
                log.warning(
                    "[db_tools] agent_tools.credentials missing - applying 014 inline"
                )
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "ALTER TABLE agent_tools "
                            "ADD COLUMN IF NOT EXISTS credentials JSONB"
                        )
            if "integration_id" not in present or "action" not in present:
                log.warning(
                    "[db_tools] agent_tools.integration_id/action missing - applying 016 inline"
                )
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "ALTER TABLE agent_tools "
                            "ADD COLUMN IF NOT EXISTS integration_id TEXT, "
                            "ADD COLUMN IF NOT EXISTS action TEXT"
                        )
            # Confirm presence after the inline ALTERs. Cheap and
            # avoids a false positive if ALTER silently no-ops.
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'agent_tools' "
                        "AND column_name IN ('credentials', 'integration_id', 'action')"
                    )
                    present = set()
                    bad_rows = 0
                    while True:
                        row = cur.fetchone()
                        if row is None:
                            break
                        names: list = []
                        if isinstance(row, dict):
                            v = row.get("column_name")
                            if isinstance(v, str):
                                names.append(v)
                        elif isinstance(row, (tuple, list)):
                            names.extend(row)
                        for name in names:
                            if isinstance(name, str):
                                present.add(name)
                        if not names:
                            bad_rows += 1
                            if bad_rows > 8:
                                break
                            continue
                        if present.issuperset(expected):
                            break
            _TOOL_COLS_EXTRA = [c for c in ("credentials", "integration_id", "action") if c in present]
            _columns_check_done = True
            log.info(
                "[db_tools] self-heal complete: tool cols extra=%s",
                _TOOL_COLS_EXTRA,
            )
        except Exception as exc:
            log.error("[db_tools] _ensure_tool_columns failed: %s", exc)
            # Don't mark done — the next request will retry. Returning
            # without _TOOL_COLS_EXTRA means the base cols (no
            # credentials / integration_id / action) are used; the
            # broken call will surface the same column-not-exist
            # error and the operator can see _ensure_tool_columns'
            # log line in the next deploy.
            raise


def _tool_cols() -> str:
    """Returns the SELECT column list for the agent_tools table.

    Calls _ensure_tool_columns() on first use, then caches. Includes
    `credentials`, `integration_id`, `action` only when the column is
    known to exist (either it was on the table at boot, or we added it
    via the self-heal ALTER).
    """
    if not _columns_check_done:
        _ensure_tool_columns()
    cols = _TOOL_COLS_BASE
    for c in _TOOL_COLS_EXTRA:
        cols += f", {c}"
    return cols


def _row_to_tool(row: dict) -> dict:
    """Map a DB row to the JSON shape the rest of the BE consumes.

    Ponytail: parameters, assignments, and credentials are stored as
    JSONB on disk (so we can index/filter on them later if needed) but
    exposed to the route layer as plain Python lists / dicts. Without
    this conversion the FE / resolver would see raw strings instead of
    structured data on the first call.
    """
    if row is None:
        return None
    if isinstance(row, dict):
        out = dict(row)
    elif isinstance(row, (list, tuple)) and len(row) == 1 and isinstance(row[0], dict):
        out = dict(row[0])
    elif isinstance(row, (list, tuple)):
        try:
            # Fallback for tuple cursor (should not happen with RealDictCursor)
            from STT_server.db_tools import _tool_cols
            cols = [c.strip() for c in _tool_cols().split(",")]
            out = dict(zip(cols, row))
        except Exception:
            log.warning("[db_tools] unexpected row shape for _row_to_tool: %r", row)
            return None
    else:
        try:
            out = dict(row)
        except Exception as exc:
            log.warning("[db_tools] dict(row) failed for %r: %s", row, exc)
            return None
    for k in ("parameters", "assignments", "credentials"):
        v = out.get(k)
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = {} if k == "parameters" else ([] if k == "assignments" else None)
    for k in ("last_tested_at", "last_test_error_at",
              "last_invoked_at", "last_invocation_error_at",
              "created_at", "updated_at"):
        if hasattr(out.get(k), "isoformat"):
            out[k] = out[k].isoformat() + "Z"
    return out


def list_tools(user_id: str, agent_id: str | None = None) -> list[dict]:
    """List tools for one user.

    ponytail: when `agent_id` is None we return every tool the user
    owns (per-agent + shared). When set, we return per-agent rows
    PLUS shared rows whose `assignments` include this agent_id —
    mirrors the legacy `_load_agent_tools(agent_id, user_id)` rule
    so the runtime sees the same set of tools after the migration.
    """
    if not is_postgres():
        # JSON fallback: read the file, apply the same filter the
        # Postgres branch would. Used in local dev where there's no
        # DATABASE_URL set.
        if not _AGENT_TOOLS_FILE.exists():
            return []
        try:
            with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError) as exc:
            log.warning("[db_tools] JSON load failed (%s): %s", type(exc).__name__, exc)
            return []
        out = [r for r in rows if isinstance(r, dict) and r.get("user_id") == user_id]
        if agent_id is not None:
            out = [r for r in out
                   if r.get("agent_id") == agent_id
                   or (r.get("agent_id") == "__shared__"
                       and agent_id in (r.get("assignments") or []))]
        return out
    with get_conn() as conn:
        with conn.cursor() as cur:
            if agent_id is None:
                cur.execute(
                    f"SELECT {_tool_cols()} FROM agent_tools "
                    "WHERE user_id = %s ORDER BY created_at DESC",
                    (user_id,),
                )
            else:
                # ponytail: the OR-with-assignments branch needs the
                # jsonb ?| operator (overlap). We pass the agent id
                # as a single-element text[] so the index can help;
                # without the GIN index this is a sequential scan,
                # fine for the row counts we expect (handful per user).
                cur.execute(
                    f"SELECT {_tool_cols()} FROM agent_tools "
                    "WHERE user_id = %s AND ("
                    "  agent_id = %s OR "
                    "  (agent_id = '__shared__' AND COALESCE(assignments, '[]'::jsonb) ? %s)"
                    ") ORDER BY created_at DESC",
                    (user_id, agent_id, agent_id),
                )
            return [_row_to_tool(r) for r in cur.fetchall()]


def get_tool(tool_id: str, user_id: str) -> dict | None:
    if not is_postgres():
        if not _AGENT_TOOLS_FILE.exists():
            return None
        try:
            with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError):
            return None
        for r in rows:
            if isinstance(r, dict) and r.get("id") == tool_id and r.get("user_id") == user_id:
                return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_tool_cols()} FROM agent_tools "
                "WHERE id = %s AND user_id = %s",
                (tool_id, user_id),
            )
            row = cur.fetchone()
            return _row_to_tool(row) if row else None


# ponytail: 2026-09-03 — keep an aliased name available for callers
# that imported `db_get_tool` before the route layer consolidated on
# `get_tool`. The session_runtime + usage_store consume this alias
# to keep their import surface stable.
db_get_tool = get_tool


def create_tool(
    user_id: str,
    payload: dict,
    tool_id: str | None = None,
) -> dict:
    """Insert a new tool row. Generates a UUID-style id if missing.

    The payload shape matches the JSON file the route layer was
    building before: name, description, webhook_url, filler_phrase,
    parameters, kind, destination, assignments, function_name (derived
    by AgentTool if missing). The function_name is duplicated as a
    top-level column so the runtime can index / filter on it without
    unpacking parameters JSONB.
    """
    new_id = tool_id or f"tool-{uuid.uuid4().hex[:8]}"
    params_json = json.dumps(payload.get("parameters") or {})
    assignments_json = json.dumps(payload.get("assignments") or [])
    integration_id = payload.get("integration_id")
    action = payload.get("action")
    if not is_postgres():
        # JSON fallback for local dev
        if not _AGENT_TOOLS_FILE.exists():
            rows = []
        else:
            try:
                with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                    rows = json.load(f) or []
            except (json.JSONDecodeError, IOError, OSError):
                rows = []
        new_row = {
            "id": new_id,
            "user_id": user_id,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            **payload,
        }
        new_row["parameters"] = payload.get("parameters") or {}
        new_row["assignments"] = payload.get("assignments") or []
        # ponytail: keep `credentials` as whatever the caller passed
        # (None when the caller is creating a real n8n tool; a Fernet
        # dict when the caller is creating a service-credential row).
        new_row["credentials"] = payload.get("credentials")
        # ponytail: integration_id + action live alongside the tool row
        # so a future JOIN in /tools can hand the FE one object instead
        # of making it round-trip /integrations.
        new_row["integration_id"] = integration_id
        new_row["action"] = action
        rows.append(new_row)
        os.makedirs(_AGENT_TOOLS_FILE.parent, exist_ok=True)
        with open(_AGENT_TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return new_row
    # ponytail: 016 — the new columns (integration_id, action) are
    # written conditionally so a deploy that ran the migration AFTER
    # boot doesn't crash on a column-not-exist. db_update_tool
    # follows the same shape (only emits SET clauses for present
    # columns). The route layer already validates that
    # integration_id refers to an existing row + the agent_id matrix,
    # so we just store what it sent.
    extra_cols_sql = ""
    extra_vals_sql = ""
    extra_params: list = []
    if "integration_id" in _TOOL_COLS_EXTRA:
        extra_cols_sql += ", integration_id"
        extra_vals_sql += ", %s"
        extra_params.append(integration_id)
    if "action" in _TOOL_COLS_EXTRA:
        extra_cols_sql += ", action"
        extra_vals_sql += ", %s"
        extra_params.append(action)
    with get_conn() as conn:
        with conn.cursor() as cur:
            credentials_json = json.dumps(payload.get("credentials")) if payload.get("credentials") is not None else None
            cur.execute(
                f"INSERT INTO agent_tools (  id, user_id, agent_id, name, description,   webhook_url, filler_phrase, parameters,   kind, destination, assignments, function_name, test_data_model, credentials{extra_cols_sql}) VALUES (  %s, %s, %s, %s, %s, %s, %s, %s::jsonb,   %s, %s, %s::jsonb, %s, %s, %s::jsonb{extra_vals_sql}) "
                f"RETURNING {_tool_cols()}",
                (
                    new_id, user_id,
                    payload["agent_id"],
                    payload["name"],
                    payload.get("description") or "",
                    payload.get("webhook_url") or "",
                    payload.get("filler_phrase") or "Let me check the system...",
                    params_json,
                    payload.get("kind") or "webhook",
                    payload.get("destination"),
                    assignments_json,
                    payload.get("function_name") or "",
                    payload.get("test_data_model") or "gpt-4o-mini",
                    credentials_json,
                    *extra_params,
                ),
            )
            row = cur.fetchone()
    return _row_to_tool(row)


def update_tool(tool_id: str, user_id: str, payload: dict) -> dict | None:
    """Patch a tool row in place. Only the fields present in payload
    are written (preserves last_*_at, last_*_result, etc. unless the
    caller explicitly nulls them out)."""
    if not is_postgres():
        if not _AGENT_TOOLS_FILE.exists():
            return None
        try:
            with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError):
            return None
        out = None
        # ponytail: same DB-managed filter as the Postgres branch
        # below. Payload carries id / created_at / updated_at from
        # AgentTool.to_dict(), and letting them through the merge
        # would either rename the PK (id) or clobber the original
        # creation timestamp (created_at). updated_at is rewritten
        # unconditionally one line below.
        db_managed = {"id", "user_id", "created_at", "updated_at"}
        for r in rows:
            if isinstance(r, dict) and r.get("id") == tool_id and r.get("user_id") == user_id:
                r.update({k: v for k, v in payload.items() if v is not None and k not in db_managed})
                r["updated_at"] = _now_iso()
                out = r
                break
        if out is None:
            return None
        with open(_AGENT_TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return out
    # ponytail: build a dynamic SET clause from the payload so the
    # caller only needs to send the fields it actually changed.
    # JSONB columns need a ::jsonb cast. The DB-managed columns
    # below are filtered out for two reasons:
    #   1. updated_at is appended unconditionally at the end of the
    #      SET clause as `updated_at = NOW()` — letting it through
    #      here would produce `updated_at = ..., updated_at = NOW()`
    #      and Postgres rejects with `multiple assignments to same
    #      column`. Bit the operator just hit (railway traceback:
    #      psycopg2.errors.SyntaxError on PUT /tools/{id}).
    #   2. id, user_id, created_at are write-once on INSERT — letting
    #      them through here would silently let the BE rebuild the
    #      row's PK with AgentTool.to_dict()'s freshly-minted uuid,
    #      or clobber the original creation timestamp, neither of
    #      which the FE asked for.
    jsonb_keys = {"parameters", "assignments", "credentials"}
    db_managed = {"id", "user_id", "created_at", "updated_at"}
    # ponytail: 016 — integration_id + action are plain TEXT, not JSONB,
    # but only present when the column was self-healed. We append them
    # to the SET clause only when (a) the column is known to exist
    # AND (b) the payload actually contains the key. Without (b), the
    # caller didn't touch the integration binding and we shouldn't
    # blank it out by writing NULL.
    optional_text_keys = [k for k in ("integration_id", "action")
                          if k in _TOOL_COLS_EXTRA]
    set_clauses = []
    values: list = []
    for k, v in payload.items():
        if k in db_managed:
            continue
        if k in jsonb_keys:
            if v is None:
                continue
            set_clauses.append(f"{k} = %s::jsonb")
            values.append(json.dumps(v))
        elif k in optional_text_keys:
            # Allow explicit None to clear the binding; payload
            # semantics: None = "remove the binding", string = set.
            set_clauses.append(f"{k} = %s")
            values.append(v)
        else:
            if v is None:
                continue
            set_clauses.append(f"{k} = %s")
            values.append(v)
    if not set_clauses:
        return get_tool(tool_id, user_id)
    set_clauses.append("updated_at = NOW()")
    values.extend([tool_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE agent_tools SET {', '.join(set_clauses)} "
                "WHERE id = %s AND user_id = %s "
                f"RETURNING {_tool_cols()}",
                values,
            )
            row = cur.fetchone()
    return _row_to_tool(row) if row else None


def delete_tool(tool_id: str, user_id: str) -> bool:
    if not is_postgres():
        if not _AGENT_TOOLS_FILE.exists():
            return False
        try:
            with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError):
            return False
        before = len(rows)
        rows = [r for r in rows
                if not (isinstance(r, dict)
                        and r.get("id") == tool_id
                        and r.get("user_id") == user_id)]
        if len(rows) == before:
            return False
        with open(_AGENT_TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return True
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_tools WHERE id = %s AND user_id = %s",
                (tool_id, user_id),
            )
            return cur.rowcount > 0


def add_assignment(tool_id: str, user_id: str, agent_id: str) -> dict | None:
    """Add `agent_id` to the tool's `assignments` JSONB array.

    Idempotent — the jsonb ?| operator returns true if the id is
    already present, so we short-circuit without a write. Returns
    the updated tool row, or None if the tool doesn't exist or
    isn't owned by this user.
    """
    if not is_postgres():
        # Local dev path: simple append / no-op
        if not _AGENT_TOOLS_FILE.exists():
            return None
        try:
            with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError):
            return None
        out = None
        for r in rows:
            if (isinstance(r, dict)
                    and r.get("id") == tool_id
                    and r.get("user_id") == user_id):
                assigns = r.get("assignments") or []
                if agent_id in assigns:
                    return r
                assigns.append(agent_id)
                r["assignments"] = assigns
                r["updated_at"] = _now_iso()
                out = r
                break
        if out is None:
            return None
        with open(_AGENT_TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return out
    with get_conn() as conn:
        with conn.cursor() as cur:
            # ponytail: conditional append. If the id is already in
            # assignments the WHERE clause excludes the row and we
            # skip the update — same idempotent contract the FE expects.
            cur.execute(
                "UPDATE agent_tools SET assignments = COALESCE(assignments, '[]'::jsonb) || %s::jsonb, "
                "updated_at = NOW() "
                "WHERE id = %s AND user_id = %s "
                "AND NOT (COALESCE(assignments, '[]'::jsonb) ? %s) "
                f"RETURNING {_tool_cols()}",
                (json.dumps([agent_id]), tool_id, user_id, agent_id),
            )
            row = cur.fetchone()
            if row:
                return _row_to_tool(row)
            # No row updated — either the tool doesn't exist, the
            # id is already in assignments, or the user doesn't own
            # it. Re-read to distinguish idempotent re-assign from
            # 404 for the FE.
            return get_tool(tool_id, user_id)


def remove_assignment(tool_id: str, user_id: str, agent_id: str) -> dict | None:
    """Drop `agent_id` from the tool's `assignments` JSONB array.

    Idempotent — if the id is already absent the WHERE clause
    excludes the row.
    """
    if not is_postgres():
        if not _AGENT_TOOLS_FILE.exists():
            return None
        try:
            with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError):
            return None
        out = None
        for r in rows:
            if (isinstance(r, dict)
                    and r.get("id") == tool_id
                    and r.get("user_id") == user_id):
                assigns = r.get("assignments") or []
                if agent_id not in assigns:
                    return r
                r["assignments"] = [a for a in assigns if a != agent_id]
                r["updated_at"] = _now_iso()
                out = r
                break
        if out is None:
            return None
        with open(_AGENT_TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return out
    with get_conn() as conn:
        with conn.cursor() as cur:
            # jsonb - operator subtracts elements from the array.
            # The WHERE clause makes the operation a no-op when the
            # id isn't present (idempotent).
            cur.execute(
                "UPDATE agent_tools SET assignments = COALESCE(assignments, '[]'::jsonb) - %s::jsonb, "
                "updated_at = NOW() "
                "WHERE id = %s AND user_id = %s "
                "AND (COALESCE(assignments, '[]'::jsonb) ? %s) "
                f"RETURNING {_tool_cols()}",
                (json.dumps([agent_id]), tool_id, user_id, agent_id),
            )
            row = cur.fetchone()
            if row:
                return _row_to_tool(row)
            return get_tool(tool_id, user_id)


def backfill_from_json() -> int:
    """One-shot: if the legacy JSON file exists and Postgres has no
    rows for any of the tool ids in it, copy them over. Idempotent.

    Called at startup so existing operators don't lose their tools
    on the first deploy after this migration runs. After the
    backfill the JSON file is orphaned and can be deleted by hand
    on the next maintenance window.
    """
    if not is_postgres() or not _AGENT_TOOLS_FILE.exists():
        return 0
    try:
        with open(_AGENT_TOOLS_FILE, "r", encoding="utf-8") as f:
            json_rows = json.load(f) or []
    except (json.JSONDecodeError, IOError, OSError) as exc:
        log.warning("[db_tools] backfill read failed (%s): %s", type(exc).__name__, exc)
        return 0
    if not isinstance(json_rows, list):
        return 0
    n = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in json_rows:
                if not isinstance(r, dict) or not r.get("user_id") or not r.get("id"):
                    continue
                cur.execute(
                    "SELECT 1 FROM agent_tools WHERE id = %s AND user_id = %s",
                    (r["id"], r["user_id"]),
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO agent_tools ("
                    "  id, user_id, agent_id, name, description, "
                    "  webhook_url, filler_phrase, parameters, "
                    "  kind, destination, assignments, function_name, test_prompt, "
                    "  last_tested_at, last_test_result, last_test_error, last_test_error_at, "
                    "  last_invoked_at, last_invocation_status, last_invocation_error, "
                    "  last_invocation_error_at, invocation_count, created_at, updated_at"
                    ") VALUES ("
                    "  %s, %s, %s, %s, %s, %s, %s, %s::jsonb, "
                    "  %s, %s, %s::jsonb, %s, %s, "
                    "  %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "  COALESCE(%s, NOW()), COALESCE(%s, NOW())"
                    ") ON CONFLICT (id, user_id) DO NOTHING",
                    (
                        r["id"], r["user_id"], r.get("agent_id", ""),
                        r.get("name", ""), r.get("description", ""),
                        r.get("webhook_url", ""),
                        r.get("filler_phrase", "Let me check the system..."),
                        json.dumps(r.get("parameters") or {}),
                        r.get("kind", "webhook"),
                        r.get("destination"),
                        json.dumps(r.get("assignments") or []),
                        r.get("function_name", ""),
                        r.get("test_prompt") or None,
                        r.get("last_tested_at"),
                        r.get("last_test_result"),
                        r.get("last_test_error"),
                        r.get("last_test_error_at"),
                        r.get("last_invoked_at"),
                        r.get("last_invocation_status"),
                        r.get("last_invocation_error"),
                        r.get("last_invocation_error_at"),
                        int(r.get("invocation_count") or 0),
                        r.get("created_at"),
                        r.get("updated_at"),
                    ),
                )
                n += 1
    if n:
        log.info("[db_tools] backfilled %d tools from JSON to Postgres", n)
    return n


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")