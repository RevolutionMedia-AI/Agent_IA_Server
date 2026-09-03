"""Postgres-backed CRUD for the integrations table.

Mirror of db_tools.py: the JSON file under STT_server/data/ is only
used as a one-time backfill source on first boot (this file does NOT
backfill — there is no legacy integrations.json, the table is new in
migration 015). Subsequent reads/writes all go through Postgres when
DATABASE_URL is set.

Shape returned by list_/get_/create_/update_:
  {
    "id":           "int-abcd1234",
    "user_id":      "user-...",
    "agent_id":     "__shared__" | "agent-<uuid8>",
    "provider":     "zendesk",
    "name":         "RevolutionMedia Support",
    "configuration": {"subdomain": "revolutionmedia", ...},   # plain JSONB
    "credentials_encrypted": bytes | None,                    # Fernet ciphertext
    "credentials_cipher":    "fernet-v1",
    "connection_status":     "unknown" | "connected" | "failed",
    "last_tested_at":        ISO8601 | None,
    "last_test_message":     "..." | None,
    "created_at":            ISO8601,
    "updated_at":            ISO8601,
  }

Credentials are stored encrypted (BYTEA) and decrypted only by the
internal endpoint /internal/integrations/{id}/credentials which
requires the service token. The /integrations endpoints NEVER return
credentials, even masked.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_integrations")

# ponytail: JSON-file fallback for local dev. Same single-writer
# assumption as db_tools — concurrent tool create + integration
# delete can race because there's no DB transaction; production runs
# Postgres so the JSON file is essentially unused outside of tests.
_DATA_DIR = Path(__file__).resolve().parent / "data"
_INTEGRATIONS_FILE = _DATA_DIR / "integrations.json"


# ── Schema self-heal ───────────────────────────────────────────────────────

_INTEGRATIONS_COLS_BASE = (
    "id, user_id, agent_id, provider, name, configuration, "
    "credentials_encrypted, credentials_cipher, connection_status, "
    "last_tested_at, last_test_message, oauth_scope, "
    "oauth_state_hash, oauth_state_expires_at, assignments, created_at, updated_at"
)
# credentials_encrypted lands as BYTEA; we kept `cipher` as a separate
# TEXT column so future migrations (rotating Fernet keys, switching to
# KMS) can mark old rows without overwriting data.
_columns_check_done: bool = False
_columns_check_lock = threading.Lock()


def _ensure_integrations_table() -> None:
    """Idempotent self-heal: confirms `integrations` table exists with
    all expected columns. Runs ALTERs inline if migration 015 didn't
    apply (cached-image / start.sh miss).

    Same pattern as db_tools._ensure_tool_columns — the FE will hit a
    500 with "relation does not exist" if the migration runner
    skipped, so we self-heal rather than chase the runner.
    """
    global _columns_check_done
    if _columns_check_done:
        return
    with _columns_check_lock:
        if _columns_check_done:
            return
        if not is_postgres():
            _columns_check_done = True
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'integrations'"
                    )
                    # ponytail: use fetchone in a loop with both
                    # dict + tuple row handling so this works under
# RealDictCursor (psycopg2 with cursor_factory=
                    # RealDictCursor — what Railway's prod uses) and
                    # the plain tuple cursors used in tests. The
                    # earlier `cur.fetchall()` + `row[0]` shape threw
                    # KeyError: 0 in production the first time the
                    # endpoint was hit.
                    present: set = set()
                    while True:
                        row = cur.fetchone()
                        if row is None:
                            break
                        if isinstance(row, dict):
                            v = row.get("column_name")
                            if isinstance(v, str):
                                present.add(v)
                        else:
                            try:
                                present.add(row[0])
                            except (KeyError, TypeError, IndexError):
                                # Mock or unexpected shape; skip.
                                pass
            if not present:
                log.warning(
                    "[db_integrations] integrations table missing — applying 015 inline"
                )
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "CREATE TABLE IF NOT EXISTS integrations ("
                            "  id                    TEXT        PRIMARY KEY,"
                            "  user_id               TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,"
                            "  agent_id              TEXT        NOT NULL DEFAULT '__shared__',"
                            "  provider              TEXT        NOT NULL,"
                            "  name                  TEXT        NOT NULL,"
                            "  configuration         JSONB       NOT NULL DEFAULT '{}'::jsonb,"
                            "  credentials_encrypted BYTEA,"
                            "  credentials_cipher    TEXT        NOT NULL DEFAULT 'fernet-v1',"
                            "  connection_status     TEXT        NOT NULL DEFAULT 'unknown',"
                            "  last_tested_at        TIMESTAMPTZ,"
                            "  last_test_message     TEXT,"
                            "  oauth_scope            TEXT,"
                            "  oauth_state_hash       TEXT,"
                            "  oauth_state_expires_at TIMESTAMPTZ,"
                            "  oauth_code_verifier_encrypted BYTEA,"
                            "  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
                            "  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                            ")"
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_integrations_user_agent "
                            "ON integrations (user_id, agent_id)"
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_integrations_provider "
                            "ON integrations (user_id, provider)"
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_integrations_oauth_state_hash "
                            "ON integrations (oauth_state_hash) "
                            "WHERE oauth_state_hash IS NOT NULL"
                        )
            else:
                # Table exists; confirm every column we expect. If a
                # partial migration left some columns off, ADD them.
                # Safe under IF NOT EXISTS so it's idempotent.
                expected = {
                    "agent_id": "TEXT NOT NULL DEFAULT '__shared__'",
                    "provider": "TEXT NOT NULL",
                    "name": "TEXT NOT NULL",
                    "configuration": "JSONB NOT NULL DEFAULT '{}'::jsonb",
                    "credentials_encrypted": "BYTEA",
                    "credentials_cipher": "TEXT NOT NULL DEFAULT 'fernet-v1'",
                    "connection_status": "TEXT NOT NULL DEFAULT 'unknown'",
                    "last_tested_at": "TIMESTAMPTZ",
                    "last_test_message": "TEXT",
                    "oauth_scope": "TEXT",
                    "oauth_state_hash": "TEXT",
                    "oauth_state_expires_at": "TIMESTAMPTZ",
                    # ponyy: 018 added the PKCE code_verifier column.
                    # Listed here so the self-heal recreates it on
                    # deploys where the migration file itself failed
                    # to apply (e.g. UTF-8 BOM at the start of the
                    # .sql makes Postgres reject the whole script).
                    # Without this the /oauth/start UPDATE would 500
                    # with "column oauth_code_verifier_encrypted
                    # does not exist" forever.
                    "oauth_code_verifier_encrypted": "BYTEA",
                    "assignments": "JSONB NOT NULL DEFAULT '[]'::jsonb",
                    "created_at": "TIMESTAMPTZ NOT NULL DEFAULT NOW()",
                    "updated_at": "TIMESTAMPTZ NOT NULL DEFAULT NOW()",
                }
                missing = {k: v for k, v in expected.items() if k not in present}
                if missing:
                    log.warning(
                        "[db_integrations] columns missing, applying ALTER: %s",
                        list(missing.keys()),
                    )
                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            for col, decl in missing.items():
                                cur.execute(
                                    f"ALTER TABLE integrations "
                                    f"ADD COLUMN IF NOT EXISTS {col} {decl}"
                                )
            _columns_check_done = True
        except Exception as exc:
            log.error("[db_integrations] _ensure_integrations_table failed: %s", exc)
            raise


def _integrations_cols() -> str:
    if not _columns_check_done:
        _ensure_integrations_table()
    return _INTEGRATIONS_COLS_BASE


# ── Row mapper ──────────────────────────────────────────────────────────────

def _row_to_integration(row: dict | None) -> dict | None:
    """DB row → wire shape.

    Ponytail: configuration is stored as JSONB and exposed as a plain
    dict. credentials_encrypted is BYTEA on disk but exposed as raw
    bytes (callers in /internal/integrations/{id}/credentials pass
    them through to decrypt_credentials). The /integrations endpoints
    in routes/api.py strip both fields before returning to the FE.

    OAuth fields (oauth_state_hash, oauth_state_expires_at) are
    internal-only — never returned by any /integrations endpoint.
    They're consumed by the OAuth helpers (start_oauth_flow,
    complete_oauth_flow, get_integration_by_oauth_state). _strip
    below trims them before any wire response.
    """
    if row is None:
        return None
    # ponytail: RealDictCursor returns dict, but handle tuple/string for safety (500 fix)
    if isinstance(row, dict):
        out = dict(row)
    elif isinstance(row, (list, tuple)) and len(row) == 1 and isinstance(row[0], dict):
        out = dict(row[0])
    elif isinstance(row, (list, tuple)):
        try:
            cols = [c.strip() for c in _INTEGRATIONS_COLS_BASE.split(",")]
            out = dict(zip(cols, row))
        except Exception:
            log.warning("[db_integrations] unexpected row shape for _row_to_integration: %r", row)
            return None
    else:
        try:
            out = dict(row)
        except Exception as exc:
            log.warning("[db_integrations] dict(row) failed for %r: %s", row, exc)
            return None
    if isinstance(out.get("configuration"), str):
        try:
            out["configuration"] = json.loads(out["configuration"])
        except (json.JSONDecodeError, TypeError):
            out["configuration"] = {}
    if not isinstance(out.get("configuration"), dict):
        out["configuration"] = {}
    for k in ("last_tested_at", "oauth_state_expires_at", "created_at", "updated_at"):
        if hasattr(out.get(k), "isoformat"):
            out[k] = out[k].isoformat() + "Z"
    # ponytail: normalize assignments to list for FE assignment UI
    if not isinstance(out.get("assignments"), list):
        if isinstance(out.get("assignments"), str):
            try:
                out["assignments"] = json.loads(out["assignments"])
            except Exception:
                out["assignments"] = []
        elif out.get("assignments") is None:
            out["assignments"] = []
        else:
            # Handle psycopg2 returning JSONB as already-parsed list or other
            try:
                out["assignments"] = list(out["assignments"])
            except Exception:
                out["assignments"] = []
    return out


def _strip_integration_for_wire_with_oauth(row: dict | None) -> dict | None:
    """Wire shape for /integrations + /integrations/{id}. Strips
    credentials AND oauth state fields. Use this from the route
    handlers instead of the local _strip_integration_for_wire helper
    in routes/api.py so OAuth internals never leak.
    """
    if row is None:
        return None
    out = dict(row)
    out.pop("credentials_encrypted", None)
    out.pop("credentials_cipher", None)
    out.pop("oauth_state_hash", None)
    out.pop("oauth_state_expires_at", None)
    return out


# ── CRUD ────────────────────────────────────────────────────────────────────

def list_integrations(user_id: str, agent_id: str | None = None) -> list[dict]:
    """List integrations for one user.

    agent_id filter:
      * None  → every integration the user owns
      * '__shared__' → only the user's shared integrations
      * 'agent-<uuid8>' → that agent's private integrations + shared ones
        (matches db_tools.list_tools' shared/private merge so the FE
        can list a per-agent view without two round-trips).
    """
    if not is_postgres():
        return _list_integrations_json(user_id, agent_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            if agent_id is None:
                cur.execute(
                    f"SELECT {_integrations_cols()} FROM integrations "
                    "WHERE user_id = %s ORDER BY created_at DESC",
                    (user_id,),
                )
            elif agent_id == "__shared__":
                cur.execute(
                    f"SELECT {_integrations_cols()} FROM integrations "
                    "WHERE user_id = %s AND agent_id = '__shared__' "
                    "ORDER BY created_at DESC",
                    (user_id,),
                )
            else:
                cur.execute(
                    f"SELECT {_integrations_cols()} FROM integrations "
                    "WHERE user_id = %s AND (agent_id = '__shared__' OR agent_id = %s) "
                    "ORDER BY created_at DESC",
                    (user_id, agent_id),
                )
            return [_row_to_integration(r) for r in cur.fetchall()]


def get_integration(integration_id: str, user_id: str) -> dict | None:
    """Fetch a single integration by id. Ownership scoped by user_id so
    a token from user A can't read user B's row."""
    if not is_postgres():
        rows = _list_integrations_json(user_id, agent_id=None)
        for r in rows:
            if r.get("id") == integration_id:
                return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_integrations_cols()} FROM integrations "
                "WHERE id = %s AND user_id = %s",
                (integration_id, user_id),
            )
            row = cur.fetchone()
            return _row_to_integration(row) if row else None


def get_integration_by_id(integration_id: str) -> dict | None:
    """Fetch a single integration by id WITHOUT user ownership scoping.

    Used by the internal endpoint /internal/integrations/{id}/credentials
    that authenticates via the shared service token (no user_id
    available). The endpoint's caller (n8n) is trusted — the service
    token is the auth. NEVER expose this lookup behind a user-bearer
    guard.
    """
    if not is_postgres():
        for rows_owner in _walk_all_integrations_json():
            for r in rows_owner:
                if r.get("id") == integration_id:
                    return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_integrations_cols()} FROM integrations WHERE id = %s",
                (integration_id,),
            )
            row = cur.fetchone()
            return _row_to_integration(row) if row else None


def _walk_all_integrations_json():
    """Yields the full list of integrations across all users. Used by
    get_integration_by_id's JSON-fallback path. In practice this is
    only the local-dev file — production runs Postgres."""
    yield _read_integrations_file()


def create_integration(
    user_id: str,
    payload: dict,
    *,
    credentials_encrypted: bytes | None = None,
    cipher: str = "fernet-v1",
) -> dict:
    """Insert a new integration row.

    payload must include: provider, name, agent_id, configuration.
    credentials_encrypted is a Fernet ciphertext (BYTES, not str).
    Pass None when the user is creating a row but hasn't filled the
    password fields yet (rare — the FE usually submits them together).
    """
    new_id = payload.get("id") or f"int-{uuid.uuid4().hex[:8]}"
    configuration = payload.get("configuration") or {}
    if not is_postgres():
        rows = _read_integrations_file()
        new_row = {
            "id": new_id,
            "user_id": user_id,
            "credentials_cipher": cipher,
            "connection_status": payload.get("connection_status") or "unknown",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            **payload,
        }
        new_row["configuration"] = configuration
        new_row["credentials_encrypted"] = credentials_encrypted
        rows.append(new_row)
        _write_integrations_file(rows)
        return new_row
    # ponytail: BYTEA column needs Binary wrapper for dicts
    _creds_for_db = credentials_encrypted
    if isinstance(_creds_for_db, dict):
        from psycopg2 import Binary
        _creds_for_db = Binary(json.dumps(_creds_for_db).encode("utf-8"))
    elif isinstance(_creds_for_db, str):
        from psycopg2 import Binary
        _creds_for_db = Binary(_creds_for_db.encode("utf-8"))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO integrations ("
                "  id, user_id, agent_id, provider, name, configuration, "
                "  credentials_encrypted, credentials_cipher, connection_status, "
                "  last_tested_at, last_test_message, created_at, updated_at"
                ") VALUES ("
                "  %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, NOW(), NOW()"
                f") RETURNING {_integrations_cols()}",
                (
                    new_id, user_id,
                    payload["agent_id"],
                    payload["provider"],
                    payload["name"],
                    json.dumps(configuration),
                    _creds_for_db,
                    cipher,
                    payload.get("connection_status") or "unknown",
                    payload.get("last_tested_at"),
                    payload.get("last_test_message"),
                ),
            )
            row = cur.fetchone()
    return _row_to_integration(row)


def update_integration(
    integration_id: str,
    user_id: str,
    payload: dict,
    *,
    credentials_encrypted: bytes | None = None,
) -> dict | None:
    """Patch an integration row.

    payload keys:
      * name             → replace
      * agent_id         → replace
      * configuration    → replace (full dict; no merge — the FE re-sends
                            the whole object so we don't need partial-merge)
      * credentials_encrypted (bytes) → caller-encrypted blob (already
                            merged: empty/missing values for individual
                            fields kept the existing encrypted value,
                            so what's passed here is the new full blob
                            or None to keep existing)
      * connection_status / last_tested_at / last_test_message → as-is

    Ponytail: the credentials merge happens in routes/api.py BEFORE
    calling update_integration — that layer is the one that knows
    "this field is empty → keep, this field is non-empty → replace
    encrypted value". db_integrations just stores the resulting blob.
    """
    if not is_postgres():
        rows = _read_integrations_file()
        updated = None
        db_managed = {"id", "user_id", "created_at"}
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                for k, v in payload.items():
                    if k in db_managed:
                        continue
                    r[k] = v
                if credentials_encrypted is not None:
                    r["credentials_encrypted"] = credentials_encrypted
                r["updated_at"] = _now_iso()
                updated = r
                break
        if updated is None:
            return None
        _write_integrations_file(rows)
        return updated
    set_clauses: list[str] = []
    values: list = []
    if "name" in payload:
        set_clauses.append("name = %s")
        values.append(payload["name"])
    if "agent_id" in payload:
        set_clauses.append("agent_id = %s")
        values.append(payload["agent_id"])
    if "configuration" in payload:
        set_clauses.append("configuration = %s::jsonb")
        values.append(json.dumps(payload["configuration"] or {}))
    if "connection_status" in payload:
        set_clauses.append("connection_status = %s")
        values.append(payload["connection_status"])
    if "last_tested_at" in payload:
        set_clauses.append("last_tested_at = %s")
        values.append(payload["last_tested_at"])
    if "last_test_message" in payload:
        set_clauses.append("last_test_message = %s")
        values.append(payload["last_test_message"])
    if credentials_encrypted is not None:
        # ponytail: BYTEA column needs Binary wrapper for dicts
        if isinstance(credentials_encrypted, dict):
            from psycopg2 import Binary
            credentials_encrypted = Binary(json.dumps(credentials_encrypted).encode("utf-8"))
        elif isinstance(credentials_encrypted, str):
            from psycopg2 import Binary
            credentials_encrypted = Binary(credentials_encrypted.encode("utf-8"))
        set_clauses.append("credentials_encrypted = %s")
        values.append(credentials_encrypted)
    if not set_clauses:
        return get_integration(integration_id, user_id)
    set_clauses.append("updated_at = NOW()")
    values.extend([integration_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE integrations SET {', '.join(set_clauses)} "
                "WHERE id = %s AND user_id = %s "
                f"RETURNING {_integrations_cols()}",
                values,
            )
            row = cur.fetchone()
    return _row_to_integration(row) if row else None


def delete_integration(integration_id: str, user_id: str) -> tuple[bool, str | None]:
    """Delete an integration. Returns (success, error_message).

    On Postgres: RESTRICT — count dependent tools inside the same
    transaction as the DELETE so a concurrent tool create can't sneak
    in between the count and the delete. If count > 0, returns
    (False, "<n> tools depend on this integration"); the caller maps
    that to 409 Conflict with the same message.

    Ponytail: there is NO real FK on agent_tools.integration_id (the
    JSON-file fallback can't enforce referential integrity), so the
    transactional count + delete here is what stands between the
    operator and orphan tools. The index on agent_tools.integration_id
    (migration 016) keeps the count cheap.
    """
    if not is_postgres():
        # ponytail: JSON-file path. Single-writer assumption; a
        # concurrent tool create + integration delete CAN race. This
        # is documented — production runs Postgres.
        rows = _read_integrations_file()
        target = next(
            (r for r in rows if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id),
            None,
        )
        if target is None:
            return False, None
        tools_file = _DATA_DIR / "agent_tools.json"
        dep_count = 0
        if tools_file.exists():
            try:
                with open(tools_file, "r", encoding="utf-8") as f:
                    tools = json.load(f) or []
                dep_count = sum(
                    1 for t in tools
                    if isinstance(t, dict) and t.get("integration_id") == integration_id
                )
            except (json.JSONDecodeError, IOError, OSError):
                dep_count = 0
        if dep_count > 0:
            return False, f"{dep_count} tools depend on this integration"
        rows = [r for r in rows if not (isinstance(r, dict) and r.get("id") == integration_id)]
        _write_integrations_file(rows)
        return True, None
    with get_conn() as conn:
        with conn.cursor() as cur:
            # COUNT inside the same transaction as the DELETE so a
            # concurrent tool create can't land between the two. The
            # default isolation level (READ COMMITTED on Postgres)
            # is enough for this — the COUNT acquires a row-level
            # shared lock implicitly through the FOR UPDATE on the
            # integration row below.
            cur.execute(
                "SELECT COUNT(*) AS n FROM agent_tools WHERE integration_id = %s",
                (integration_id,),
            )
            # ponytail: cursor-shape-agnostic scalar read (see
            # _scalar() docstring). Same defensive fix as
            # count_dependent_tools — the previous (dep_count,) =
            # cur.fetchone() shape fails when the cursor returns a
            # dict (RealDictCursor) because you can't unpack a dict
            # by position. The actual production error surfaced as
            # "invalid literal for int() with base 10: 'count'" when
            # the dict's column name was stringified into the
            # unpack target.
            dep_count = _scalar(cur.fetchone())
            if dep_count > 0:
                return False, f"{dep_count} tools depend on this integration"
            cur.execute(
                "DELETE FROM integrations WHERE id = %s AND user_id = %s",
                (integration_id, user_id),
            )
            return cur.rowcount > 0, None


def count_dependent_tools(integration_id: str, user_id: str | None = None) -> int:
    """Count tools pointing at this integration. user_id is optional —
    when provided, only counts tools the user owns (consistent with
    the ownership scoping every other helper uses)."""
    if not is_postgres():
        tools_file = _DATA_DIR / "agent_tools.json"
        if not tools_file.exists():
            return 0
        try:
            with open(tools_file, "r", encoding="utf-8") as f:
                tools = json.load(f) or []
        except (json.JSONDecodeError, IOError, OSError):
            return 0
        return sum(
            1 for t in tools
            if isinstance(t, dict)
            and t.get("integration_id") == integration_id
            and (user_id is None or t.get("user_id") == user_id)
        )
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM agent_tools WHERE integration_id = %s",
                    (integration_id,),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM agent_tools "
                    "WHERE integration_id = %s AND user_id = %s",
                    (integration_id, user_id),
                )
            # ponytail: handle BOTH cursor shapes. RealDictCursor
            # (psycopg2.cursor_factory=RealDictCursor — what Railway
            # uses) returns the row as a dict `{'n': 0}`. A plain
            # tuple cursor returns a tuple `(0,)`. The previous
            # `(n,) = cur.fetchone()` shape assumes tuple shape and
            # throws on dicts; the dict shape `[k] = cur.fetchone()`
            # assumes dict shape. Casting both via _scalar() is the
            # only shape-agnostic way.
            row = cur.fetchone()
            return _scalar(row) or 0


def add_integration_assignment(integration_id: str, user_id: str, agent_id: str) -> dict | None:
    """Add `agent_id` to the integration's `assignments` JSONB array. Idempotent."""
    if not is_postgres():
        rows = _read_integrations_file()
        out = None
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
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
        _write_integrations_file(rows)
        return out
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE integrations SET assignments = COALESCE(assignments, '[]'::jsonb) || %s::jsonb, updated_at = NOW() "
                "WHERE id = %s AND user_id = %s AND NOT (COALESCE(assignments, '[]'::jsonb) ? %s) "
                f"RETURNING {_integrations_cols()}",
                (json.dumps([agent_id]), integration_id, user_id, agent_id),
            )
            row = cur.fetchone()
            if row:
                return _row_to_integration(row)
            return get_integration(integration_id, user_id)


def remove_integration_assignment(integration_id: str, user_id: str, agent_id: str) -> dict | None:
    """Drop `agent_id` from the integration's `assignments` JSONB array. Idempotent."""
    if not is_postgres():
        rows = _read_integrations_file()
        out = None
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                assigns = r.get("assignments") or []
                if agent_id not in assigns:
                    return r
                r["assignments"] = [a for a in assigns if a != agent_id]
                r["updated_at"] = _now_iso()
                out = r
                break
        if out is None:
            return None
        _write_integrations_file(rows)
        return out
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE integrations SET assignments = COALESCE(assignments, '[]'::jsonb) - %s, updated_at = NOW() "
                "WHERE id = %s AND user_id = %s AND COALESCE(assignments, '[]'::jsonb) ? %s "
                f"RETURNING {_integrations_cols()}",
                (agent_id, integration_id, user_id, agent_id),
            )
            row = cur.fetchone()
            if row:
                return _row_to_integration(row)
            return get_integration(integration_id, user_id)


def _scalar(row) -> int:
    """Extract the first value of a fetchone() result regardless of
    cursor shape. RealDictCursor → dict; tuple cursor → tuple.

    The function name and return type are deliberately narrow —
    only used for COUNT-style single-value reads. If row is a dict
    that contains a non-numeric value (which shouldn't happen for
    `SELECT COUNT(*)` but could for a malformed query), we fall
    back to 0 rather than crash — disconnect is best-effort and
    a bad count shouldn't 500 the whole request.
    """
    if row is None:
        return 0
    if isinstance(row, dict):
        # RealDictCursor: the row is a dict keyed by column name.
        # COUNT(*) is aliased to "n" in our queries, so row['n'] is
        # the integer. Fall back to 'count' for un-aliased queries.
        v = row.get("n", row.get("count", 0))
    else:
        # Plain tuple cursor: the row is a 1-tuple.
        v = row[0] if row else 0
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        log.warning("[db_integrations] non-numeric scalar from query: %r", v)
        return 0


# ── JSON-file fallback (local dev / tests) ──────────────────────────────────

def _list_integrations_json(user_id: str, agent_id: str | None) -> list[dict]:
    rows = _read_integrations_file()
    out = [r for r in rows if isinstance(r, dict) and r.get("user_id") == user_id]
    if agent_id is None:
        return out
    if agent_id == "__shared__":
        return [r for r in out if r.get("agent_id") == "__shared__"]
    return [r for r in out if r.get("agent_id") in ("__shared__", agent_id)]


def _read_integrations_file() -> list[dict]:
    if not _INTEGRATIONS_FILE.exists():
        return []
    try:
        with open(_INTEGRATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except (json.JSONDecodeError, IOError, OSError) as exc:
        log.warning("[db_integrations] JSON load failed (%s): %s", type(exc).__name__, exc)
        return []


def _write_integrations_file(rows: list[dict]) -> None:
    os.makedirs(_INTEGRATIONS_FILE.parent, exist_ok=True)
    with open(_INTEGRATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ─ ─ ─ OAuth state + refresh helpers ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
# ponytail: the OAuth helpers below are intentionally separate from the
# generic CRUD above. The OAuth flow has its own concerns (state hash
# storage, single-use semantics, refresh-on-read under advisory lock,
# revoke + status reset) and inlining them into the CRUD layer would
# make both harder to read. Future providers (Dynamics, Google,
# HubSpot) reuse these helpers unchanged.


def start_oauth_flow(
    integration_id: str,
    user_id: str,
    state_hash: str,
    *,
    code_verifier_encrypted: bytes | None = None,
    ttl_seconds: int = 600,
) -> dict | None:
    """Persist the OAuth state hash + expiry on the integration row.

    Sets connection_status='pending' (the operator hasn't completed
    the dance yet). Returns the refreshed row so the caller can
    surface the new status without an extra round-trip.

    `code_verifier_encrypted` is the PKCE verifier encrypted with
    Fernet (same key as `credentials_encrypted`). Salesforce's
    External Client Apps require it in the token-exchange POST. We
    only need the original value, never its hash — the verifier is
    a bearer secret, so it's encrypted at rest. NULL when the
    provider doesn't require PKCE (defensive; in V1 only
    Salesforce exists and it does require PKCE).

    `ttl_seconds` defaults to 10 minutes — long enough for the
    operator to log in to Salesforce and approve, short enough that
    a stolen state can't be replayed for long.
    """
    # ponyy: ensure the PKCE column exists. Migration 018 had a UTF-8 BOM
    # that Postgres rejected, so the column is missing on deploys that
    # already ran the migration. The self-heal in _ensure_integrations_table
    # will add it if missing (idempotent). Without this, the UPDATE below
    # 500s with UndefinedColumn.
    _ensure_integrations_table()
    from datetime import datetime, timezone, timedelta
    if not is_postgres():
        rows = _read_integrations_file()
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                r["oauth_state_hash"] = state_hash
                r["oauth_state_expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
                r["oauth_code_verifier_encrypted"] = code_verifier_encrypted
                r["connection_status"] = "pending"
                r["updated_at"] = _now_iso()
                _write_integrations_file(rows)
                return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            cur.execute(
                f"UPDATE integrations SET oauth_state_hash = %s, "
                "oauth_state_expires_at = %s, oauth_code_verifier_encrypted = %s, "
                "connection_status = 'pending', updated_at = NOW() "
                "WHERE id = %s AND user_id = %s "
                f"RETURNING {_integrations_cols()}",
                (state_hash, expires, code_verifier_encrypted, integration_id, user_id),
            )
            row = cur.fetchone()
    return _row_to_integration(row) if row else None


def get_integration_by_oauth_state(state_hash: str) -> dict | None:
    """Read-only lookup by OAuth state hash. Use this ONLY for
    diagnostics / debugging. The callback path MUST use
    consume_oauth_state instead — that path atomically clears the
    state hash inside the same UPDATE, so a duplicate callback can't
    re-exchange the same code. This function leaves the state hash
    intact, which is exactly what an attacker would want."""
    if not is_postgres():
        for rows_owner in _walk_all_integrations_json():
            for r in rows_owner:
                if isinstance(r, dict) and r.get("oauth_state_hash") == state_hash:
                    return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_integrations_cols()} FROM integrations "
                "WHERE oauth_state_hash = %s",
                (state_hash,),
            )
            row = cur.fetchone()
            return _row_to_integration(row) if row else None


def consume_oauth_state(state_hash: str, cur=None) -> dict | None:
    """Atomically consume the OAuth state — single-use.

    UPDATE integrations SET oauth_state_hash = NULL,
    oauth_state_expires_at = NULL WHERE oauth_state_hash = %s
    AND oauth_state_expires_at > NOW()
    RETURNING id, user_id, provider, name, configuration;

    ponyy: this is the cornerstone of OAuth replay protection. Two
    callbacks (A and B) arriving at the same time both look up by
    state_hash; if either of them used the old `lookup → exchange →
    clear` pattern, both could observe a valid state and both could
    exchange the same code. With this atomic consume, only ONE of
    them gets the row back (the UPDATE matches one row, the other
    call's UPDATE matches zero rows). The other callback's exchange
    happens against no row → fail.

    Replay protection in detail:
      * TTL check inside the WHERE filters out expired states — the
        state can't be replayed after 10 min regardless of how many
        copies of the URL the operator bookmarked.
      * Setting oauth_state_hash = NULL inside the same UPDATE means
        the next concurrent / parallel callback with the same state
        gets zero rows and is rejected as invalid.
      * The match is on `oauth_state_hash = %s` only — the user_id
        check is implicit (a given state belongs to exactly one row
        which belongs to exactly one user), so we don't need a
        separate user_id filter.
    """
    if cur is not None:
        return _consume_oauth_state_cursor(state_hash, cur)
    if not is_postgres():
        # JSON-file fallback: no atomic compare-and-swap, but the
        # operator's dev environment doesn't have concurrent
        # callbacks in flight. Best-effort: clear the hash + read.
        rows = _read_integrations_file()
        for r in rows:
            if isinstance(r, dict) and r.get("oauth_state_hash") == state_hash:
                verifier = r.get("oauth_code_verifier_encrypted")
                r["oauth_state_hash"] = None
                r["oauth_state_expires_at"] = None
                r["updated_at"] = _now_iso()
                _write_integrations_file(rows)
                if verifier is not None:
                    r["_oauth_code_verifier_encrypted"] = verifier
                return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur_:
            return _consume_oauth_state_cursor(state_hash, cur_)


def _consume_oauth_state_cursor(state_hash: str, cur) -> dict | None:
    # ponyy: we also need to return the encrypted code_verifier so
    # the callback handler can decrypt it and pass the plaintext to
    # the token-exchange POST. The verifier is a bearer secret —
    # we keep it encrypted on the row and only decrypt at the
    # exact moment we need to send it to Salesforce.
    cur.execute(
        "UPDATE integrations "
        "SET oauth_state_hash = NULL, oauth_state_expires_at = NULL, updated_at = NOW() "
        "WHERE oauth_state_hash = %s "
        "AND oauth_state_expires_at > NOW() "
        f"RETURNING {_integrations_cols()}, oauth_code_verifier_encrypted",
        (state_hash,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    # The verifier lives in the same RETURNING row but is NOT in
    # _integrations_cols (intentionally — it's transient). Extract
    # it before passing the rest through _row_to_integration.
    verifier_encrypted = row.get("oauth_code_verifier_encrypted")
    integ = _row_to_integration(row)
    if integ is not None:
        integ["_oauth_code_verifier_encrypted"] = verifier_encrypted
    return integ


def complete_oauth_flow(
    integration_id: str,
    user_id: str,
    *,
    credentials_encrypted: bytes,
    configuration: dict,
    scope: str | None,
    connection_status: str = "connected",
    cur=None,
) -> dict | None:
    """Persist the OAuth tokens + clear the state hash in one shot.

    Called from the /oauth/callback handler AFTER consume_oauth_state
    has already cleared the state hash atomically. The state hash
    was already NULL when this runs — this just writes the tokens +
    status. cur is the SAME cursor the callback used for the
    consume UPDATE so the whole OAuth dance is one transaction.
    """
    if cur is not None:
        return _complete_oauth_flow_cursor(
            integration_id, user_id,
            credentials_encrypted=credentials_encrypted,
            configuration=configuration, scope=scope,
            connection_status=connection_status, cur=cur,
        )
    if not is_postgres():
        rows = _read_integrations_file()
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                r["credentials_encrypted"] = credentials_encrypted
                r["configuration"] = configuration
                r["oauth_scope"] = scope
                r["oauth_state_hash"] = None
                r["oauth_state_expires_at"] = None
                r["oauth_code_verifier_encrypted"] = None
                r["connection_status"] = connection_status
                r["updated_at"] = _now_iso()
                _write_integrations_file(rows)
                return r
        return None
    with get_conn() as conn:
        with conn.cursor() as cur_:
            return _complete_oauth_flow_cursor(
                integration_id, user_id,
                credentials_encrypted=credentials_encrypted,
                configuration=configuration, scope=scope,
                connection_status=connection_status, cur=cur_,
            )


def _complete_oauth_flow_cursor(
    integration_id: str, user_id: str,
    *, credentials_encrypted: bytes, configuration: dict, scope: str | None,
    connection_status: str, cur,
) -> dict | None:
    # ponyy: clear oauth_code_verifier_encrypted here too. The
    # consume UPDATE already NULLed oauth_state_hash; this is the
    # second half of the cleanup. Verifier is single-use by design
    # (RFC 7636) so it must NOT survive the callback.
    # ponytail: psycopg2 can't adapt a dict for BYTEA — the caller
    # passes the dict from encrypt_credentials, which must be
    # serialized to JSON bytes first. Use Binary for the BYTEA
    # column and Json for the JSONB column (the previous
    # json.dumps() + ::jsonb cast also works, but Json() is the
    # canonical psycopg2 adapter for dict->jsonb and handles
    # escaping correctly).
    from psycopg2.extras import Json
    from psycopg2 import Binary
    # credentials_encrypted is a dict like {"access_token": "gAAAA..."}.
    # Serialize to JSON bytes for the BYTEA column.
    if isinstance(credentials_encrypted, dict):
        credentials_encrypted = Binary(json.dumps(credentials_encrypted).encode("utf-8"))
    elif isinstance(credentials_encrypted, str):
        credentials_encrypted = Binary(credentials_encrypted.encode("utf-8"))
    cur.execute(
        f"UPDATE integrations SET credentials_encrypted = %s, "
        "configuration = %s, oauth_scope = %s, "
        "oauth_code_verifier_encrypted = NULL, "
        "connection_status = %s, updated_at = NOW() "
        "WHERE id = %s AND user_id = %s "
        f"RETURNING {_integrations_cols()}",
        (credentials_encrypted, Json(configuration),
         scope, connection_status, integration_id, user_id),
    )
    row = cur.fetchone()
    return _row_to_integration(row) if row else None


def clear_oauth_state(integration_id: str, user_id: str) -> None:
    """Drop the state hash + expiry without touching credentials.

    Used when the callback fails (e.g. provider returned ?error=access_denied
    or our token exchange 5xx'd). The next /oauth/start for the same
    integration can then write a fresh state hash without the unique-
    index collision.
    """
    if not is_postgres():
        rows = _read_integrations_file()
        changed = False
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                r["oauth_state_hash"] = None
                r["oauth_state_expires_at"] = None
                r["oauth_code_verifier_encrypted"] = None
                r["updated_at"] = _now_iso()
                changed = True
                break
        if changed:
            _write_integrations_file(rows)
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE integrations SET oauth_state_hash = NULL, "
                "oauth_state_expires_at = NULL, oauth_code_verifier_encrypted = NULL, "
                "updated_at = NOW() "
                "WHERE id = %s AND user_id = %s",
                (integration_id, user_id),
            )


def disconnect_integration(
    integration_id: str,
    user_id: str,
) -> tuple[bool, str | None]:
    """Wipe credentials + flip status to 'disconnected'.

    Caller (the route handler) is expected to have already best-effort
    revoked at the provider. We don't gate on dependent tools here —
    that's done in the route layer via count_dependent_tools so the
    409 message carries the count. Returns (success, error_message)
    matching the pattern used by delete_integration.
    """
    if not is_postgres():
        rows = _read_integrations_file()
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                r["credentials_encrypted"] = None
                r["connection_status"] = "disconnected"
                r["last_tested_at"] = None
                r["last_test_message"] = None
                r["updated_at"] = _now_iso()
                _write_integrations_file(rows)
                return True, None
        return False, None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE integrations SET credentials_encrypted = NULL, "
                "connection_status = 'disconnected', "
                "last_tested_at = NULL, last_test_message = NULL, "
                "updated_at = NOW() "
                "WHERE id = %s AND user_id = %s "
                f"RETURNING {_integrations_cols()}",
                (integration_id, user_id),
            )
            row = cur.fetchone()
            return (row is not None, None)


def update_integration_credentials(
    integration_id: str,
    user_id: str,
    credentials_encrypted: bytes,
    *,
    cur=None,
) -> None:
    """Persist refreshed tokens back to the row. Called from the
    /internal/.../credentials handler after a successful refresh.

    Caller holds the advisory lock on the integration row (see the
    route handler) so concurrent refreshes serialize here. We do NOT
    clear connection_status — if the row was 'failed' for some
    reason, refresh succeeded means we should mark it 'connected'.
    Caller passes the right status via mark_integration_status().

    ponyy: when called from the /internal/.../credentials handler,
    the SAME cursor is passed in so the lock → read → refresh →
    persist → status update is one transaction. No connection hop
    between the SELECT and the UPDATE means no chance of another
    process racing in between.
    """
    # ponytail: BYTEA column needs Binary wrapper for dicts
    if isinstance(credentials_encrypted, dict):
        from psycopg2 import Binary
        credentials_encrypted = Binary(json.dumps(credentials_encrypted).encode("utf-8"))
    elif isinstance(credentials_encrypted, str):
        from psycopg2 import Binary
        credentials_encrypted = Binary(credentials_encrypted.encode("utf-8"))
    if cur is not None:
        cur.execute(
            "UPDATE integrations SET credentials_encrypted = %s, "
            "updated_at = NOW() "
            "WHERE id = %s AND user_id = %s",
            (credentials_encrypted, integration_id, user_id),
        )
        return
    if not is_postgres():
        rows = _read_integrations_file()
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                r["credentials_encrypted"] = credentials_encrypted
                r["updated_at"] = _now_iso()
                _write_integrations_file(rows)
                return
        return
    with get_conn() as conn:
        with conn.cursor() as cur_:
            cur_.execute(
                "UPDATE integrations SET credentials_encrypted = %s, "
                "updated_at = NOW() "
                "WHERE id = %s AND user_id = %s",
                (credentials_encrypted, integration_id, user_id),
            )


def mark_integration_status(
    integration_id: str,
    user_id: str,
    status: str,
    *,
    last_tested_at=None,
    last_test_message: str | None = None,
    cur=None,
) -> None:
    """Flip connection_status (and optionally last_tested_*). Used by
    the /test endpoint and by the refresh-failure handler.

    ponyy: when called from the /internal/.../credentials handler
    on refresh, the SAME cursor is passed in so the lock → read →
    refresh → persist → status update is one transaction.
    """
    if status not in {"unknown", "pending", "connected", "failed", "disconnected"}:
        raise ValueError(f"invalid connection_status: {status!r}")
    if cur is not None:
        if last_tested_at is not None:
            cur.execute(
                "UPDATE integrations SET connection_status = %s, "
                "last_tested_at = %s, last_test_message = %s, "
                "updated_at = NOW() "
                "WHERE id = %s AND user_id = %s",
                (status, last_tested_at, last_test_message,
                 integration_id, user_id),
            )
        else:
            cur.execute(
                "UPDATE integrations SET connection_status = %s, "
                "last_test_message = %s, updated_at = NOW() "
                "WHERE id = %s AND user_id = %s",
                (status, last_test_message, integration_id, user_id),
            )
        return
    if not is_postgres():
        rows = _read_integrations_file()
        for r in rows:
            if isinstance(r, dict) and r.get("id") == integration_id and r.get("user_id") == user_id:
                r["connection_status"] = status
                if last_tested_at is not None:
                    r["last_tested_at"] = last_tested_at
                if last_test_message is not None:
                    r["last_test_message"] = last_test_message
                r["updated_at"] = _now_iso()
                _write_integrations_file(rows)
                return
        return
    if last_tested_at is not None:
        with get_conn() as conn:
            with conn.cursor() as cur_:
                cur_.execute(
                    "UPDATE integrations SET connection_status = %s, "
                    "last_tested_at = %s, last_test_message = %s, "
                    "updated_at = NOW() "
                    "WHERE id = %s AND user_id = %s",
                    (status, last_tested_at, last_test_message,
                     integration_id, user_id),
                )
    else:
        with get_conn() as conn:
            with conn.cursor() as cur_:
                cur_.execute(
                    "UPDATE integrations SET connection_status = %s, "
                    "last_test_message = %s, updated_at = NOW() "
                    "WHERE id = %s AND user_id = %s",
                    (status, last_test_message, integration_id, user_id),
                )


def acquire_advisory_xact_lock(cur, lock_key: str) -> None:
    """Take a transaction-scoped advisory lock on `lock_key`.

    Released automatically when the surrounding transaction commits
    or rolls back. Concurrent requests with the same key serialize on
    this lock. Used by the refresh-on-read path so 5 parallel n8n
    requests on the same integration don't all try to refresh the
    Salesforce token in parallel — first one does the work, the rest
    read the freshly-persisted row.

    ponytail: we use pg_advisory_xact_lock (transaction-scoped) over
    pg_advisory_lock (session-scoped) so a forgotten RELEASE call
    can never leak a lock. Postgres's hashtext() hashes the key
    into a 32-bit signed range — same input always maps to the same
    lock id, no collisions within a single deployment.
    """
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtext('integration:' || %s))",
        (lock_key,),
    )
