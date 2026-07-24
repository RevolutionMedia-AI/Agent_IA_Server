"""Postgres-backed implementations for tools_integrations.

This is the storage of provider credentials per user. The
tools_integrations table has a composite PK (user_id, id) so each
user can have their own row per provider.

The runtime (services/credentials_resolver.py) reads from this table on
every outbound call via list_tools() — DO NOT remove the table or the
list/get primitives, even though the FE-facing /tools endpoints are
gone. The Settings → API keys UI (/settings/api-keys/*) still uses
upsert/delete here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from STT_server.db import get_conn, is_postgres
from STT_server.security.credentials import encrypt_credentials

log = logging.getLogger("stt_server.db_tools")

# ponytail: legacy JSON file the local-dev backend used to write to
# before the migration to Postgres. The startup backfill in start.sh
# imports backfill_from_json() from this module — it has to exist or
# the warning 'cannot import name backfill_from_json' fires on every
# boot. On a greenfield deploy the file doesn't exist (no local-dev
# data) and the function returns 0; on an upgrade from a pre-Postgres
# deploy this is the path that preserves the user's per-user API keys.
_TOOLS_INTEGRATIONS_FILE = Path(__file__).resolve().parent / "data" / "tools_integrations.json"


def backfill_from_json() -> int:
    """One-shot helper: if the legacy JSON file exists and Postgres is
    empty for the same (user_id, id), copy the rows over — encrypting
    the credentials on the way in so a leaked DB row still doesn't leak
    plaintext keys. Called at startup so existing local-dev users
    don't lose their provider keys on the first Postgres-backed deploy.
    """
    if not is_postgres() or not _TOOLS_INTEGRATIONS_FILE.exists():
        return 0
    try:
        with open(_TOOLS_INTEGRATIONS_FILE, "r", encoding="utf-8") as f:
            json_data = json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return 0
    if not isinstance(json_data, list):
        return 0
    n = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for entry in json_data:
                user_id = entry.get("user_id")
                tool_id = entry.get("id")
                if not user_id or not tool_id:
                    continue
                # Idempotent: skip if (user_id, id) already in DB.
                cur.execute(
                    "SELECT 1 FROM tools_integrations WHERE user_id = %s AND id = %s",
                    (user_id, tool_id),
                )
                if cur.fetchone():
                    continue
                creds_plain = entry.get("credentials") or {}
                if not isinstance(creds_plain, dict):
                    creds_plain = {}
                # Mirror the runtime path: encrypt every string value
                # before insert so a leaked DB row still doesn't leak
                # plaintext keys.
                creds_enc = encrypt_credentials(creds_plain)
                cur.execute(
                    "INSERT INTO tools_integrations "
                    "(user_id, id, connected, credentials, connected_at, "
                    " display_name, category, updated_at) "
                    "VALUES (%s, %s, %s, %s::jsonb, COALESCE(%s, NOW()), %s, %s, NOW()) "
                    "ON CONFLICT (user_id, id) DO NOTHING",
                    (
                        user_id, tool_id,
                        bool(entry.get("connected", True)),
                        json.dumps(creds_enc),
                        entry.get("connected_at"),
                        entry.get("display_name") or tool_id,
                        entry.get("category"),
                    ),
                )
                n += 1
    if n:
        log.info("[db_tools] backfilled %d per-user provider keys from JSON to Postgres", n)
    return n


def _row_to_tool(row: dict) -> dict:
    if row is None:
        return None
    out = dict(row)
    if isinstance(out.get("credentials"), str):
        try:
            out["credentials"] = json.loads(out["credentials"])
        except (json.JSONDecodeError, TypeError):
            out["credentials"] = {}
    for k in ("connected_at", "updated_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat() + "Z"
    return out


def list_tools(user_id: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, id, connected, credentials, connected_at, "
                "display_name, category, updated_at "
                "FROM tools_integrations WHERE user_id = %s ORDER BY id",
                (user_id,),
            )
            return [_row_to_tool(r) for r in cur.fetchall()]


def get_tool(user_id: str, tool_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, id, connected, credentials, connected_at, "
                "display_name, category, updated_at "
                "FROM tools_integrations WHERE user_id = %s AND id = %s",
                (user_id, tool_id),
            )
            row = cur.fetchone()
            return _row_to_tool(row) if row else None


def upsert_tool(user_id: str, tool_id: str, payload: dict) -> dict:
    credentials = payload.get("credentials") or {}
    connected = bool(payload.get("connected", True))
    display_name = payload.get("display_name") or tool_id
    category = payload.get("category")
    creds_json = json.dumps(credentials)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tools_integrations "
                "(user_id, id, connected, credentials, connected_at, display_name, category, updated_at) "
                "VALUES (%s, %s, %s, %s::jsonb, NOW(), %s, %s, NOW()) "
                "ON CONFLICT (user_id, id) DO UPDATE SET "
                "connected = EXCLUDED.connected, "
                "credentials = EXCLUDED.credentials, "
                "connected_at = CASE WHEN EXCLUDED.connected THEN NOW() ELSE tools_integrations.connected_at END, "
                "display_name = EXCLUDED.display_name, "
                "category = EXCLUDED.category, "
                "updated_at = NOW()",
                (user_id, tool_id, connected, creds_json, display_name, category),
            )
    return get_tool(user_id, tool_id)


def delete_tool(user_id: str, tool_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM tools_integrations WHERE user_id = %s AND id = %s",
                (user_id, tool_id),
            )
            return cur.rowcount > 0
