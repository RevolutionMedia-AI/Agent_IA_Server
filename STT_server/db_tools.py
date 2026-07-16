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

from STT_server.db import get_conn

log = logging.getLogger("stt_server.db_tools")


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
