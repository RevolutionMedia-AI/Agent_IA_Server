"""Postgres-backed implementations for tools_integrations.

This is the storage of API keys / provider credentials. The
tools_integrations table has a composite PK (user_id, id) so each
user can have their own row per provider.

JSON shape returned by the FE matches what we used to put in
data/tools_integrations.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_tools")

DATA_DIR = Path(__file__).resolve().parent / "data"
TOOLS_FILE = DATA_DIR / "tools_integrations.json"


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
    if not is_postgres():
        if not TOOLS_FILE.exists():
            return []
        try:
            with open(TOOLS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return []
        return [t for t in data if t.get("user_id") == user_id]
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
    if not is_postgres():
        if not TOOLS_FILE.exists():
            return None
        try:
            with open(TOOLS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return None
        for t in data:
            if t.get("id") == tool_id and t.get("user_id") == user_id:
                return t
        return None
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
    if not is_postgres():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(TOOLS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            data = []
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for t in data:
            if t.get("user_id") == user_id and t.get("id") == tool_id:
                t.update({"connected": connected, "credentials": credentials,
                          "display_name": display_name, "category": category,
                          "updated_at": now})
                break
        else:
            data.append({"user_id": user_id, "id": tool_id, "connected": connected,
                         "credentials": credentials, "display_name": display_name,
                         "category": category, "connected_at": now,
                         "updated_at": now})
        with open(TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return get_tool(user_id, tool_id)
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
    if not is_postgres():
        if not TOOLS_FILE.exists():
            return False
        try:
            with open(TOOLS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return False
        new_data = [t for t in data if not (t.get("id") == tool_id and t.get("user_id") == user_id)]
        if len(new_data) == len(data):
            return False
        with open(TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        return True
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM tools_integrations WHERE user_id = %s AND id = %s",
                (user_id, tool_id),
            )
            return cur.rowcount > 0


def backfill_from_json() -> int:
    if not is_postgres() or not TOOLS_FILE.exists():
        return 0
    try:
        with open(TOOLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return 0
    if not data:
        return 0
    n = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for t in data:
                if not t.get("user_id") or not t.get("id"):
                    continue
                cur.execute(
                    "SELECT 1 FROM tools_integrations WHERE user_id = %s AND id = %s",
                    (t["user_id"], t["id"]),
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO tools_integrations "
                    "(user_id, id, connected, credentials, connected_at, display_name, category) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)",
                    (
                        t["user_id"], t["id"], bool(t.get("connected", True)),
                        json.dumps(t.get("credentials") or {}),
                        t.get("connected_at"),
                        t.get("display_name", t["id"]),
                        t.get("category"),
                    ),
                )
                n += 1
    if n:
        log.info("[db_tools] backfilled %d tools from JSON to Postgres", n)
    return n
