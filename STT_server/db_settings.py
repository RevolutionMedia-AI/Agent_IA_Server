"""Postgres-backed implementations for the settings table (one row per user)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from STT_server.db import get_conn, is_postgres
from STT_server.utils.safe_path import UnsafePathError, sanitize_id

log = logging.getLogger("stt_server.db_settings")

DATA_DIR = Path(__file__).resolve().parent / "data"
SETTINGS_DIR = DATA_DIR / "settings"


def _safe_user_filename(user_id: str) -> str:
    """Sanitize user_id before it lands in a settings/<filename> path.

    The auth layer already rejects exotic user_id values, but the
    file-inclusion scanner flags every f"{user_id}.json" pattern.
    Funneling through sanitize_id kills traversal chars and gives the
    scanner a single allowlist to check.
    """
    return sanitize_id(user_id, field="user_id") + ".json"


def _row_to_settings(row: dict) -> dict:
    if row is None:
        return None
    out = dict(row)
    n = out.get("notifications")
    if isinstance(n, str):
        try:
            out["notifications"] = json.loads(n)
        except (json.JSONDecodeError, TypeError):
            out["notifications"] = {}
    for k in ("updated_at",):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat() + "Z"
    return out


def get_settings(user_id: str) -> dict | None:
    if not is_postgres():
        path = SETTINGS_DIR / _safe_user_filename(user_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, name, company, timezone, notifications, "
                "test_data_model, updated_at "
                "FROM settings WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return _row_to_settings(row) if row else None


def upsert_settings(user_id: str, payload: dict) -> dict:
    name = payload.get("name")
    company = payload.get("company")
    timezone = payload.get("timezone") or "America/Mexico_City"
    notifications = payload.get("notifications")
    # ponytail: test_data_model is a new optional column. The FE
    # picks the LLM (default 'gpt-4o-mini'); the BE persists
    # whatever the user chose and the test_data_generator reads it
    # back on every Test button click. Empty string falls back to
    # the DB default.
    test_data_model = (payload.get("test_data_model") or "").strip() or "gpt-4o-mini"
    if not isinstance(notifications, dict):
        notifications = {"calls": True, "qa": True, "weekly": False, "marketing": False}
    if not is_postgres():
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = SETTINGS_DIR / _safe_user_filename(user_id)
        existing = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except (json.JSONDecodeError, IOError):
                existing = {}
        merged = {**existing, **{
            "user_id": user_id, "name": name, "company": company,
            "timezone": timezone, "notifications": notifications,
            "test_data_model": test_data_model,
        }}
        merged = {k: v for k, v in merged.items() if v is not None}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        return merged
    notif_json = json.dumps(notifications)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO settings (user_id, name, company, timezone, "
                "notifications, test_data_model, updated_at) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s, NOW()) "
                "ON CONFLICT (user_id) DO UPDATE SET "
                "name = COALESCE(EXCLUDED.name, settings.name), "
                "company = COALESCE(EXCLUDED.company, settings.company), "
                "timezone = EXCLUDED.timezone, "
                "notifications = EXCLUDED.notifications, "
                "test_data_model = EXCLUDED.test_data_model, "
                "updated_at = NOW()",
                (user_id, name, company, timezone, notif_json, test_data_model),
            )
    return get_settings(user_id)
