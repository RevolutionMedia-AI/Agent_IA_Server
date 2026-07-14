"""Postgres-backed implementations for the tenants table.

This is the storage layer for TenantConfig (Twilio sub-account + per-call
config). The dataclass and the public facade live in
STT_server/domain/tenant.py — this module is the persistence backend only.

Schema (db/migrations/001_schema.sql):
  tenants(
    tenant_id            TEXT PRIMARY KEY,
    user_id              TEXT REFERENCES users(id) ON DELETE CASCADE,
    name                 TEXT,
    twilio_account_sid   TEXT,
    twilio_auth_token    TEXT,
    twilio_phone_number  TEXT,
    custom_prompt        TEXT,
    tts_provider         TEXT NOT NULL DEFAULT 'elevenlabs',
    preferred_language   TEXT NOT NULL DEFAULT 'es',
    webhook_configured   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ
  )

Per the Opcion 2 design, provider API keys (openai_api_key, etc.) are NOT
stored on tenants — they live on tools_integrations and are resolved at
call time by credentials_resolver. The legacy fields stay on TenantConfig
for backwards-compat with the route layer but are stripped on DB writes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_tenants")

DATA_DIR = Path(__file__).resolve().parent / "data"
TENANTS_FILE = DATA_DIR / "tenants.json"

# ponytail: the tenant dict shape carries all the columns the route
# layer reads via TenantConfig.to_dict(), plus the four legacy provider
# key fields (always None on a fresh row). The SELECT lists below match
# this ordering so callers can index the dict directly.
_TENANT_COLS = (
    "tenant_id, user_id, name, twilio_account_sid, twilio_auth_token, "
    "twilio_phone_number, custom_prompt, tts_provider, preferred_language, "
    "webhook_configured, created_at, updated_at"
)

# Opcion 2: these fields exist on the TenantConfig dataclass (legacy) but
# are NEVER written to the tenants table. The dict adapter strips them
# on the way in and defaults them to None on the way out.
_LEGACY_KEY_FIELDS = frozenset({
    "openai_api_key", "elevenlabs_api_key", "elevenlabs_voice_id", "deepgram_api_key",
})


def _row_to_tenant(row: dict | None) -> dict | None:
    """Map a DB row to the dict shape callers expect. Datetime -> ISO
    string; the facade in domain/tenant.py turns that back into float
    epoch seconds for the dataclass."""
    if row is None:
        return None
    out = dict(row)
    for k in ("created_at", "updated_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    # The route layer's to_dict() reads these even when secrets are masked,
    # so carry them as None to keep the shape stable.
    for f in _LEGACY_KEY_FIELDS:
        out.setdefault(f, None)
    return out


# ── JSON backend (preserved for local dev, no DATABASE_URL) ────────────

def _read_json() -> list[dict]:
    if not TENANTS_FILE.exists():
        return []
    try:
        with open(TENANTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return []


def _write_json(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TENANTS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# ── Public API ────────────────────────────────────────────────────────

def list_tenants(user_id: Optional[str] = None) -> list[dict]:
    if not is_postgres():
        rows = _read_json()
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
        return [_row_to_tenant(r) for r in rows]
    sql = f"SELECT {_TENANT_COLS} FROM tenants"
    args: tuple = ()
    if user_id is not None:
        sql += " WHERE user_id = %s"
        args = (user_id,)
    sql += " ORDER BY created_at DESC"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return [_row_to_tenant(r) for r in cur.fetchall()]


def get_tenant(tenant_id: str, user_id: Optional[str] = None) -> dict | None:
    if not is_postgres():
        for r in _read_json():
            if r.get("tenant_id") == tenant_id and (user_id is None or r.get("user_id") == user_id):
                return _row_to_tenant(r)
        return None
    sql = f"SELECT {_TENANT_COLS} FROM tenants WHERE tenant_id = %s"
    args: tuple = (tenant_id,)
    if user_id is not None:
        sql += " AND user_id = %s"
        args = (tenant_id, user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            row = cur.fetchone()
            return _row_to_tenant(row) if row else None


def get_tenant_by_phone(phone_number: str, user_id: Optional[str] = None) -> dict | None:
    """Twilio webhook resolves the tenant from the dialed number. The
    incoming number is E.164; the stored number may be missing the
    country prefix. Match on the digit-only suffix (same approach as
    db_phone_numbers.find_by_number)."""
    if not phone_number:
        return None
    digits = "".join(c for c in phone_number if c.isdigit())
    if not is_postgres():
        for r in _read_json():
            stored = "".join(c for c in (r.get("twilio_phone_number") or "") if c.isdigit())
            if stored and (stored.endswith(digits) or digits.endswith(stored)):
                if user_id is None or r.get("user_id") == user_id:
                    return _row_to_tenant(r)
        return None
    sql = (
        f"SELECT {_TENANT_COLS} FROM tenants "
        "WHERE %s LIKE '%%' || regexp_replace(twilio_phone_number, '\\D', '', 'g') "
        "OR regexp_replace(twilio_phone_number, '\\D', '', 'g') LIKE '%%' || %s"
    )
    args: tuple = (digits, digits)
    if user_id is not None:
        sql = (
            f"SELECT {_TENANT_COLS} FROM tenants WHERE user_id = %s AND ("
            "%s LIKE '%%' || regexp_replace(twilio_phone_number, '\\D', '', 'g') "
            "OR regexp_replace(twilio_phone_number, '\\D', '', 'g') LIKE '%%' || %s)"
        )
        args = (user_id, digits, digits)
    sql += " ORDER BY length(twilio_phone_number) DESC LIMIT 1"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            row = cur.fetchone()
            return _row_to_tenant(row) if row else None


def upsert_tenant(tenant: dict) -> dict:
    """Insert or update a tenant. Dict-in / dict-out matching the JSON
    shape. tenant_id is required; everything else is optional.

    Provider key fields (openai_api_key, etc.) are silently dropped per
    Opcion 2 — they live on tools_integrations, not on tenants."""
    tenant_id = tenant.get("tenant_id")
    if not tenant_id:
        raise ValueError("tenant_id is required")
    clean = {k: v for k, v in tenant.items() if k not in _LEGACY_KEY_FIELDS}
    if not is_postgres():
        rows = _read_json()
        for i, r in enumerate(rows):
            if r.get("tenant_id") == tenant_id:
                rows[i] = {**r, **clean}
                _write_json(rows)
                return _row_to_tenant(rows[i])
        rows.append(clean)
        _write_json(rows)
        return _row_to_tenant(clean)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants ("
                "  tenant_id, user_id, name, twilio_account_sid, twilio_auth_token, "
                "  twilio_phone_number, custom_prompt, tts_provider, preferred_language, "
                "  webhook_configured, created_at, updated_at"
                ") VALUES ("
                "  %s, %s, %s, %s, %s, %s, %s, "
                "  COALESCE(NULLIF(%s, ''), 'elevenlabs'), "
                "  COALESCE(NULLIF(%s, ''), 'es'), "
                "  %s, "
                "  COALESCE(%s::timestamptz, NOW()), COALESCE(%s::timestamptz, NOW())"
                ") ON CONFLICT (tenant_id) DO UPDATE SET "
                "  user_id = EXCLUDED.user_id, "
                "  name = EXCLUDED.name, "
                "  twilio_account_sid = EXCLUDED.twilio_account_sid, "
                "  twilio_auth_token = EXCLUDED.twilio_auth_token, "
                "  twilio_phone_number = EXCLUDED.twilio_phone_number, "
                "  custom_prompt = EXCLUDED.custom_prompt, "
                "  tts_provider = EXCLUDED.tts_provider, "
                "  preferred_language = EXCLUDED.preferred_language, "
                "  webhook_configured = EXCLUDED.webhook_configured, "
                "  updated_at = NOW() "
                f"RETURNING {_TENANT_COLS}",
                (
                    clean.get("tenant_id"),
                    clean.get("user_id"),
                    clean.get("name"),
                    clean.get("twilio_account_sid"),
                    clean.get("twilio_auth_token"),
                    clean.get("twilio_phone_number"),
                    clean.get("custom_prompt"),
                    clean.get("tts_provider") or "",
                    clean.get("preferred_language") or "",
                    bool(clean.get("webhook_configured", False)),
                    clean.get("created_at"),
                    clean.get("updated_at"),
                ),
            )
            row = cur.fetchone()
    return _row_to_tenant(row) if row else _row_to_tenant(clean)


def delete_tenant(tenant_id: str, user_id: Optional[str] = None) -> bool:
    if not is_postgres():
        rows = _read_json()
        new_rows = [
            r for r in rows
            if not (r.get("tenant_id") == tenant_id
                    and (user_id is None or r.get("user_id") == user_id))
        ]
        if len(new_rows) == len(rows):
            return False
        _write_json(new_rows)
        return True
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute("DELETE FROM tenants WHERE tenant_id = %s", (tenant_id,))
            else:
                cur.execute(
                    "DELETE FROM tenants WHERE tenant_id = %s AND user_id = %s",
                    (tenant_id, user_id),
                )
            return cur.rowcount > 0


def list_tenant_ids() -> list[str]:
    """Cheap variant of list_tenants() that returns just the IDs."""
    if not is_postgres():
        return [r.get("tenant_id") for r in _read_json() if r.get("tenant_id")]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tenant_id FROM tenants ORDER BY tenant_id")
            return [r["tenant_id"] for r in cur.fetchall()]


# ── Startup helpers ──────────────────────────────────────────────────

def backfill_from_json() -> int:
    """One-shot helper: if data/tenants.json exists with rows the DB
    doesn't have, copy them over. Called at server startup so existing
    local-dev users don't lose their tenants on the first Postgres deploy.
    The JSON file doesn't exist yet (current in-memory store is ephemeral)
    so this is a no-op on greenfield — but matches the db_users /
    db_agents pattern so the same hook runs everywhere."""
    if not is_postgres() or not TENANTS_FILE.exists():
        return 0
    try:
        with open(TENANTS_FILE, "r", encoding="utf-8") as f:
            json_data = json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return 0
    if not json_data:
        return 0
    n = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for t in json_data:
                tid = t.get("tenant_id")
                if not tid:
                    continue
                cur.execute("SELECT 1 FROM tenants WHERE tenant_id = %s", (tid,))
                if cur.fetchone():
                    continue
                clean = {k: v for k, v in t.items() if k not in _LEGACY_KEY_FIELDS}
                cur.execute(
                    "INSERT INTO tenants ("
                    "  tenant_id, user_id, name, twilio_account_sid, twilio_auth_token, "
                    "  twilio_phone_number, custom_prompt, tts_provider, preferred_language, "
                    "  webhook_configured, created_at"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, "
                    "  COALESCE(NULLIF(%s, ''), 'elevenlabs'), "
                    "  COALESCE(NULLIF(%s, ''), 'es'), "
                    "  %s, COALESCE(%s::timestamptz, NOW()))",
                    (
                        clean.get("tenant_id"),
                        clean.get("user_id"),
                        clean.get("name"),
                        clean.get("twilio_account_sid"),
                        clean.get("twilio_auth_token"),
                        clean.get("twilio_phone_number"),
                        clean.get("custom_prompt"),
                        clean.get("tts_provider") or "",
                        clean.get("preferred_language") or "",
                        bool(clean.get("webhook_configured", False)),
                        clean.get("created_at"),
                    ),
                )
                n += 1
    if n:
        log.info("[db_tenants] backfilled %d tenants from JSON to Postgres", n)
    return n
