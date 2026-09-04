"""Postgres-backed CRUD for the per-user Twilio credential store.

Twilio sub-accounts used to live in `agent_tools` (one row per user,
`id='twilio'`, `agent_id='__shared__'`). Multi-sub-account organisations
needed to register more than one SID+Token pair; the old single-row
shape couldn't carry two. Migration 019 introduces a dedicated
`twilio_credentials` table so the Settings UI can list, edit and remove
each set independently.

ponytail: storage convention. Account SID and Auth Token are encrypted
at rest via Fernet (security/credentials.encrypt_value) just like the
`credentials` JSONB blob on agent_tools. We store the last 4 chars of
the SID in plaintext so the Settings list can render a card
("…AB12") without round-tripping every row through decrypt.

ponytail: JSON-fallback path mirrors the SQL contract for local dev.
The file lives under STT_server/data/twilio_credentials.json — it
ships only when DATABASE_URL is unset (ephemeral dev mode). Production
deployments always go through Postgres.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from STT_server.db import get_conn, is_postgres
from STT_server.security.credentials import encrypt_value, decrypt_value

log = logging.getLogger("stt_server.db_twilio_credentials")

DATA_DIR = Path(__file__).resolve().parent / "data"
CREDENTIALS_FILE = DATA_DIR / "twilio_credentials.json"

# ponytail: never expose the full SID or token in list responses. The
# FE only needs the last 4 chars of the SID (routing numbers live long
# enough that the suffix is a stable identifier; the FE also shows the
# user-given name as the primary label). Auth tokens never leak, not
# even truncated.
SID_LAST4_LEN = 4


def _row_to_dict(row: dict, *, include_secrets: bool = False) -> dict:
    """Map a Postgres row to the JSON shape the FE consumes.

    `include_secrets=True` is reserved for the per-row reveal endpoint
    and is never used by the bulk list endpoint.
    """
    if row is None:
        return None
    out = {
        "id": row["id"],
        "name": row.get("name") or "",
        "account_sid_last4": row.get("account_sid_last4") or "",
        "status": row.get("status") or "unknown",
        "last_tested_at": (
            row["last_tested_at"].isoformat() + "Z"
            if hasattr(row.get("last_tested_at"), "isoformat")
            else row.get("last_tested_at")
        ),
        "last_test_message": row.get("last_test_message") or "",
        "created_at": (
            row["created_at"].isoformat() + "Z"
            if hasattr(row.get("created_at"), "isoformat")
            else row.get("created_at")
        ),
        "updated_at": (
            row["updated_at"].isoformat() + "Z"
            if hasattr(row.get("updated_at"), "isoformat")
            else row.get("updated_at")
        ),
    }
    if include_secrets:
        out["account_sid"] = decrypt_value(row["account_sid_encrypted"]) if row.get("account_sid_encrypted") else ""
        out["auth_token"] = decrypt_value(row["auth_token_encrypted"]) if row.get("auth_token_encrypted") else ""
    return out


def _last4(account_sid: str) -> str:
    return account_sid[-SID_LAST4_LEN:] if account_sid else ""


# ── JSON-file fallback (mirrors the Postgres shape) ───────────────────────


def _load_file() -> list[dict]:
    if not CREDENTIALS_FILE.exists():
        return []
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return []


def _save_file(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _row_from_file(r: dict, *, include_secrets: bool = False) -> dict:
    out = {
        "id": r.get("id"),
        "name": r.get("name") or "",
        "account_sid_last4": r.get("account_sid_last4") or "",
        "status": r.get("status") or "unknown",
        "last_tested_at": r.get("last_tested_at"),
        "last_test_message": r.get("last_test_message") or "",
        "created_at": r.get("created_at"),
        "updated_at": r.get("updated_at"),
    }
    if include_secrets:
        out["account_sid"] = decrypt_value(r.get("account_sid_encrypted") or "")
        out["auth_token"] = decrypt_value(r.get("auth_token_encrypted") or "")
    return out


# ── CRUD ─────────────────────────────────────────────────────────────────


def list_credentials(user_id: str) -> list[dict]:
    """List the user's credentials. NEVER returns the auth_token."""
    if not user_id:
        return []
    if not is_postgres():
        rows = [r for r in _load_file() if r.get("user_id") == user_id]
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return [_row_from_file(r) for r in rows]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, account_sid_last4, status, last_tested_at, "
                "last_test_message, created_at, updated_at "
                "FROM twilio_credentials WHERE user_id = %s "
                "ORDER BY created_at DESC",
                (user_id,),
            )
            return [_row_to_dict(r) for r in cur.fetchall()]


def get_credential(credential_id: str, user_id: str, *, include_secrets: bool = False) -> Optional[dict]:
    """Fetch one credential. Only the explicit reveal endpoint sets
    `include_secrets=True`; everywhere else the secrets stay encrypted."""
    if not credential_id or not user_id:
        return None
    if not is_postgres():
        for r in _load_file():
            if r.get("id") == credential_id and r.get("user_id") == user_id:
                return _row_from_file(r, include_secrets=include_secrets)
        return None
    cols = (
        "id, name, account_sid_last4, status, last_tested_at, "
        "last_test_message, created_at, updated_at"
    )
    if include_secrets:
        cols = "id, name, account_sid_last4, account_sid_encrypted, auth_token_encrypted, status, last_tested_at, last_test_message, created_at, updated_at"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM twilio_credentials "
                "WHERE id = %s AND user_id = %s",
                (credential_id, user_id),
            )
            row = cur.fetchone()
            return _row_to_dict(row, include_secrets=include_secrets) if row else None


def create_credential(user_id: str, *, name: str, account_sid: str, auth_token: str, status: str = "unknown", last_test_message: str = "") -> dict:
    """Persist a new credential. Encrypts the SID and token; the last4
    of the SID is stored plaintext for display."""
    cid = f"twcred-{uuid.uuid4().hex[:12]}"
    sid_enc = encrypt_value(account_sid)
    tok_enc = encrypt_value(auth_token)
    if not is_postgres():
        rows = _load_file()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rows.append({
            "id": cid,
            "user_id": user_id,
            "name": name.strip(),
            "account_sid_encrypted": sid_enc,
            "auth_token_encrypted": tok_enc,
            "account_sid_last4": _last4(account_sid),
            "status": status,
            "last_tested_at": None,
            "last_test_message": last_test_message,
            "created_at": now,
            "updated_at": now,
        })
        _save_file(rows)
        return get_credential(cid, user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO twilio_credentials "
                "(id, user_id, name, account_sid_encrypted, auth_token_encrypted, "
                " account_sid_last4, status, last_test_message) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id, name, account_sid_last4, status, last_tested_at, "
                "last_test_message, created_at, updated_at",
                (cid, user_id, name.strip(), sid_enc, tok_enc, _last4(account_sid), status, last_test_message),
            )
            row = cur.fetchone()
    return _row_to_dict(row)


def update_credential(credential_id: str, user_id: str, *, name: Optional[str] = None, account_sid: Optional[str] = None, auth_token: Optional[str] = None, status: Optional[str] = None, last_test_message: Optional[str] = None) -> Optional[dict]:
    """Patch a credential. Only fields present in the kwargs are touched.

    ponytail: callers that want to flip `status` only (after a Test call)
    pass `status` + `last_test_message` and leave the secrets alone.
    Callers that rotate the SID/Token pass the new values; the row's
    `account_sid_last4` is recomputed when the SID changes."""
    if not credential_id or not user_id:
        return None
    if not is_postgres():
        rows = _load_file()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for r in rows:
            if r.get("id") == credential_id and r.get("user_id") == user_id:
                if name is not None:
                    r["name"] = name.strip()
                if account_sid is not None:
                    r["account_sid_encrypted"] = encrypt_value(account_sid)
                    r["account_sid_last4"] = _last4(account_sid)
                if auth_token is not None:
                    r["auth_token_encrypted"] = encrypt_value(auth_token)
                if status is not None:
                    r["status"] = status
                if last_test_message is not None:
                    r["last_test_message"] = last_test_message
                r["updated_at"] = now
                _save_file(rows)
                return get_credential(credential_id, user_id)
        return None
    sets: list[str] = []
    values: list = []
    if name is not None:
        sets.append("name = %s")
        values.append(name.strip())
    if account_sid is not None:
        sets.append("account_sid_encrypted = %s")
        values.append(encrypt_value(account_sid))
        sets.append("account_sid_last4 = %s")
        values.append(_last4(account_sid))
    if auth_token is not None:
        sets.append("auth_token_encrypted = %s")
        values.append(encrypt_value(auth_token))
    if status is not None:
        sets.append("status = %s")
        values.append(status)
    if last_test_message is not None:
        sets.append("last_test_message = %s")
        values.append(last_test_message)
    if not sets:
        return get_credential(credential_id, user_id)
    sets.append("updated_at = NOW()")
    values.extend([credential_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE twilio_credentials SET {', '.join(sets)} "
                "WHERE id = %s AND user_id = %s "
                "RETURNING id, name, account_sid_last4, status, last_tested_at, "
                "last_test_message, created_at, updated_at",
                values,
            )
            row = cur.fetchone()
    return _row_to_dict(row) if row else None


def delete_credential(credential_id: str, user_id: str) -> bool:
    if not credential_id or not user_id:
        return False
    if not is_postgres():
        rows = _load_file()
        new = [r for r in rows if not (r.get("id") == credential_id and r.get("user_id") == user_id)]
        if len(new) == len(rows):
            return False
        _save_file(new)
        return True
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM twilio_credentials WHERE id = %s AND user_id = %s",
                (credential_id, user_id),
            )
            return cur.rowcount > 0


def find_for_phone_number(credential_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    """Resolve a credential by id and decrypt SID+Token for the
    phone-number / call path. Caller passes user_id to keep the query
    tenant-scoped (defence-in-depth)."""
    return get_credential(credential_id, user_id, include_secrets=True) if user_id else None
