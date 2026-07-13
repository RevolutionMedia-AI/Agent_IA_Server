"""Postgres-backed implementations for the phone_numbers table.

Schema (001 + 004):
  phone_numbers(
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL DEFAULT 'twilio',
    country TEXT NOT NULL DEFAULT '+1',
    number TEXT NOT NULL,
    display TEXT,
    label TEXT,
    agent TEXT REFERENCES agents(id) ON DELETE SET NULL,
    calls TEXT NOT NULL DEFAULT '0',
    status TEXT NOT NULL DEFAULT 'Active',
    twilio_account_sid TEXT,
    twilio_auth_token TEXT,
    sip_host TEXT,
    sip_username TEXT,
    sip_password TEXT,
    whatsapp_phone_number_id TEXT,
    whatsapp_access_token TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
  )
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_phone_numbers")

DATA_DIR = Path(__file__).resolve().parent / "data"
NUMBERS_FILE = DATA_DIR / "phone_numbers.json"


def _format(country: str, number: str) -> str:
    digits = re.sub(r"\D", "", number)
    country = country or "+1"
    if digits.startswith(country.lstrip("+")):
        digits = digits[len(country.lstrip("+")):]
    out = ""
    for i, d in enumerate(digits):
        if i and i % 2 == 0:
            out += " "
        out += d
    return f"{country} {out}".strip()


def _row_to_number(row: dict) -> dict:
    if row is None:
        return None
    out = dict(row)
    for k in ("created_at", "updated_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat() + "Z"
    return out


def list_numbers(user_id: str) -> list[dict]:
    if not is_postgres():
        if not NUMBERS_FILE.exists():
            return []
        try:
            with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return []
        return [n for n in data if n.get("user_id") == user_id]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, provider, country, number, display, label, name, campaign, agent, "
                "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                "sip_username, sip_password, whatsapp_phone_number_id, "
                "whatsapp_access_token, created_at, updated_at "
                "FROM phone_numbers WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            return [_row_to_number(r) for r in cur.fetchall()]


def get_number(number_id: str, user_id: str | None = None) -> dict | None:
    if not is_postgres():
        if not NUMBERS_FILE.exists():
            return None
        try:
            with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return None
        for n in data:
            if n.get("id") == number_id and (user_id is None or n.get("user_id") == user_id):
                return n
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute(
                    "SELECT id, user_id, provider, country, number, display, label, agent, "
                    "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                    "sip_username, sip_password, whatsapp_phone_number_id, "
                    "whatsapp_access_token, created_at, updated_at "
                    "FROM phone_numbers WHERE id = %s",
                    (number_id,),
                )
            else:
                cur.execute(
                    "SELECT id, user_id, provider, country, number, display, label, agent, "
                    "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                    "sip_username, sip_password, whatsapp_phone_number_id, "
                    "whatsapp_access_token, created_at, updated_at "
                    "FROM phone_numbers WHERE id = %s AND user_id = %s",
                    (number_id, user_id),
                )
            row = cur.fetchone()
            return _row_to_number(row) if row else None


def create_number(user_id: str, payload: dict) -> dict:
    number_id = f"num-{uuid.uuid4().hex[:8]}"
    display = _format(payload.get("country", "+1"), payload.get("number", ""))
    record = {
        "id": number_id,
        "user_id": user_id,
        "provider": payload.get("provider", "twilio"),
        "country": payload.get("country", "+1"),
        "number": payload.get("number"),
        "display": display,
        # ponytail: name wins over label/display so the operator sees a
        # friendly label first ("Soporte principal") before falling
        # back to the auto-generated display ("+52 55 1234 5678").
        "label": payload.get("label") or payload.get("name") or display,
        "name": payload.get("name"),
        "campaign": payload.get("campaign"),
        "agent": payload.get("agent"),
        "calls": "0",
        "status": "Active",
    }
    for opt in ("twilio_account_sid", "twilio_auth_token", "sip_host",
                 "sip_username", "sip_password",
                 "whatsapp_phone_number_id", "whatsapp_access_token"):
        v = payload.get(opt)
        if v:
            record[opt] = v
    if not is_postgres():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            data = []
        from datetime import datetime, timezone
        record["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        data.append(record)
        with open(NUMBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return record
    cols = ["id", "user_id", "provider", "country", "number", "display", "label",
            "name", "campaign",
            "agent", "calls", "status",
            "twilio_account_sid", "twilio_auth_token", "sip_host", "sip_username",
            "sip_password", "whatsapp_phone_number_id", "whatsapp_access_token"]
    placeholders = ", ".join(["%s"] * len(cols))
    values = [record[c] if c in record else None for c in cols]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO phone_numbers ({', '.join(cols)}) VALUES ({placeholders}) "
                "RETURNING id, user_id, provider, country, number, display, label, name, campaign, agent, "
                "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                "sip_username, sip_password, whatsapp_phone_number_id, "
                "whatsapp_access_token, created_at, updated_at",
                values,
            )
            row = cur.fetchone()
    return _row_to_number(row)


def update_number(number_id: str, user_id: str, payload: dict) -> dict | None:
    if not is_postgres():
        if not NUMBERS_FILE.exists():
            return None
        try:
            with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return None
        for n in data:
            if n.get("id") == number_id and n.get("user_id") == user_id:
                n.update({k: v for k, v in payload.items() if v is not None})
                with open(NUMBERS_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return n
        return None
    allowed = {"agent", "status", "label", "name", "campaign"}
    set_clauses, values = [], []
    for k, v in payload.items():
        if v is None or k not in allowed:
            continue
        set_clauses.append(f"{k} = %s")
        values.append(v)
    if not set_clauses:
        return get_number(number_id, user_id)
    set_clauses.append("updated_at = NOW()")
    values.extend([number_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE phone_numbers SET {', '.join(set_clauses)} "
                "WHERE id = %s AND user_id = %s "
                "RETURNING id, user_id, provider, country, number, display, label, name, campaign, agent, "
                "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                "sip_username, sip_password, whatsapp_phone_number_id, "
                "whatsapp_access_token, created_at, updated_at",
                values,
            )
            row = cur.fetchone()
    return _row_to_number(row) if row else None


def delete_number(number_id: str, user_id: str) -> bool:
    if not is_postgres():
        if not NUMBERS_FILE.exists():
            return False
        try:
            with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return False
        new_data = [n for n in data if not (n.get("id") == number_id and n.get("user_id") == user_id)]
        if len(new_data) == len(data):
            return False
        with open(NUMBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        return True
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM phone_numbers WHERE id = %s AND user_id = %s",
                (number_id, user_id),
            )
            return cur.rowcount > 0


def find_by_number(to_number: str, user_id: str | None = None) -> dict | None:
    """Lookup a phone number by its E.164-style 'number' field. Used by
    the inbound Twilio webhook to find which agent handles the call.
    The number is matched as a suffix (so "+521551234567' matches the
    '21551234567' stored row).
    """
    digits = re.sub(r"\D", "", to_number)
    if not is_postgres():
        if not NUMBERS_FILE.exists():
            return None
        try:
            with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return None
        for n in data:
            stored = re.sub(r"\D", "", n.get("number", ""))
            if stored and stored.endswith(digits) or digits.endswith(stored):
                if user_id is None or n.get("user_id") == user_id:
                    return n
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute(
                    "SELECT id, user_id, provider, country, number, display, label, agent, "
                    "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                    "sip_username, sip_password, whatsapp_phone_number_id, "
                    "whatsapp_access_token, created_at, updated_at "
                    "FROM phone_numbers WHERE %s LIKE '%%' || regexp_replace(number, '\\D', '', 'g') "
                    "OR regexp_replace(number, '\\D', '', 'g') LIKE '%%' || %s "
                    "ORDER BY length(number) DESC LIMIT 1",
                    (digits, digits),
                )
            else:
                cur.execute(
                    "SELECT id, user_id, provider, country, number, display, label, agent, "
                    "calls, status, twilio_account_sid, twilio_auth_token, sip_host, "
                    "sip_username, sip_password, whatsapp_phone_number_id, "
                    "whatsapp_access_token, created_at, updated_at "
                    "FROM phone_numbers WHERE user_id = %s AND "
                    "(%s LIKE '%%' || regexp_replace(number, '\\D', '', 'g') "
                    "OR regexp_replace(number, '\\D', '', 'g') LIKE '%%' || %s) "
                    "ORDER BY length(number) DESC LIMIT 1",
                    (user_id, digits, digits),
                )
            row = cur.fetchone()
            return _row_to_number(row) if row else None


def backfill_from_json() -> int:
    if not is_postgres() or not NUMBERS_FILE.exists():
        return 0
    try:
        with open(NUMBERS_FILE, "r", encoding="utf-8") as f:
            json_data = json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return 0
    if not json_data:
        return 0
    n = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for x in json_data:
                if not x.get("user_id") or not x.get("number"):
                    continue
                cur.execute("SELECT 1 FROM phone_numbers WHERE id = %s", (x["id"],))
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO phone_numbers (id, user_id, provider, country, number, "
                    "display, label, agent, calls, status, twilio_account_sid, "
                    "twilio_auth_token, sip_host, sip_username, sip_password, "
                    "whatsapp_phone_number_id, whatsapp_access_token, created_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    "COALESCE(%s, NOW()))",
                    (
                        x["id"], x["user_id"], x.get("provider", "twilio"),
                        x.get("country", "+1"), x["number"], x.get("display"),
                        x.get("label"), x.get("agent"),
                        x.get("calls", "0"), x.get("status", "Active"),
                        x.get("twilio_account_sid"), x.get("twilio_auth_token"),
                        x.get("sip_host"), x.get("sip_username"), x.get("sip_password"),
                        x.get("whatsapp_phone_number_id"), x.get("whatsapp_access_token"),
                        x.get("created_at"),
                    ),
                )
                n += 1
    if n:
        log.info("[db_phone_numbers] backfilled %d numbers from JSON to Postgres", n)
    return n
