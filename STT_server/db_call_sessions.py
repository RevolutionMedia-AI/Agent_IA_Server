"""Postgres-backed implementations for the call_sessions table.

The runtime today is in-memory (STT_server/services/session_runtime.py).
This module is the DB-backed equivalent, structured to match the other
db_*.py modules so the migration is a mechanical swap: replace each
`sessions[key] = ...` in the runtime with the corresponding function
call here. The runtime is intentionally NOT modified yet — flip the
imports in session_runtime.py when the cutover lands.

Schema (db/migrations/001_schema.sql):
  call_sessions(
    session_key        TEXT PRIMARY KEY,
    tenant_id          TEXT REFERENCES tenants(tenant_id) ON DELETE SET NULL,
    call_sid           TEXT,
    preferred_language TEXT,
    tts_provider       TEXT,
    custom_prompt      TEXT,
    assistant_speaking BOOLEAN NOT NULL DEFAULT FALSE,
    closed             BOOLEAN NOT NULL DEFAULT FALSE,
    started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at           TIMESTAMPTZ
  )

Only DB-persistable fields are written here. Runtime state (audio
queues, tasks, speech_frames) stays in the in-memory CallSession and
is NOT round-tripped to Postgres — those belong in Redis, not the DB.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_call_sessions")

DATA_DIR = Path(__file__).resolve().parent / "data"
CALL_SESSIONS_FILE = DATA_DIR / "call_sessions.json"

_SESSION_COLS = (
    "session_key, tenant_id, call_sid, preferred_language, "
    "tts_provider, custom_prompt, assistant_speaking, closed, "
    "started_at, ended_at"
)

# ponytail: 2026-09-03 — dashboard live-calls needs per-tenant counts.
# The runtime never stored user_id or agent_id on the session row, so
# `live_calls` always read as 0 even with active calls. We read them
# defensively off the GET response: callers that send `agent_id` +
# `user_id` get them back, the rest keep working.
_EXTRA_COLS = "user_id, agent_id"


def _row_to_session(row):
    if row is None:
        return None
    out = dict(row)
    for k in ("started_at", "ended_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


# ── JSON backend (preserved for local dev, no DATABASE_URL) ────────────

def _read_json():
    if not CALL_SESSIONS_FILE.exists():
        return []
    try:
        with open(CALL_SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return []


def _write_json(rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALL_SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _float_to_iso(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Public API ────────────────────────────────────────────────────────

def register_session(
    session_key: str,
    *,
    tenant_id: Optional[str] = None,
    call_sid: Optional[str] = None,
    preferred_language: Optional[str] = None,
    tts_provider: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    started_at: Optional[float] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> dict:
    """Persist a new call session. Idempotent on session_key — a second
    call with the same key returns the existing row instead of erroring.

    Mirrors session_runtime.register_session: one row per call, used
    for /dashboard history and post-mortem on dropped WebSockets.

    ponytail: 2026-09-03 — `user_id` + `agent_id` are optional kwargs.
    Existing callers that don't pass them keep working; the dashboard
    only counts live rows that the BE actually registered with these
    fields. Without them we can't tell whose call is active, so the
    'Live calls' KPI always read 0.
    """
    if not is_postgres():
        rows = _read_json()
        for s in rows:
            if s.get("session_key") == session_key:
                # ponytail: refresh ownership on the JSON backend too so
                # a re-registration (e.g. WS reconnect) carries the agent
                # forward into the row that already exists.
                if user_id is not None:
                    s["user_id"] = user_id
                if agent_id is not None:
                    s["agent_id"] = agent_id
                return s
        record = {
            "session_key": session_key,
            "tenant_id": tenant_id,
            "call_sid": call_sid,
            "preferred_language": preferred_language,
            "tts_provider": tts_provider,
            "custom_prompt": custom_prompt,
            "assistant_speaking": False,
            "closed": False,
            "started_at": _float_to_iso(started_at) if started_at is not None else None,
            "ended_at": None,
            "user_id": user_id,
            "agent_id": agent_id,
        }
        rows.append(record)
        _write_json(rows)
        return record
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO call_sessions "
                "(session_key, tenant_id, call_sid, preferred_language, "
                " tts_provider, custom_prompt, started_at, user_id, agent_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, "
                "        COALESCE(to_timestamp(%s), NOW()), %s, %s) "
                "ON CONFLICT (session_key) DO NOTHING "
                f"RETURNING {_SESSION_COLS}",
                (session_key, tenant_id, call_sid, preferred_language,
                 tts_provider, custom_prompt, started_at, user_id, agent_id),
            )
            row = cur.fetchone()
            if row:
                out = _row_to_session(row)
                out["user_id"] = user_id
                out["agent_id"] = agent_id
                return out
    existing = get_session(session_key) or {}
    if user_id is not None:
        existing["user_id"] = user_id
    if agent_id is not None:
        existing["agent_id"] = agent_id
    return existing


def get_session(session_key: str) -> Optional[dict]:
    if not is_postgres():
        for s in _read_json():
            if s.get("session_key") == session_key:
                return s
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_SESSION_COLS} FROM call_sessions WHERE session_key = %s",
                (session_key,),
            )
            row = cur.fetchone()
            return _row_to_session(row) if row else None


def list_open_sessions() -> list:
    """Return every still-open session, marking them closed=true. Used
    at server startup so a new instance can emit cleanup events and
    write usage records for calls the previous instance dropped on the
    floor when it crashed."""
    if not is_postgres():
        rows = _read_json()
        recovered = []
        for s in rows:
            if not s.get("closed"):
                s["closed"] = True
                s["ended_at"] = _float_to_iso(time.time())
                recovered.append(s)
        if recovered:
            _write_json(rows)
        return recovered
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE call_sessions SET closed = TRUE, ended_at = NOW() "
                "WHERE closed = FALSE "
                f"RETURNING {_SESSION_COLS}"
            )
            return [_row_to_session(r) for r in cur.fetchall()]


def count_open_sessions(*, user_id: Optional[str] = None) -> int:
    """Live counter for the dashboard. READ-ONLY.

    `list_open_sessions()` flips every open row to closed=True, which is
    correct at server startup (recover dropped calls) but destructive on
    a polling dashboard. Use this helper instead — it never mutates.
    Optional `user_id` narrows the count to a single owner's calls so
    multi-tenant dashboards don't leak peer counts.
    """
    if not is_postgres():
        return sum(
            1
            for s in _read_json()
            if not s.get("closed")
            and (user_id is None or s.get("user_id") == user_id)
        )
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute(
                    "SELECT COUNT(*) FROM call_sessions WHERE closed = FALSE"
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM call_sessions "
                    "WHERE closed = FALSE AND user_id = %s",
                    (user_id,),
                )
            row = cur.fetchone() or {}
    return int(row.get("count") or 0)


def list_active_for_user(
    *, user_id: Optional[str] = None, limit: int = 50
) -> list[dict]:
    """Active rows for the dashboard, READ-ONLY.

    Returns the latest open rows (one per call) so the FE can render a
    roster. Rows without `user_id` (legacy callers that pre-date the
    field) are excluded from the per-user view so we don't leak
    anonymous calls into a tenant's dashboard.
    """
    if not is_postgres():
        rows = [
            s
            for s in _read_json()
            if not s.get("closed")
            and (user_id is None or s.get("user_id") == user_id)
        ]
        rows.sort(key=lambda s: s.get("started_at") or "", reverse=True)
        return rows[: max(0, limit)]

    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute(
                    f"SELECT {_SESSION_COLS} FROM call_sessions "
                    "WHERE closed = FALSE "
                    "ORDER BY started_at DESC LIMIT %s",
                    (max(0, limit),),
                )
            else:
                cur.execute(
                    f"SELECT {_SESSION_COLS} FROM call_sessions "
                    "WHERE closed = FALSE AND user_id = %s "
                    "ORDER BY started_at DESC LIMIT %s",
                    (user_id, max(0, limit)),
                )
            return [_row_to_session(r) for r in cur.fetchall()]


def close_session(session_key: str, ended_at: Optional[float] = None) -> bool:
    """Mark a session closed. Returns False if the row was already closed
    or doesn't exist (idempotent — calling twice is fine)."""
    if not is_postgres():
        rows = _read_json()
        changed = False
        for s in rows:
            if s.get("session_key") == session_key and not s.get("closed"):
                s["closed"] = True
                s["ended_at"] = _float_to_iso(ended_at) if ended_at is not None else _float_to_iso(time.time())
                changed = True
        if changed:
            _write_json(rows)
        return changed
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE call_sessions SET closed = TRUE, ended_at = "
                "  COALESCE(to_timestamp(%s), NOW()) "
                "WHERE session_key = %s AND closed = FALSE",
                (ended_at, session_key),
            )
            return cur.rowcount > 0


def set_assistant_speaking(session_key: str, speaking: bool) -> bool:
    """Toggle the assistant_speaking flag. Hot-path call from the TTS
    worker — a single UPDATE on the JSON backend too (no read-modify-
    write of the whole list)."""
    if not is_postgres():
        rows = _read_json()
        changed = False
        for s in rows:
            if s.get("session_key") == session_key:
                s["assistant_speaking"] = bool(speaking)
                changed = True
        if changed:
            _write_json(rows)
        return changed
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE call_sessions SET assistant_speaking = %s "
                "WHERE session_key = %s",
                (bool(speaking), session_key),
            )
            return cur.rowcount > 0


def list_sessions_for_tenant(tenant_id: str, limit: int = 100) -> list:
    """Convenience for /dashboard: most recent N sessions for a tenant.
    Used by the future-DB-backed session_history endpoint."""
    if not is_postgres():
        rows = [s for s in _read_json() if s.get("tenant_id") == tenant_id]
        rows.sort(key=lambda s: s.get("started_at") or "", reverse=True)
        return rows[:limit]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_SESSION_COLS} FROM call_sessions "
                "WHERE tenant_id = %s "
                "ORDER BY started_at DESC LIMIT %s",
                (tenant_id, limit),
            )
            return [_row_to_session(r) for r in cur.fetchall()]
