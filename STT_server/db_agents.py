"""Postgres-backed implementations for the agents table.

Mirrors the JSON-file shape returned by routes/api.py today so the
route layer can swap one import without changing call sites.

Schema (001_schema.sql + 006_agent_runtime_params.sql):
  agents(
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    voice TEXT,
    voice_id TEXT,
    language TEXT NOT NULL DEFAULT 'English',
    campaign TEXT,
    status TEXT NOT NULL DEFAULT 'Active',
    description TEXT,
    tone TEXT,
    prompt TEXT,
    welcome_message TEXT,
    stt_provider TEXT,
    stt_model TEXT,
    tts_provider TEXT,
    tts_model TEXT,
    llm_provider TEXT,
    llm_model TEXT,
    -- runtime knobs added by 006_agent_runtime_params.sql
    llm_temperature REAL,     -- 0.0..2.0 (NULL = adapter default 0.2)
    llm_max_tokens  INTEGER,  -- >0..4096  (NULL = config.MAX_RESPONSE_TOKENS)
    tts_speed      REAL,     -- 0.5..2.0  (NULL = provider default)
    -- per-agent idle/silence detection (008_agent_idle_settings.sql).
    -- NULL on every column = fall back to global IDLE_SILENCE_TIMEOUT_SEC
    -- (the legacy single-timeout-then-close behaviour).
    idle_enabled                BOOLEAN,     -- explicit opt-in
    idle_first_timeout_sec      INTEGER,     -- >0
    idle_first_message          TEXT,        -- <=1000 chars
    idle_subsequent_timeout_sec INTEGER,     -- >0
    idle_final_message          TEXT,        -- <=1000 chars
    idle_disconnect_timeout_sec INTEGER,     -- >0
    idle_max_attempts           INTEGER,     -- 1..10
    calls TEXT NOT NULL DEFAULT '0',
    perf INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
  )
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_agents")

# ponytail: idle/silence-detection columns added by 008_agent_idle_settings.sql.
# Single source of truth for SELECT / INSERT / UPDATE — every other place that
# lists agent columns reads from here so a future field can't be silently
# dropped by half the call sites (the 006 migration bug pattern).
_IDLE_COLS = (
    "idle_enabled, idle_first_timeout_sec, idle_first_message, "
    "idle_subsequent_timeout_sec, idle_final_message, "
    "idle_disconnect_timeout_sec, idle_max_attempts"
)

# Every SELECT/UPDATE/INSERT RETURNING in this module uses the same column
# list. Keeping it as a constant stops "I added the column to the SELECT but
# forgot the INSERT" bugs — one place to extend when 009 lands.
_AGENT_COLS = (
    "id, user_id, name, voice, voice_id, language, campaign, status, "
    "description, tone, prompt, welcome_message, "
    "stt_provider, stt_model, tts_provider, tts_model, "
    "llm_provider, llm_model, "
    "llm_temperature, llm_max_tokens, tts_speed, "
    f"{_IDLE_COLS}, "
    "calls, perf, created_at, updated_at"
)

# ponytail: keep the JSON-file path so we can read from it on first
# boot to backfill Postgres when the migration runs against a project
# that already has data in data/agents.json. Reads go to Postgres on
# a DATABASE_URL deployment; reads from the JSON file otherwise.
DATA_DIR = Path(__file__).resolve().parent / "data"
AGENTS_FILE = DATA_DIR / "agents.json"


def _row_to_agent(row: dict) -> dict:
    """Map a DB row to the JSON shape the FE expects."""
    if row is None:
        return None
    out = dict(row)
    # ponytail: the FE reads "created_at" as ISO string. psycopg2 hands
    # us a datetime; convert so the FE doesn't choke.
    for k in ("created_at", "updated_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat() + "Z"
    # The DB stores a few optional columns as None; the FE is happy with
    # either null or empty string but null is the contract we kept.
    return out


def list_agents(user_id: str) -> list[dict]:
    if not is_postgres():
        # JSON fallback - same shape the route layer expects.
        if not AGENTS_FILE.exists():
            return []
        try:
            with open(AGENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return []
        return [a for a in data if a.get("user_id") == user_id]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_AGENT_COLS} FROM agents "
                "WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            return [_row_to_agent(r) for r in cur.fetchall()]


def get_agent(agent_id: str, user_id: str | None = None) -> dict | None:
    """Lookup one agent. user_id is optional because the call path
    may pass just the agent id (Twilio custom parameter)."""
    if not is_postgres():
        if not AGENTS_FILE.exists():
            return None
        try:
            with open(AGENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, TypeError):
            return None
        for a in data:
            if a.get("id") == agent_id:
                if user_id is None or a.get("user_id") == user_id:
                    return a
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_id is None:
                cur.execute(
                    f"SELECT {_AGENT_COLS} FROM agents WHERE id = %s",
                    (agent_id,),
                )
            else:
                cur.execute(
                    f"SELECT {_AGENT_COLS} FROM agents "
                    "WHERE id = %s AND user_id = %s",
                    (agent_id, user_id),
                )
            row = cur.fetchone()
            return _row_to_agent(row) if row else None


def create_agent(user_id: str, payload: dict) -> dict:
    agent_id = f"agent-{uuid.uuid4().hex[:8]}"
    if not is_postgres():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(AGENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            data = []
        new_agent = {
            "id": agent_id,
            "user_id": user_id,
            "calls": "0",
            "perf": 0,
            "created_at": _now_iso(),
            **payload,
        }
        data.append(new_agent)
        with open(AGENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return new_agent
    cols = ["id", "user_id", "name", "calls", "perf", "voice", "voice_id", "language",
            "campaign", "status", "description", "tone", "prompt", "welcome_message",
            "stt_provider", "stt_model", "tts_provider", "tts_model",
            "llm_provider", "llm_model",
            "llm_temperature", "llm_max_tokens", "tts_speed",
            "idle_enabled", "idle_first_timeout_sec", "idle_first_message",
            "idle_subsequent_timeout_sec", "idle_final_message",
            "idle_disconnect_timeout_sec", "idle_max_attempts"]
    placeholders = ", ".join(["%s"] * len(cols))
    insert_cols = ", ".join(cols)
    values = [agent_id, user_id, payload.get("name", "Untitled"),
              payload.get("calls", "0"), int(payload.get("perf", 0)),
              payload.get("voice"), payload.get("voice_id"),
              payload.get("language", "English"), payload.get("campaign"),
              payload.get("status", "Active"), payload.get("description"),
              payload.get("tone"), payload.get("prompt"),
              payload.get("welcome_message"),
              payload.get("stt_provider"), payload.get("stt_model"),
              payload.get("tts_provider"), payload.get("tts_model"),
              payload.get("llm_provider"), payload.get("llm_model"),
              payload.get("llm_temperature"), payload.get("llm_max_tokens"),
              payload.get("tts_speed"),
              payload.get("idle_enabled"),
              payload.get("idle_first_timeout_sec"),
              payload.get("idle_first_message"),
              payload.get("idle_subsequent_timeout_sec"),
              payload.get("idle_final_message"),
              payload.get("idle_disconnect_timeout_sec"),
              payload.get("idle_max_attempts")]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO agents ({insert_cols}) VALUES ({placeholders}) "
                f"RETURNING {_AGENT_COLS}",
                values,
            )
            row = cur.fetchone()
    return _row_to_agent(row)


def update_agent(agent_id: str, user_id: str, payload: dict) -> dict | None:
    if not payload:
        return get_agent(agent_id, user_id)
    if not is_postgres():
        if not AGENTS_FILE.exists():
            return None
        try:
            with open(AGENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return None
        for a in data:
            if a.get("id") == agent_id and a.get("user_id") == user_id:
                a.update({k: v for k, v in payload.items() if v is not None})
                with open(AGENTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return a
        return None
    # ponytail: only update fields the caller passed (exclude_none), so a
    # PUT with {"name": "X"} doesn't blank out tts_provider. The set
    # below also matches columns added in 006_agent_runtime_params.sql +
    # 008_agent_idle_settings.sql so the FE can PATCH temperature /
    # max_tokens / tts_speed / idle_* without the BE silently dropping them.
    set_clauses = []
    values = []
    for k, v in payload.items():
        if v is None:
            continue
        if k not in {"name", "voice", "voice_id", "language", "campaign", "status",
                     "description", "tone", "prompt", "welcome_message",
                     "stt_provider", "stt_model", "tts_provider", "tts_model",
                     "llm_provider", "llm_model",
                     "llm_temperature", "llm_max_tokens", "tts_speed",
                     "idle_enabled", "idle_first_timeout_sec", "idle_first_message",
                     "idle_subsequent_timeout_sec", "idle_final_message",
                     "idle_disconnect_timeout_sec", "idle_max_attempts"}:
            continue
        set_clauses.append(f"{k} = %s")
        values.append(v)
    if not set_clauses:
        return get_agent(agent_id, user_id)
    set_clauses.append("updated_at = NOW()")
    values.extend([agent_id, user_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE agents SET {', '.join(set_clauses)} "
                "WHERE id = %s AND user_id = %s "
                f"RETURNING {_AGENT_COLS}",
                values,
            )
            row = cur.fetchone()
    return _row_to_agent(row) if row else None


def delete_agent(agent_id: str, user_id: str) -> bool:
    if not is_postgres():
        if not AGENTS_FILE.exists():
            return False
        try:
            with open(AGENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            return False
        new_data = [a for a in data if not (a.get("id") == agent_id and a.get("user_id") == user_id)]
        if len(new_data) == len(data):
            return False
        with open(AGENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        return True
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agents WHERE id = %s AND user_id = %s",
                (agent_id, user_id),
            )
            return cur.rowcount > 0


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def backfill_from_json() -> int:
    """One-shot helper: if a JSON file exists and Postgres is empty for
    the same user, copy the rows over. Called at startup so existing
    local-dev users don't lose their agents on the first deploy."""
    if not is_postgres() or not AGENTS_FILE.exists():
        return 0
    try:
        with open(AGENTS_FILE, "r", encoding="utf-8") as f:
            json_data = json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return 0
    if not json_data:
        return 0
    n = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for a in json_data:
                if not a.get("user_id") or not a.get("name"):
                    continue
                # Idempotent: skip if id already in DB.
                cur.execute("SELECT 1 FROM agents WHERE id = %s", (a["id"],))
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO agents (id, user_id, name, voice, language, campaign, "
                    "status, description, tone, prompt, calls, perf, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))",
                    (
                        a["id"], a["user_id"], a["name"],
                        a.get("voice"), a.get("language", "English"),
                        a.get("campaign"), a.get("status", "Active"),
                        a.get("description"), a.get("tone"), a.get("prompt"),
                        a.get("calls", "0"), int(a.get("perf", 0)),
                        a.get("created_at"),
                    ),
                )
                n += 1
    if n:
        log.info("[db_agents] backfilled %d agents from JSON to Postgres", n)
    return n
