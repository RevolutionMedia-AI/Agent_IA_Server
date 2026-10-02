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
    -- 027_agent_stt_latency_mode.sql. OpenAI transcription latency dial
    -- (minimal/low/medium/high/xhigh). NULL = platform default for the
    -- model, and is the ONLY valid stored value for gpt-transcribe, which
    -- is committed-turn and has no dial. Intentionally no DEFAULT: a
    -- default would push an invalid value into committed-turn rows.
    stt_latency_mode TEXT,
    tts_provider TEXT,
    tts_model TEXT,
    llm_provider TEXT,
    llm_model TEXT,
    -- runtime knobs added by 006_agent_runtime_params.sql
    llm_temperature REAL,     -- 0.0..2.0 (NULL = adapter default 0.2)
    llm_max_tokens  INTEGER,  -- >0..4096  (NULL = config.MAX_RESPONSE_TOKENS)
    tts_speed      REAL,     -- 0.5..2.0  (NULL = provider default)
    -- 028_agent_tts_instructions.sql. Free-text voice steering sent to the
    -- provider's `instructions` field. Only honoured by models that
    -- accept it (OpenAI gpt-4o-mini-tts); the adapter drops it for
    -- tts-1 / tts-1-hd. NULL = no instructions, which is valid.
    tts_instructions TEXT,   -- <=600 chars
-- per-agent idle/silence detection (008_agent_idle_settings.sql).
    -- NULL on every column = fall back to the global IDLE_SILENCE_TIMEOUT_SEC
    -- (the legacy single-timeout-then-close behaviour).
    idle_enabled                BOOLEAN,     -- explicit opt-in
    idle_first_timeout_sec      INTEGER,     -- >0
    idle_first_message          TEXT,        -- <=1000 chars
    idle_subsequent_timeout_sec INTEGER,     -- >0
    idle_final_message          TEXT,        -- <=1000 chars
    idle_disconnect_timeout_sec INTEGER,     -- >0
    idle_max_attempts           INTEGER,     -- 1..10
    -- ponytail: per-agent credential-source toggle
    -- (009_agent_use_own_key.sql). false = resolver may fall back to
    -- platform env vars (Railway OPENAI_API_KEY etc.) when no per-user
    -- key is stored. true = resolver must use ONLY the per-user /
    -- per-agent credential. The flag is a behavioural switch that the
    -- FE toggle drives — the BE never overrides a stored credential
    -- just because the toggle is false.
    stt_use_own_key             BOOLEAN NOT NULL DEFAULT FALSE,
    llm_use_own_key             BOOLEAN NOT NULL DEFAULT FALSE,
    tts_use_own_key             BOOLEAN NOT NULL DEFAULT FALSE,
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

from STT_server.utils.iso import iso_utc
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
    "stt_provider, stt_model, stt_latency_mode, "
    "tts_provider, tts_model, "
    "llm_provider, llm_model, "
    "llm_temperature, llm_max_tokens, tts_speed, tts_instructions, "
    "stt_use_own_key, llm_use_own_key, tts_use_own_key, "
    f"{_IDLE_COLS}, "
    "transfer_cascade, "
    "transfer_chain, "
    "transfer_enabled, "
    "transfer_unavailable_message, "
    "ai_first_dates, "
    "calls, perf, created_at, updated_at"
)

# ponytail: keep the JSON-file path so we can read from it on first
# boot to backfill Postgres when the migration runs against a project
# that already has data in data/agents.json. Reads go to Postgres on
# a DATABASE_URL deployment; reads from the JSON file otherwise.
DATA_DIR = Path(__file__).resolve().parent / "data"
AGENTS_FILE = DATA_DIR / "agents.json"

# Columns update_agent is allowed to SET. Extracted from the old inline set
# so the payload loop and the clear_fields loop validate against the same
# list — previously a new column had to be added in two places and forgetting
# the second one silently dropped every clear for that column.
_UPDATABLE_COLS = frozenset({
    "name", "voice", "voice_id", "language", "campaign", "status",
    "description", "tone", "prompt", "welcome_message",
    "stt_provider", "stt_model", "stt_latency_mode",
"tts_provider", "tts_model",
    "llm_provider", "llm_model",
    "llm_temperature", "llm_max_tokens", "tts_speed",
    # 028_agent_tts_instructions.sql
    "tts_instructions",
    "stt_use_own_key", "llm_use_own_key", "tts_use_own_key",
    "idle_enabled", "idle_first_timeout_sec", "idle_first_message",
    "idle_subsequent_timeout_sec", "idle_final_message",
    "idle_disconnect_timeout_sec", "idle_max_attempts",
    "transfer_cascade", "transfer_chain", "transfer_enabled",
    "transfer_unavailable_message", "ai_first_dates",
})


def _row_to_agent(row: dict) -> dict:
    """Map a DB row to the JSON shape the FE expects."""
    if row is None:
        return None
    out = dict(row)
    # ponytail: the FE reads "created_at" as ISO string. psycopg2 hands
    # us a datetime; convert so the FE doesn't choke.
    for k in ("created_at", "updated_at"):
        out[k] = iso_utc(out.get(k))
    # ponytail: transfer_cascade (021) is JSONB. psycopg2+RealDictCursor
    # already parses it to a list; the JSON-file backend stores
    # whatever the FE sent. Normalize both to a plain list so the
    # route layer never has to defend against str/None downstream.
    tc = out.get("transfer_cascade")
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except (json.JSONDecodeError, TypeError):
            tc = []
    out["transfer_cascade"] = tc if isinstance(tc, list) else []
    # ponytail: transfer_chain (023) — same JSONB normalize as the
    # cascade above. Ordered call_transfer tool ids; [] = no chain.
    ch = out.get("transfer_chain")
    if isinstance(ch, str):
        try:
            ch = json.loads(ch)
        except (json.JSONDecodeError, TypeError):
            ch = []
    out["transfer_chain"] = [c for c in ch] if isinstance(ch, list) else []
    # ponytail: ai_first_dates (026) — JSONB list of "YYYY-MM-DD" strings on
    # which the AI answers before any human. Same normalize as the cascade.
    # A JSON legacy row (pre-migration) has no key at all → [].
    afd = out.get("ai_first_dates")
    if isinstance(afd, str):
        try:
            afd = json.loads(afd)
        except (json.JSONDecodeError, TypeError):
            afd = []
    out["ai_first_dates"] = [d for d in afd if isinstance(d, str)] if isinstance(afd, list) else []
    # ponytail: transfer_enabled (024) — BOOL NOT NULL DEFAULT TRUE.
    # None (JSON legacy row) means "never set" → default on.
    if out.get("transfer_enabled") is None:
        out["transfer_enabled"] = True
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
            "stt_provider", "stt_model", "stt_latency_mode",
"tts_provider", "tts_model",
            "llm_provider", "llm_model",
            "llm_temperature", "llm_max_tokens", "tts_speed", "tts_instructions",
            "stt_use_own_key", "llm_use_own_key", "tts_use_own_key",
            "idle_enabled", "idle_first_timeout_sec", "idle_first_message",
            "idle_subsequent_timeout_sec", "idle_final_message",
            "idle_disconnect_timeout_sec", "idle_max_attempts",
            "transfer_cascade", "transfer_chain", "transfer_enabled",
            "transfer_unavailable_message", "ai_first_dates"]
    # ponytail: transfer_cascade + transfer_chain + ai_first_dates are the
    # JSONB columns on this table — all need an explicit ::jsonb cast,
    # the rest stay plain %s.
    placeholders = ", ".join(
        "%s::jsonb" if c in ("transfer_cascade", "transfer_chain", "ai_first_dates") else "%s"
        for c in cols
    )
    insert_cols = ", ".join(cols)
    values = [agent_id, user_id, payload.get("name", "Untitled"),
              payload.get("calls", "0"), int(payload.get("perf", 0)),
              payload.get("voice"), payload.get("voice_id"),
              payload.get("language", "English"), payload.get("campaign"),
              payload.get("status", "Active"), payload.get("description"),
              payload.get("tone"), payload.get("prompt"),
              payload.get("welcome_message"),
              payload.get("stt_provider"), payload.get("stt_model"),
              payload.get("stt_latency_mode"),
              payload.get("tts_provider"), payload.get("tts_model"),
              payload.get("llm_provider"), payload.get("llm_model"),
              payload.get("llm_temperature"), payload.get("llm_max_tokens"),
              payload.get("tts_speed"),
              payload.get("tts_instructions"),
              payload.get("stt_use_own_key"),
              payload.get("llm_use_own_key"),
              payload.get("tts_use_own_key"),
              payload.get("idle_enabled"),
              payload.get("idle_first_timeout_sec"),
              payload.get("idle_first_message"),
              payload.get("idle_subsequent_timeout_sec"),
              payload.get("idle_final_message"),
              payload.get("idle_disconnect_timeout_sec"),
              payload.get("idle_max_attempts"),
              json.dumps(payload.get("transfer_cascade") or []),
              json.dumps(payload.get("transfer_chain") or []),
              # ponytail: None (FE didn't send) = default on. Only an
              # explicit false disables handoff.
              True if payload.get("transfer_enabled") is None else bool(payload.get("transfer_enabled")),
              payload.get("transfer_unavailable_message"),
              json.dumps(payload.get("ai_first_dates") or [])]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO agents ({insert_cols}) VALUES ({placeholders}) "
                f"RETURNING {_AGENT_COLS}",
                values,
            )
            row = cur.fetchone()
    return _row_to_agent(row)


def update_agent(
    agent_id: str,
    user_id: str,
    payload: dict,
    clear_fields: set[str] | None = None,
) -> dict | None:
    """Patch an agent row.

    *clear_fields* names columns to set to NULL explicitly. Needed because
    None already means "don't touch" everywhere else in this function
    (the FE sends a partial PUT), so clearing a column is otherwise
    inexpressible: switching an agent from a model that has a latency dial
    to one that does not would leave the old value stored forever.

    Kept separate from payload rather than overloading a sentinel value so
    a caller cannot accidentally blank a column by sending null.
    """
    clear_fields = clear_fields or set()
    if not payload and not clear_fields:
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
                for k in clear_fields:
                    a[k] = None
                with open(AGENTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return a
        return None
    # ponytail: only update fields the caller passed (exclude_none), so a
    # PUT with {"name": "X"} doesn't blank out tts_provider. Columns added
    # by 006 / 008 / 027 are in _UPDATABLE_COLS so the FE can PATCH them
    # without the BE silently dropping them.
    set_clauses = []
    values = []
    # A column in clear_fields is emitted as a bare NULL and its value from
    # payload is DISCARDED, not appended as a second clause: two assignments
    # to one column in a single UPDATE is a Postgres syntax error, so a
    # caller passing both would have taken the whole save down.
    for k, v in payload.items():
        if v is None or k in clear_fields:
            continue
        if k not in _UPDATABLE_COLS:
            continue
        if k in ("transfer_cascade", "transfer_chain", "ai_first_dates"):
            # ponytail: same ::jsonb cast as the INSERT above. Accept
            # list (normal) or pre-serialized str (defensive).
            v = v if isinstance(v, str) else json.dumps(v or [])
            set_clauses.append(f"{k} = %s::jsonb")
        else:
            set_clauses.append(f"{k} = %s")
        values.append(v)

    for k in clear_fields:
        if k not in _UPDATABLE_COLS:
            continue
        set_clauses.append(f"{k} = NULL")

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
