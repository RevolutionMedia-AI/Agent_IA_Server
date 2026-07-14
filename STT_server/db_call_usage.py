"""Postgres-backed implementations for the call_usage table.

Replaces the legacy calls.json ledger that services/usage_store.py
used to write to. One row per completed call, appended at
cleanup_session() time. The /usage endpoint reads from this table
directly via SQL aggregation (SUM + FILTER) so totals + per-agent
breakdown come back in a single round-trip.

Schema (db/migrations/001_schema.sql):
  call_usage(
    id                  BIGSERIAL PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id            TEXT,
    tenant_id           TEXT REFERENCES tenants(tenant_id) ON DELETE SET NULL,
    call_sid            TEXT,
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ NOT NULL,
    duration_seconds    REAL NOT NULL,
    stt_provider        TEXT,
    llm_provider        TEXT,
    tts_provider        TEXT,
    used_platform_keys  BOOLEAN NOT NULL DEFAULT FALSE,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from STT_server.config import (
    PRICE_OWN_KEY_PER_MIN,
    PRICE_PLATFORM_KEY_PER_MIN,
)
from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_call_usage")

DATA_DIR = Path(__file__).resolve().parent / "data"
LEGACY_CALLS_FILE = DATA_DIR / "calls.json"


def _row_to_record(row):
    out = dict(row)
    for k in ("started_at", "ended_at", "created_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


def _cost_usd(seconds, used_platform_keys):
    minutes = seconds / 60.0
    rate = PRICE_PLATFORM_KEY_PER_MIN if used_platform_keys else PRICE_OWN_KEY_PER_MIN
    return round(minutes * rate, 4)


def _safe_float(row, key):
    """Read a float column, defaulting to 0.0 if NULL or missing.
    Pulled out of the aggregation helpers so the per-column logic
    doesn't get buried in 4 layers of `or 0.0`."""
    v = row.get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def record_call(
    *,
    user_id,
    agent_id,
    tenant_id,
    call_sid,
    started_at,
    ended_at,
    providers,
    used_platform_keys,
):
    """Append one billing row. Best-effort: a DB failure logs and
    returns; the call path must never crash on billing. Skips when
    user_id or timestamps are missing (anonymous test calls or
    mid-startup calls)."""
    if not user_id:
        return
    if started_at is None or ended_at is None:
        return
    duration = max(0.0, ended_at - started_at)
    cost = _cost_usd(duration, used_platform_keys)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO call_usage ("
                    "  user_id, agent_id, tenant_id, call_sid,"
                    "  started_at, ended_at, duration_seconds,"
                    "  stt_provider, llm_provider, tts_provider,"
                    "  used_platform_keys, cost_usd"
                    ") VALUES ("
                    "  %s, %s, %s, %s,"
                    "  to_timestamp(%s), to_timestamp(%s), %s,"
                    "  %s, %s, %s, %s, %s"
                    ")",
                    (
                        user_id, agent_id, tenant_id, call_sid,
                        started_at, ended_at, duration,
                        providers.get("stt"),
                        providers.get("llm"),
                        providers.get("tts"),
                        bool(used_platform_keys),
                        cost,
                    ),
                )
            conn.commit()
    except Exception as exc:
        log.warning("[call_usage] insert failed for user=%s: %s", user_id, exc)


def aggregate_usage(user_id, *, agent_name_lookup=None, limit_recent=50):
    """Return totals + per-agent breakdown + recent calls for one user.

    Single SQL round-trip for the totals, a second for the per-agent
    breakdown, a third for the recent feed. Avoids the legacy
    in-memory aggregation that read the whole ledger in Python.
    """
    if not agent_name_lookup:
        agent_name_lookup = {}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT "
                "  COUNT(*) AS calls, "
                "  COALESCE(SUM(duration_seconds), 0) AS total_duration, "
                "  COALESCE(SUM(duration_seconds) FILTER (WHERE used_platform_keys), 0) AS platform_duration, "
                "  COALESCE(SUM(duration_seconds) FILTER (WHERE NOT used_platform_keys), 0) AS own_duration, "
                "  COALESCE(SUM(cost_usd), 0) AS total_cost "
                "FROM call_usage WHERE user_id = %s",
                (user_id,),
            )
            t = cur.fetchone() or {}

            cur.execute(
                "SELECT "
                "  COALESCE(agent_id, '_unassigned') AS bucket, "
                "  COUNT(*) AS calls, "
                "  COALESCE(SUM(duration_seconds), 0) AS total_duration, "
                "  COALESCE(SUM(duration_seconds) FILTER (WHERE used_platform_keys), 0) AS platform_duration, "
                "  COALESCE(SUM(duration_seconds) FILTER (WHERE NOT used_platform_keys), 0) AS own_duration, "
                "  COALESCE(SUM(cost_usd), 0) AS total_cost "
                "FROM call_usage WHERE user_id = %s "
                "GROUP BY COALESCE(agent_id, '_unassigned') "
                "ORDER BY total_duration DESC",
                (user_id,),
            )
            agent_rows = cur.fetchall() or []

            cur.execute(
                "SELECT id, user_id, agent_id, tenant_id, call_sid, "
                "  started_at, ended_at, duration_seconds, "
                "  stt_provider, llm_provider, tts_provider, "
                "  used_platform_keys, cost_usd, created_at "
                "FROM call_usage WHERE user_id = %s "
                "ORDER BY started_at DESC LIMIT %s",
                (user_id, limit_recent),
            )
            recent = [_row_to_record(r) for r in (cur.fetchall() or [])]

    totals = {
        "calls": int(t.get("calls") or 0),
        "duration_seconds": round(_safe_float(t, "total_duration"), 1),
        "platform_duration_seconds": round(_safe_float(t, "platform_duration"), 1),
        "own_duration_seconds": round(_safe_float(t, "own_duration"), 1),
        "cost_usd": round(_safe_float(t, "total_cost"), 2),
    }

    per_agent = []
    for r in agent_rows:
        bucket = r["bucket"]
        per_agent.append({
            "agent_id": bucket,
            "agent_name": agent_name_lookup.get(bucket) or bucket,
            "calls": int(r.get("calls") or 0),
            "duration_seconds": round(_safe_float(r, "total_duration"), 1),
            "platform_duration_seconds": round(_safe_float(r, "platform_duration"), 1),
            "own_duration_seconds": round(_safe_float(r, "own_duration"), 1),
            "cost_usd": round(_safe_float(r, "total_cost"), 2),
        })

    return {
        "totals": totals,
        "per_agent": per_agent,
        "recent_calls": recent,
        "rates": {
            "own_per_min": PRICE_OWN_KEY_PER_MIN,
            "platform_per_min": PRICE_PLATFORM_KEY_PER_MIN,
            "currency": "USD",
        },
    }


def backfill_from_json():
    """One-shot: read the legacy calls.json ledger and INSERT every
    row into call_usage. Idempotent: re-runs after a partial
    backfill skip rows that already exist (de-dup on
    user_id+call_sid+started_at).

    Called at server startup so existing local-dev data survives
    the first Postgres-backed deploy. After it runs, the file is
    orphaned (new code never reads it); delete it manually after
    one release.
    """
    if not is_postgres() or not os.path.exists(LEGACY_CALLS_FILE):
        return 0
    try:
        with open(LEGACY_CALLS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f) or []
    except (json.JSONDecodeError, IOError):
        return 0
    if not records:
        return 0
    n = 0
    skipped = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in records:
                started_iso = r.get("started_at")
                ended_iso = r.get("ended_at")
                if not started_iso or not ended_iso:
                    skipped += 1
                    continue
                try:
                    started_dt = datetime.fromisoformat(
                        started_iso.replace("Z", "+00:00")
                    )
                    ended_dt = datetime.fromisoformat(
                        ended_iso.replace("Z", "+00:00")
                    )
                    started_ts = started_dt.timestamp()
                    ended_ts = ended_dt.timestamp()
                except (ValueError, TypeError):
                    skipped += 1
                    continue

                cur.execute(
                    "SELECT 1 FROM call_usage "
                    "WHERE user_id = %s "
                    "  AND COALESCE(call_sid, '') = COALESCE(%s, '') "
                    "  AND started_at = to_timestamp(%s)",
                    (r.get("user_id"), r.get("call_sid"), started_ts),
                )
                if cur.fetchone():
                    skipped += 1
                    continue

                providers = r.get("providers") or {}
                duration = float(
                    r.get("duration_seconds") or max(0.0, ended_ts - started_ts)
                )
                cost = _cost_usd(duration, bool(r.get("used_platform_keys")))
                cur.execute(
                    "INSERT INTO call_usage ("
                    "  user_id, agent_id, tenant_id, call_sid, "
                    "  started_at, ended_at, duration_seconds, "
                    "  stt_provider, llm_provider, tts_provider, "
                    "  used_platform_keys, cost_usd"
                    ") VALUES ("
                    "  %s, %s, %s, %s, "
                    "  to_timestamp(%s), to_timestamp(%s), %s, "
                    "  %s, %s, %s, %s, %s"
                    ")",
                    (
                        r.get("user_id"),
                        r.get("agent_id"),
                        r.get("tenant_id"),
                        r.get("call_sid"),
                        started_ts,
                        ended_ts,
                        duration,
                        providers.get("stt"),
                        providers.get("llm"),
                        providers.get("tts"),
                        bool(r.get("used_platform_keys")),
                        cost,
                    ),
                )
                n += 1
        conn.commit()
    if n or skipped:
        log.info(
            "[db_call_usage] backfilled %d record(s) from calls.json, skipped %d",
            n,
            skipped,
        )
    return n
