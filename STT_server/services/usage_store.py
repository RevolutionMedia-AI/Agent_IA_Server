"""Call-usage ledger.

Per-call record gets written when the call websocket closes
(cleanup_session). GET /api/usage aggregates those records for the
authenticated user and returns totals + a per-agent breakdown.

Storage:
  - One JSON file (calls.json) holding a list of records.
  - Same simple pattern as the rest of the BE's data layer. Volume
    per user is low (hundreds of calls/month); reading the whole file
    and grouping in Python is fine. Swap to SQLite when this stops
    being fine.

Cost tiers (config.py):
  PRICE_OWN_KEY_PER_MIN     — user brought their own provider key;
                              we only charge the platform fee.
  PRICE_PLATFORM_KEY_PER_MIN — user used our provisioned key; the
                              provider cost is baked in.

A call is classified as `used_platform_keys=True` when at least one
of the providers it actually used fell back to the env-var key
(because the user hadn't stored a credential for that provider).
Otherwise the call is `used_platform_keys=False`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from STT_server.config import PRICE_OWN_KEY_PER_MIN, PRICE_PLATFORM_KEY_PER_MIN


log = logging.getLogger("stt_server.usage_store")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CALLS_FILE = os.path.join(DATA_DIR, "calls.json")

_io_lock = threading.Lock()


@contextmanager
def _data_lock():
    # ponytail: serialise RMW against the calls file. Same approach the
    # rest of the data layer uses. Fine at MVP volume.
    with _io_lock:
        yield


def _load() -> list[dict]:
    if not os.path.exists(CALLS_FILE):
        return []
    try:
        with open(CALLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save(records: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = CALLS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CALLS_FILE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_call(
    *,
    user_id: str | None,
    agent_id: str | None,
    tenant_id: str | None,
    call_sid: str | None,
    started_at: float | None,
    ended_at: float | None,
    providers: dict[str, str | None],
    used_platform_keys: bool,
) -> None:
    """Append a per-call record. Best-effort: never raises into the
    websocket cleanup path. Skips records without a user_id since we
    have nothing to attribute them to.
    """
    if not user_id:
        return
    if started_at is None or ended_at is None:
        return
    duration = max(0.0, ended_at - started_at)
    record = {
        "user_id": user_id,
        "agent_id": agent_id,
        "tenant_id": tenant_id,
        "call_sid": call_sid,
        "started_at": _iso_from_ts(started_at),
        "ended_at": _iso_from_ts(ended_at),
        "duration_seconds": round(duration, 3),
        "providers": {
            "stt": providers.get("stt"),
            "llm": providers.get("llm"),
            "tts": providers.get("tts"),
        },
        "used_platform_keys": bool(used_platform_keys),
    }
    try:
        with _data_lock():
            records = _load()
            records.append(record)
            _save(records)
    except Exception as exc:  # noqa: BLE001 — ledger must never crash a call
        log.warning("[usage] failed to record call for user=%s: %s", user_id, exc)


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _cost_usd(seconds: float, used_platform_keys: bool) -> float:
    minutes = seconds / 60.0
    rate = PRICE_PLATFORM_KEY_PER_MIN if used_platform_keys else PRICE_OWN_KEY_PER_MIN
    return round(minutes * rate, 4)


def aggregate_usage(
    user_id: str,
    *,
    agent_name_lookup: dict[str, str] | None = None,
    records: Iterable[dict] | None = None,
) -> dict[str, Any]:
    """Return totals + per-agent breakdown + recent calls for one user.

    `records` is an override hook so the route can pass its own load
    (e.g. filtered by date). Defaults to the full ledger.
    """
    if records is None:
        with _data_lock():
            records = _load()

    user_records = [r for r in records if r.get("user_id") == user_id]
    user_records.sort(key=lambda r: r.get("started_at") or "", reverse=True)

    totals = {
        "calls": len(user_records),
        "duration_seconds": 0.0,
        "platform_duration_seconds": 0.0,
        "own_duration_seconds": 0.0,
        "cost_usd": 0.0,
    }
    per_agent: dict[str, dict] = {}
    for r in user_records:
        dur = float(r.get("duration_seconds") or 0)
        platform = bool(r.get("used_platform_keys"))
        totals["duration_seconds"] += dur
        if platform:
            totals["platform_duration_seconds"] += dur
        else:
            totals["own_duration_seconds"] += dur
        totals["cost_usd"] += _cost_usd(dur, platform)

        aid = r.get("agent_id") or "_unassigned"
        bucket = per_agent.setdefault(aid, {
            "agent_id": aid,
            "agent_name": (agent_name_lookup or {}).get(aid) or aid,
            "calls": 0,
            "duration_seconds": 0.0,
            "platform_duration_seconds": 0.0,
            "own_duration_seconds": 0.0,
            "cost_usd": 0.0,
        })
        bucket["calls"] += 1
        bucket["duration_seconds"] += dur
        if platform:
            bucket["platform_duration_seconds"] += dur
        else:
            bucket["own_duration_seconds"] += dur
        bucket["cost_usd"] += _cost_usd(dur, platform)

    totals["cost_usd"] = round(totals["cost_usd"], 2)
    agent_rows = sorted(
        per_agent.values(),
        key=lambda b: b["duration_seconds"],
        reverse=True,
    )
    for b in agent_rows:
        b["cost_usd"] = round(b["cost_usd"], 2)
        b["duration_seconds"] = round(b["duration_seconds"], 1)
        b["platform_duration_seconds"] = round(b["platform_duration_seconds"], 1)
        b["own_duration_seconds"] = round(b["own_duration_seconds"], 1)
    totals["duration_seconds"] = round(totals["duration_seconds"], 1)
    totals["platform_duration_seconds"] = round(totals["platform_duration_seconds"], 1)
    totals["own_duration_seconds"] = round(totals["own_duration_seconds"], 1)

    return {
        "totals": totals,
        "per_agent": agent_rows,
        "recent_calls": user_records[:50],
        "rates": {
            "own_per_min": PRICE_OWN_KEY_PER_MIN,
            "platform_per_min": PRICE_PLATFORM_KEY_PER_MIN,
            "currency": "USD",
        },
    }


def has_user_stored_key(user_id: str, provider: str | None) -> bool:
    """True if the user has a connected api-key record for this provider.

    Used by session_runtime to decide whether a call used the user's
    own key or fell back to the env-var (platform) key. Reads the
    tools_integrations.json that the settings router manages.
    """
    if not provider or not user_id:
        return False
    tools_file = os.path.join(DATA_DIR, "tools_integrations.json")
    if not os.path.exists(tools_file):
        return False
    try:
        with open(tools_file, "r", encoding="utf-8") as f:
            tools = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False
    if not isinstance(tools, list):
        return False
    return any(
        isinstance(t, dict)
        and t.get("id") == provider
        and t.get("user_id") == user_id
        and t.get("connected")
        for t in tools
    )