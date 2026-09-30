"""Per-agent "the AI answers first on these dates" gate (migration 026).

Why this exists
---------------
The operator does not want a business-hours grid. They want a short list of
specific calendar days — company holidays, special closures — on which the
pre-AI cascade must not ring anybody: the AI picks up on the first ring and
the humans become the post-AI chain instead.

The whole feature is one boolean at the call entry point. Everything here
exists to answer "is today one of this agent's dates?" correctly and safely,
which is deceptively fiddly for three reasons:

1. "Today" is the company's wall clock, not UTC. A call at 23:30 in Mexico
   City is already the next day in UTC; reading UTC would misroute New
   Year's Eve for six hours.
2. The list is operator input crossing a trust boundary, so it is validated
   rather than trusted.
3. The date the operator typed and the date we compare must be the same
   string shape, or every holiday silently misses.

Pure (no DB, no clock) helpers take an injected `today` so the whole thing
is testable without freezing time or standing up Postgres.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

log = logging.getLogger("stt_server.ai_first_dates")

# A stored date is "YYYY-MM-DD" and nothing else. Compared as a string, so
# the shape is the contract: no "2026-1-5", no "25/12/2026", no timestamps.
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Fallback when the operator has not saved a timezone in Settings. Matches
# db_settings.upsert_settings' own default so an unconfigured deployment
# still resolves "today" the way the rest of the app already does.
DEFAULT_TZ = "America/Mexico_City"

# A year of dates is already past "holiday calendar". Cap it so a runaway
# client can't turn this column into a blob and slow every call's normalize.
MAX_DATES = 366


def normalize_dates(raw) -> list[str]:
    """Coerce whatever the API handed us into a sorted, de-duped ISO list.

    Accepts a list of strings, or a JSON string (the JSON-file backend and
    a hand-edited DB row both produce that). Anything unparseable is
    DROPPED, never raised: a bad date must not take a live call down, and
    one bad date must not discard the twenty good ones next to it.
    """
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not DATE_RE.match(value):
            log.warning("[ai_first_dates] ignoring malformed date %r", item[:32])
            continue
        # Reject impossible dates ("2026-02-30") that match the shape.
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            log.warning("[ai_first_dates] ignoring non-existent date %r", value)
            continue
        seen.add(value)
    out = sorted(seen)
    if len(out) > MAX_DATES:
        log.warning(
            "[ai_first_dates] %d dates stored, keeping the first %d",
            len(out), MAX_DATES,
        )
        out = out[:MAX_DATES]
    return out


def is_ai_first_date(dates, today_iso: str) -> bool:
    """True when `today_iso` ("YYYY-MM-DD") is in the agent's list.

    Both sides are normalized first, so a hand-edited DB row with spaces or
    a JSON string still matches. A malformed `today_iso` returns False —
    the safe direction is "ring the humans", because that is the behaviour
    that exists today.
    """
    if not today_iso or not isinstance(today_iso, str):
        return False
    wanted = today_iso.strip()
    if not DATE_RE.match(wanted):
        return False
    return wanted in normalize_dates(dates)


def local_date(now_utc: datetime, tz_name: str | None = None) -> str:
    """The calendar date `now_utc` falls on in `tz_name`, as "YYYY-MM-DD".

    Pure: the caller passes the instant, so the UTC-vs-local day boundary is
    testable without freezing a clock. Falls back to UTC on an unknown zone
    name rather than raising — a typo in a settings field must not fail
    every inbound call.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    name = (tz_name or "").strip() or DEFAULT_TZ
    try:
        from zoneinfo import ZoneInfo

        return now_utc.astimezone(ZoneInfo(name)).strftime("%Y-%m-%d")
    except Exception as exc:
        log.warning("[ai_first_dates] bad timezone %r (%s) — using UTC", name, exc)
        return now_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")


def today_in_timezone(tz_name: str | None = None) -> str:
    """Today's date in the operator's timezone, as "YYYY-MM-DD"."""
    return local_date(datetime.now(timezone.utc), tz_name)


def agent_answers_ai_first(agent_row: dict | None, tz_name: str | None = None) -> bool:
    """The one call /voice asks. True = skip the pre-AI cascade, ring the AI.

    `tz_name` is passed in by the caller because /voice has already resolved
    the agent row and does not want a second settings lookup here; omitting
    it falls back to the platform default.
    """
    if not agent_row:
        return False
    dates = agent_row.get("ai_first_dates")
    if not dates:
        return False
    return is_ai_first_date(dates, today_in_timezone(tz_name))
