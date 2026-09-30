"""ai_first_dates: does the AI answer first on this date?

The gate decides whether an inbound call rings a human or the AI, so the
failure that matters is a FALSE NEGATIVE (a holiday that silently rings the
team at 2am) and a timezone slip (New Year's Eve read as the 31st in UTC
when it is already the 1st in Mexico City).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT_server.services.ai_first_dates import (  # noqa: E402
    MAX_DATES,
    agent_answers_ai_first,
    is_ai_first_date,
    local_date,
    normalize_dates,
    today_in_timezone,
)


def test_matches_only_the_listed_day() -> None:
    dates = ["2026-12-25", "2026-01-01"]
    assert is_ai_first_date(dates, "2026-12-25") is True
    assert is_ai_first_date(dates, "2026-01-01") is True
    # The day AFTER a holiday must NOT inherit the flag.
    assert is_ai_first_date(dates, "2026-12-26") is False
    assert is_ai_first_date(dates, "2026-12-24") is False


def test_empty_list_is_always_false() -> None:
    for empty in (None, [], "", {}, "not json"):
        assert is_ai_first_date(empty, "2026-12-25") is False


def test_malformed_entries_are_dropped_not_fatal() -> None:
    # One bad date must not discard the good ones beside it, and must not
    # raise on a live call path.
    got = normalize_dates(["2026-12-25", "25/12/2026", "", None, 42, "2026-1-5"])
    assert got == ["2026-12-25"]


def test_impossible_dates_are_rejected() -> None:
    # Matches the shape but is not a real day.
    assert normalize_dates(["2026-02-30", "2026-13-01"]) == []


def test_json_string_column_is_accepted() -> None:
    # A hand-edited DB row or the JSON-file backend can hand us a str.
    assert is_ai_first_date('["2026-12-25"]', "2026-12-25") is True


def test_surrounding_whitespace_does_not_break_the_match() -> None:
    assert is_ai_first_date([" 2026-12-25 "], "2026-12-25") is True


def test_bad_today_falls_back_to_ringing_humans() -> None:
    # Today's behaviour: ring the humans. Never the AI on a parse failure.
    assert is_ai_first_date(["2026-12-25"], "not-a-date") is False
    assert is_ai_first_date(["2026-12-25"], "") is False
    assert is_ai_first_date(["2026-12-25"], None) is False


def test_duplicates_collapse_and_order_is_stable() -> None:
    assert normalize_dates(["2026-03-01", "2026-01-01", "2026-03-01"]) == [
        "2026-01-01",
        "2026-03-01",
    ]


def test_list_is_capped() -> None:
    # Real consecutive days, so the validator keeps all of them and the
    # assertion measures the cap and not date validity.
    start = datetime(2026, 1, 1)
    many = [
        (start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(MAX_DATES + 50)
    ]
    assert len(normalize_dates(many)) == MAX_DATES


def test_today_is_the_operators_day_not_utcs() -> None:
    # 02:30 UTC on Jan 1 is still 20:30 on Dec 31 in Mexico City. If the
    # gate read UTC, New Year's Eve would ring the team instead of the AI.
    just_after_utc_midnight = datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc)
    assert local_date(just_after_utc_midnight, "America/Mexico_City") == "2025-12-31"
    assert local_date(just_after_utc_midnight, "UTC") == "2026-01-01"


def test_utc_only_offset_keeps_the_same_day() -> None:
    # Noon UTC is 06:00 in Mexico City — still the same day.
    noon = datetime(2026, 12, 25, 12, 0, tzinfo=timezone.utc)
    assert local_date(noon, "America/Mexico_City") == "2026-12-25"


def test_naive_input_is_treated_as_utc() -> None:
    naive = datetime(2026, 1, 1, 2, 30)
    assert local_date(naive, "America/Mexico_City") == "2025-12-31"


def test_bad_timezone_degrades_instead_of_raising() -> None:
    instant = datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc)
    assert local_date(instant, "Not/AZone") == "2026-01-01"
    assert local_date(instant, None) == "2025-12-31"  # DEFAULT_TZ
    assert local_date(instant, "") == "2025-12-31"
    # The live helper returns a well-formed date whatever it is handed.
    assert len(today_in_timezone("Not/AZone")) == 10


def test_agent_gate_reads_the_row() -> None:
    # No row / no dates -> normal handoff order, never a crash.
    assert agent_answers_ai_first(None) is False
    assert agent_answers_ai_first({}) is False
    assert agent_answers_ai_first({"ai_first_dates": []}) is False
    # And a row WITH dates resolves against the injected timezone without
    # touching the clock: today in Mexico City right now.
    today = today_in_timezone("America/Mexico_City")
    assert agent_answers_ai_first(
        {"ai_first_dates": [today]}, "America/Mexico_City"
    ) is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ai_first_dates: all checks passed")
