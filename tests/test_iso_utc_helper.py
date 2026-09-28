"""One ISO rule for every row mapper (STT_server/utils/iso.py).

Extends tests/test_timestamp_iso_suffix.py, which covered only the
integrations + tools mappers, to the five row mappers that open-coded
the same broken idiom:

    db_agents.py            created_at / updated_at
    db_settings.py          updated_at
    db_phone_numbers.py     created_at / updated_at
    db_pricing_overrides.py updated_at
    db_twilio_credentials.py last_tested_at / created_at / updated_at

db_twilio_credentials is the one with a user-visible symptom of its own:
its `last_tested_at` rendered as "never" in Settings -> Twilio for the
same reason the connections page did.

The rule: a tz-aware datetime's isoformat() already ends in the offset,
so appending "Z" produced "...+00:00Z", which JS Date.parse rejects as
NaN. Append "Z" only for naive datetimes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from STT_server.utils.iso import iso_utc

AWARE_UTC = datetime(2026, 9, 28, 12, 34, 56, 789000, tzinfo=timezone.utc)
AWARE_NEG = datetime(2026, 9, 28, 12, 34, 56, 789000, tzinfo=timezone(timedelta(hours=-5)))
NAIVE = datetime(2026, 9, 28, 12, 34, 56, 789000)


def _js_parseable(iso: str) -> bool:
    """Structural guard, not a round-trip: Python's fromisoformat accepts
    the trailing 'Z' from 3.11, but the double-suffixed string is only
    rejected by the JS engine."""
    return not iso.endswith("+00:00Z") and not iso.endswith("-05:00Z")


# ── the helper itself ──────────────────────────────────────────────────────

def test_aware_keeps_offset_and_is_js_readable():
    out = iso_utc(AWARE_UTC)
    assert out == "2026-09-28T12:34:56.789000+00:00"
    assert _js_parseable(out)
    assert not out.endswith("Z")


def test_negative_offset_keeps_its_sign():
    out = iso_utc(AWARE_NEG)
    assert out.endswith("-05:00")
    assert _js_parseable(out)
    assert out.count("-") >= 1


def test_naive_gets_an_explicit_z():
    out = iso_utc(NAIVE)
    assert out == "2026-09-28T12:34:56.789000Z"
    assert out.endswith("Z")
    assert _js_parseable(out)


@pytest.mark.parametrize("passthrough", [
    None, "", "2026-09-28T12:34:56.789Z", "already-a-string", 0, 12345, True,
])
def test_non_datetimes_pass_through_untouched(passthrough):
    """Every call site previously guarded on hasattr(v, "isoformat") and
    left non-datetimes alone, so the helper must not start coercing them."""
    assert iso_utc(passthrough) == passthrough


def test_a_string_that_already_ends_in_z_is_not_doubled():
    class Fake:
        def isoformat(self):
            return "2026-09-28T12:34:56.789Z"

    assert iso_utc(Fake()) == "2026-09-28T12:34:56.789Z"


# ── the row mappers now delegate to it ─────────────────────────────────────

def test_agents_mapper_serializes_both_stamps():
    from STT_server.db_agents import _row_to_agent

    out = _row_to_agent({"id": "a1", "created_at": AWARE_UTC, "updated_at": AWARE_UTC})
    assert _js_parseable(out["created_at"])
    assert _js_parseable(out["updated_at"])


def test_settings_mapper_serializes_updated_at():
    from STT_server.db_settings import _row_to_settings

    out = _row_to_settings({"updated_at": AWARE_UTC})
    assert _js_parseable(out["updated_at"])


def test_phone_numbers_mapper_serializes_both_stamps():
    from STT_server.db_phone_numbers import _row_to_number

    out = _row_to_number({"id": "p1", "created_at": AWARE_UTC, "updated_at": AWARE_UTC})
    assert _js_parseable(out["created_at"])
    assert _js_parseable(out["updated_at"])


def test_pricing_overrides_mapper_serializes_updated_at():
    from STT_server.db_pricing_overrides import _row_to_override

    out = _row_to_override({"agent_id": "a1", "updated_at": AWARE_UTC})
    assert _js_parseable(out["updated_at"])


def test_twilio_credentials_mapper_serializes_all_three():
    from STT_server.db_twilio_credentials import _row_to_dict

    out = _row_to_dict(
        {"id": "c1", "last_tested_at": AWARE_UTC, "created_at": AWARE_UTC, "updated_at": AWARE_UTC}
    )
    for key in ("last_tested_at", "created_at", "updated_at"):
        assert _js_parseable(out[key]), f"{key}={out[key]!r} is rejected by JS Date.parse"


def test_twilio_credentials_tolerates_a_missing_last_tested_at():
    """The old inline expression read row["last_tested_at"] unguarded, so
    a row without that column raised KeyError. iso_utc(row.get(...))
    returns None instead, which the UI renders as "never"."""
    from STT_server.db_twilio_credentials import _row_to_dict

    out = _row_to_dict({"id": "c1"})
    assert out["last_tested_at"] is None
