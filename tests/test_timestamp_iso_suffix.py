"""Timestamp serialization must stay parseable by the browser.

Regression: both row mappers did `out[k] = out[k].isoformat() + "Z"`.
A tz-aware datetime's isoformat() ALREADY ends in "+00:00", so the
result was "2026-09-28T12:34:56.789000+00:00Z". JS `new Date(...)` /
`Date.parse(...)` return NaN for that, which is why every connection
card rendered "Never tested" and IntegrationDetail rendered
"Last test · never" no matter what the BE had actually stored.

The fix only appends "Z" when isoformat() left the offset off (naive).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from STT_server.db_integrations import _row_to_integration
from STT_server.db_tools import _row_to_tool

AWARE = datetime(2026, 9, 28, 12, 34, 56, 789000, tzinfo=timezone.utc)
NAIVE = datetime(2026, 9, 28, 12, 34, 56, 789000)


def _js_parseable(iso: str) -> bool:
    """Mirror what `new Date(s)` does for the two shapes we can emit.

    Python's fromisoformat accepts the trailing 'Z' from 3.11 but the
    double-suffixed string is only rejected by the JS engine, so the
    guard is structural, not a round-trip.
    """
    return not iso.endswith("+00:00Z") and iso[-1] in ("Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


@pytest.mark.parametrize("value", [AWARE, NAIVE])
def test_integration_last_tested_at_is_not_double_suffixed(value):
    out = _row_to_integration({"id": "int_1", "last_tested_at": value})
    iso = out["last_tested_at"]
    assert _js_parseable(iso), f"{iso!r} is rejected by JS Date.parse"
    assert iso.count("+") <= 1


@pytest.mark.parametrize("value", [AWARE, NAIVE])
def test_tool_timestamps_are_not_double_suffixed(value):
    out = _row_to_tool({"id": "tool_1", "last_tested_at": value, "last_invoked_at": value})
    for key in ("last_tested_at", "last_invoked_at"):
        iso = out[key]
        assert _js_parseable(iso), f"{key}={iso!r} is rejected by JS Date.parse"
        assert iso.count("+") <= 1


def test_aware_keeps_offset_naive_gets_z():
    """The distinction that matters: tz-aware keeps '+00:00', naive
    gets an explicit 'Z' so the browser still reads it as UTC."""
    aware = _row_to_integration({"id": "i", "last_tested_at": AWARE})["last_tested_at"]
    naive = _row_to_integration({"id": "i", "last_tested_at": NAIVE})["last_tested_at"]
    assert aware.endswith("+00:00")
    assert naive.endswith("Z")
    assert not aware.endswith("Z")
    assert not naive.endswith("+00:00")


def test_plain_strings_pass_through_untouched():
    """The JSON-file fallback stores strings; it must not be mangled."""
    out = _row_to_integration({"id": "i", "last_tested_at": "2026-09-28T12:34:56.789Z"})
    assert out["last_tested_at"] == "2026-09-28T12:34:56.789Z"
