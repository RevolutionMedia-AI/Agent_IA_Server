"""Tests for _speaking_stuck_reason (STT_Server watchdog decision).

2026-10-02 production: the greeting played but its mark ack never arrived,
so assistant_speaking stayed True for the flat 30 s watchdog window. User
speech inside that window was treated as barge-in (phantom generation cut)
instead of a normal turn — "se interrumpe de manera abrupta". The fix
compares against the actual audio length (started_at + frames * 20 ms +
margin) instead of a fixed 30 s for every turn.
"""
from __future__ import annotations

import time

from STT_server.domain.session import CallSession
from STT_server.STT_Server import _speaking_stuck_reason


def _speaking_session(*, started_ago: float, expected_in: float | None) -> CallSession:
    """A session mid-utterance. expected_in is seconds from now (negative
    means the expected end already passed); None means no frames counted."""
    now = time.perf_counter()
    s = CallSession(session_key="watchdog-probe")
    s.assistant_speaking = True
    s.assistant_started_at = now - started_ago
    s.assistant_expected_end_at = (
        None if expected_in is None else now + expected_in
    )
    return s


def test_not_speaking_never_resets():
    s = CallSession(session_key="idle")
    s.assistant_speaking = False
    assert _speaking_stuck_reason(s, time.perf_counter()) is None


def test_speaking_without_start_time_never_resets():
    s = CallSession(session_key="no-start")
    s.assistant_speaking = True
    s.assistant_started_at = None
    assert _speaking_stuck_reason(s, time.perf_counter()) is None


def test_expected_end_in_future_does_not_reset():
    """Audio should still be playing; the mark may simply be slow."""
    s = _speaking_session(started_ago=2.0, expected_in=5.0)
    assert _speaking_stuck_reason(s, time.perf_counter()) is None


def test_past_expected_end_resets_before_the_flat_timeout():
    """A lost mark on a 2 s reply unsticks in ~5 s, not 30 s. This is the
    case that used to wedge the call: the flag stayed True for the full
    30 s window no matter how short the audio was."""
    s = _speaking_session(started_ago=6.0, expected_in=-1.0)
    assert _speaking_stuck_reason(s, time.perf_counter()) == "past-expected-end"


def test_stale_expected_end_from_previous_turn_is_ignored():
    """A fresh turn that has not sent frames yet still carries the old
    turn's (past) expected_end. Without the guard, the watchdog would kill
    the new turn instantly. It must fall through to the 30 s backstop."""
    now = time.perf_counter()
    s = CallSession(session_key="fresh-turn")
    s.assistant_speaking = True
    s.assistant_started_at = now - 1.0
    # previous turn's value: in the past AND before this turn started
    s.assistant_expected_end_at = now - 10.0
    assert _speaking_stuck_reason(s, now) is None


def test_no_expected_end_falls_back_to_absolute_cap():
    """Turns where no frames were ever counted (TTS failed after the flag
    was set) still unstick via the 30 s backstop."""
    s = _speaking_session(started_ago=35.0, expected_in=None)
    assert _speaking_stuck_reason(s, time.perf_counter()) == "over-absolute-cap"


def test_no_expected_end_yet_still_speaking_does_not_reset():
    s = _speaking_session(started_ago=5.0, expected_in=None)
    assert _speaking_stuck_reason(s, time.perf_counter()) is None
