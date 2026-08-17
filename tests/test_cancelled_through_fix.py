"""Tests for the 2026-08-17 production incident: in-band greeting
silently skipped on the first generation.

The user's complaint: "nunca sale el audio de bienvenida y no pasa
nada dentro de la llamada". The call summary had no `frames=`
field while `playback_queue.high_water` was 28 — the audio was queued
but never sent.

Root cause: PlaybackLoop's cancelled-generation filter
(`if generation <= session.cancelled_through: continue`) skipped
items with the INITIAL generation (gen=0) because
``cancelled_through`` defaulted to 0, and ``0 <= 0`` is True. Every
in-band greeting on the first turn was dropped before the WS send
loop ever got a chance to push it to Twilio.

The fix: initialize ``cancelled_through`` to -1 (smaller than any
valid generation). The check ``0 <= -1`` is False on the first
generation, so the greeting is processed. After barge-in,
``interrupt_current_turn`` sets ``cancelled_through = max(-1, 0) = 0``
on the first barge-in (gen 0 → 1), so old items with gen=0 are
correctly skipped via ``0 <= 0``.

These tests pin the fix so a future refactor can't revert
cancelled_through to 0.
"""
from __future__ import annotations

import pytest


def test_cancelled_through_default_is_minus_one() -> None:
    """The dataclass must default cancelled_through to -1, not 0.

    A regression to 0 makes the playback_loop's cancelled-gen
    filter drop the first generation's audio on every call.
    """
    from STT_server.domain.session import CallSession
    s = CallSession(session_key="test")
    assert s.cancelled_through == -1, (
        f"cancelled_through must start at -1 (not 0); current={s.cancelled_through}. "
        "See 2026-08-17 production incident — with default=0, the "
        "in-band greeting is silently skipped because the check "
        "`if generation <= cancelled_through` is True for gen=0."
    )


def test_first_generation_passes_cancelled_filter() -> None:
    """An item with gen=0 on a fresh session must NOT be
    cancelled. The check is ``generation <= cancelled_through``; with
    cancelled_through=-1 the inequality is False for gen=0."""
    from STT_server.domain.session import CallSession
    s = CallSession(session_key="test")
    assert not (0 <= s.cancelled_through), (
        "Expected gen=0 to NOT satisfy `<= -1` (i.e. NOT cancelled). "
        "If this fails, the first generation's audio is being skipped."
    )


def test_barge_in_cancels_old_generation() -> None:
    """After barge-in from gen 0 to gen 1, items with gen=0 must
    be cancelled. The check is ``0 <= cancelled_through`` which the
    interrupt_current_turn handler sets to max(-1, 1-1)=0."""
    from STT_server.domain.session import CallSession
    from STT_server.services.playback_service import interrupt_current_turn
    import asyncio
    s = CallSession(session_key="test")
    # Simulate barge-in: ASYNC call to interrupt_current_turn (it
    # awaits drain_queue_nowait etc.). We just need the side effect
    # of bumping active_generation and cancelled_through.
    asyncio.run(interrupt_current_turn(s))
    assert s.active_generation == 1
    assert s.cancelled_through == 0
    # Old gen=0 items are now cancelled: 0 <= 0 is True → skip.
    assert (0 <= s.cancelled_through)


def test_two_barge_ins_cancel_two_generations() -> None:
    """Two barge-ins: active=2, cancelled=max(0, 1)=1.
    Items with gen=0 or gen=1 are skipped; gen=2 is preserved."""
    from STT_server.domain.session import CallSession
    from STT_server.services.playback_service import interrupt_current_turn
    import asyncio
    s = CallSession(session_key="test")
    asyncio.run(interrupt_current_turn(s))
    asyncio.run(interrupt_current_turn(s))
    assert s.active_generation == 2
    assert s.cancelled_through == 1
    # Both gen=0 and gen=1 are cancelled.
    assert (0 <= s.cancelled_through)
    assert (1 <= s.cancelled_through)
    # gen=2 is the current one — not cancelled.
    assert not (2 <= s.cancelled_through)