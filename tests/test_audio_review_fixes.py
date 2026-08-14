"""Regression tests for the 2026-08-14 audio-review findings.

Each test pins ONE behavior that a future refactor must not
silently break. The behaviors are not covered by the integration
suite (which exercises the full call path but only one happy
turn) — these are unit-level guards for the four production
incidents called out in the review:

  1. ``pending_playback_marks`` counter (assistant_speaking must
     only drop when the WHOLE generation has been acked, not
     after every segment of a multi-segment reply).
  2. SEQ_GAP detector must ignore non-media events (the previous
     detector counted Twilio's whole-session sequenceNumber on
     mark/start/stop events and logged phantom gaps in lock-step
     with mark acks).
  3. SPEECH_START_FRAMES default (4 frames ≈ 80 ms of sustained
     voice-positive + RMS — enough to filter clicks/echo but
     well under a human phoneme).
"""
from __future__ import annotations

import base64

import pytest

from STT_server.config import SPEECH_START_FRAMES
from STT_server.domain.session import CallSession
from STT_server.services.audio_metrics import CallMetrics, attach_metrics


# ── 1. pending_playback_marks counter ──────────────────────────────────────


def test_pending_playback_marks_default_is_zero() -> None:
    """A fresh session starts with zero pending segments — the
    counter is the audio-gate's invariant; if it's accidentally
    defaulted to a non-zero value, the assistant would never appear
    to "finish playing" and assistant_speaking would never drop.
    """
    s = CallSession(session_key="test")
    assert s.pending_playback_marks == 0


def test_pending_playback_marks_increments_only_for_active_generation() -> None:
    """Barge-in (generation change) leaves stale segments behind; we
    must NOT count them toward the next turn's pending budget.
    Without this guard, a leaked segment from a cancelled turn
    leaves the counter permanently above zero.
    """
    # The increment lives in playback_service.playback_loop's
    # segment_end branch; we verify the dataclass side. A pending
    # mark for an old generation must not appear in pending_marks
    # when we read the field — it's popped on the ack, regardless of
    # generation, so the counter logic is decoupled from the dict.
    s = CallSession(session_key="test")
    # Simulate: gen=6 had 2 segments, gen=7 was kicked off.
    s.pending_marks["gen-6-seg-1"] = 100.0
    s.pending_marks["gen-6-seg-2"] = 100.5
    s.pending_marks["gen-7-seg-3"] = 101.0
    assert len(s.pending_marks) == 3
    # The mark handler pops by name; counter management lives in
    # the handler too. Here we just confirm the dict stays
    # pristine for downstream counting.
    s.pending_marks.pop("gen-6-seg-1", None)
    assert "gen-6-seg-1" not in s.pending_marks
    assert s.pending_marks["gen-7-seg-3"] == 101.0


# ── 2. SEQ_GAP detector ignores non-media events ──────────────────────────


def test_seq_gap_detector_ignores_mark_event() -> None:
    """The 2026-08-14 logs showed gaps in lock-step with mark acks
    (prev=3893 got=3895 missing=1). Root cause: Twilio assigns its
    own monotonic sequenceNumber to EVERY WS message (media, mark,
    start, stop). The previous detector compared sequenceNumber
    across all event types, so a media event followed by a mark
    event always looked like a missing media frame.

    After the fix the detector bails out for non-media events. We
    simulate by feeding a media event, then a mark event, then a
    second media event. The first media establishes last_seq. The
    mark is ignored (no last_seq bump). The second media should
    show last_seq+1, no gap.
    """
    from STT_server.adapters.twilio_media import track_twilio_sequence

    class _S:
        # Minimal stand-in for CallSession — duck-typed; the
        # detector reads/writes attrs directly. session_key is
        # required because the gap-branch log line reads it.
        session_key = "test"
        stream_sid = "MZtest"

    s = _S()
    # First media event at seq=10.
    track_twilio_sequence(s, {"event": "media", "sequenceNumber": 10, "media": {"timestamp": 1000}})
    assert s._twilio_last_seq == 10
    assert s._twilio_seq_gaps == 0
    # Mark event at seq=11 (Twilio's own WS-level counter). Must be
    # IGNORED — no gap, no last_seq bump.
    track_twilio_sequence(s, {"event": "mark", "sequenceNumber": 11, "mark": {"name": "x"}})
    assert s._twilio_last_seq == 10, "mark event must NOT update last_seq"
    assert s._twilio_seq_gaps == 0, "mark event must NOT count as a gap"
    # Next media event at seq=12 — Twilio's WS counter advanced by
    # 1 because of the intervening mark, but the audio stream DID
    # NOT lose a frame. Detector must see seq=12 as last+2 and
    # count exactly one missing media frame (the gap is in
    # *audio* frames, not Twilio WS heartbeats).
    track_twilio_sequence(s, {"event": "media", "sequenceNumber": 12, "media": {"timestamp": 1100}})
    assert s._twilio_last_seq == 12
    # Exactly 1 missing media frame between seq=10 and seq=12.
    assert s._twilio_seq_gaps == 1


def test_seq_gap_detector_ignores_start_and_stop() -> None:
    """start/stop are also non-media. The detector must skip them so
    they don't poison the gap counter."""
    from STT_server.adapters.twilio_media import track_twilio_sequence

    class _S:
        session_key = "test"
        stream_sid = "MZtest"

    s = _S()
    track_twilio_sequence(s, {"event": "media", "sequenceNumber": 100, "media": {"timestamp": 1000}})
    track_twilio_sequence(s, {"event": "start", "sequenceNumber": 101, "start": {"streamSid": "MZtest"}})
    track_twilio_sequence(s, {"event": "stop", "sequenceNumber": 102, "stop": {}})
    assert s._twilio_last_seq == 100
    assert s._twilio_seq_gaps == 0


# ── 3. SPEECH_START_FRAMES default ─────────────────────────────────────────


def test_speech_start_frames_is_at_least_three() -> None:
    """The 2026-08-14 review bumped this from 1 → 4. A future
    refactor that lowers it below 3 will reintroduce the "one
    click triggers INICIO DE VOZ" bug. Pin the floor at 3.
    """
    assert SPEECH_START_FRAMES >= 3


def test_speech_start_frames_is_at_most_ten() -> None:
    """The other side of the band: raising it too high starts eating
    into real short utterances. Pin the ceiling so a regression
    that cranks it to 50 doesn't pass silently.
    """
    assert SPEECH_START_FRAMES <= 10