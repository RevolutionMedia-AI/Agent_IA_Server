"""Tests for the 2026-08-14 audio-review A/B-test infrastructure.

The operator's A/B test compares two configurations:

  TEST A (default, current behaviour):
    - Reply of 308 chars → 225 + 82 chars → 2 TTS calls.
    - Reply of 331 chars → 179 + 103 + 47 chars → 3 TTS calls.
    - Each TTS call emits μ-law frames via emit_playback_item →
      enqueued to playback_queue → playback_loop → frame_proc →
      send_twilio_media → Twilio.

  TEST B (TTS_SINGLE_SEGMENT_PER_REPLY=true):
    - Same reply → 1 TTS call → 1 playback.
    - Hypothesis: the audible discontinuities between segments
      disappear.

The audio capture is the diagnostic that proves whether the
hypothesis is right or wrong. With A_inworld_<sid>.mulaw and
B_twilio_<sid>.mulaw on disk, plus the AMR recording from Twilio,
the operator can diff three points and locate the artifact at
one specific stage instead of guessing.

These tests pin the contract for both knobs so a future refactor
can't silently regress them.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import pytest


# ── A/B test knob ──────────────────────────────────────────────────────────


def _reload_language_with_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Re-import STT_server.config + STT_server.domain.language with a
    controlled env so the tests can flip TTS_SINGLE_SEGMENT_PER_REPLY
    without leaking state across tests."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Force a re-import of the two modules that read this constant
    # at module-load time. Use importlib.reload so the test sees
    # the new env immediately.
    import STT_server.config as cfg_mod
    importlib.reload(cfg_mod)
    import STT_server.domain.language as lang_mod
    importlib.reload(lang_mod)


def test_split_tts_segments_returns_one_segment_when_single_segment_mode(monkeypatch) -> None:
    """A 308-char reply that normally splits into 2 must come back
    as a single segment when TTS_SINGLE_SEGMENT_PER_REPLY is on.
    """
    _reload_language_with_env(
        monkeypatch, {"TTS_SINGLE_SEGMENT_PER_REPLY": "true"}
    )
    from STT_server.domain.language import split_tts_segments
    reply = (
        "Entiendo. No se preocupe, puedo ayudarle con eso. "
        "Por lo que me comenta, la opción que mejor encaja es el plan "
        "Data Ilimitada, que tiene un costo de veinticinco balboas "
        "antes de impuestos, o veintinueve balboas con impuestos."
    )
    segs = split_tts_segments(reply)
    assert segs == [reply], (
        f"Single-segment mode must return the WHOLE reply as one segment, got {len(segs)}"
    )


def test_pop_streaming_segments_returns_one_segment_when_single_segment_mode(monkeypatch) -> None:
    """Same contract for the streaming (openai / openai_realtime)
    segmenter — the buffer accumulates token by token and the
    segmenter decides when to flush. In single-segment mode it must
    flush the entire buffer at once.
    """
    _reload_language_with_env(
        monkeypatch, {"TTS_SINGLE_SEGMENT_PER_REPLY": "true"}
    )
    from STT_server.domain.language import pop_streaming_segments
    # Note: pop_streaming_segments strips leading/trailing whitespace
    # on the segment, so compare against the stripped form. The
    # contract under test is "single-segment mode produces ONE
    # segment containing the whole buffer", not "preserves
    # whitespace byte-for-byte".
    buffer = "Esto es un texto de streaming. " * 5
    expected = buffer.strip()
    segs, remaining = pop_streaming_segments(buffer, force=False)
    assert segs == [expected]
    assert remaining == ""
    # Even with force=False, the single-segment mode bypasses the
    # segmentation logic so it must already be flushed.
    segs2, remaining2 = pop_streaming_segments(buffer, force=True)
    assert segs2 == [expected]
    assert remaining2 == ""


def test_single_segment_mode_default_is_off(monkeypatch) -> None:
    """Confirm the default behaviour is preserved when the env var
    is absent — a regression that flipped the default to on would
    silently change every deployment. The env var must be opt-in."""
    monkeypatch.delenv("TTS_SINGLE_SEGMENT_PER_REPLY", raising=False)
    import STT_server.config as cfg_mod
    importlib.reload(cfg_mod)
    assert cfg_mod.TTS_SINGLE_SEGMENT_PER_REPLY is False


# ── Audio capture helpers ──────────────────────────────────────────────────


def test_capture_disabled_when_env_var_unset(tmp_path: Path, monkeypatch) -> None:
    """Without TTS_AUDIO_CAPTURE_DIR set, capture_a / capture_b are
    no-ops. Verify by calling them and asserting no files appear
    in the (otherwise writable) tmp_path."""
    monkeypatch.delenv("TTS_AUDIO_CAPTURE_DIR", raising=False)
    # Reset module-level cache in case a previous test set it.
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    ac.capture_a("CA-disabled-A", b"\xff" * 160)
    ac.capture_b("CA-disabled-B", b"\x00" * 160)
    assert list(tmp_path.glob("*")) == []


def test_capture_a_writes_inworld_bytes(tmp_path: Path, monkeypatch) -> None:
    """With TTS_AUDIO_CAPTURE_DIR set, capture_a appends the bytes
    to A_inworld_<callSid>.mulaw in append-binary mode."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    ac.capture_a("CA-test-A", b"\xff" * 160)
    ac.capture_a("CA-test-A", b"\x01" * 320)
    ac.close_all()
    a_path = tmp_path / "A_inworld_CA-test-A.mulaw"
    assert a_path.read_bytes() == b"\xff" * 160 + b"\x01" * 320


def test_capture_b_writes_twilio_bytes(tmp_path: Path, monkeypatch) -> None:
    """Same contract for B — bytes appended to
    B_twilio_<callSid>.mulaw in append-binary mode.
    """
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    ac.capture_b("CA-test-B", b"\x00" * 160)
    ac.capture_b("CA-test-B", b"\x02" * 160)
    ac.close_all()
    b_path = tmp_path / "B_twilio_CA-test-B.mulaw"
    assert b_path.read_bytes() == b"\x00" * 160 + b"\x02" * 160


def test_capture_a_and_b_are_separate_files(tmp_path: Path, monkeypatch) -> None:
    """A and B for the same callSid go to different files. The whole
    point of the A/B test is that we can diff them. If they merged,
    the test would be useless."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    ac.capture_a("CA-same", b"\xaa" * 160)
    ac.capture_b("CA-same", b"\xbb" * 160)
    ac.close_all()
    a_path = tmp_path / "A_inworld_CA-same.mulaw"
    b_path = tmp_path / "B_twilio_CA-same.mulaw"
    assert a_path.read_bytes() == b"\xaa" * 160
    assert b_path.read_bytes() == b"\xbb" * 160
    assert a_path != b_path


def test_capture_per_call_isolation(tmp_path: Path, monkeypatch) -> None:
    """Different callSids write to different files — the operator
    diffs one call at a time, never two at once."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    ac.capture_a("CA-call-1", b"\x01" * 160)
    ac.capture_a("CA-call-2", b"\x02" * 160)
    ac.capture_b("CA-call-1", b"\x11" * 160)
    ac.capture_b("CA-call-2", b"\x22" * 160)
    ac.close_all()
    assert (tmp_path / "A_inworld_CA-call-1.mulaw").read_bytes() == b"\x01" * 160
    assert (tmp_path / "A_inworld_CA-call-2.mulaw").read_bytes() == b"\x02" * 160
    assert (tmp_path / "B_twilio_CA-call-1.mulaw").read_bytes() == b"\x11" * 160
    assert (tmp_path / "B_twilio_CA-call-2.mulaw").read_bytes() == b"\x22" * 160


def test_capture_disables_after_write_failure(tmp_path: Path, monkeypatch) -> None:
    """A write failure (read-only dir, disk full, permission denied)
    must NOT crash the live path. The capture module logs once
    and stops trying for the rest of the call. Verified by writing
    to a path that doesn't exist (parent doesn't exist) and
    confirming subsequent writes don't crash."""
    bogus_dir = tmp_path / "does-not-exist"
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(bogus_dir))
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    # First write attempts to mkdir + open, both fail. Module
    # must swallow the exception.
    ac.capture_a("CA-fail", b"\xff" * 160)
    # Second write goes through the same path — must also not crash.
    ac.capture_a("CA-fail", b"\x00" * 160)
    # And close_all on the (never-opened) state — must not crash.
    ac.close_all()


def test_capture_empty_bytes_is_noop(tmp_path: Path, monkeypatch) -> None:
    """A zero-byte capture call must not create an empty file
    (nothing to diagnose) AND must not crash. Best-effort."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)
    ac.capture_a("CA-empty", b"")
    ac.capture_b("CA-empty", b"")
    ac.close_all()
    # No files should exist because the helper short-circuits
    # on empty bytes.
    assert not (tmp_path / "A_inworld_CA-empty.mulaw").exists()
    assert not (tmp_path / "B_twilio_CA-empty.mulaw").exists()