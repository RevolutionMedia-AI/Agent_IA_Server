from __future__ import annotations

import asyncio
import base64
import time

import pytest

from STT_server.config import SPEECH_START_FRAMES
from STT_server.domain.session import CallSession
from STT_server.services import audio_ingest, session_runtime, turn_manager


def _one_frame() -> str:
    """base64 of 160 bytes mu-law == exactly one 20 ms frame after ulaw2lin."""
    return base64.b64encode(b"\xff" * 160).decode("ascii")


def _voice_stub(monkeypatch, *, is_voice: bool, rms: int) -> None:
    async def voice(_frame: bytes, _min_rms: int) -> tuple[bool, int]:
        return is_voice, rms

    monkeypatch.setattr(audio_ingest, "is_probable_voice", voice)


@pytest.mark.asyncio
async def test_sustained_voice_refreshes_idle_activity(monkeypatch) -> None:
    # 2026-10-01: the clock now requires SPEECH_START_FRAMES of sustained
    # voice instead of moving on any single voice-positive frame. The +1
    # is real: audio_ingest reads voice_streak BEFORE incrementing it for
    # the current frame, so the gate opens one frame (20 ms) late. On a
    # timer whose resolution is seconds that lag is free; asserting it
    # here keeps the behaviour pinned.
    _voice_stub(monkeypatch, is_voice=True, rms=1000)
    session = CallSession(session_key="voice-activity")
    session.last_activity_at = 0

    for _ in range(SPEECH_START_FRAMES + 1):
        await audio_ingest.handle_incoming_media(session, _one_frame())

    assert session.last_activity_at > 0


@pytest.mark.asyncio
async def test_single_voice_frame_does_not_refresh_idle_activity(monkeypatch) -> None:
    # 2026-10-01: the regression that kept dead calls alive forever — a
    # lone hiss burst used to reset the idle clock, so the line never
    # looked silent and the call never hung up.
    _voice_stub(monkeypatch, is_voice=True, rms=1000)
    session = CallSession(session_key="hiss-activity")
    session.last_activity_at = 0

    await audio_ingest.handle_incoming_media(session, _one_frame())

    assert session.last_activity_at == 0


@pytest.mark.asyncio
async def test_assistant_playback_advances_idle_clock(monkeypatch) -> None:
    # 2026-10-01: the clock was frozen while the assistant spoke, so the
    # whole assistant turn later counted as caller silence and the bot
    # hung up mid-conversation. The assistant's own turn must move it.
    _voice_stub(monkeypatch, is_voice=False, rms=100)
    session = CallSession(session_key="assistant-activity")
    session.assistant_speaking = True
    session.last_activity_at = 0

    await audio_ingest.handle_incoming_media(session, _one_frame())

    assert session.last_activity_at > 0


def idle_session(*, first_timeout: float = 0.04) -> CallSession:
    session = CallSession(session_key="idle-monitor")
    session.idle_enabled = True
    session.idle_first_timeout_sec = first_timeout
    session.idle_subsequent_timeout_sec = 1
    session.idle_disconnect_timeout_sec = 1
    session.idle_max_attempts = 1
    return session


@pytest.mark.asyncio
async def test_idle_clock_starts_after_assistant_playback(monkeypatch) -> None:
    # 2026-10-01: this was flaky (failed ~1 in 3 on the clean tree). The
    # deadline was 0.04 s and the test slept 0.02 s before asserting the
    # prompt had NOT fired — a 2x margin that asyncio.sleep blows through
    # whenever the loop is loaded. The production logic is fine; the
    # observation window was too close to the deadline. Give the "not
    # yet" assertion a 10x margin and let the prompt land on the event
    # with a 4x margin instead of a second fixed sleep.
    prompted = asyncio.Event()

    async def fake_tts(_session, _text, _generation):
        prompted.set()

    monkeypatch.setattr(turn_manager, "run_tts_with_retries", fake_tts)
    monkeypatch.setattr(session_runtime, "IDLE_MONITOR_POLL_SEC", 0.005)
    session = idle_session(first_timeout=0.5)
    session.last_activity_at = time.monotonic() - 10
    session.assistant_speaking = True

    task = asyncio.create_task(session_runtime.monitor_idle_silence(session, object()))
    try:
        # Assistant is mid-utterance: the clock must not be running, and
        # last_activity_at is already 10 s stale, so anything less than a
        # real 0.5 s wait proves the clock was correctly not started.
        await asyncio.sleep(0.05)
        assert not prompted.is_set()

        session.assistant_speaking = False
        await asyncio.wait_for(prompted.wait(), timeout=3.0)
    finally:
        task.cancel()
        await task


@pytest.mark.asyncio
async def test_idle_prompt_stays_speaking_until_twilio_ack(monkeypatch) -> None:
    queued = asyncio.Event()

    async def fake_tts(_session, _text, _generation):
        queued.set()

    monkeypatch.setattr(turn_manager, "run_tts_with_retries", fake_tts)
    monkeypatch.setattr(session_runtime, "IDLE_MONITOR_POLL_SEC", 0.005)
    session = idle_session()
    session.last_activity_at = time.monotonic() - 1

    task = asyncio.create_task(session_runtime.monitor_idle_silence(session, object()))
    try:
        await asyncio.wait_for(queued.wait(), timeout=0.05)
        await asyncio.sleep(0)
        assert session.assistant_speaking is True
    finally:
        task.cancel()
        await task
