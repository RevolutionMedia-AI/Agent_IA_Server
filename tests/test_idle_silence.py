from __future__ import annotations

import asyncio
import base64
import time

import pytest

from STT_server.domain.session import CallSession
from STT_server.services import audio_ingest, session_runtime, turn_manager


@pytest.mark.asyncio
async def test_voice_refreshes_idle_activity_before_turn_is_final(monkeypatch) -> None:
    async def voice(_frame: bytes) -> tuple[bool, int]:
        return True, 1000

    monkeypatch.setattr(audio_ingest, "is_probable_voice", voice)
    session = CallSession(session_key="voice-activity")
    session.last_activity_at = 0

    payload = base64.b64encode(b"\xff" * 160).decode("ascii")
    await audio_ingest.handle_incoming_media(session, payload)

    assert session.last_activity_at > 0


def idle_session() -> CallSession:
    session = CallSession(session_key="idle-monitor")
    session.idle_enabled = True
    session.idle_first_timeout_sec = 0.04
    session.idle_subsequent_timeout_sec = 1
    session.idle_disconnect_timeout_sec = 1
    session.idle_max_attempts = 1
    return session


@pytest.mark.asyncio
async def test_idle_clock_starts_after_assistant_playback(monkeypatch) -> None:
    prompted = asyncio.Event()

    async def fake_tts(_session, _text, _generation):
        prompted.set()

    monkeypatch.setattr(turn_manager, "run_tts_with_retries", fake_tts)
    monkeypatch.setattr(session_runtime, "IDLE_MONITOR_POLL_SEC", 0.005)
    session = idle_session()
    session.last_activity_at = time.monotonic() - 10
    session.assistant_speaking = True

    task = asyncio.create_task(session_runtime.monitor_idle_silence(session, object()))
    try:
        await asyncio.sleep(0.02)
        session.assistant_speaking = False
        await asyncio.sleep(0.02)
        assert not prompted.is_set()
        await asyncio.wait_for(prompted.wait(), timeout=0.2)
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
