#!/usr/bin/env python3
"""Run full TTS -> playback path locally and capture frames that would be sent to Twilio.

This script:
- enables RIME_SAVE_AUDIO and SAVE_TWILIO_FRAMES via env
- monkeypatches the playback sender to append raw mu-law frames to a file
- runs a TTS generation and lets playback_loop send frames to the fake sender
"""
import sys
import os
import asyncio
import time
from contextlib import suppress

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Enable diagnostics before importing modules that read config
os.environ.setdefault("RIME_SAVE_AUDIO", "1")
os.environ.setdefault("SAVE_TWILIO_FRAMES", "1")

from STT_server.domain.session import CallSession
import STT_server.services.playback_service as playback_service
from STT_server.adapters.rime_tts import stream_tts_segment


async def fake_send_twilio_media(ws, stream_sid, mulaw_audio: bytes) -> None:
    """Fake sender: append raw mulaw bytes to disk (one file per session/generation)."""
    # In this test harness we rely on playback_service's own
    # SAVE_TWILIO_FRAMES behavior to capture frames. Avoid writing here
    # as playback_loop already appends frames when SAVE_TWILIO_FRAMES=1,
    # which would otherwise duplicate frames in the capture file.
    await asyncio.sleep(0.001)


async def fake_send_twilio_mark(ws, stream_sid, mark_name: str) -> None:
    # noop for test
    await asyncio.sleep(0)


async def fake_send_twilio_clear(ws, stream_sid) -> None:
    await asyncio.sleep(0)


async def main():
    session = CallSession(session_key="localtest")
    session.stream_sid = "SIMULATED"
    # Ensure generation matches what stream_tts_segment will emit
    session.active_generation = 1

    # Remove any previous capture file so we start fresh
    out_fname = f"twilio_out_{session.session_key}_1.mulaw"
    try:
        if os.path.exists(out_fname):
            os.remove(out_fname)
    except Exception:
        pass

    # Monkeypatch the senders in playback_service module
    playback_service.send_twilio_media = fake_send_twilio_media
    playback_service.send_twilio_mark = fake_send_twilio_mark
    playback_service.send_twilio_clear = fake_send_twilio_clear
    playback_service._test_session_key = session.session_key

    # Start playback loop
    pb_task = asyncio.create_task(playback_service.playback_loop(None, session))

    # Emit TTS into playback via the adapter
    text = "Hola, esto es una prueba completa de reproducción y captura para comparar frames." 
    try:
        await stream_tts_segment(session, text, 1, lambda item: playback_service.emit_playback_item(session, item))
    except Exception as exc:
        print("stream_tts_segment error:", exc)

    # Wait sufficiently long for playback to finish sending all frames.
    # Typical duration = frames * 20ms; use 10s to be safe for longer replies.
    await asyncio.sleep(10.0)

    pb_task.cancel()
    with suppress(asyncio.CancelledError):
        await pb_task

    print("Done. Files: rime_tts_localtest_1.mulaw (Rime) and twilio_out_localtest_1.mulaw (sent)")


if __name__ == '__main__':
    asyncio.run(main())
