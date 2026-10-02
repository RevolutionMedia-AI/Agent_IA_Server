"""Tests for adapters/openai_stt_transcription.py.

The failure this adapter exists to prevent is a session that OPENS and
then never transcribes, because one field of the wire contract was
wrong. So the tests pin the exact session.update shape rather than
just asserting the function returns something.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct

import pytest

from STT_server.adapters.openai_stt_transcription import (
    DEFAULT_MODEL_ID,
    TARGET_SAMPLE_RATE,
    TRANSCRIPTION_MODELS,
    _mulaw_8k_to_pcm16_24k,
    build_session_update,
)
from STT_server.services import audio_codec


def test_catalog_is_exactly_the_three_documented_ids():
    """The picker, the router and the fallback catalog must agree.

    If this drifts, an agent can be saved with an stt_model the
    transcription adapter refuses and silently falls back to the
    speech-to-speech adapter.
    """
    from STT_server.services.credentials_resolver import _HARDCODED_STT_MODELS

    catalog = {m["id"] for m in _HARDCODED_STT_MODELS["openai"]}
    assert catalog == set(TRANSCRIPTION_MODELS)
    assert catalog == {
        "gpt-live-transcribe",
        "gpt-transcribe",
        "gpt-realtime-whisper",
    }
    # the default must itself be a member, never a fourth id
    assert DEFAULT_MODEL_ID in catalog


def test_session_update_is_a_transcription_session_at_24k():
    """Pin the four fields whose absence or wrong value produces a
    session that connects but never returns text.
    """
    msg = build_session_update("gpt-live-transcribe", "en")

    assert msg["type"] == "session.update"
    sess = msg["session"]
    assert sess["type"] == "transcription"

    audio_in = sess["audio"]["input"]
    # 24 kHz, NOT the 8 kHz Twilio hands us. Getting this wrong opens a
    # session that transcribes 3x fast (chipmunk) or not at all.
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_in["format"]["rate"] == TARGET_SAMPLE_RATE
    # gpt-live-transcribe has no server_vad/semantic_vad; we commit turns
    # ourselves from audio_ingest's VAD.
    assert audio_in["turn_detection"] is None
    assert audio_in["transcription"]["model"] == "gpt-live-transcribe"


def test_live_transcribe_sends_plural_languages_not_language():
    """The docs are explicit: gpt-live-transcribe / gpt-transcribe use
    `languages`. Sending `language` alongside is rejected, and
    gpt-realtime-whisper keeps the singular form.
    """
    modern = build_session_update("gpt-live-transcribe", "es")
    tr = modern["session"]["audio"]["input"]["transcription"]
    assert tr["languages"] == ["es"]
    assert "language" not in tr

    legacy = build_session_update("gpt-realtime-whisper", "es")
    tr_legacy = legacy["session"]["audio"]["input"]["transcription"]
    assert tr_legacy["language"] == "es"
    assert "languages" not in tr_legacy


def test_resampler_triples_the_sample_count_and_stays_in_range():
    """8 kHz -> 24 kHz is 3x. Also guards int16 overflow, which would
    wrap to full-scale negative and sound like loud static.

    Sample-exact equality at the stride positions is deliberately NOT
    asserted. That was a property of the old zero-stuffing code, which
    copied input samples into every 3rd output slot verbatim. A
    band-limited resample (resample_poly plus an anti-aliasing FIR) filters
    every output sample, so those positions are legitimately different.
    That the waveform survives is covered in test_openai_stt_resampler.py.
    """
    from STT_server.services.audio_codec import lin2ulaw

    n_8k = 40
    samples = [(i * 500) - 10000 for i in range(n_8k)]
    pcm_8k = struct.pack(f"<{n_8k}h", *samples)

    out = _mulaw_8k_to_pcm16_24k(lin2ulaw(pcm_8k, 2))

    got = struct.unpack(f"<{len(out) // 2}h", out)
    assert len(got) == n_8k * 3
    assert all(-32768 <= v <= 32767 for v in got)
    # the signal is still the caller's, not noise and not silence
    assert max(got) > 0


def test_resampler_handles_empty_and_odd_length_input():
    assert _mulaw_8k_to_pcm16_24k(b"") == b""
    # odd number of bytes -> ulaw2lin yields an odd PCM16 length; must
    # not raise on the //2 truncation
    assert _mulaw_8k_to_pcm16_24k(b"\xff") is not None


@pytest.mark.asyncio
async def test_turn_end_seq_from_vad_is_what_drives_commit(monkeypatch):
    """The adapter must commit on audio_ingest's VAD, not a private
    detector. If these drift the agent commits at the wrong time.
    """
    from STT_server.adapters import openai_stt_transcription as mod
    from STT_server.domain.session import CallSession

    sent: list[dict] = []

    class FakeWS:
        async def send(self, raw: str) -> None:
            sent.append(json.loads(raw))

    session = CallSession(session_key="commit-probe")
    session.stt_audio_queue.put_nowait(b"\xff" * 160)

    task = asyncio.create_task(
        mod._audio_sender(FakeWS(), session, "gpt-live-transcribe")
    )
    try:
        # real elapsed time, not sleep(0): the sender polls the turn-end
        # seq on a wait_for(timeout=...), so a zero-sleep spin would never
        # let the clock advance far enough for the poll to fire.
        for _ in range(50):
            await asyncio.sleep(0.005)
            if sent:
                break
        assert sent and sent[0]["type"] == "input_audio_buffer.append"
        assert not any(
            m["type"] == "input_audio_buffer.commit" for m in sent
        ), "a turn must not commit before the VAD says end-of-speech"

        # audio_ingest bumps this at FIN DE VOZ
        session.stt_turn_end_seq += 1
        for _ in range(60):
            await asyncio.sleep(0.005)
            if any(
                m["type"] == "input_audio_buffer.commit" for m in sent
            ):
                break
    finally:
        task.cancel()

    assert any(m["type"] == "input_audio_buffer.commit" for m in sent), (
        "a VAD end-of-speech must produce an input_audio_buffer.commit"
    )
