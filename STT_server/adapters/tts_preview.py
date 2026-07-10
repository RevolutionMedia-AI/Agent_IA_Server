"""TTS preview helper.

Build a synthetic CallSession with the user's TTS config and run
one TTS call for a sample text. Returns the collected mu-law audio
bytes so the FE can play them in a regular <audio> element. This
is the same path the live call uses, so what the user hears in
the preview is what callers will hear in production.
"""
from __future__ import annotations

import io
import logging
import wave
from typing import Optional

import numpy as np

from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider

log = logging.getLogger("stt_server.tts_preview")


async def preview_tts(
    user_id: Optional[str],
    provider: str,
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> bytes:
    """Run a single TTS call and return the raw mu-law 8 kHz audio bytes.

    `voice_id` and `model_id` come from the agent's per-user config or
    the per-provider defaults. `text` should be short (1-2 sentences)
    so the preview is fast.
    """
    # Build a synthetic session with the user's resolved TTS config.
    # `stream_tts_segment` reads credentials via resolve_provider(user_id,
    # provider) so we don't need to set anything beyond what the
    # resolution path expects.
    creds = resolve_provider(user_id, provider) if user_id else {}
    session = CallSession(session_key="preview")
    session.user_id = user_id
    session.tts_provider = provider
    if voice_id:
        session.voice_id = voice_id
    if model_id:
        session.model_id = model_id

    chunks: list[bytes] = []

    def _collect(item: dict) -> None:
        if item.get("type") == "audio":
            data = item.get("data")
            if isinstance(data, (bytes, bytearray)):
                chunks.append(bytes(data))

    # Dispatch to the configured provider. Mirrors the logic in
    # tts_dispatcher but we drive the call directly so we can collect
    # the audio bytes here.
    if provider == "elevenlabs":
        from STT_server.adapters.elevenlabs_tts import stream_tts_segment
        await stream_tts_segment(session, text, 0, _collect)
    elif provider == "rime":
        from STT_server.adapters.rime_tts import stream_tts_segment
        await stream_tts_segment(session, text, 0, _collect)
    else:
        raise ValueError(f"Unsupported TTS provider for preview: {provider!r}")

    mulaw_bytes = b"".join(chunks)
    return _wrap_mulaw_as_wav_pcm16(mulaw_bytes)


_MULAW_BIAS = 0x84


def _mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    # ponytail: ITU G.711 mu-law decode via numpy (the BE already ships
    # scipy/numpy — see rime_tts.py). audioop is deprecated in 3.13 so
    # we don't use it.
    arr = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    arr = np.bitwise_and(np.bitwise_not(arr).astype(np.int16), 0xFF)
    sign = (arr & 0x80) != 0
    exponent = (arr & 0x70) >> 4
    mantissa = arr & 0x0F
    sample = (mantissa << 3) + 0x84
    sample = sample << exponent
    sample = np.where(sign, sample - _MULAW_BIAS, _MULAW_BIAS - sample)
    return sample.astype(np.int16).tobytes()


def _wrap_mulaw_as_wav_pcm16(mulaw_bytes: bytes, sample_rate: int = 8000) -> bytes:
    # ponytail: emit PCM16 mono WAV (universally decoded by every browser
    # audio element). mu-law-WAV (format code 7) works in Chrome/Safari
    # but spotty in Firefox, so we decode and rewrap.
    pcm = _mulaw_to_pcm16(mulaw_bytes)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()
