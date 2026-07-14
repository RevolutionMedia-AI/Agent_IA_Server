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
import asyncio
import json
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np

from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider, test_provider

log = logging.getLogger("stt_server.tts_preview")


async def preview_tts(
    user_id: Optional[str],
    provider: str,
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """Run a single TTS call and return the raw mu-law 8 kHz audio bytes.

    `voice_id` and `model_id` come from the agent's per-user config or
    the per-provider defaults. `text` should be short (1-2 sentences)
    so the preview is fast.

    `api_key` (optional) lets the FE inline a fresh key for the test
    without going through the persisted credential. Same priority as
    the test endpoint: caller-supplied > stored > env.
    """
    # Build a synthetic session with the user's resolved TTS config.
    # `stream_tts_segment` reads credentials via resolve_provider(user_id,
    # provider) so we don't need to set anything beyond what the
    # resolution path expects.
    inline_key = api_key.strip() if api_key and api_key.strip() else None
    if inline_key:
        creds = {"api_key": inline_key}
    else:
        creds = resolve_provider(user_id, provider) if user_id else {}
    session = CallSession(session_key="preview")
    session.user_id = user_id
    session.tts_provider = provider
    if voice_id:
        session.voice_id = voice_id
    if model_id:
        # ponytail: L3 from the call-flow audit. The dataclass has
        # `tts_model` (not `model_id`); the dispatcher and the
        # Inworld/Rime adapters read tts_model. The preview does its
        # own dispatch below (it doesn't go through tts_dispatcher)
        # so this is currently dead, but keeping the field name
        # consistent avoids future drift if the preview ever does
        # route through the dispatcher.
        session.tts_model = model_id

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
    elif provider == "openai":
        # ponytail: OpenAI TTS preview. We don't have a streaming
        # adapter (the live call path is a future feature); the
        # preview hits /v1/audio/speech once and converts the raw
        # PCM16 LE 24 kHz response to mu-law 8 kHz via the helpers
        # rime_tts already ships. Inline-key wins over stored.
        if not inline_key:
            raise RuntimeError("OpenAI API key not configured.")
        body = json.dumps({
            "model": model_id or "tts-1",
            "input": text,
            "voice": voice_id or "alloy",
            "response_format": "pcm",  # raw PCM16 LE mono @ 24 kHz
            "speed": 1.0,
        }).encode("utf-8")

        def _oai_fetch() -> bytes:
            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/speech",
                data=body,
                headers={"Authorization": f"Bearer {inline_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()

        try:
            pcm_bytes = await asyncio.to_thread(_oai_fetch)
        except Exception as exc:
            log.warning("[tts_preview] provider=openai fetch failed: %s", exc)
            raise
        from STT_server.adapters.rime_tts import _pcm16_bytes_to_mulaw_8k
        mulaw, _ = _pcm16_bytes_to_mulaw_8k(pcm_bytes, 24000)
        if mulaw:
            chunks.append(mulaw)
    elif provider == "deepgram":
        # ponytail: Deepgram TTS preview. We bypass deepgram_tts.py
        # (its stream_tts_segment ignores inline keys) and POST
        # /v1/speak directly with mu-law/8000/container=none, the
        # exact params the live call uses. The response is already
        # mu-law 8 kHz so no conversion is needed.
        if not inline_key:
            raise RuntimeError("Deepgram API key not configured.")
        params = urllib.parse.urlencode({
            "model": model_id or "aura-asteria-en",
            "encoding": "mulaw",
            "sample_rate": "8000",
            "container": "none",
        })
        body = json.dumps({"text": text}).encode("utf-8")
        url = f"https://api.deepgram.com/v1/speak?{params}"

        def _dg_fetch() -> bytes:
            req = urllib.request.Request(
                url, data=body,
                headers={"Authorization": f"Token {inline_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()

        try:
            raw = await asyncio.to_thread(_dg_fetch)
        except Exception as exc:
            log.warning("[tts_preview] provider=deepgram fetch failed: %s", exc)
            raise
        if raw:
            chunks.append(bytes(raw))
    elif provider == "inworld":
        # ponytail: Inworld TTS preview. Asks the non-streaming
        # /tts/v1/voice for MULAW@8k; the bytes Inworld returns go
        # straight to the WAV wrapper with no conversion. Live
        # streaming path lives in adapters/inworld_tts.py and shares
        # the same audioConfig.
        if not inline_key:
            raise RuntimeError("Inworld API key not configured.")
        from STT_server.adapters.inworld_tts import fetch_preview as _iw_fetch

        def _iw_run() -> bytes:
            return _iw_fetch(text, voice_id or "", model_id or "", inline_key)

        try:
            mulaw = await asyncio.to_thread(_iw_run)
        except Exception as exc:
            log.warning("[tts_preview] provider=inworld fetch failed: %s", exc)
            raise
        if mulaw:
            chunks.append(bytes(mulaw))
    else:
        # ponytail: validation-only path. Providers without a
        # streaming adapter (Inworld today, future ones) still need
        # a working /tts/preview so the FE can confirm "your key
        # works against this provider's catalog endpoint and the
        # text you typed was received". We hit the provider's
        # catalog/auth check, then return a tiny silent WAV as
        # placeholder audio. The FE renders a "preview not
        # available for this provider" message instead of the audio.
        try:
            result = test_provider(user_id, provider, inline_key)
            if not result.get("valid"):
                raise RuntimeError(result.get("message") or "validation failed")
            log.info(
                "[tts_preview] provider=%s validation-only OK (text=%d chars, voice=%s, model=%s)",
                provider, len(text or ""), voice_id or "-", model_id or "-",
            )
        except Exception as exc:
            log.warning("[tts_preview] provider=%s validation failed: %s", provider, exc)
            raise
        # 1 second of mu-law silence = 0xFF byte repeated. Wrapped
        # in PCM16 WAV so the FE <audio> can decode it on every
        # browser.
        return _wrap_mulaw_as_wav_pcm16(b'\xff' * 8000)

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
