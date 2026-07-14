"""TTS Dispatcher — routes TTS requests to the correct provider based on session config.

Supports:
  - elevenlabs: ElevenLabs WebSocket TTS (ulaw_8000 output)
  - rime: Rime WebSocket TTS (PCM -> mu-law conversion)
  - openai: OpenAI /v1/audio/speech (PCM -> mu-law conversion)
  - deepgram: Deepgram /v1/speak with mulaw/8000/container=none

The provider is determined by:
  1. session.tts_provider (per-session override from frontend)
  2. DEFAULT_TTS_PROVIDER (global config fallback)
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request

from STT_server.config import DEFAULT_TTS_PROVIDER
from STT_server.domain.session import CallSession, VALID_TTS_PROVIDERS
from STT_server.services.credentials_resolver import resolve_provider

log = logging.getLogger("stt_server")


def _resolve_provider(session: CallSession) -> str:
    """Return the effective TTS provider for a session."""
    provider = getattr(session, "tts_provider", None) or DEFAULT_TTS_PROVIDER
    provider = provider.strip().lower()
    if provider not in VALID_TTS_PROVIDERS:
        log.warning(
            "[TTS] Invalid tts_provider '%s' on session %s, falling back to '%s'",
            provider, session.session_key, DEFAULT_TTS_PROVIDER,
        )
        provider = DEFAULT_TTS_PROVIDER
    return provider


def _resolve_api_key(session: CallSession, provider: str) -> str:
    """Return the per-user API key for the given provider, or ''."""
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, provider) if user_id else {}
    return (creds.get("api_key") or "").strip()


async def stream_tts_segment(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
) -> tuple[float | None, float]:
    """Stream TTS audio using the session's configured provider.

    Dispatches to the appropriate adapter's ``stream_tts_segment`` function.
    """
    provider = _resolve_provider(session)
    log.info(
        "[TTS] Dispatching to provider='%s' session=%s gen=%s text_len=%d",
        provider, session.session_key, generation, len(text),
    )

    if provider == "elevenlabs":
        from STT_server.adapters.elevenlabs_tts import stream_tts_segment as _elevenlabs
        return await _elevenlabs(session, text, generation, emit_item)

    if provider == "rime":
        from STT_server.adapters.rime_tts import stream_tts_segment as _rime
        return await _rime(session, text, generation, emit_item)

    if provider == "inworld":
        from STT_server.adapters.inworld_tts import stream_tts_segment as _inworld
        return await _inworld(session, text, generation, emit_item)

    # ponytail: HTTP-only providers (no streaming adapter) get an inline
    # implementation here. They collect one response and emit it as a
    # single chunk - latency is dominated by the provider's first byte
    # anyway, so an inline path keeps the call simple.
    api_key = _resolve_api_key(session, provider)
    if not api_key:
        raise RuntimeError(f"{provider} API key not configured.")

    if provider == "openai":
        return await _stream_openai(session, text, generation, emit_item, api_key)
    if provider == "deepgram":
        return await _stream_deepgram(session, text, generation, emit_item, api_key)

    # Should not reach here due to validation, but just in case
    raise RuntimeError(f"Unknown TTS provider: {provider}")


async def _stream_openai(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
    api_key: str,
) -> tuple[float | None, float]:
    import time
    started = time.perf_counter()
    body = json.dumps({
        "model": getattr(session, "tts_model", None) or "tts-1",
        "input": text,
        "voice": getattr(session, "voice_id", None) or "alloy",
        "response_format": "pcm",  # raw PCM16 LE 24 kHz mono
        "speed": 1.0,
    }).encode("utf-8")

    def _fetch() -> bytes:
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read()

    pcm_bytes = await asyncio.to_thread(_fetch)
    from STT_server.adapters.rime_tts import _pcm16_bytes_to_mulaw_8k
    mulaw, _ = _pcm16_bytes_to_mulaw_8k(pcm_bytes, 24000)
    ttfb_ms = (time.perf_counter() - started) * 1000
    if mulaw:
        emit_item({"type": "audio", "generation": generation, "data": mulaw})
    emit_item({"type": "segment_end", "generation": generation})
    return ttfb_ms, (time.perf_counter() - started) * 1000


async def _stream_deepgram(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
    api_key: str,
) -> tuple[float | None, float]:
    import time
    started = time.perf_counter()
    params = urllib.parse.urlencode({
        "model": getattr(session, "voice_id", None) or "aura-asteria-en",
        "encoding": "mulaw",
        "sample_rate": "8000",
        "container": "none",
    })
    body = json.dumps({"text": text}).encode("utf-8")
    url = f"https://api.deepgram.com/v1/speak?{params}"

    def _fetch() -> bytes:
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read()

    raw = await asyncio.to_thread(_fetch)
    ttfb_ms = (time.perf_counter() - started) * 1000
    if raw:
        emit_item({"type": "audio", "generation": generation, "data": bytes(raw)})
    emit_item({"type": "segment_end", "generation": generation})
    return ttfb_ms, (time.perf_counter() - started) * 1000