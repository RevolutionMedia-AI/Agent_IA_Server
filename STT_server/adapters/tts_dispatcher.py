"""TTS Dispatcher — routes TTS requests to the correct provider based on session config.

Supports:
  - elevenlabs: ElevenLabs WebSocket TTS (ulaw_8000 output)
  - rime: Rime WebSocket TTS (PCM -> mu-law conversion)

The provider is determined by:
  1. session.tts_provider (per-session override from frontend)
  2. DEFAULT_TTS_PROVIDER (global config fallback)
"""

import logging

from STT_server.config import DEFAULT_TTS_PROVIDER
from STT_server.domain.session import CallSession, VALID_TTS_PROVIDERS

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

    # Should not reach here due to validation, but just in case
    raise RuntimeError(f"Unknown TTS provider: {provider}")