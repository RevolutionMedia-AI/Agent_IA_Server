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
from STT_server.services._instrumentation import StageTimer, Stages
from STT_server.services.audio_frame_processor import AudioFrameProcessor
from STT_server.services.credentials_resolver import resolve_provider, resolve_for_session
from STT_server.services.thread_pool import to_thread as _to_thread

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
    """Return the API key for the given TTS provider, or ''.

    ponytail: 009_agent_use_own_key.sql. Delegates to
    resolve_for_session so the resolver picks platform env vs.
    per-user key based on session.tts_use_own_key. False = platform
    env (Railway OPENAI_API_KEY etc.) fills the gap; True = per-user
    wins. Empty dict means the operator must save a key somewhere.
    """
    creds = resolve_for_session(session, "tts", provider)
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
        # ponytail: P3 — surface the missing-key error to the playback queue
        # so the operator sees a structured item instead of a silent mute.
        # The exception is still raised so callers can branch, but emit the
        # marker FIRST so the queue advances cleanly with the consumer's
        # error-handling branch (playback_service.playback_loop already
        # handles `item_type == "error"`).
        log.error(
            "[TTS] %s API key not configured for session=%s; emitting error marker",
            provider, session.session_key,
        )
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"{provider} API key not configured",
        })
        emit_item({"type": "segment_end", "generation": generation})
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
    from STT_server.services._instrumentation import Stages  # ponytail: lazy per spec
    import time
    started = time.perf_counter()
    if not api_key:
        # ponytail: P3 — defense-in-depth. _stream_openai should not be
        # called without an api_key, but emit a structured error if it is.
        log.error("[TTS] openai called without api_key for session=%s", session.session_key)
        emit_item({"type": "error", "generation": generation,
                   "message": "openai TTS: API key not configured"})
        emit_item({"type": "segment_end", "generation": generation})
        return None, 0.0
    # ponytail: per-agent speed override (006_agent_runtime_params.sql).
    # OpenAI TTS accepts 0.25..4.0; we clamp to the adapter's safe
    # range so a typo doesn't trip an HTTP 400.
    _speed = getattr(session, "tts_speed", None)
    speed = max(0.25, min(4.0, _speed if _speed is not None else 1.0))
    body = json.dumps({
        "model": getattr(session, "tts_model", None) or "tts-1",
        "input": text,
        "voice": getattr(session, "voice_id", None) or "alloy",
        "response_format": "pcm",  # raw PCM16 LE 24 kHz mono
        "speed": speed,
    }).encode("utf-8")

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None

    def _emit_frame(frame: bytes) -> None:
        nonlocal ttfb_ms
        if ttfb_ms is None:
            ttfb_ms = (time.perf_counter() - started) * 1000
            # ponytail: stamp TTS_FIRST_BYTE on the first 160-byte frame emitted.
            session._stage_timer = session._stage_timer or StageTimer(
                call_id=session.session_key,
                turn_id=0,
                generation=session.active_generation,
            )
            if Stages.TTS_FIRST_BYTE not in session._stage_timer._stages:
                session._stage_timer.mark(Stages.TTS_FIRST_BYTE)
        loop.call_soon_threadsafe(
            emit_item, {"type": "audio", "generation": generation, "data": frame}
        )

    def _fetch() -> None:
        from STT_server.adapters.rime_tts import _pcm16_bytes_to_mulaw_8k
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        # ponytail: AudioFrameProcessor owns 20ms framing; emit_silence_tail=False
        # drops the partial trailing frame to avoid a <20ms packet boundary click.
        proc = AudioFrameProcessor(emit_silence_tail=False)
        pcm_remainder = b""
        with urllib.request.urlopen(req, timeout=45) as resp:
            while True:
                # ponytail: cheap closed-check per chunk; thread stays sync.
                if getattr(session, "closed", False):
                    break
                chunk = resp.read(8192)
                if not chunk:
                    break
                mulaw_bytes, pcm_remainder = _pcm16_bytes_to_mulaw_8k(chunk, 24000, pcm_remainder)
                if not mulaw_bytes:
                    continue
                for frame in proc.feed(mulaw_bytes):
                    _emit_frame(frame)
        for frame in proc.flush():
            _emit_frame(frame)

    await _to_thread(_fetch)
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
    if not api_key:
        # ponytail: P3 — defense-in-depth. _stream_deepgram should not be
        # called without an api_key, but emit a structured error if it is.
        log.error("[TTS] deepgram called without api_key for session=%s", session.session_key)
        emit_item({"type": "error", "generation": generation,
                   "message": "deepgram TTS: API key not configured"})
        emit_item({"type": "segment_end", "generation": generation})
        return None, 0.0
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

    raw = await _to_thread(_fetch)
    ttfb_ms = (time.perf_counter() - started) * 1000
    if raw:
        # ponytail: stamp TTS_FIRST_BYTE on the first audio byte from this adapter.
        session._stage_timer = session._stage_timer or StageTimer(
            call_id=session.session_key,
            turn_id=0,
            generation=session.active_generation,
        )
        if Stages.TTS_FIRST_BYTE not in session._stage_timer._stages:
            session._stage_timer.mark(Stages.TTS_FIRST_BYTE)
        # ponytail: AudioFrameProcessor is the single owner of frame buffering;
        # emit_silence_tail=False drops the partial trailing frame to avoid a
        # <20ms packet boundary click.
        proc = AudioFrameProcessor(emit_silence_tail=False)
        for frame in proc.feed(bytes(raw)):
            emit_item({"type": "audio", "generation": generation, "data": frame})
        proc.flush()
    emit_item({"type": "segment_end", "generation": generation})
    return ttfb_ms, (time.perf_counter() - started) * 1000