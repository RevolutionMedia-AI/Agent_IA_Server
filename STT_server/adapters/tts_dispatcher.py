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
    seg_idx: int = 0,
) -> tuple[float | None, float]:
    """Stream TTS audio using the session's configured provider.

    Dispatches to the appropriate adapter's ``stream_tts_segment`` function.

    `seg_idx` is forwarded to the adapter so the TTS observability
    chain (`TTS_RAW_SEGMENT` → `TTS_SANITIZED_SEGMENT` →
    `TSS_INWORLD_BODY`) can join on the same value across the three
    log sites.
    """
    provider = _resolve_provider(session)
    # ponytail: 2026-10-02 — INFO→DEBUG. One line per segment; the
    # provider is a per-agent constant, so this repeated a known value.
    log.debug(
        "[TTS] Dispatching to provider='%s' session=%s gen=%s seg=%d text_len=%d",
        provider, session.session_key, generation, seg_idx, len(text),
    )

    if provider == "elevenlabs":
        from STT_server.adapters.elevenlabs_tts import stream_tts_segment as _elevenlabs
        return await _elevenlabs(session, text, generation, emit_item, seg_idx=seg_idx)

    if provider == "rime":
        from STT_server.adapters.rime_tts import stream_tts_segment as _rime
        return await _rime(session, text, generation, emit_item, seg_idx=seg_idx)

    if provider == "inworld":
        from STT_server.adapters.inworld_tts import stream_tts_segment as _inworld
        return await _inworld(session, text, generation, emit_item, seg_idx=seg_idx)

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
    """Stream OpenAI /v1/audio/speech into the mu-law playback pipeline.

    ponytail: 2026-10-02 — was a bare urllib.urlopen(timeout=45) with the
    model defaulting to tts-1, no voice validation, no `instructions`, no
    error classification and no cancellation of the HTTP stream on
    barge-in. Changes:
      * model / voice / instructions / input / speed are validated and
        clamped by services.openai_tts.build_speech_request, so a bad
        frontend value cannot 400 mid-call and an unknown stored model
        still places calls (falls back to gpt-4o-mini-tts).
      * http.client instead of urllib so connect/first-byte and mid-stream
        reads get SEPARATE budgets and conn.close() actually releases the
        socket when a barge-in cancels us mid-body.
      * the read loop aborts as soon as `generation` stops being the active
        generation, so a barge-in'd turn stops burning CPU and stops
        holding an HTTP connection instead of draining the rest of the body
        into the void.
      * resample+encode time is measured (they happen inside one converter
        call, so a single figure, not two).
    """
    from STT_server.services._instrumentation import Stages  # ponytail: lazy per spec
    from STT_server.services.openai_tts import (
        OPENAI_TTS_CONNECT_TIMEOUT_SEC,
        OPENAI_TTS_FIRST_BYTE_TIMEOUT_SEC,
        OPENAI_TTS_READ_TIMEOUT_SEC,
        OpenAITtsError,
        build_speech_request,
        classify_http_status,
    )
    import http.client
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
    # OpenAI TTS accepts 0.25..4.0; build_speech_request clamps so a typo
    # can't trip an HTTP 400.
    _speed = getattr(session, "tts_speed", None)
    payload, src_rate = build_speech_request(
        text,
        model_id=getattr(session, "tts_model", None),
        voice_id=getattr(session, "voice_id", None),
        instructions=getattr(session, "tts_instructions", None),
        speed=_speed,
    )
    body = json.dumps(payload).encode("utf-8")
    # ponytail: the TTS request starts HERE, not when the first byte lands.
    # playback_loop reads this to compute tts_first_audio_sent_ms, i.e. the
    # true request -> caller-hears-it number instead of request -> provider.
    session._tts_request_started_at = time.monotonic()

    log.debug(
        "[TTS_OPENAI] session=%s gen=%d model=%s voice=%s speed=%s "
        "instructions=%s input_chars=%d",
        session.session_key, generation, payload["model"], payload["voice"],
        payload.get("speed"), bool(payload.get("instructions")),
        len(payload["input"]),
    )

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None
    total_ms = 0.0
    cancelled = False
    convert_ms = 0.0

    metrics = getattr(session, "metrics", None)

    def _observe(name: str, value: float) -> None:
        if metrics is not None:
            try:
                metrics.observe_ms(name, value)
            except Exception:
                pass

    def _emit_frame(frame: bytes) -> None:
        nonlocal ttfb_ms
        if ttfb_ms is None:
            ttfb_ms = (time.perf_counter() - started) * 1000
            _observe("tts_first_byte_ms", ttfb_ms)
            # ponytail: stamp TTS_FIRST_BYTE on the first 160-byte frame emitted.
            # getattr, not attribute access: _stage_timer is not a declared
            # CallSession field, it is attached by session_runtime. The
            # sibling adapters (inworld/rime/elevenlabs) all getattr it for
            # that reason; reading it directly raised AttributeError on any
            # path that reaches TTS before session_runtime has run.
            _timer = getattr(session, "_stage_timer", None)
            if _timer is None:
                _timer = StageTimer(
                    call_id=session.session_key,
                    turn_id=0,
                    generation=session.active_generation,
                )
                session._stage_timer = _timer
            if Stages.TTS_FIRST_BYTE not in _timer._stages:
                _timer.mark(Stages.TTS_FIRST_BYTE)
        loop.call_soon_threadsafe(
            emit_item, {"type": "audio", "generation": generation, "data": frame}
        )

    def _fetch() -> None:
        nonlocal cancelled, convert_ms
        from STT_server.adapters.rime_tts import _pcm16_bytes_to_mulaw_8k
        # AudioFrameProcessor owns 20ms framing; emit_silence_tail=False
        # drops the partial trailing frame to avoid a <20ms packet boundary click.
        proc = AudioFrameProcessor(emit_silence_tail=False)
        pcm_remainder = b""
        # NO Authorization header is ever logged (spec §15/§19): only the
        # status and a truncated error body reach the logs.
        conn = http.client.HTTPSConnection(
            "api.openai.com", timeout=OPENAI_TTS_CONNECT_TIMEOUT_SEC,
        )
        try:
            conn.request(
                "POST", "/v1/audio/speech", body=body,
                headers={
                    # ponytail: key comes from resolve_for_session (stored
                    # per-user credential or platform env). Never from the
                    # frontend, never from a request body.
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=OPENAI_TTS_FIRST_BYTE_TIMEOUT_SEC,
            )
            if conn.sock is not None:
                conn.sock.settimeout(OPENAI_TTS_FIRST_BYTE_TIMEOUT_SEC)
            resp = conn.getresponse()
            if resp.status != 200:
                snippet = ""
                try:
                    snippet = resp.read(300).decode("utf-8", "replace")
                except Exception:
                    pass
                classify_http_status(resp.status, snippet)
            # ponytail: once bytes are flowing, bound each individual read
            # separately. A stalled mid-stream socket must not hold the
            # thread for the whole first-byte budget.
            if conn.sock is not None:
                conn.sock.settimeout(OPENAI_TTS_READ_TIMEOUT_SEC)
            while True:
                if getattr(session, "closed", False):
                    break
                # ponytail: barge-in. When the user talks, active_generation
                # moves on and cancelled_through advances; this turn's audio
                # is dead. Stop consuming instead of decoding a body nobody
                # will hear — playback_loop would drop every frame anyway.
                if generation != getattr(session, "active_generation", generation):
                    cancelled = True
                    break
                chunk = resp.read(8192)
                if not chunk:
                    break
                c0 = time.perf_counter()
                mulaw_bytes, pcm_remainder = _pcm16_bytes_to_mulaw_8k(
                    chunk, src_rate, pcm_remainder, session,
                )
                convert_ms += (time.perf_counter() - c0) * 1000
                if not mulaw_bytes:
                    continue
                for frame in proc.feed(mulaw_bytes):
                    _emit_frame(frame)
            for frame in proc.flush():
                _emit_frame(frame)
        finally:
            # Always release the socket: barge-in, timeout, 5xx and normal
            # end-of-stream all land here.
            try:
                conn.close()
            except Exception:
                pass

    try:
        await _to_thread(_fetch)
    except OpenAITtsError as exc:
        # ponytail: structured failure. Emit the error marker + segment_end
        # so playback_loop advances and the call does not sit in SPEAKING
        # forever waiting on a mark that will never come.
        log.warning(
            "[TTS_OPENAI] %s status=%s stage=%s retryable=%s session=%s gen=%d",
            exc, exc.status, exc.stage, exc.retryable, session.session_key, generation,
        )
        if metrics is not None:
            try:
                metrics.incr("tts_error_total", 1)
                metrics.observe_ms("tts_error_ms", (time.perf_counter() - started) * 1000)
            except Exception:
                pass
        emit_item({"type": "error", "generation": generation,
                   "message": f"openai TTS: {exc}"})
        emit_item({"type": "segment_end", "generation": generation})
        return None, (time.perf_counter() - started) * 1000
    except Exception as exc:
        log.exception(
            "[TTS_OPENAI] transport failure session=%s gen=%d", session.session_key, generation,
        )
        if metrics is not None:
            try:
                metrics.incr("tts_error_total", 1)
            except Exception:
                pass
        emit_item({"type": "error", "generation": generation,
                   "message": f"openai TTS: {type(exc).__name__}"})
        emit_item({"type": "segment_end", "generation": generation})
        return None, (time.perf_counter() - started) * 1000
    finally:
        total_ms = (time.perf_counter() - started) * 1000
        if metrics is not None:
            try:
                metrics.incr("tts_cancelled_total", 1) if cancelled else metrics.incr("tts_completed_total", 1)
            except Exception:
                pass
        _observe("tts_resample_encode_ms", convert_ms)
        _observe("tts_generation_ms", total_ms)

    emit_item({"type": "segment_end", "generation": generation})
    return ttfb_ms, total_ms


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