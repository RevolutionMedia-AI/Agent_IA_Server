import asyncio
import base64
import json
import math
import logging
import struct
import time
import os

import websockets

from STT_server.config import (
    RIME_TTS_MODEL_ID,
    RIME_TTS_SAMPLE_RATE,
    TTS_IDLE_TIMEOUT_SEC,
    TTS_TTFB_TIMEOUT_SEC,
)
from STT_server.domain.language import (
    get_tts_model,
    infer_supported_language_from_text,
    normalize_supported_language,
    sanitize_tts_text,
)
from STT_server.domain.session import CallSession
from STT_server.services._instrumentation import StageTimer, Stages
from STT_server.services.audio_frame_processor import AudioFrameProcessor
from STT_server.services.credentials_resolver import resolve_for_session


log = logging.getLogger("stt_server")

RIME_WS_URL = "wss://users-ws.rime.ai/ws3"
TWILIO_SAMPLE_RATE = 8000
_SCIPY_AVAILABLE = None
_np = None
_resample_poly = None

# ── mu-law encoder (lookup-table, no audioop needed) ─────────────────────
_MULAW_BIAS = 33
_MULAW_CLIP = 32635


def _encode_mulaw_sample(sample: int) -> int:
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    sample = min(sample + _MULAW_BIAS, _MULAW_CLIP)
    mask = 0x4000
    for exponent in range(7, -1, -1):
        if sample & mask:
            break
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


_MULAW_TABLE = bytes(_encode_mulaw_sample(s) for s in range(32768))
_MULAW_TABLE_NEG = bytes(_encode_mulaw_sample(-s) for s in range(32769))


def _pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data)
    return bytes(
        _MULAW_TABLE[s] if s >= 0 else _MULAW_TABLE_NEG[-s]
        for s in samples
    )


def _downsample_linear(samples: list[int], src_rate: int, dst_rate: int) -> list[int]:
    """Simple linear-interpolation downsampler."""
    if src_rate == dst_rate:
        return samples
    ratio = src_rate / dst_rate
    dst_len = int(len(samples) / ratio)
    out = []
    for i in range(dst_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx
        s0 = samples[idx]
        s1 = samples[idx + 1] if idx + 1 < len(samples) else s0
        out.append(int(s0 + frac * (s1 - s0)))
    return out


def _resample_samples(samples: list[int], src_rate: int, dst_rate: int) -> list[int]:
    """Resample using scipy.signal.resample_poly when available, fallback to linear.

    This function uses a lazy, cached import so the optional scipy dependency
    doesn't need to be present at module import time.
    """
    if src_rate == dst_rate:
        return samples
    if _scipy_available():
        try:
            arr = _np.asarray(samples, dtype=_np.int16)
            g = math.gcd(src_rate, dst_rate)
            up = dst_rate // g
            down = src_rate // g
            res = _resample_poly(arr.astype(_np.float64), up, down)
            res = _np.round(res).astype(_np.int16)
            log.debug("[TTS] Resampled with scipy.signal.resample_poly")
            return res.tolist()
        except Exception:
            log.debug("[TTS] scipy resample failed, falling back to linear")
            return _downsample_linear(samples, src_rate, dst_rate)
    else:
        log.debug("[TTS] scipy not available, using linear downsample")
        return _downsample_linear(samples, src_rate, dst_rate)


def _scipy_available() -> bool:
    """Lazy import and cache scipy/numpy resampling helpers."""
    global _SCIPY_AVAILABLE, _np, _resample_poly
    if _SCIPY_AVAILABLE is not None:
        return _SCIPY_AVAILABLE
    try:
        import numpy as _np  # type: ignore
        from scipy.signal import resample_poly as _resample_poly  # type: ignore
        _SCIPY_AVAILABLE = True
    except Exception:
        _SCIPY_AVAILABLE = False
        _np = None
        _resample_poly = None
    return _SCIPY_AVAILABLE


def _pcm16_bytes_to_mulaw_8k(
    pcm_bytes: bytes,
    src_rate: int,
    remainder: bytes = b"",
    session: CallSession | None = None,
) -> tuple[bytes, bytes]:
    """Down-convert signed PCM16 LE samples from ``src_rate`` to mu-law 8 kHz mono.

    Returns ``(mulaw_bytes_for_complete_frames, leftover_pcm_bytes_for_next_call)``.
    Uses ``scipy.signal.resample_poly`` with a Kaiser-window FIR (beta=8.0,
    ~60 dB stopband) for anti-aliased down-conversion when scipy is available;
    otherwise degrades to nearest-neighbour decimation.

    ``session`` is optional and only used to attribute the AUDIO-007
    resample-path counter (``rime_resample_scipy_segments`` vs
    ``rime_resample_fallback_segments``) to the right call. External
    callers (tts_preview, tts_dispatcher) can omit it — the metric
    is just skipped, never wrong.
    """
    try:
        import numpy as np
        from scipy.signal import resample_poly
        HAVE_SCIPY = True
    except ImportError:
        HAVE_SCIPY = False

    data = remainder + pcm_bytes
    n_samples = len(data) // 2
    if n_samples == 0:
        log.debug(f"[RIME_TTS] No usable audio samples. Data len: {len(data)}")
        return b"", data

    g = math.gcd(src_rate, TWILIO_SAMPLE_RATE)
    up = TWILIO_SAMPLE_RATE // g
    down = src_rate // g
    FRAME_SAMPLES = 160  # 20 ms @ 8 kHz

    if src_rate == TWILIO_SAMPLE_RATE:
        # Already at 8 kHz — just trim to frame boundary and encode.
        usable_out = (n_samples // FRAME_SAMPLES) * FRAME_SAMPLES
        if usable_out == 0:
            return b"", data
        mulaw = _pcm16_to_mulaw(data[: usable_out * 2])
        log.debug(f"[RIME_TTS] PCM->mulaw (8k passthrough): src={n_samples} out={usable_out} mulaw_bytes={len(mulaw)}")
        return mulaw, data[usable_out * 2:]

    # Need at least `down` input samples to produce one output sample.
    n_aligned = (n_samples // down) * down
    if n_aligned < down:
        return b"", data
    expected_out = (n_aligned * up) // down  # exact integer output count

    if HAVE_SCIPY:
        samples = np.frombuffer(data[: n_aligned * 2], dtype=np.int16).astype(np.float64)
        # ponytail: pad one edge sample each side so the kaiser FIR boundaries
        # don't smear the first/last output samples; the extra output sample(s)
        # are trimmed from the front below.
        padded = np.pad(samples, (1, 1), mode="edge")
        res = resample_poly(padded, up=up, down=down, window=("kaiser", 8.0))
        trim_front = max(0, len(res) - expected_out)
        res = res[trim_front: trim_front + expected_out]
        out_int = np.clip(np.round(res), -32768, 32767).astype(np.int16)
        pcm_8k = out_int.tobytes()
        log.debug(f"[RIME_TTS] Resampled {src_rate}->{TWILIO_SAMPLE_RATE} via kaiser FIR: src={n_aligned} out={expected_out}")
        # ponytail: AUDIO-007 — record which resample path served this
        # segment. The audit notes the fallback degrades quality silently
        # when scipy is missing. Per-call summary exposes the counter
        # so a missing-scipy image is loud.
        try:
            _m = getattr(session, "metrics", None) if session is not None else None
            if _m is not None:
                _m.incr("rime_resample_scipy_segments")
        except Exception:
            pass
    else:
        # ponytail: degrades to decimation if scipy missing; install scipy==1.11+ in requirements.txt
        n_in = struct.unpack(f"<{n_aligned}h", data[: n_aligned * 2])
        pcm_8k = struct.pack(f"<{expected_out}h", *n_in[::down])
        log.warning(f"[RIME_TTS] scipy unavailable — decimation fallback {src_rate}->{TWILIO_SAMPLE_RATE}; audio quality degraded")
        # ponytail: AUDIO-007 — same counter, fallback path. With both
        # counters on, ops can see rime_resample_fallback_segments > 0
        # at a glance and start the scipy install.
        try:
            _m = getattr(session, "metrics", None) if session is not None else None
            if _m is not None:
                _m.incr("rime_resample_fallback_segments")
        except Exception:
            pass

    usable_out = (expected_out // FRAME_SAMPLES) * FRAME_SAMPLES
    if usable_out == 0:
        return b"", data

    mulaw = _pcm16_to_mulaw(pcm_8k[: usable_out * 2])
    # Source samples consumed for the usable output, rounded down to a multiple
    # of `down` so the next call's data length is also a clean multiple.
    src_samples_used = (usable_out * down) // up
    src_samples_used = (src_samples_used // down) * down
    log.debug(f"[RIME_TTS] Converted PCM->mulaw: src_in={n_samples} used_src={src_samples_used} dest_out={usable_out} mulaw_bytes={len(mulaw)}")
    return mulaw, data[src_samples_used * 2:]


async def stream_tts_segment(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
    seg_idx: int = 0,
) -> tuple[float | None, float]:
    """Stream TTS audio from Rime via WebSocket, emitting mulaw chunks as they arrive."""
    # ponytail: AudioFrameProcessor is the single owner of frame buffering.
    # One per stream; emit_silence_tail=False drops the partial trailing
    # frame at segment end (boundary click fix).
    frame_proc = AudioFrameProcessor(emit_silence_tail=False)
    # ponytail: stamp TTS_FIRST_BYTE on the first audio byte from this adapter.
    def _mark_first_byte() -> None:
        session._stage_timer = session._stage_timer or StageTimer(
            call_id=session.session_key,
            turn_id=0,
            generation=session.active_generation,
        )
        if Stages.TTS_FIRST_BYTE not in session._stage_timer._stages:
            session._stage_timer.mark(Stages.TTS_FIRST_BYTE)
    user_id = getattr(session, "user_id", None)
    creds = resolve_for_session(session, "tts", "rime")
    api_key = creds.get("api_key")
    if not api_key:
        raise RuntimeError("Rime no configurado. Sube tu key en Settings → API o en el campo inline de ModalAgents.")

    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    emitted_audio = False

    tts_language = (
        session.preferred_language
        if session.preferred_language
        else infer_supported_language_from_text(text, fallback="en")
    )
    # ponytail: M9 from the call-flow audit. Per-agent voice
    # (session.voice_id) wins over the per-user speaker_en/speaker_es
    # which wins over the system default. The agent's voice selection
    # was previously ignored by Rime.
    lang_norm = normalize_supported_language(tts_language)
    session_voice = getattr(session, "voice_id", None)
    if session_voice:
        speaker = session_voice
    elif lang_norm == "en" and creds.get("speaker_en"):
        speaker = creds["speaker_en"]
    elif lang_norm == "es" and creds.get("speaker_es"):
        speaker = creds["speaker_es"]
    else:
        speaker = get_tts_model(tts_language, provider="rime")
    model_id = (
        getattr(session, "tts_model", None)
        or creds.get("model_id")
        or RIME_TTS_MODEL_ID
    )
    lang_code = "eng" if lang_norm == "en" else "spa"

    # Request 8 kHz directly so we avoid downsampling most of the time.
    sample_rate = RIME_TTS_SAMPLE_RATE

    log.debug(
        "[TTS] Rime WS TTS request: speaker=%s model=%s lang=%s rate=%d text_len=%d text=%.40r",
        speaker, model_id, lang_code, sample_rate, len(text), text[:40]
    )

    # Rime WS3 requires ALL config as query params; message body is text-only.
    from urllib.parse import urlencode
    qs = urlencode({
        "speaker": speaker,
        "modelId": model_id,
        "lang": lang_code,
        "audioFormat": "pcm",
        "samplingRate": str(sample_rate),
    })
    ws_url = f"{RIME_WS_URL}?{qs}"

    # Sanitize text to avoid problematic characters confusing the TTS engine
    try:
        safe_text = sanitize_tts_text(text)
    except Exception:
        safe_text = text
    if safe_text != text:
        log.info("[TTS] Sanitized text for Rime request: %.120r -> %.120r", text[:120], safe_text[:120])
    ws_message = json.dumps({"text": safe_text})

    # Registro esencial: qué dirá el TTS
    try:
        log.warning(
            "[TTS] Texto a decir (session=%s gen=%s): %.512r",
            getattr(session, "session_key", "?"),
            generation,
            safe_text,
        )
    except Exception:
        pass

    extra_headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # --- Guardado de audio para análisis ---
    # Can be enabled via env var RIME_SAVE_AUDIO=1 for diagnostics.
    save_audio = os.getenv("RIME_SAVE_AUDIO", "false").strip().lower() in {"1", "true", "yes", "on"}
    audio_accum = bytearray()

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=extra_headers,
            close_timeout=5,
            open_timeout=10,
        ) as ws:
            await ws.send(ws_message)
            log.debug("[TTS] Rime WS message sent, waiting for audio... text=%.40r", text[:40])

            pcm_remainder = b""  # carry odd trailing byte across chunks

            while True:
                per_recv_timeout = TTS_TTFB_TIMEOUT_SEC if not emitted_audio else TTS_IDLE_TIMEOUT_SEC
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=per_recv_timeout)
                except asyncio.CancelledError:
                    # Propagate cancellation (e.g. generation change / overall segment timeout)
                    raise
                except (asyncio.TimeoutError, TimeoutError):
                    if not emitted_audio:
                        # No audio ever arrived within TTFB timeout.
                        raise
                    # If audio already started and Rime goes idle, treat as end-of-stream.
                    log.warning(
                        "Rime WS idle timeout after audio started: session=%s gen=%s",
                        session.session_key,
                        generation,
                    )
                    break

                # Binary frame = raw audio data (shouldn't happen on /ws3, but handle it)
                if isinstance(raw_msg, bytes):
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started_at) * 1000
                        log.warning("[TTS] Rime WS TTFB (binary) ms=%.1f session=%s gen=%s", ttfb_ms, getattr(session, 'session_key', '?'), generation)
                    mulaw_bytes, pcm_remainder = _pcm16_bytes_to_mulaw_8k(
                        raw_msg, sample_rate, pcm_remainder, session=session,
                    )
                    if mulaw_bytes:
                        _mark_first_byte()
                    if save_audio:
                        audio_accum.extend(mulaw_bytes)
                    for frame in frame_proc.feed(mulaw_bytes):
                        log.debug("[TTS] Emitting audio frame: session=%s gen=%s bytes=%d", getattr(session, 'session_key', '?'), generation, len(frame))
                        emit_item({"type": "audio", "generation": generation, "data": frame, "source": "tts"})
                        emitted_audio = True
                    continue

                # Text frame — JSON
                msg = json.loads(raw_msg)
                msg_type = msg.get("type", "")

                if msg_type == "error" or "error" in msg:
                    log.error("Rime WS TTS error: %s", msg.get("error", msg))
                    emit_item({
                        "type": "error",
                        "generation": generation,
                        "message": f"Rime WS error: {msg.get('error', msg)}",
                    })
                    break

                if msg_type == "done":
                    log.info("Rime WS TTS complete (done frame)")
                    break

                if msg_type == "timestamps":
                    # Metadata frame — not end-of-stream; continue reading audio.
                    continue

                if msg_type == "chunk":
                    audio_b64 = msg.get("data", "")
                    if not audio_b64:
                        continue
                    pcm_bytes = base64.b64decode(audio_b64)

                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started_at) * 1000
                        log.warning("[TTS] Rime WS TTFB ms=%.1f session=%s gen=%s", ttfb_ms, getattr(session, 'session_key', '?'), generation)

                    mulaw_bytes, pcm_remainder = _pcm16_bytes_to_mulaw_8k(
                        pcm_bytes, sample_rate, pcm_remainder, session=session,
                    )
                    if mulaw_bytes:
                        _mark_first_byte()
                    if save_audio:
                        audio_accum.extend(mulaw_bytes)
                    for frame in frame_proc.feed(mulaw_bytes):
                        log.debug("[TTS] Emitting audio frame: session=%s gen=%s bytes=%d", getattr(session, 'session_key', '?'), generation, len(frame))
                        emit_item({"type": "audio", "generation": generation, "data": frame, "source": "tts"})
                        emitted_audio = True
                    continue

                # Unknown frame type — log and skip
                log.warning("Rime WS unknown frame type=%s keys=%s", msg_type, list(msg.keys()))

    except (asyncio.TimeoutError, TimeoutError):
        log.warning(
            "Rime WS recv timeout: session=%s gen=%s emitted_audio=%s",
            session.session_key,
            generation,
            emitted_audio,
        )
        emit_item({
            "type": "error",
            "generation": generation,
            "message": "Rime WS timeout while waiting for audio",
        })

    except websockets.exceptions.InvalidStatus as exc:
        body = ""
        if hasattr(exc, "response") and exc.response:
            try:
                body = exc.response.body.decode("utf-8", errors="replace") if exc.response.body else ""
            except Exception:
                pass
        log.error(
            "Rime WS handshake rejected HTTP %s — body: %s",
            exc.response.status_code if hasattr(exc, "response") and exc.response else "?",
            body,
        )
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"Rime WS handshake error: {exc}",
        })

    except websockets.exceptions.ConnectionClosed as exc:
        log.error("Rime WS connection closed unexpectedly: %s", exc)
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"Rime WS closed: {exc}",
        })

    except asyncio.CancelledError:
        # Do not emit error items on cancellation; caller requested stop.
        raise

    except Exception as exc:
        log.exception("Rime WS TTS error in %s", session.session_key)
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"Rime WS error: {exc}",
        })

    finally:
        # Guardar el audio acumulado si corresponde
        if save_audio and audio_accum:
            try:
                # ponytail: sanitize session_key before it lands in a
                # filesystem path. Closes the "file inclusion via
                # reading file" scanner finding.
                from STT_server.utils.safe_path import UnsafePathError, sanitize_id
                safe_key = sanitize_id(
                    str(getattr(session, "session_key", "unknown")),
                    field="session_key",
                )
                fname = f"rime_tts_{safe_key}_{generation}.mulaw"
                with open(fname, "wb") as f:
                    f.write(audio_accum)
                    log.debug(f"[TTS] Audio guardado en {fname} ({len(audio_accum)} bytes)")
            except (UnsafePathError, OSError) as exc:
                log.warning(f"[TTS] Skipping rime audio dump: {exc}")
            except Exception as e:
                log.error(f"[TTS] Error guardando audio: {e}")

        # Siempre emitir segment_end, sin importar qué excepción ocurrió
        # ponytail: drop the partial trailing frame at segment end
        # (emit_silence_tail=False).
        frame_proc.flush()
        emit_item({"type": "segment_end", "generation": generation, "has_audio": emitted_audio})

    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms