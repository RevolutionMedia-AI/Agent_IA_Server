import asyncio
import base64
import json
import logging
import struct
import time

import websockets

from STT_server.config import (
    RIME_API_KEY,
    RIME_TTS_MODEL_ID,
    RIME_TTS_SAMPLE_RATE,
    TTS_IDLE_TIMEOUT_SEC,
    TTS_TTFB_TIMEOUT_SEC,
)
from STT_server.domain.language import get_tts_model, infer_supported_language_from_text, normalize_supported_language
from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")

RIME_WS_URL = "wss://users-ws.rime.ai/ws3"
TWILIO_SAMPLE_RATE = 8000

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


def _pcm16_bytes_to_mulaw_8k(pcm_bytes: bytes, src_rate: int, remainder: bytes = b"") -> tuple[bytes, bytes]:
    """Convert raw PCM16-LE bytes at *src_rate* to mu-law 8 kHz.

    Returns ``(mulaw_bytes, leftover)`` where *leftover* is a 0-or-1 byte
    remainder that should be prepended to the next chunk so sample
    boundaries stay aligned across WebSocket messages.
    """
    data = remainder + pcm_bytes
    CHUNK_SIZE = 160  # 20 ms de audio a 8000 Hz, 16 bits, 1 canal
    usable = len(data) - (len(data) % CHUNK_SIZE)
    leftover = data[usable:]  # bytes restantes para el siguiente chunk
    if usable == 0:
        log.debug(f"[RIME_TTS] No usable audio chunk. Data len: {len(data)}")
        return b"", leftover
    n_samples = usable // 2
    try:
        samples = list(struct.unpack(f"<{n_samples}h", data[:usable]))
    except Exception as e:
        log.error(f"[RIME_TTS] Error unpacking PCM data: {e}, usable={usable}, data_len={len(data)}")
        return b"", leftover
    if src_rate != TWILIO_SAMPLE_RATE:
        log.debug(f"[RIME_TTS] Downsampling from {src_rate}Hz to {TWILIO_SAMPLE_RATE}Hz, samples={len(samples)}")
        samples = _downsample_linear(samples, src_rate, TWILIO_SAMPLE_RATE)
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    mulaw = _pcm16_to_mulaw(pcm)
    log.debug(f"[RIME_TTS] Converted PCM to mulaw: input_samples={len(samples)}, mulaw_bytes={len(mulaw)}")
    return mulaw, leftover


async def stream_tts_segment(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
) -> tuple[float | None, float]:
    """Stream TTS audio from Rime via WebSocket, emitting mulaw chunks as they arrive."""
    if not RIME_API_KEY:
        raise RuntimeError("RIME_API_KEY no configurada")

    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    emitted_audio = False

    tts_language = (
        session.preferred_language
        if session.preferred_language
        else infer_supported_language_from_text(text, fallback="en")
    )
    speaker = get_tts_model(tts_language)
    lang_code = "eng" if normalize_supported_language(tts_language) == "en" else "spa"

    # Request 8 kHz directly so we avoid downsampling most of the time.
    sample_rate = RIME_TTS_SAMPLE_RATE

    log.info(
        "[TTS] Rime WS TTS request: speaker=%s model=%s lang=%s rate=%d text_len=%d text=%.40r",
        speaker, RIME_TTS_MODEL_ID, lang_code, sample_rate, len(text), text[:40]
    )

    # Rime WS3 requires ALL config as query params; message body is text-only.
    from urllib.parse import urlencode
    qs = urlencode({
        "speaker": speaker,
        "modelId": RIME_TTS_MODEL_ID,
        "lang": lang_code,
        "audioFormat": "pcm",
        "samplingRate": str(sample_rate),
    })
    ws_url = f"{RIME_WS_URL}?{qs}"

    ws_message = json.dumps({"text": text})

    extra_headers = {
        "Authorization": f"Bearer {RIME_API_KEY}",
    }

    # --- Guardado de audio para análisis ---
    save_audio = True  # Cambia a False para desactivar
    audio_accum = bytearray()

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=extra_headers,
            close_timeout=5,
            open_timeout=10,
        ) as ws:
            await ws.send(ws_message)
            log.info("[TTS] Rime WS message sent, waiting for audio... text=%.40r", text[:40])

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
                        log.info("Rime WS TTS TTFB (binary): %.1f ms", ttfb_ms)
                    mulaw_bytes, pcm_remainder = _pcm16_bytes_to_mulaw_8k(raw_msg, sample_rate, pcm_remainder)
                    if save_audio:
                        audio_accum.extend(mulaw_bytes)
                    for i in range(0, len(mulaw_bytes), 4096):
                        chunk = mulaw_bytes[i : i + 4096]
                        log.debug("[TTS] Emitting audio chunk: session=%s gen=%s bytes=%d", getattr(session, 'session_key', '?'), generation, len(chunk))
                        emit_item({"type": "audio", "generation": generation, "data": chunk})
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
                        log.info("Rime WS TTS TTFB: %.1f ms", ttfb_ms)

                    mulaw_bytes, pcm_remainder = _pcm16_bytes_to_mulaw_8k(pcm_bytes, sample_rate, pcm_remainder)
                    if save_audio:
                        audio_accum.extend(mulaw_bytes)
                    for i in range(0, len(mulaw_bytes), 4096):
                        chunk = mulaw_bytes[i : i + 4096]
                        log.debug("[TTS] Emitting audio chunk: session=%s gen=%s bytes=%d", getattr(session, 'session_key', '?'), generation, len(chunk))
                        emit_item({"type": "audio", "generation": generation, "data": chunk})
                        emitted_audio = True
                    continue

                # Unknown frame type — log and skip
                log.warning("Rime WS unknown frame type=%s keys=%s", msg_type, list(msg.keys()))

    except (asyncio.TimeoutError, TimeoutError):
        # Treat as expected failure mode; retry logic lives upstream.
        log.warning(
            "Rime WS recv timeout: session=%s gen=%s emitted_audio=%s",
            session.session_key,
            generation,
            emitted_audio,
        )
    finally:
        # Guardar el audio acumulado si corresponde
        if save_audio and audio_accum:
            try:
                fname = f"rime_tts_{getattr(session, 'session_key', 'unknown')}_{generation}.mulaw"
                with open(fname, "wb") as f:
                    f.write(audio_accum)
                log.info(f"[TTS] Audio guardado en {fname} ({len(audio_accum)} bytes)")
                # Enviar el archivo por correo
                try:
                    from STT_server.utils.send_audio_email import send_audio_email
                    send_audio_email(fname)
                    log.info(f"[TTS] Audio enviado por correo a kevin.escalante@revolutionmedia.ai")
                except Exception as e:
                    log.error(f"[TTS] Error enviando audio por correo: {e}")
            except Exception as e:
                log.error(f"[TTS] Error guardando audio: {e}")
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
        log.error("Rime WS handshake rejected HTTP %s — body: %s", exc.response.status_code if hasattr(exc, "response") and exc.response else "?", body)
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
        emit_item({"type": "segment_end", "generation": generation, "has_audio": emitted_audio})

    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms
