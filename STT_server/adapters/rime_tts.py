import asyncio
import base64
import json
import logging
import struct
import time

import websockets

from STT_server.config import RIME_API_KEY, RIME_TTS_MODEL_ID, RIME_TTS_SAMPLE_RATE
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


def _pcm16_bytes_to_mulaw_8k(pcm_bytes: bytes, src_rate: int) -> bytes:
    """Convert raw PCM16-LE bytes at *src_rate* to mu-law 8 kHz."""
    n_samples = len(pcm_bytes) // 2
    if n_samples == 0:
        return b""
    samples = list(struct.unpack(f"<{n_samples}h", pcm_bytes))
    if src_rate != TWILIO_SAMPLE_RATE:
        samples = _downsample_linear(samples, src_rate, TWILIO_SAMPLE_RATE)
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    return _pcm16_to_mulaw(pcm)


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

    tts_language = (
        session.preferred_language
        if session.preferred_language
        else infer_supported_language_from_text(text, fallback="en")
    )
    speaker = get_tts_model(tts_language)
    lang_code = "eng" if normalize_supported_language(tts_language) == "en" else "spa"

    # Request 8 kHz directly so we avoid downsampling most of the time.
    sample_rate = RIME_TTS_SAMPLE_RATE

    ws_request = {
        "text": text,
        "speaker": speaker,
        "modelId": RIME_TTS_MODEL_ID,
        "lang": lang_code,
        "audioFormat": "pcm",
        "samplingRate": sample_rate,
    }

    log.info(
        "Rime WS TTS request: speaker=%s model=%s lang=%s rate=%d text_len=%d",
        speaker, RIME_TTS_MODEL_ID, lang_code, sample_rate, len(text),
    )

    # Rime WS authenticates via query parameter, not HTTP header.
    ws_url = f"{RIME_WS_URL}?authToken={RIME_API_KEY}"

    try:
        async with websockets.connect(
            ws_url,
            close_timeout=5,
            open_timeout=10,
        ) as ws:
            await ws.send(json.dumps(ws_request))

            async for raw_msg in ws:
                msg = json.loads(raw_msg)

                # Error frame from Rime
                if "error" in msg:
                    log.error("Rime WS TTS error: %s", msg["error"])
                    emit_item({
                        "type": "error",
                        "generation": generation,
                        "message": f"Rime WS error: {msg['error']}",
                    })
                    break

                audio_b64 = msg.get("audio")
                if not audio_b64:
                    # Could be a status/metadata frame — skip.
                    if msg.get("is_final"):
                        break
                    continue

                pcm_bytes = base64.b64decode(audio_b64)

                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - started_at) * 1000
                    log.info("Rime WS TTS TTFB: %.1f ms", ttfb_ms)

                mulaw_bytes = _pcm16_bytes_to_mulaw_8k(pcm_bytes, sample_rate)

                # Emit in Twilio-friendly 4096-byte chunks
                for i in range(0, len(mulaw_bytes), 4096):
                    chunk = mulaw_bytes[i : i + 4096]
                    emit_item({"type": "audio", "generation": generation, "data": chunk})

                if msg.get("is_final"):
                    break

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
    except Exception as exc:
        log.exception("Rime WS TTS error in %s", session.session_key)
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"Rime WS error: {exc}",
        })
    finally:
        emit_item({"type": "segment_end", "generation": generation})

    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms
