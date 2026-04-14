import asyncio
import json
import logging
import time
import os

import websockets

from STT_server.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_TTS_MODEL_ID,
    ELEVENLABS_TTS_VOICE_ID,
    TTS_IDLE_TIMEOUT_SEC,
    TTS_TTFB_TIMEOUT_SEC,
)
from STT_server.domain.language import (
    infer_supported_language_from_text,
    sanitize_tts_text,
)
from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")

ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech"


async def stream_tts_segment(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
) -> tuple[float | None, float]:
    """Stream TTS audio from ElevenLabs via WebSocket, emitting mulaw chunks as they arrive.

    Uses the ElevenLabs WebSocket streaming API with ulaw_8000 output format,
    which produces mu-law 8 kHz audio directly compatible with Twilio -- no
    resampling or mu-law encoding needed.
    """
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY no configurada")

    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    emitted_audio = False

    # Sanitize text to avoid problematic characters confusing the TTS engine
    try:
        safe_text = sanitize_tts_text(text)
    except Exception:
        safe_text = text
    if safe_text != text:
        log.info("[TTS] Sanitized text for ElevenLabs request: %.120r -> %.120r", text[:120], safe_text[:120])

    # Registro esencial: que dira el TTS
    try:
        log.warning(
            "[TTS] Texto a decir (session=%s gen=%s): %.512r",
            getattr(session, "session_key", "?"),
            generation,
            safe_text,
        )
    except Exception:
        pass

    # Build WebSocket URL with voice_id, output format, and model
    ws_url = (
        f"{ELEVENLABS_WS_URL}/{ELEVENLABS_TTS_VOICE_ID}/stream-input"
        f"?output_format=ulaw_8000&model_id={ELEVENLABS_TTS_MODEL_ID}"
    )

    extra_headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
    }

    # --- Guardado de audio para analisis ---
    save_audio = os.getenv("ELEVENLABS_SAVE_AUDIO", "false").strip().lower() in {"1", "true", "yes", "on"}
    audio_accum = bytearray()

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=extra_headers,
            close_timeout=5,
            open_timeout=10,
        ) as ws:
            # Send the text as a single message with flush=True
            message = json.dumps({
                "text": safe_text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
                "flush": True,
            })
            await ws.send(message)
            log.debug("[TTS] ElevenLabs WS message sent, waiting for audio... text=%.40r", text[:40])

            while True:
                per_recv_timeout = TTS_TTFB_TIMEOUT_SEC if not emitted_audio else TTS_IDLE_TIMEOUT_SEC
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=per_recv_timeout)
                except asyncio.CancelledError:
                    raise
                except (asyncio.TimeoutError, TimeoutError):
                    if not emitted_audio:
                        raise
                    log.warning(
                        "ElevenLabs WS idle timeout after audio started: session=%s gen=%s",
                        session.session_key,
                        generation,
                    )
                    break

                # Binary frame = raw audio data (mu-law 8kHz from ElevenLabs)
                if isinstance(raw_msg, bytes):
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started_at) * 1000
                        log.warning("[TTS] ElevenLabs WS TTFB ms=%.1f session=%s gen=%s", ttfb_ms, getattr(session, 'session_key', '?'), generation)

                    mulaw_bytes = raw_msg  # Already mu-law 8kHz -- no conversion needed
                    if save_audio:
                        audio_accum.extend(mulaw_bytes)
                    for i in range(0, len(mulaw_bytes), 4096):
                        chunk = mulaw_bytes[i : i + 4096]
                        log.debug("[TTS] Emitting audio chunk: session=%s gen=%s bytes=%d", getattr(session, 'session_key', '?'), generation, len(chunk))
                        emit_item({"type": "audio", "generation": generation, "data": chunk, "source": "tts"})
                        emitted_audio = True
                    continue

                # Text frame -- JSON (could be error, done, or alignment info)
                try:
                    msg = json.loads(raw_msg)
                except Exception:
                    log.warning("[TTS] ElevenLabs unknown text frame: %s", raw_msg[:200])
                    continue

                # Check for errors
                if msg.get("status") == "error" or "error" in msg:
                    error_msg = msg.get("error", msg.get("message", str(msg)))
                    log.error("ElevenLabs WS TTS error: %s", error_msg)
                    emit_item({
                        "type": "error",
                        "generation": generation,
                        "message": f"ElevenLabs WS error: {error_msg}",
                    })
                    break

                # Check for completion
                if msg.get("status") == "done":
                    log.info("ElevenLabs WS TTS complete (done frame)")
                    break

                # Alignment/timing info -- not end-of-stream; continue reading audio.
                if msg.get("type") == "alignment":
                    continue

                # Unknown frame type -- log and skip
                log.warning("ElevenLabs WS unknown frame: %s", str(msg)[:200])

    except (asyncio.TimeoutError, TimeoutError):
        log.warning(
            "ElevenLabs WS recv timeout: session=%s gen=%s emitted_audio=%s",
            session.session_key,
            generation,
            emitted_audio,
        )
        emit_item({
            "type": "error",
            "generation": generation,
            "message": "ElevenLabs WS timeout while waiting for audio",
        })

    except websockets.exceptions.InvalidStatus as exc:
        body = ""
        if hasattr(exc, "response") and exc.response:
            try:
                body = exc.response.body.decode("utf-8", errors="replace") if exc.response.body else ""
            except Exception:
                pass
        log.error(
            "ElevenLabs WS handshake rejected HTTP %s -- body: %s",
            exc.response.status_code if hasattr(exc, "response") and exc.response else "?",
            body,
        )
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"ElevenLabs WS handshake error: {exc}",
        })

    except websockets.exceptions.ConnectionClosed as exc:
        log.error("ElevenLabs WS connection closed unexpectedly: %s", exc)
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"ElevenLabs WS closed: {exc}",
        })

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log.exception("ElevenLabs WS TTS error in %s", session.session_key)
        emit_item({
            "type": "error",
            "generation": generation,
            "message": f"ElevenLabs WS error: {exc}",
        })

    finally:
        if save_audio and audio_accum:
            try:
                fname = f"elevenlabs_tts_{getattr(session, 'session_key', 'unknown')}_{generation}.mulaw"
                with open(fname, "wb") as f:
                    f.write(audio_accum)
                    log.debug(f"[TTS] Audio guardado en {fname} ({len(audio_accum)} bytes)")
            except Exception as e:
                log.error(f"[TTS] Error guardando audio: {e}")

        emit_item({"type": "segment_end", "generation": generation, "has_audio": emitted_audio})

    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms