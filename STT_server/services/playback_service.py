import asyncio
import contextlib
import logging
import time
import json

from fastapi import WebSocket

from STT_server.adapters.rime_tts import stream_tts_segment
from STT_server.adapters.twilio_media import send_twilio_clear, send_twilio_mark, send_twilio_media
from STT_server.config import (
    INITIAL_GREETING_ENABLED,
    INITIAL_GREETING_TEXT,
    LOG_TWILIO_PLAYBACK,
    OPENAI_API_KEY,
    RIME_API_KEY,
    TTS_MAX_RETRIES,
    TTS_RETRY_BACKOFF_MS,
    TTS_TIMEOUT_SEC,
    TWILIO_OUTBOUND_CHUNK_BYTES,
    TWILIO_OUTBOUND_PACING_MS,
    SAVE_TWILIO_FRAMES,
    TWIML_INITIAL_GREETING_ENABLED,
)
from STT_server.domain.language import split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.common import drain_queue_nowait, enqueue_nowait_with_drop, enqueue_with_drop
import os
# RNNoise removed: playback sends mu-law frames directly to Twilio.


log = logging.getLogger("stt_server")


async def run_tts_with_retries(session: CallSession, text: str, generation: int) -> tuple[float | None, float]:
    attempts = max(0, TTS_MAX_RETRIES) + 1
    last_timeout = False
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(
                stream_tts_segment(session, text, generation, lambda item: emit_playback_item(session, item)),
                timeout=TTS_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            last_timeout = True
            log.warning(
                "TTS timeout en saludo %s (attempt %s/%s, text_len=%s)",
                session.session_key,
                attempt,
                attempts,
                len(text),
            )
            if attempt < attempts:
                await asyncio.sleep(max(0, TTS_RETRY_BACKOFF_MS) / 1000.0)
        except Exception:
            log.exception("TTS error en saludo %s (attempt %s/%s)", session.session_key, attempt, attempts)
            raise

    if last_timeout:
        raise asyncio.TimeoutError()
    raise RuntimeError("TTS greeting failed without timeout detail")


def emit_playback_item(session: CallSession, item: dict) -> bool:
    log.debug("[PLAYBACK] Enqueue playback item: session=%s type=%s gen=%s bytes=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'), len(item.get('data', b'')) if 'data' in item else '-')
    ok = enqueue_nowait_with_drop(session.playback_queue, item, "playback_queue")
    if not ok:
        log.warning("[PLAYBACK] Failed to enqueue playback item (queue full): session=%s type=%s gen=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'))
    return ok


async def enqueue_playback_clear(session: CallSession) -> None:
    await enqueue_with_drop(
        session.playback_queue,
        {"type": "clear", "generation": session.active_generation},
        "playback_queue",
    )


async def interrupt_current_turn(session: CallSession) -> None:
    session.active_generation += 1

    # Stop any pending response generation and prefetch.
    if session.reply_task and not session.reply_task.done():
        session.reply_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session.reply_task
        session.reply_task = None

    if session.prefetched_reply_task and not session.prefetched_reply_task.done():
        session.prefetched_reply_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session.prefetched_reply_task
        session.prefetched_reply_task = None

    session.reply_source_text = ""
    session.partial_reply_task = None
    session.prefetched_reply_source_text = ""
    session.prefetched_reply_text = ""

    session.assistant_speaking = False
    session.assistant_started_at = None
    session.pending_marks.clear()
    drain_queue_nowait(session.playback_queue)
    await enqueue_playback_clear(session)
    session.generation_changed.set()


async def play_initial_greeting(session: CallSession) -> None:
    # Initial greeting functionality removed — keep function as a no-op to
    # preserve callers that may still schedule it.
    log.debug("[PLAYBACK] play_initial_greeting skipped (initial greeting disabled)")
    return


async def playback_loop(ws: WebSocket, session: CallSession) -> None:
    try:
        while True:
            item = await session.playback_queue.get()
            item_type = item.get("type")
            generation = item.get("generation")

            if item_type == "clear":
                if session.stream_sid:
                    await send_twilio_clear(ws, session.stream_sid)
                session.pending_marks.clear()
                session.assistant_speaking = False
                session.assistant_started_at = None
                continue

            if generation != session.active_generation:
                continue

            if item_type == "audio":
                if not session.stream_sid:
                    # Wait briefly for Twilio to send the stream SID.
                    # Increased from 50 (2.5s) to 100 (5s) to avoid race conditions.
                    for _ in range(100):  # up to 5.0 s
                        await asyncio.sleep(0.05)
                        if session.stream_sid:
                            break
                if not session.stream_sid:
                    log.warning("[PLAYBACK] No stream_sid for audio item, skipping")
                    continue

                if not session.assistant_speaking:
                    session.assistant_started_at = time.perf_counter()
                    session.last_activity_at = time.monotonic()
                session.assistant_speaking = True
                chunk = item["data"]
                # Direct pass-through of mu-law frames; no denoising applied.
                sent_frames = 0
                timings = []
                for start in range(0, len(chunk), TWILIO_OUTBOUND_CHUNK_BYTES):
                    frame = chunk[start : start + TWILIO_OUTBOUND_CHUNK_BYTES]
                    log.debug("[PLAYBACK] Sending Twilio frame: session=%s gen=%s frame_bytes=%d", session.session_key, generation, len(frame))
                    if frame:
                        if sent_frames == 0:
                            log.debug("[PLAYBACK] Sending first Twilio frame: session=%s gen=%s bytes=%d", session.session_key, generation, len(frame))
                        # Optionally save frames to disk for diagnostics
                        if SAVE_TWILIO_FRAMES:
                            try:
                                fname = f"twilio_out_{session.session_key}_{generation}.mulaw"
                                with open(fname, "ab") as f:
                                    f.write(frame)
                            except Exception:
                                log.exception("Error escribiendo frame Twilio para %s", session.session_key)

                        # No denoiser: send frame directly to Twilio.

                        send_start = time.perf_counter()
                        await send_twilio_media(ws, session.stream_sid, frame)
                        sent_frames += 1
                        # Pace outgoing frames proportionally to their duration.
                        # A full frame (TWILIO_OUTBOUND_CHUNK_BYTES) represents
                        # TWILIO_OUTBOUND_PACING_MS milliseconds of audio.
                        try:
                            pacing_ms = (len(frame) / TWILIO_OUTBOUND_CHUNK_BYTES) * TWILIO_OUTBOUND_PACING_MS
                        except Exception:
                            pacing_ms = TWILIO_OUTBOUND_PACING_MS
                        elapsed = time.perf_counter() - send_start
                        wait = (pacing_ms / 1000.0) - elapsed if pacing_ms > 0 else 0.0
                        if wait > 0:
                            await asyncio.sleep(wait)

                        # Record timing diagnostic for this frame when saving frames
                        timings.append({"idx": sent_frames - 1, "bytes": len(frame), "send_elapsed": elapsed, "pacing_ms": pacing_ms, "wait_applied_s": max(wait, 0.0)})
                        if SAVE_TWILIO_FRAMES:
                            try:
                                tname = f"twilio_out_{session.session_key}_{generation}.timings.jsonl"
                                with open(tname, "a", encoding="utf-8") as tf:
                                    tf.write(json.dumps(timings[-1]) + "\n")
                            except Exception:
                                log.exception("Error escribiendo timings Twilio para %s", session.session_key)
                if LOG_TWILIO_PLAYBACK and sent_frames:
                    log.debug(
                        "[PLAYBACK] Playback audio %s gen=%s bytes=%s frames=%s",
                        session.session_key,
                        generation,
                        len(chunk),
                        sent_frames,
                    )
                # If we collected timings, compute simple stats and log
                if timings:
                    try:
                        avg_send = sum(t["send_elapsed"] for t in timings) / len(timings)
                        avg_wait = sum(t["wait_applied_s"] for t in timings) / len(timings)
                        log.debug("[PLAYBACK] Timing stats session=%s gen=%s frames=%s avg_send_s=%.5f avg_wait_s=%.5f", session.session_key, generation, len(timings), avg_send, avg_wait)
                    except Exception:
                        log.exception("Error computing timing stats for playback")
                continue

            if item_type == "segment_end":
                    # Forzar unmute del STT al terminar cualquier segmento, con o sin audio
                    session.assistant_speaking = False
                    session.assistant_started_at = None
                    # Enviar mark para rastrear segmento (si tenemos stream_sid)
                    try:
                        session.mark_counter += 1
                        mark_name = f"gen-{generation}-seg-{session.mark_counter}"
                        session.pending_marks.add(mark_name)
                        if session.stream_sid:
                            await send_twilio_mark(ws, session.stream_sid, mark_name)
                            if LOG_TWILIO_PLAYBACK:
                                log.info("Playback mark enviado %s %s", session.session_key, mark_name)
                    except Exception:
                        log.exception("Error enviando mark de playback para %s", session.session_key)
                    continue

            if item_type == "error":
                log.error("Playback error en %s: %s", session.session_key, item.get("message"))
                if not session.pending_marks:
                    session.assistant_speaking = False
                    session.assistant_started_at = None
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Error en playback_loop")