import asyncio
import contextlib
import logging
import time

from fastapi import WebSocket

from STT_server.adapters.rime_tts import stream_tts_segment
from STT_server.adapters.twilio_media import send_twilio_clear, send_twilio_mark, send_twilio_media
from STT_server.config import (
    INITIAL_GREETING_ENABLED,
    INITIAL_GREETING_TEXT,
    LOG_TWILIO_PLAYBACK,
    OPENAI_API_KEY,
    RIME_API_KEY,
    TTS_TIMEOUT_SEC,
    TWILIO_OUTBOUND_CHUNK_BYTES,
    TWILIO_OUTBOUND_PACING_MS,
)
from STT_server.domain.language import split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.common import drain_queue_nowait, enqueue_nowait_with_drop, enqueue_with_drop


log = logging.getLogger("stt_server")


def emit_playback_item(session: CallSession, item: dict) -> bool:
    log.debug("[PLAYBACK] Enqueue playback item: session=%s type=%s gen=%s bytes=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'), len(item.get('data', b'')) if 'data' in item else '-')
    return enqueue_nowait_with_drop(session.playback_queue, item, "playback_queue")


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
    if not INITIAL_GREETING_ENABLED or not INITIAL_GREETING_TEXT or not (OPENAI_API_KEY or RIME_API_KEY):
        return

    session.active_generation += 1
    generation = session.active_generation
    session.history.append({"role": "assistant", "content": INITIAL_GREETING_TEXT})

    try:
        for segment in split_tts_segments(INITIAL_GREETING_TEXT):
            if generation != session.active_generation:
                return
            await asyncio.wait_for(
                stream_tts_segment(session, segment, generation, lambda item: emit_playback_item(session, item)),
                timeout=TTS_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        log.warning("TTS timeout en saludo inicial para %s", session.session_key)
    except Exception:
        log.exception("Error en saludo inicial para %s", session.session_key)


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
                    for _ in range(50):  # up to 2.5 s
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
                sent_frames = 0
                for start in range(0, len(chunk), TWILIO_OUTBOUND_CHUNK_BYTES):
                    frame = chunk[start : start + TWILIO_OUTBOUND_CHUNK_BYTES]
                    log.debug("[PLAYBACK] Sending Twilio frame: session=%s gen=%s frame_bytes=%d", session.session_key, generation, len(frame))
                    if frame:
                        await send_twilio_media(ws, session.stream_sid, frame)
                        sent_frames += 1
                        if len(frame) == TWILIO_OUTBOUND_CHUNK_BYTES:
                            await asyncio.sleep(TWILIO_OUTBOUND_PACING_MS / 1000.0)
                if LOG_TWILIO_PLAYBACK and sent_frames:
                    log.debug(
                        "[PLAYBACK] Playback audio %s gen=%s bytes=%s frames=%s",
                        session.session_key,
                        generation,
                        len(chunk),
                        sent_frames,
                    )
                continue

            if item_type == "segment_end":
                if not session.stream_sid:
                    session.assistant_speaking = False
                    session.assistant_started_at = None
                    continue

                session.mark_counter += 1
                mark_name = f"gen-{generation}-seg-{session.mark_counter}"
                session.pending_marks.add(mark_name)
                await send_twilio_mark(ws, session.stream_sid, mark_name)
                if LOG_TWILIO_PLAYBACK:
                    log.info("Playback mark enviado %s %s", session.session_key, mark_name)
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