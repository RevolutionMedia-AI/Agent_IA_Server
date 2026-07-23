import asyncio
import contextlib
import logging
import time
import json

from fastapi import WebSocket

from STT_server.adapters.tts_dispatcher import stream_tts_segment
from STT_server.adapters.twilio_media import send_twilio_clear, send_twilio_mark, send_twilio_media
from STT_server.config import (
    LOG_TWILIO_PLAYBACK,
    STREAM_SID_WAIT_MAX_MS,
    STREAM_SID_WAIT_POLL_MS,
    TWILIO_OUTBOUND_CHUNK_BYTES,
    TWILIO_OUTBOUND_PACING_MS,
    SAVE_TWILIO_FRAMES,
)
from STT_server.domain.language import split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.common import drain_queue_nowait, enqueue_nowait_with_drop, enqueue_with_drop
from STT_server.utils.safe_path import UnsafePathError, sanitize_id
import os
# RNNoise removed: playback sends mu-law frames directly to Twilio.


log = logging.getLogger("stt_server")


def emit_playback_item(session: CallSession, item: dict) -> bool:
    log.debug("[PLAYBACK] Enqueue playback item: session=%s type=%s gen=%s bytes=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'), len(item.get('data', b'')) if 'data' in item else '-')
    ok = enqueue_nowait_with_drop(session.playback_queue, item, "playback_queue")
    if not ok:
        log.warning("[PLAYBACK] Failed to enqueue playback item (queue full): session=%s type=%s gen=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'))
    return ok


def enqueue_playback_clear(session: CallSession) -> None:
    """M2: was async with `await enqueue_with_drop`, but the body is sync
    (enqueue_with_drop just wraps a put_nowait). The await was a no-op
    that confused every caller. Now plain sync."""
    enqueue_with_drop(
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
    # ponytail: H4 from the call-flow audit. The stt_mute_buffer
    # holds ~500ms of user audio captured while the assistant was
    # talking. After barge-in, that audio is the user's voice
    # bleeding into the TTS playback — NOT a fresh user turn.
    # Draining it (the previous behaviour) caused phantom
    # transcripts of the user's own echo. Clear instead.
    session.stt_mute_buffer.clear()
    drain_queue_nowait(session.playback_queue)
    enqueue_playback_clear(session)
    session.generation_changed.set()


async def play_initial_greeting(session: CallSession) -> None:
    """Play the agent's welcome_message (set by media_stream at call start)
    via the TTS pipeline so the caller hears the agent greet them first.
    No-op if welcome_message is empty / not configured.

    ponytail: set assistant_speaking = True BEFORE scheduling the TTS
    so monitor_idle_silence knows the agent is about to speak and
    doesn't count the TTFB window as user silence. The flag gets
    cleared by playback_loop when Twilio confirms playback completion
    via the mark event.
    """
    welcome = getattr(session, 'welcome_message', None)
    if not welcome or not welcome.strip():
        log.debug("[PLAYBACK] play_initial_greeting skipped (no welcome_message)")
        return
    log.info("[PLAYBACK] playing initial greeting (%d chars) for session=%s", len(welcome), session.session_key)
    # Mark the agent as speaking NOW (before the TTS provider responds).
    # Without this the idle monitor could fire during the TTFB window
    # (some TTS providers take 1-2 s to first byte), think the user is
    # silent, and hang up the call.
    session.assistant_speaking = True
    session.assistant_started_at = time.perf_counter()
    session.last_activity_at = time.monotonic()
    # Wait a moment for the audio stream to be established.
    await asyncio.sleep(0.4)
    # Generate the TTS via the session's configured TTS provider. The
    # playback_loop will pick up the queued audio and stream it to Twilio.
    from STT_server.services.turn_manager import run_tts_with_retries
    try:
        await run_tts_with_retries(session, welcome.strip(), session.active_generation)
    except Exception as exc:
        log.warning("[PLAYBACK] initial greeting TTS failed: %s", exc)
        session.assistant_speaking = False
        session.assistant_started_at = None


async def play_error_and_hangup(
    session: CallSession,
    ws: WebSocket,
    message: str = "Lo sentimos, hubo un problema de configuracion. Adios.",
) -> None:
    """Best-effort: speak a short error message via the configured TTS
    provider, then close the WebSocket so Twilio tears the call down.

    Falls back to a plain WebSocket close if no TTS provider is
    configured (the user explicitly asked for no default fallbacks,
    so we can't TTS without one). In either case the call ends
    cleanly — never a silent limbo where Twilio keeps charging.
    """
    log.error("[HANGUP] session %s: %s", session.session_key, message)
    tts_provider = getattr(session, 'tts_provider', None) or ""
    if tts_provider:
        try:
            from STT_server.services.turn_manager import run_tts_with_retries
            # Generation bump so the playback_loop sees this as a new
            # turn (not a continuation of the welcome greeting).
            session.active_generation += 1
            await asyncio.wait_for(
                run_tts_with_retries(session, message, session.active_generation),
                timeout=10.0,
            )
        except Exception as exc:
            log.warning("[HANGUP] error TTS failed for %s: %s", session.session_key, exc)
    else:
        log.error(
            "[HANGUP] session %s: no TTS provider configured, closing WS without error audio",
            session.session_key,
        )
    # Close the WS → Twilio sees the close and tears down the call. The
    # exact same code path cleanup_session uses, but called explicitly
    # so the operator can see this code path in a traceback if it
    # fails.
    try:
        await ws.close()
    except Exception as exc:
        log.warning("[HANGUP] ws.close() failed for %s: %s", session.session_key, exc)


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
                    # L6: was hardcoded 100 iterations * 50 ms = 5 s.
                    # Now driven by STREAM_SID_WAIT_MAX_MS / STREAM_SID_WAIT_POLL_MS.
                    wait_iterations = max(1, STREAM_SID_WAIT_MAX_MS // max(1, STREAM_SID_WAIT_POLL_MS))
                    for _ in range(wait_iterations):
                        await asyncio.sleep(STREAM_SID_WAIT_POLL_MS / 1000.0)
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
                # ponytail: removed per-frame log.debug - one chunk can
                # contain 50+ frames, and at INFO that's a flood. The
                # one-line summary at the bottom covers the same info.
                # ponytail: validate Twilio's 20ms alignment. Inworld
                # sends mu-law at 8000 Hz where each byte is exactly
                # 0.125 ms - so 160 bytes = 20 ms, which is the size
                # Twilio expects for one media packet. If Inworld
                # ever returns a stream at a different rate (or a
                # chunk that doesn't end on a frame boundary), the
                # leftover gets emitted as a short final packet with
                # a "pacing_ms" that doesn't match Twilio's clock -
                # the user perceives that as a tiny click/pop at the
                # boundary. Log a one-line summary if any non-aligned
                # chunk comes in so we can see it once per turn
                # instead of once per frame.
                sent_frames = 0
                for start in range(0, len(chunk), TWILIO_OUTBOUND_CHUNK_BYTES):
                    frame = chunk[start : start + TWILIO_OUTBOUND_CHUNK_BYTES]
                    if frame:
                        if SAVE_TWILIO_FRAMES:
                            try:
                                # ponytail: sanitize_id kills any path-
                                # traversal chars in session_key before
                                # it goes into a filesystem path. The
                                # server-issued values (id(ws), Twilio
                                # call_sid) already match, but defending
                                # at the write site is the cheapest
                                # place to neutralise the file-include
                                # scanner finding.
                                safe_key = sanitize_id(
                                    str(getattr(session, "session_key", "unknown")),
                                    field="session_key",
                                )
                                fname = f"twilio_out_{safe_key}_{generation}.mulaw"
                                with open(fname, "ab") as f:
                                    f.write(frame)
                            except (UnsafePathError, OSError) as exc:
                                log.warning(
                                    "Skipping twilio_out frame for %s: %s",
                                    session.session_key, exc,
                                )
                            except Exception:
                                log.exception("Error escribiendo frame Twilio para %s", session.session_key)

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
                # ponytail: collapsed the previous "Playback audio / Timing
                # stats / per-frame debug" stack into a single INFO line.
                # Old logging fired 2-3 messages per audio chunk; with
                # 20ms frames at 8 kHz the original code emitted ~50
                # debug lines per turn. One summary line per turn.
                tail = len(chunk) % TWILIO_OUTBOUND_CHUNK_BYTES
                if tail:
                    log.info(
                        "[PLAYBACK] session=%s gen=%s bytes=%d frames=%d "
                        "tail_bytes=%d (non-aligned remainder - possible "
                        "boundary click; Inworld may be returning non-"
                        "20ms-aligned chunks)",
                        session.session_key, generation, len(chunk),
                        sent_frames, tail,
                    )
                elif LOG_TWILIO_PLAYBACK:
                    log.debug(
                        "[PLAYBACK] session=%s gen=%s bytes=%d frames=%d",
                        session.session_key, generation, len(chunk), sent_frames,
                    )
                continue

            if item_type == "segment_end":
                    # Do NOT set assistant_speaking = False here!
                    # The audio is still being played by Twilio. We must wait
                    # for the Twilio "mark" event to confirm playback finished.
                    # Setting assistant_speaking = False here causes the STT to
                    # pick up the agent's own voice as user input (echo loop).
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