"""OpenAI Realtime API adapter — STT + LLM via audio-in / text-out.

Audio from Twilio (g711_ulaw 8 kHz) is forwarded directly to the
Realtime WebSocket.  OpenAI performs STT + turn detection + LLM
inference.  Text deltas are segmented via pop_streaming_segments
and fed to the existing ElevenLabs TTS pipeline.
"""

import asyncio
import base64
import json
import logging
import time

import websockets

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    MAX_HISTORY_MESSAGES,
    MAX_RESPONSE_TOKENS,
    OPENAI_API_KEY,
    OPENAI_REALTIME_MODEL,
    REALTIME_TTS_STREAMING,
    TEXT_SEGMENT_QUEUE_MAXSIZE,
)
from STT_server.domain.language import (
    extract_structured_data,
    get_language_instruction,
    get_system_prompt,
)
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_nowait_with_drop


log = logging.getLogger("stt_server")

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"


# ── Helpers ──────────────────────────────────────────────────────────

def _build_instructions(session: CallSession) -> str:
    """Compose the system instructions including dynamic session state."""
    parts = [get_system_prompt(session.preferred_language or DEFAULT_CALL_LANGUAGE)]
    lang = session.preferred_language or DEFAULT_CALL_LANGUAGE
    parts.append(get_language_instruction(lang))

    if session.collected_data:
        collected = ", ".join(f"{k}: {v}" for k, v in session.collected_data.items())
        parts.append(
            f"User state already collected in this session: {collected}. "
            "Do not ask for these details again."
        )

    _ORDER_PHRASES = ("order number", "order #", "número de orden", "numero de pedido")
    ask_count = sum(
        1 for e in session.history
        if e["role"] == "assistant"
        and any(p in e["content"].lower() for p in _ORDER_PHRASES)
    )
    if ask_count >= 2:
        parts.append(
            f"WARNING: You have already asked for the order number {ask_count} times. "
            "The speech recognition system is having difficulty. "
            "Do NOT ask again. Transfer the caller to a live agent immediately using TRANSFER_AGENT."
        )

    return "\n\n".join(parts)


# ── Main entry ───────────────────────────────────────────────────────

async def run_realtime_session(session: CallSession) -> None:
    """Connect to OpenAI Realtime and run for the lifetime of the call."""
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set — cannot start Realtime session")
        return

    url = f"{REALTIME_WS_URL}?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        try:
            ws_connect = websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=10,
                close_timeout=5,
                max_size=2**24,
            )
        except TypeError:
            ws_connect = websockets.connect(
                url,
                extra_headers=headers,
                open_timeout=10,
                close_timeout=5,
                max_size=2**24,
            )

        async with ws_connect as ws:
            log.info(
                "OpenAI Realtime connected for %s model=%s",
                session.session_key,
                OPENAI_REALTIME_MODEL,
            )

            # ── Configure session ──
            from STT_server.config import OPENAI_REALTIME_TEMPERATURE
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "instructions": _build_instructions(session),
                    "input_audio_format": "g711_ulaw",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                    "temperature": OPENAI_REALTIME_TEMPERATURE,
                    "max_response_output_tokens": MAX_RESPONSE_TOKENS,
                },
            }))

            # Initial greeting injection removed — no assistant message pre-seeded.

            sender_task = asyncio.create_task(_audio_sender(ws, session))
            watcher_task = asyncio.create_task(_barge_in_watcher(ws, session))

            try:
                await _event_receiver(ws, session)
            finally:
                sender_task.cancel()
                watcher_task.cancel()
                for t in (sender_task, watcher_task):
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("OpenAI Realtime session error for %s", session.session_key)


# ── Audio sender ─────────────────────────────────────────────────────

async def _audio_sender(ws, session: CallSession) -> None:
    """Read mulaw chunks from the session queue and forward to OpenAI."""
    try:
        while not session.closed:
            try:
                chunk = await asyncio.wait_for(
                    session.realtime_audio_queue.get(), timeout=5.0,
                )
            except asyncio.TimeoutError:
                continue
            if chunk is None:
                return
            audio_b64 = base64.b64encode(chunk).decode("ascii")
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }))
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Realtime audio sender error for %s", session.session_key)


# ── Barge-in watcher ─────────────────────────────────────────────────

async def _barge_in_watcher(ws, session: CallSession) -> None:
    """When local VAD triggers a barge-in (generation_changed), cancel
    the in-progress OpenAI response so it stops generating."""
    try:
        while not session.closed:
            await session.generation_changed.wait()
            session.generation_changed.clear()
            # Only send a cancel if the session knows a response is active.
            if not getattr(session, "response_active", False):
                log.debug(
                    "generation_changed but no active realtime response for %s; ignoring cancel",
                    session.session_key,
                )
                continue

            tq = session.realtime_text_queue
            if tq is None:
                log.debug(
                    "response_active True but realtime_text_queue is None for %s; sending cancel anyway",
                    session.session_key,
                )

            try:
                await ws.send(json.dumps({"type": "response.cancel"}))
            except Exception:
                # Log and continue watching — transient network or server errors
                log.exception(
                    "Failed to send response.cancel for %s",
                    session.session_key,
                )
                continue

            # Unblock and clear the TTS consumer queue after requesting cancel
            if tq is not None:
                enqueue_nowait_with_drop(tq, None, "text_segment_queue")
            session.realtime_text_queue = None
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Realtime barge-in watcher error for %s", session.session_key)


# ── Event receiver ───────────────────────────────────────────────────

async def _event_receiver(ws, session: CallSession) -> None:
    """Process server events: transcription, response streaming, errors."""
    from STT_server.services.turn_manager import play_tts_from_text_queue

    pending = ""
    playback_task: asyncio.Task | None = None
    response_started_at: float | None = None
    current_response_text = ""

    try:
        async for raw_msg in ws:
            if session.closed:
                break

            event = json.loads(raw_msg)
            etype = event.get("type", "")

            # ── Session lifecycle ──
            if etype == "session.created":
                log.info("Realtime session created for %s", session.session_key)
                continue

            if etype == "session.updated":
                continue

            # ── User speech events ──
            if etype == "input_audio_buffer.speech_started":
                session.last_activity_at = time.monotonic()
                continue

            if etype in (
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
            ):
                continue

            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = (event.get("transcript") or "").strip()
                if transcript:
                    log.info("Usuario (%s) [realtime]: %s", session.session_key, transcript)
                    session.current_transcript = transcript
                    session.last_activity_at = time.monotonic()
                    session.history.append({"role": "user", "content": transcript})
                    if len(session.history) > MAX_HISTORY_MESSAGES:
                        session.history[:] = session.history[-MAX_HISTORY_MESSAGES:]

                    try:
                        structured = extract_structured_data(transcript)
                    except Exception:
                        log.exception(
                            "extract_structured_data failed for %s transcript=%r",
                            session.session_key,
                            transcript[:120],
                        )
                        structured = {}
                    if structured:
                        for k, v in structured.items():
                            session.collected_data[k] = v
                        await ws.send(json.dumps({
                            "type": "session.update",
                            "session": {"instructions": _build_instructions(session)},
                        }))
                continue

            # ── Response lifecycle ──
            if etype == "response.created":
                response_started_at = time.perf_counter()
                current_response_text = ""
                pending = ""
                text_queue: asyncio.Queue[str | None] = asyncio.Queue(
                    maxsize=TEXT_SEGMENT_QUEUE_MAXSIZE,
                )
                session.realtime_text_queue = text_queue
                session.active_generation += 1
                # Mark that a server-side response is now active (used to gate cancels)
                session.response_active = True
                generation = session.active_generation
                playback_task = asyncio.create_task(
                    play_tts_from_text_queue(session, generation, text_queue),
                )
                session.tasks.add(playback_task)
                playback_task.add_done_callback(session.tasks.discard)
                continue

            if etype == "response.text.delta":
                delta = event.get("delta", "")
                tq = session.realtime_text_queue
                if not delta or tq is None:
                    continue
                current_response_text += delta
                if REALTIME_TTS_STREAMING:
                    pending += delta
                    from STT_server.domain.language import pop_streaming_segments
                    segments, pending = pop_streaming_segments(pending)
                    for seg in segments:
                        enqueue_nowait_with_drop(tq, seg, "text_segment_queue")
                continue

            if etype == "response.text.done":
                tq = session.realtime_text_queue
                if REALTIME_TTS_STREAMING and tq is not None and pending.strip():
                    from STT_server.domain.language import pop_streaming_segments
                    segments, _ = pop_streaming_segments(pending, force=True)
                    for seg in segments:
                        enqueue_nowait_with_drop(tq, seg, "text_segment_queue")
                    pending = ""
                continue

            if etype == "response.done":
                # If we are NOT streaming TTS, enqueue the full reply once.
                if not REALTIME_TTS_STREAMING:
                    tq = session.realtime_text_queue
                    full_reply = current_response_text.strip()
                    if tq is not None and full_reply:
                        enqueue_nowait_with_drop(tq, full_reply, "text_segment_queue")
                # Signal end-of-stream to TTS consumer
                tq = session.realtime_text_queue
                if tq is not None:
                    enqueue_nowait_with_drop(tq, None, "text_segment_queue")
                session.realtime_text_queue = None

                status = (event.get("response") or {}).get("status", "completed")

                if status == "cancelled":
                    # Barge-in or explicit cancel — discard partial output
                    if playback_task and not playback_task.done():
                        playback_task.cancel()
                        try:
                            await playback_task
                        except asyncio.CancelledError:
                            pass
                    playback_task = None
                    response_started_at = None
                    current_response_text = ""
                    pending = ""
                    # Server-side response no longer active
                    session.response_active = False
                    continue

                # Normal completion
                reply = current_response_text.strip()
                if reply:
                    session.history.append({"role": "assistant", "content": reply})
                    if len(session.history) > MAX_HISTORY_MESSAGES:
                        session.history[:] = session.history[-MAX_HISTORY_MESSAGES:]
                    session.last_processed_user_text = session.current_transcript
                    log.info("Agente (%s): %s", session.session_key, reply)

                # Wait for TTS playback to finish
                tts_metrics: list[tuple[float | None, float]] = []
                if playback_task:
                    try:
                        tts_metrics = await playback_task
                    except Exception:
                        log.exception("Playback error for %s", session.session_key)

                first_tts_ms = next(
                    (m[0] for m in tts_metrics if m[0] is not None), None,
                )
                total_ms = (
                    (time.perf_counter() - response_started_at) * 1000
                    if response_started_at
                    else 0
                )
                log.info(
                    "Turno %s gen=%s tts_ttfb_ms=%s total_ms=%.1f",
                    session.session_key,
                    session.active_generation,
                    f"{first_tts_ms:.1f}" if first_tts_ms is not None else "n/a",
                    total_ms,
                )

                playback_task = None
                response_started_at = None
                current_response_text = ""
                # Server-side response finished normally
                session.response_active = False
                continue

            # ── Errors ──
            if etype == "error":
                err = event.get("error", {})
                # Treat cancellation-not-active as non-fatal (likely a race).
                code = err.get("code") if isinstance(err, dict) else None
                if code == "response_cancel_not_active":
                    log.debug(
                        "Realtime API non-fatal cancel for %s: %s",
                        session.session_key,
                        err,
                    )
                else:
                    log.error(
                        "Realtime API error for %s: %s",
                        session.session_key,
                        err,
                    )
                continue

            # ── Known metadata events — ignore silently ──
            if etype in (
                "response.output_item.added",
                "response.output_item.done",
                "response.content_part.added",
                "response.content_part.done",
                "conversation.item.created",
                "rate_limits.updated",
            ):
                continue

            log.debug("Realtime unknown event %s for %s", etype, session.session_key)

    except websockets.exceptions.ConnectionClosed as exc:
        log.warning("Realtime WS closed for %s: %s", session.session_key, exc)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Realtime event receiver error for %s", session.session_key)
    finally:
        tq = session.realtime_text_queue
        if tq is not None:
            enqueue_nowait_with_drop(tq, None, "text_segment_queue")
            session.realtime_text_queue = None
        if playback_task and not playback_task.done():
            playback_task.cancel()
            try:
                await playback_task
            except asyncio.CancelledError:
                pass
        # Ensure response_active is cleared on exit
        session.response_active = False
