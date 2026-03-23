import asyncio
import contextlib
import logging
import time

from STT_server.adapters.deepgram_tts import stream_tts_segment
from STT_server.adapters.openai_llm import build_messages, call_llm, stream_llm_reply_sync
from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    FILLER_DELAY_MS,
    FINAL_RESTART_DELTA_CHARS,
    LLM_TIMEOUT_SEC,
    PARTIAL_TRANSCRIPT_DEBOUNCE_MS,
    PARTIAL_TRANSCRIPT_START_CHARS,
    TEXT_SEGMENT_QUEUE_MAXSIZE,
    TTS_TIMEOUT_SEC,
)
from STT_server.domain.language import detect_language, get_filler_text, get_stt_failure_prompt, normalize_supported_language, split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_nowait_with_drop, enqueue_with_drop
from STT_server.services.playback_service import emit_playback_item, interrupt_current_turn


log = logging.getLogger("stt_server")


def trim_history(session: CallSession) -> None:
    from STT_server.config import MAX_HISTORY_MESSAGES

    if len(session.history) > MAX_HISTORY_MESSAGES:
        session.history[:] = session.history[-MAX_HISTORY_MESSAGES:]


async def play_tts_from_text_queue(
    session: CallSession,
    generation: int,
    text_queue: asyncio.Queue[str | None],
) -> list[tuple[float | None, float]]:
    metrics: list[tuple[float | None, float]] = []

    while True:
        text = await text_queue.get()
        if text is None:
            break
        if generation != session.active_generation:
            break

        try:
            metric = await asyncio.wait_for(
                stream_tts_segment(session, text, generation, lambda item: emit_playback_item(session, item)),
                timeout=TTS_TIMEOUT_SEC,
            )
            metrics.append(metric)
        except asyncio.TimeoutError:
            log.warning("TTS timeout en %s", session.session_key)
            break
        except Exception:
            log.exception("TTS error en %s", session.session_key)
            break

    return metrics


async def enqueue_delayed_filler(
    text_queue: asyncio.Queue[str | None],
    filler_text: str,
    first_segment_event: asyncio.Event,
    generation: int,
    session: CallSession,
) -> None:
    if not filler_text:
        return

    try:
        await asyncio.sleep(FILLER_DELAY_MS / 1000.0)
    except asyncio.CancelledError:
        return

    if generation != session.active_generation or session.closed or first_segment_event.is_set():
        return

    await enqueue_with_drop(text_queue, filler_text, "text_segment_queue")


async def stream_llm_reply_with_tts(
    session: CallSession,
    user_text: str,
    generation: int,
) -> tuple[str, float, list[tuple[float | None, float]], str | None]:
    lang = session.preferred_language or detect_language(user_text)
    filler_text = get_filler_text(lang)
    loop = asyncio.get_running_loop()
    text_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=TEXT_SEGMENT_QUEUE_MAXSIZE)
    first_segment_event = asyncio.Event()
    playback_task = asyncio.create_task(play_tts_from_text_queue(session, generation, text_queue))
    filler_task = asyncio.create_task(enqueue_delayed_filler(text_queue, filler_text, first_segment_event, generation, session))

    def emit_segment(segment: str) -> None:
        loop.call_soon_threadsafe(enqueue_nowait_with_drop, text_queue, segment, "text_segment_queue")

    def emit_done() -> None:
        loop.call_soon_threadsafe(enqueue_nowait_with_drop, text_queue, None, "text_segment_queue")

    llm_started = time.perf_counter()
    producer_task = asyncio.create_task(
        asyncio.to_thread(
            stream_llm_reply_sync,
            build_messages(session, user_text),
            lambda: generation != session.active_generation or session.closed,
            emit_segment,
            emit_done,
            first_segment_event.set,
        )
    )

    llm_error: str | None = None
    reply = ""
    try:
        if LLM_TIMEOUT_SEC > 0:
            reply, llm_error = await asyncio.wait_for(producer_task, timeout=LLM_TIMEOUT_SEC)
        else:
            reply, llm_error = await producer_task
    except asyncio.TimeoutError:
        llm_error = "timeout"
        await enqueue_with_drop(text_queue, None, "text_segment_queue")
    finally:
        filler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await filler_task

    llm_ms = (time.perf_counter() - llm_started) * 1000
    tts_metrics = await playback_task

    if llm_error and not reply and generation == session.active_generation:
        reply = "Lo siento, estoy tardando mas de lo normal. Puedes repetirlo?"
        try:
            fallback_metric = await asyncio.wait_for(
                stream_tts_segment(session, reply, generation, lambda item: emit_playback_item(session, item)),
                timeout=TTS_TIMEOUT_SEC,
            )
            tts_metrics.append(fallback_metric)
        except asyncio.TimeoutError:
            log.warning("TTS timeout en fallback de %s", session.session_key)
        except Exception:
            log.exception("TTS error en fallback de %s", session.session_key)

    return reply.strip(), llm_ms, tts_metrics, llm_error


async def handle_agent_reply(session: CallSession, user_text: str, generation: int, trigger: str) -> None:
    started_at = time.perf_counter()
    log.info("Usuario (%s) [%s]: %s", session.session_key, trigger, user_text)

    reply, llm_ms, tts_metrics, llm_error = await stream_llm_reply_with_tts(session, user_text, generation)
    if generation != session.active_generation or not reply:
        return

    session.history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
    )
    trim_history(session)

    log.info("Agente (%s): %s", session.session_key, reply)

    total_ms = (time.perf_counter() - started_at) * 1000
    first_tts_ms = next((metric[0] for metric in tts_metrics if metric[0] is not None), None)
    log.info(
        "Turno %s gen=%s trigger=%s llm_ms=%.1f tts_ttfb_ms=%s total_ms=%.1f llm_error=%s",
        session.session_key,
        generation,
        trigger,
        llm_ms,
        f"{first_tts_ms:.1f}" if first_tts_ms is not None else "n/a",
        total_ms,
        llm_error or "none",
    )


async def launch_reply_pipeline(
    session: CallSession,
    user_text: str,
    trigger: str,
    replace_current: bool = False,
) -> None:
    normalized_text = user_text.strip()
    if not normalized_text:
        return

    existing_task = session.reply_task
    if existing_task and not existing_task.done():
        if session.reply_source_text == normalized_text and not replace_current:
            return
        if not replace_current:
            return
        await interrupt_current_turn(session)
        existing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await existing_task
    else:
        session.active_generation += 1

    session.reply_source_text = normalized_text
    generation = session.active_generation
    task = asyncio.create_task(handle_agent_reply(session, normalized_text, generation, trigger))
    session.reply_task = task
    session.tasks.add(task)
    task.add_done_callback(session.tasks.discard)


async def schedule_partial_reply(session: CallSession, transcript: str) -> None:
    try:
        await asyncio.sleep(PARTIAL_TRANSCRIPT_DEBOUNCE_MS / 1000.0)
    except asyncio.CancelledError:
        return

    if session.closed or transcript != session.current_transcript:
        return
    if session.reply_task and not session.reply_task.done():
        return

    await launch_reply_pipeline(session, transcript, trigger="partial")


async def enqueue_transcript_event(session: CallSession, event: dict) -> None:
    await enqueue_with_drop(session.transcript_queue, event, "transcript_queue")


async def process_transcripts(session: CallSession) -> None:
    try:
        while True:
            item = await session.transcript_queue.get()
            text = (item.get("text") or "").strip()
            if not text:
                continue

            language = normalize_supported_language(item.get("language") or session.preferred_language or DEFAULT_CALL_LANGUAGE)
            session.preferred_language = language
            session.current_transcript = text
            is_final = bool(item.get("is_final"))
            speech_final = bool(item.get("speech_final"))

            if is_final:
                pending_partial = session.partial_reply_task
                if pending_partial and not pending_partial.done():
                    pending_partial.cancel()

                replace_current = bool(
                    session.reply_task
                    and not session.reply_task.done()
                    and text != session.reply_source_text
                    and len(text) >= len(session.reply_source_text) + FINAL_RESTART_DELTA_CHARS
                )
                await launch_reply_pipeline(session, text, trigger="final", replace_current=replace_current)

                if speech_final:
                    session.current_transcript = ""
                continue

            if len(text) < PARTIAL_TRANSCRIPT_START_CHARS:
                continue
            if session.reply_task and not session.reply_task.done():
                continue

            pending_partial = session.partial_reply_task
            if pending_partial and not pending_partial.done():
                pending_partial.cancel()

            task = asyncio.create_task(schedule_partial_reply(session, text))
            session.partial_reply_task = task
            session.tasks.add(task)
            task.add_done_callback(session.tasks.discard)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Error en process_transcripts")


async def announce_stt_failure_once(session: CallSession) -> None:
    if session.closed or session.stt_failure_announced or not session.stream_sid:
        return

    session.stt_failure_announced = True
    await interrupt_current_turn(session)
    generation = session.active_generation
    prompt = get_stt_failure_prompt(session.preferred_language or DEFAULT_CALL_LANGUAGE)

    try:
        await asyncio.wait_for(
            stream_tts_segment(session, prompt, generation, lambda item: emit_playback_item(session, item)),
            timeout=TTS_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        log.warning("TTS timeout en fallback STT para %s", session.session_key)
    except Exception:
        log.exception("TTS error en fallback STT para %s", session.session_key)