import asyncio
import contextlib
import logging
import time

from STT_server.adapters.deepgram_stt_batch import transcribe_block
from STT_server.adapters.deepgram_tts import stream_tts_segment
from STT_server.adapters.openai_llm import build_messages, call_llm, stream_llm_reply_sync
from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    DEEPGRAM_STT_LANGUAGE_HINT,
    FILLER_DELAY_MS,
    FINAL_TRANSCRIPT_GRACE_MS,
    FINAL_RESTART_DELTA_CHARS,
    LLM_TIMEOUT_SEC,
    PARTIAL_PREFETCH_MAX_DELTA_CHARS,
    PARTIAL_TRANSCRIPT_DEBOUNCE_MS,
    PARTIAL_TRANSCRIPT_START_CHARS,
    SHORT_FINAL_MAX_WORDS,
    TEXT_SEGMENT_QUEUE_MAXSIZE,
    TTS_TIMEOUT_SEC,
)
from STT_server.domain.language import detect_language, get_filler_text, get_stt_failure_prompt, looks_like_incomplete_utterance, normalize_supported_language, split_tts_segments
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
    first_emitted = False

    while True:
        text = await text_queue.get()
        if text is None:
            break
        if generation != session.active_generation:
            break

        # After the first segment is spoken, merge any already-queued
        # segments into a single TTS call to reduce inter-segment gaps.
        end_of_stream = False
        if first_emitted:
            while not text_queue.empty():
                try:
                    peek = text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if peek is None:
                    end_of_stream = True
                    break
                text += " " + peek

        first_emitted = True

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

        if end_of_stream:
            break

    return metrics


async def speak_precomputed_reply(
    session: CallSession,
    reply: str,
    generation: int,
) -> list[tuple[float | None, float]]:
    metrics: list[tuple[float | None, float]] = []

    for segment in split_tts_segments(reply):
        if generation != session.active_generation:
            break

        try:
            metric = await asyncio.wait_for(
                stream_tts_segment(session, segment, generation, lambda item: emit_playback_item(session, item)),
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


def consume_prefetched_reply(session: CallSession, final_text: str) -> str | None:
    draft_text = session.prefetched_reply_source_text.strip()
    draft_reply = session.prefetched_reply_text.strip()
    if not draft_text or not draft_reply:
        return None

    normalized_final = final_text.strip()
    if normalized_final == draft_text:
        return draft_reply

    if normalized_final.startswith(draft_text):
        delta = len(normalized_final) - len(draft_text)
        if delta <= PARTIAL_PREFETCH_MAX_DELTA_CHARS:
            return draft_reply

    return None


async def cancel_prefetch_task(session: CallSession) -> None:
    task = session.prefetched_reply_task
    session.prefetched_reply_task = None
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def clear_prefetched_reply(session: CallSession) -> None:
    session.prefetched_reply_source_text = ""
    session.prefetched_reply_text = ""


async def cancel_deferred_final_flush(session: CallSession) -> None:
    task = session.deferred_final_flush_task
    session.deferred_final_flush_task = None
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def should_defer_final_transcript(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if looks_like_incomplete_utterance(stripped):
        return True

    if stripped[-1] in ".!?":
        return False

    word_count = len(stripped.split())
    return word_count <= SHORT_FINAL_MAX_WORDS


async def process_final_transcript(
    session: CallSession,
    text: str,
    language: str,
    speech_final: bool,
) -> None:
    pending_partial = session.partial_reply_task
    if pending_partial and not pending_partial.done():
        pending_partial.cancel()

    prepared_reply = consume_prefetched_reply(session, text)
    await cancel_prefetch_task(session)
    clear_prefetched_reply(session)
    session.pending_realtime_final = None

    replace_current = bool(
        session.reply_task
        and not session.reply_task.done()
        and text != session.reply_source_text
        and len(text) >= len(session.reply_source_text) + FINAL_RESTART_DELTA_CHARS
    )
    await launch_reply_pipeline(
        session,
        text,
        trigger="final",
        replace_current=replace_current,
        prepared_reply=prepared_reply,
    )

    if speech_final:
        session.current_transcript = ""


async def flush_deferred_final_after_grace(session: CallSession) -> None:
    try:
        await asyncio.sleep(FINAL_TRANSCRIPT_GRACE_MS / 1000.0)
    except asyncio.CancelledError:
        return

    text = session.deferred_final_text.strip()
    if not text or session.closed:
        return

    if session.awaiting_local_final:
        try:
            await asyncio.sleep(FINAL_TRANSCRIPT_GRACE_MS / 1000.0)
        except asyncio.CancelledError:
            return
        text = session.deferred_final_text.strip()
        if not text or session.closed:
            return

    # If still incomplete after grace, give one more grace period for
    # the continuation to arrive before processing a partial sentence.
    if looks_like_incomplete_utterance(text):
        log.info("Texto aún incompleto tras grace en %s, esperando más: %s", session.session_key, text)
        try:
            await asyncio.sleep(FINAL_TRANSCRIPT_GRACE_MS / 1000.0)
        except asyncio.CancelledError:
            return
        text = session.deferred_final_text.strip()
        if not text or session.closed:
            return

    # If the assistant is still speaking, wait until playback finishes
    # before flushing so we don't interrupt the current response.
    if session.assistant_speaking:
        for _ in range(20):  # up to ~10s extra wait
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
            if not session.assistant_speaking or session.closed:
                break
        text = session.deferred_final_text.strip()
        if not text or session.closed:
            return

    language = normalize_supported_language(session.deferred_final_language or session.preferred_language or DEFAULT_CALL_LANGUAGE)
    session.deferred_final_text = ""
    session.deferred_final_language = None
    session.deferred_final_flush_task = None
    log.info("Procesando final diferido en %s: %s", session.session_key, text)
    await process_final_transcript(session, text, language, speech_final=True)


async def prefetch_agent_reply(session: CallSession, user_text: str) -> None:
    normalized_text = user_text.strip()
    if not normalized_text:
        return

    try:
        if LLM_TIMEOUT_SEC > 0:
            reply = await asyncio.wait_for(call_llm(session, normalized_text), timeout=LLM_TIMEOUT_SEC)
        else:
            reply = await call_llm(session, normalized_text)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Prefetch LLM error en %s", session.session_key)
        return

    if session.closed or session.prefetched_reply_source_text != normalized_text:
        return

    session.prefetched_reply_text = reply.strip()


async def launch_reply_prefetch(session: CallSession, user_text: str) -> None:
    normalized_text = user_text.strip()
    if not normalized_text:
        return

    existing_task = session.prefetched_reply_task
    if existing_task and not existing_task.done() and session.prefetched_reply_source_text == normalized_text:
        return

    await cancel_prefetch_task(session)
    clear_prefetched_reply(session)
    session.prefetched_reply_source_text = normalized_text
    task = asyncio.create_task(prefetch_agent_reply(session, normalized_text))
    session.prefetched_reply_task = task
    session.tasks.add(task)
    task.add_done_callback(session.tasks.discard)


async def handle_agent_reply(
    session: CallSession,
    user_text: str,
    generation: int,
    trigger: str,
    prepared_reply: str | None = None,
) -> None:
    started_at = time.perf_counter()
    log.info("Usuario (%s) [%s]: %s", session.session_key, trigger, user_text)

    if prepared_reply:
        reply = prepared_reply.strip()
        llm_ms = 0.0
        llm_error = None
        tts_metrics = await speak_precomputed_reply(session, reply, generation)
    else:
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
    prepared_reply: str | None = None,
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
    task = asyncio.create_task(handle_agent_reply(session, normalized_text, generation, trigger, prepared_reply=prepared_reply))
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

    await launch_reply_prefetch(session, transcript)


async def enqueue_transcript_event(session: CallSession, event: dict) -> None:
    await enqueue_with_drop(session.transcript_queue, event, "transcript_queue")


async def process_local_utterances(session: CallSession) -> None:
    try:
        while True:
            generation, utterance = await session.utterance_queue.get()
            if generation != session.active_generation:
                session.awaiting_local_final = False
                session.pending_realtime_final = None
                continue

            try:
                texts, language = await transcribe_block(
                    utterance,
                    language_hint=session.preferred_language or DEEPGRAM_STT_LANGUAGE_HINT,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Batch STT error en %s", session.session_key)
                texts = []
                language = session.preferred_language or DEFAULT_CALL_LANGUAGE

            final_text = " ".join(text.strip() for text in texts).strip()
            session.awaiting_local_final = False

            if final_text:
                session.pending_realtime_final = None
                await enqueue_transcript_event(
                    session,
                    {
                        "text": final_text,
                        "language": language,
                        "is_final": True,
                        "speech_final": True,
                        "source": "batch_final",
                    },
                )
                continue

            pending_realtime = session.pending_realtime_final
            session.pending_realtime_final = None
            if pending_realtime:
                await enqueue_transcript_event(session, pending_realtime)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Error en process_local_utterances")


async def process_transcripts(session: CallSession) -> None:
    try:
        while True:
            item = await session.transcript_queue.get()
            text = (item.get("text") or "").strip()
            if not text:
                continue

            language = normalize_supported_language(item.get("language") or session.preferred_language or DEFAULT_CALL_LANGUAGE)
            source = item.get("source") or "realtime"
            if source in {"batch_final", "realtime_fallback"} or (source == "realtime" and not session.awaiting_local_final and bool(item.get("is_final"))):
                session.preferred_language = language
            session.current_transcript = text
            is_final = bool(item.get("is_final"))
            speech_final = bool(item.get("speech_final"))

            if is_final:
                if source == "realtime" and session.awaiting_local_final:
                    session.pending_realtime_final = {
                        "text": text,
                        "language": language,
                        "is_final": True,
                        "speech_final": speech_final,
                        "source": "realtime_fallback",
                    }
                    continue

                if source == "batch_final" and session.deferred_final_text:
                    await cancel_deferred_final_flush(session)
                    session.deferred_final_text = ""
                    session.deferred_final_language = None
                elif session.deferred_final_text:
                    await cancel_deferred_final_flush(session)
                    text = f"{session.deferred_final_text} {text}".strip()
                    language = normalize_supported_language(session.deferred_final_language or language)
                    session.deferred_final_text = ""
                    session.deferred_final_language = None

                # While the assistant is speaking, always defer the final
                # so we don't cut off the current response with a new turn.
                if session.assistant_speaking or should_defer_final_transcript(text):
                    session.deferred_final_text = text
                    session.deferred_final_language = language
                    if session.assistant_speaking:
                        log.info("Defiriendo final (asistente hablando) en %s: %s", session.session_key, text)
                    else:
                        log.info("Defiriendo final incompleto en %s: %s", session.session_key, text)
                    await cancel_deferred_final_flush(session)
                    task = asyncio.create_task(flush_deferred_final_after_grace(session))
                    session.deferred_final_flush_task = task
                    session.tasks.add(task)
                    task.add_done_callback(session.tasks.discard)
                    continue

                await process_final_transcript(session, text, language, speech_final=speech_final)
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