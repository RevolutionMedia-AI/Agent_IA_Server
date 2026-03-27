import asyncio
import contextlib
import logging
import re
import time

from STT_server.adapters.rime_tts import stream_tts_segment
from STT_server.adapters.openai_llm import build_messages, call_llm, stream_llm_reply_sync
from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    DIGIT_DICTATION_GRACE_MS,
    FILLER_DELAY_MS,
    FINAL_TRANSCRIPT_GRACE_MS,
    FINAL_RESTART_DELTA_CHARS,
    LLM_TIMEOUT_SEC,
    MAX_HISTORY_MESSAGES,
    PARTIAL_PREFETCH_MAX_DELTA_CHARS,
    PARTIAL_TRANSCRIPT_DEBOUNCE_MS,
    PARTIAL_TRANSCRIPT_START_CHARS,
    SHORT_FINAL_MAX_WORDS,
    TEXT_SEGMENT_QUEUE_MAXSIZE,
    TTS_MAX_RETRIES,
    TTS_RETRY_BACKOFF_MS,
    TTS_TIMEOUT_SEC,
)
from STT_server.domain.language import detect_language, extract_structured_data, get_filler_text, get_stt_failure_prompt, is_duplicate_collected_data, is_non_actionable_utterance, looks_like_digit_dictation, looks_like_incomplete_utterance, normalize_digits_in_text, normalize_supported_language, split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_nowait_with_drop, enqueue_with_drop
from STT_server.services.playback_service import emit_playback_item, interrupt_current_turn


log = logging.getLogger("stt_server")


async def run_tts_with_retries(session: CallSession, text: str, generation: int) -> tuple[float | None, float]:
    """Run one TTS segment with bounded retries.

    Important: we avoid wrapping the whole streaming call in wait_for(),
    because cancellation truncates audio mid-stream. The adapter itself
    handles TTFB/idle timeouts.
    """
    attempts = max(0, TTS_MAX_RETRIES) + 1
    for attempt in range(1, attempts + 1):
        try:
            return await stream_tts_segment(session, text, generation, lambda item: emit_playback_item(session, item))
        except (asyncio.TimeoutError, TimeoutError):
            log.warning(
                "TTS timeout en %s (attempt %s/%s, text_len=%s)",
                session.session_key,
                attempt,
                attempts,
                len(text),
            )
            if attempt < attempts:
                await asyncio.sleep(max(0, TTS_RETRY_BACKOFF_MS) / 1000.0)
        except Exception:
            log.exception("TTS error en %s (attempt %s/%s)", session.session_key, attempt, attempts)
            raise

    raise asyncio.TimeoutError()


# ── Echo / hallucination detection ──────────────────────────────────

def _has_excessive_repetition(text: str) -> bool:
    """True when any single word repeats 3+ times consecutively."""
    words = text.lower().split()
    streak = 1
    for i in range(1, len(words)):
        prev = words[i - 1].rstrip(".,!?;:")
        curr = words[i].rstrip(".,!?;:")
        if curr and curr == prev:
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 1
    return False


def _echoes_agent_speech(text: str, session: CallSession) -> bool:
    """True when >60 % of words overlap with the agent's last utterance."""
    words = text.lower().split()
    if len(words) < 4:
        return False
    for entry in reversed(session.history[-6:]):
        if entry["role"] == "assistant":
            agent_words = set(entry["content"].lower().split())
            overlap = sum(1 for w in words if w.rstrip(".,!?;:") in agent_words)
            return overlap > len(words) * 0.6
    return False


def is_echo_hallucination(text: str, session: CallSession) -> bool:
    """Detects likely Deepgram hallucinations — repetitive garbage or
    text that mirrors the agent's own speech (TTS echo)."""
    return _has_excessive_repetition(text) or _echoes_agent_speech(text, session)


def trim_history(session: CallSession) -> None:
    if len(session.history) > MAX_HISTORY_MESSAGES:
        session.history[:] = session.history[-MAX_HISTORY_MESSAGES:]


def update_memory(session: CallSession, transcript: str) -> None:
    structured = extract_structured_data(transcript)
    for k, v in structured.items():
        session.collected_data[k] = v


def should_generate_response(session: CallSession, transcript: str) -> bool:
    structured = extract_structured_data(transcript)
    if not structured:
        return True
    if is_duplicate_collected_data(session, structured):
        # Only suppress when the transcript is *purely* a data repeat
        # (e.g. the bare order number).  If the user wrapped it in
        # conversational text ("Oh yeah sure, 123451") we must still
        # process so the assistant can acknowledge and continue.
        stripped = transcript.strip().rstrip(".,!?;:")
        data_values = {v.lower() for v in structured.values()}
        normalized = normalize_digits_in_text(stripped)
        stripped_norm = normalized.strip().rstrip(".,!?;:").lower()
        # Pure repeat: every non-whitespace token is a known data value.
        words = stripped_norm.split()
        if words and all(
            w.rstrip(".,!?;:") in data_values for w in words
        ):
            log.info("Texto puramente duplicado en collected_data, ignora: %s", transcript)
            return False
        # Conversational confirmation — process normally.
        log.info("Datos duplicados pero con contexto conversacional, procesa: %s", transcript)

    return True

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
            metric = await run_tts_with_retries(session, text, generation)
            metrics.append(metric)
        except asyncio.TimeoutError:
            break
        except Exception:
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
            metric = await run_tts_with_retries(session, segment, generation)
            metrics.append(metric)
        except asyncio.TimeoutError:
            break
        except Exception:
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
            fallback_metric = await run_tts_with_retries(session, reply, generation)
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

    # Final starts with the draft partial → small continuation appended
    if normalized_final.startswith(draft_text):
        delta = len(normalized_final) - len(draft_text)
        if delta <= PARTIAL_PREFETCH_MAX_DELTA_CHARS:
            return draft_reply

    # Draft starts with the final → user said less than predicted
    if draft_text.startswith(normalized_final):
        delta = len(draft_text) - len(normalized_final)
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


# Very short utterances (≤ this many words) are always deferred even with
# punctuation so the user has time to continue speaking.
# Only single-word fragments like "Um" or "Okay" are deferred —
# anything 2+ words with terminal punctuation is a valid turn.
VERY_SHORT_WORD_LIMIT = 1


def user_is_speaking(session: CallSession) -> bool:
    return bool(session.speech_frames or session.voice_streak > 0)


def final_transcript_ready(session: CallSession, is_final: bool) -> bool:
    if not is_final:
        return False
    return not user_is_speaking(session)


def should_defer_final_transcript(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    word_count = len(stripped.split())

    # Single-word fragments without punctuation ("Um", "Okay") → defer.
    if word_count <= VERY_SHORT_WORD_LIMIT and stripped[-1] not in ".!?":
        return True

    # Digit dictation: if the user seems to be spelling out an order
    # number but hasn't yet reached 5 digits, keep deferring even if
    # Deepgram added terminal punctuation — the user is likely still
    # going.
    if looks_like_digit_dictation(stripped):
        normalized = normalize_digits_in_text(stripped)
        digit_runs = re.findall(r"\d+", normalized)
        max_digits = max((len(r) for r in digit_runs), default=0)
        if max_digits < 5:
            return True

    # Anything with terminal punctuation (.!?) is a complete turn — process now.
    if stripped[-1] in ".!?":
        return False

    # Structurally incomplete sentences (trailing comma, "because", etc.)
    if looks_like_incomplete_utterance(stripped):
        return True

    # Unpunctuated but longer than SHORT_FINAL_MAX_WORDS → treat as complete.
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

    # Collect structured data from final transcript and check duplication.
    update_memory(session, text)
    if not should_generate_response(session, text):
        log.info("No se genera respuesta para final duplicado en %s: %s", session.session_key, text)
        if speech_final:
            session.current_transcript = ""
        return

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
    # Use a longer initial grace period when the user is dictating digits
    # (e.g. spelling out an order number one digit at a time).
    text_peek = session.deferred_final_text.strip()
    grace_ms = (
        DIGIT_DICTATION_GRACE_MS
        if text_peek and looks_like_digit_dictation(text_peek)
        else FINAL_TRANSCRIPT_GRACE_MS
    )
    try:
        await asyncio.sleep(grace_ms / 1000.0)
    except asyncio.CancelledError:
        return

    text = session.deferred_final_text.strip()
    if not text or session.closed:
        return

    # If the text still looks incomplete, allow ONE extra short wait
    # (half grace) to accumulate more text.
    if looks_like_incomplete_utterance(text):
        ext_ms = (
            DIGIT_DICTATION_GRACE_MS / 2
            if looks_like_digit_dictation(text)
            else FINAL_TRANSCRIPT_GRACE_MS / 2
        )
        try:
            await asyncio.sleep(ext_ms / 1000.0)
        except asyncio.CancelledError:
            return
        text = session.deferred_final_text.strip()
        if not text or session.closed:
            return

    # If the assistant is still speaking, wait briefly for playback to
    # finish — but cap at ~3 s to avoid the "amnesia" effect where user
    # input is recognised too late.
    if session.assistant_speaking:
        for _ in range(6):  # up to ~3s extra wait
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
            if not session.assistant_speaking or session.closed:
                break
        text = session.deferred_final_text.strip()
        if not text or session.closed:
            return

    # ── Discard Deepgram hallucinations / TTS echo ──
    if is_echo_hallucination(text, session):
        log.info(
            "Descartando hallucination/eco en %s: %s",
            session.session_key, text,
        )
        session.deferred_final_text = ""
        session.deferred_final_language = None
        session.deferred_final_flush_task = None
        return

    # ── Discard fragment echoes ──
    # Only discard if the deferred text is an exact duplicate of the last
    # processed user text — substring matching was too aggressive and
    # discarded valid short answers like "Five" that happened to appear
    # inside a previous longer utterance.
    text_norm = text.strip().lower()
    last = session.last_processed_user_text.strip().lower()
    if last and text_norm == last:
        log.info(
            "Descartando final diferido (duplicado exacto) en %s: %s",
            session.session_key, text,
        )
        session.deferred_final_text = ""
        session.deferred_final_language = None
        session.deferred_final_flush_task = None
        return

    # Also discard if it duplicates the last user entry in history.
    if session.history:
        for entry in reversed(session.history[-6:]):
            if entry["role"] == "user":
                if text_norm == entry["content"].strip().lower():
                    log.info(
                        "Descartando final diferido (duplica historial) en %s: %s",
                        session.session_key, text,
                    )
                    session.deferred_final_text = ""
                    session.deferred_final_language = None
                    session.deferred_final_flush_task = None
                    return
                break  # only check the most recent user message

    language = normalize_supported_language(session.deferred_final_language or session.preferred_language or DEFAULT_CALL_LANGUAGE)
    session.deferred_final_text = ""
    session.deferred_final_language = None
    session.deferred_final_flush_task = None

    # Clear stale prefetched replies so the deferred final always uses
    # a fresh LLM call — the prefetch was computed for a different partial.
    await cancel_prefetch_task(session)
    clear_prefetched_reply(session)

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

    session.last_processed_user_text = user_text
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


async def process_transcripts(session: CallSession) -> None:
    try:
        while True:
            item = await session.transcript_queue.get()
            text = (item.get("text") or "").strip()
            if not text:
                continue

            language = normalize_supported_language(item.get("language") or session.preferred_language or DEFAULT_CALL_LANGUAGE)
            source = item.get("source") or "realtime"
            is_final = bool(item.get("is_final"))
            speech_final = bool(item.get("speech_final"))
            if is_final:
                session.preferred_language = language
            session.current_transcript = text

            if is_final and not final_transcript_ready(session, is_final):
                log.info("Final recibido pero usuario sigue hablando, pausar procesamiento: %s", session.session_key)
                session.deferred_final_text = text
                session.deferred_final_language = language
                await cancel_deferred_final_flush(session)
                task = asyncio.create_task(flush_deferred_final_after_grace(session))
                session.deferred_final_flush_task = task
                session.tasks.add(task)
                task.add_done_callback(session.tasks.discard)
                continue

            if is_final:
                # Merge with any pending deferred text.
                if session.deferred_final_text:
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

                # If the text is purely non-actionable (greeting, filler,
                # name mention) don't trigger a turn — just defer it so it
                # can be merged with the real request when it arrives.
                if is_non_actionable_utterance(text):
                    session.deferred_final_text = text
                    session.deferred_final_language = language
                    log.info("Defiriendo final no-accionable en %s: %s", session.session_key, text)
                    await cancel_deferred_final_flush(session)
                    task = asyncio.create_task(flush_deferred_final_after_grace(session))
                    session.deferred_final_flush_task = task
                    session.tasks.add(task)
                    task.add_done_callback(session.tasks.discard)
                    continue

                await process_final_transcript(session, text, language, speech_final=speech_final)
                continue

            # Partial transcripts: no action — wait for the final.
            continue
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
        await run_tts_with_retries(session, prompt, generation)
    except asyncio.TimeoutError:
        log.warning("TTS timeout en fallback STT para %s", session.session_key)
    except Exception:
        log.exception("TTS error en fallback STT para %s", session.session_key)