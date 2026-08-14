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
    TWILIO_OUTBOUND_CHUNK_BYTES,
    TWILIO_OUTBOUND_PACING_MS,
    SAVE_TWILIO_FRAMES,
)
from STT_server.domain.language import split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.wait_signals import wait_stream_ready
from STT_server.services._instrumentation import Stages
from STT_server.services.audio_frame_processor import AudioFrameProcessor
from STT_server.services.common import drain_queue_nowait, enqueue_nowait_with_drop, enqueue_with_drop
from STT_server.utils.safe_path import UnsafePathError, sanitize_id
# RNNoise removed: playback sends mu-law frames directly to Twilio.


log = logging.getLogger("stt_server")


def emit_playback_item(session: CallSession, item: dict) -> bool:
    log.debug("[PLAYBACK] Enqueue playback item: session=%s type=%s gen=%s bytes=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'), len(item.get('data', b'')) if 'data' in item else '-')
    # ponytail: 2026-08-14 audio review — A/B-test capture.
    # Write the exact μ-law bytes the TTS adapter produced
    # (post-resample, post-μ-law encode) to the A file. This is
    # the single chokepoint every TTS adapter routes through, so
    # one write covers all providers (elevenlabs / rime / inworld
    # / openai / deepgram). The companion B capture (exact 160-byte
    # frames going to Twilio) lives in playback_loop right before
    # send_twilio_media — diff A against B against the AMR recording
    # to locate which stage introduces the artifacts.
    if item.get("type") == "audio":
        data = item.get("data") or b""
        if data:
            from STT_server.services.audio_capture import capture_a
            capture_a(getattr(session, "call_sid", "") or "", data)
    ok = enqueue_nowait_with_drop(session.playback_queue, item, "playback_queue")
    if not ok:
        log.warning("[PLAYBACK] Failed to enqueue playback item (queue full): session=%s type=%s gen=%s", getattr(session, 'session_key', '?'), item.get('type'), item.get('generation'))
    return ok


def enqueue_playback_clear(session: CallSession) -> None:
    """M2: was async with `await enqueue_with_drop`, but the body is sync
    (enqueue_with_drop just wraps a put_nowait). The await was a no-op
    that confused every caller. Now plain sync."""
    # ponytail: hotfix — call the SYNC variant directly. The previous
    # code invoked the async `enqueue_with_drop` without `await`,
    # producing a RuntimeWarning every barge-in AND silently dropping
    # the clear item (the coroutine was never awaited so the queue
    # was never actually cleared). enqueue_nowait_with_drop does the
    # same work in a non-async wrapper.
    enqueue_nowait_with_drop(
        session.playback_queue,
        {"type": "clear", "generation": session.active_generation},
        "playback_queue",
    )


async def interrupt_current_turn(session: CallSession) -> None:
    """Barge-in: invalidate every in-flight TTS item by bumping
    ``active_generation`` and advancing ``cancelled_through`` to the
    bumped-1 value.

    Convention (Phase-2 refactor):
      - Every item pushed to ``playback_queue`` carries
        ``{"generation": <active_generation_at_enqueue>}``.
      - ``playback_loop`` drops any item where
        ``generation != session.active_generation`` OR
        ``generation <= session.cancelled_through``.
      - Adapters (inworld_tts, rime_tts, openai_tts, etc.) MUST read
        ``session.active_generation`` at each emit — NOT at task spawn
        time — so producer threads that can't be cancelled mid-stream
        still tag their late frames with the live generation and the
        consumer drops them silently.
    """
    session.active_generation += 1
    session.cancelled_through = max(session.cancelled_through, session.active_generation - 1)
    session.barge_in_at = time.monotonic()
    log.info(
        "[BARGE-IN] session=%s active_gen=%s cancelled_through=%s",
        session.session_key, session.active_generation, session.cancelled_through,
    )
    # TODO other adapters should check session.active_generation at each emit
    # (inworld_tts.py, rime_tts.py, openai_tts.py — out of scope for this phase).

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
    # ponytail: AUDIO-005 — clear any in-flight speech_frames too.
    # They predate the barge-in and belong to the cancelled turn;
    # letting them persist would mix frames across turns the next
    # time INICIO DE VOZ fires.
    session.speech_frames.clear()
    session._speech_frames_cap_warned = False
    # ponytail: AUDIO echo gate — reset the per-generation
    # playback-marks counter on barge-in. The previous generation's
    # segment_ends are already drained by the next line; if any
    # leaked through (race with the playback loop) their Twilio
    # marks will arrive, get popped from ``pending_marks``, and
    # decrement the counter below zero — that's fine, the
    # ``max(0, ...)`` floor in the mark-ack handler keeps the metric
    # honest. We must reset to zero on barge-in so the next turn's
    # ack-count starts fresh.
    session.pending_playback_marks = 0
    drain_queue_nowait(session.playback_queue)
    enqueue_playback_clear(session)
    session.generation_changed.set()


async def play_initial_greeting(session: CallSession) -> None:
    """Speak the initial greeting as soon as the call connects.

    ponytail: 2026-08-14 — the user reported "the initial greeting
    prompt isn't firing". Root cause: the previous version required
    ``session.welcome_message`` to be set on the agent row; agents
    without one configured produced dead-air silence until the
    caller spoke. Fix: pick the greeting text with this priority —
      1. ``session.welcome_message`` (per-agent override, if set)
      2. ``INITIAL_GREETING_TEXT_ES`` / ``_EN`` (platform fallback,
         picked by the call's preferred_language)
      3. None of the above → no-op (silent start, preserved for
         agents that explicitly opt out via INITIAL_GREETING_ENABLED=false)

    Sent DIRECTLY to TTS via ``run_tts_with_retries`` — no STT,
    no LLM intermediate. The caller hears the agent greet them
    within the TTS provider's TTFB (typically 200-300 ms for
    Inworld, with the previous warmup path adding a pre-cached
    file we can drop in).

    The caller-side perception is: connect → silence ≤ 300 ms →
    agent greets. No more dead-air wait-for-the-customer-to-speak.
    """
    # Per-agent override wins (lets the agent personalize the
    # greeting without touching env vars).
    welcome = getattr(session, 'welcome_message', None)
    greeting: str | None = None
    greeting_source = ""

    if welcome and welcome.strip():
        greeting = welcome.strip()
        greeting_source = "agent.welcome_message"
    else:
        # ponytail: platform fallback. The agent's preferred_language
        # decides which variant; default to ES because most operator
        # deployments on this platform target es-419 customers.
        from STT_server.config import (
            INITIAL_GREETING_ENABLED,
            INITIAL_GREETING_TEXT_EN,
            INITIAL_GREETING_TEXT_ES,
        )
        if not INITIAL_GREETING_ENABLED:
            log.debug(
                "[PLAYBACK] play_initial_greeting skipped "
                "(INITIAL_GREETING_ENABLED=false and no agent welcome_message) session=%s",
                session.session_key,
            )
            return
        lang = (getattr(session, 'preferred_language', None) or "es").strip().lower()
        if lang.startswith("en"):
            greeting = INITIAL_GREETING_TEXT_EN or None
            greeting_source = "INITIAL_GREETING_TEXT_EN"
        else:
            greeting = INITIAL_GREETING_TEXT_ES or None
            greeting_source = "INITIAL_GREETING_TEXT_ES"

    if not greeting:
        log.debug(
            "[PLAYBACK] play_initial_greeting skipped (no greeting text resolved) session=%s",
            session.session_key,
        )
        return

    log.info(
        "[PLAYBACK] playing initial greeting (source=%s, %d chars, lang=%s) for session=%s",
        greeting_source,
        len(greeting),
        getattr(session, 'preferred_language', None) or "?",
        session.session_key,
    )
    # Mark the agent as speaking NOW (before the TTS provider responds).
    # Without this the idle monitor could fire during the TTFB window
    # (some TTS providers take 1-2 s to first byte), think the user is
    # silent, and hang up the call.
    session.assistant_speaking = True
    session.assistant_started_at = time.perf_counter()
    session.last_activity_at = time.monotonic()
    # P0: replaced hardcoded 0.4s sleep with event-based wait for the
    # stream_sid_ready signal. If the start event already arrived the
    # wait returns instantly; if not, it returns the moment Twilio sends
    # it (or a timeout, configurable via STREAM_SID_WAIT_TIMEOUT_MS).
    await wait_stream_ready(session)
    # Generate the TTS via the session's configured TTS provider. The
    # playback_loop will pick up the queued audio and stream it to Twilio.
    from STT_server.services.turn_manager import run_tts_with_retries
    try:
        await run_tts_with_retries(session, greeting, session.active_generation)
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
    # ponytail: AudioFrameProcessor is the defensive consumer-side
    # framer. TTS adapters pre-frame their bytes (one 160-byte item
    # per emit), so feed() here is a no-op for aligned input. It
    # guarantees Twilio only ever sees 160-byte mu-law frames even
    # if a non-own-file adapter bypasses the producer-side framing.
    # emit_silence_tail=False drops the partial trailing frame at
    # call end (boundary click fix).
    frame_proc = AudioFrameProcessor(emit_silence_tail=False)
    # ponytail: AUDIO-001 + AUDIO-004 — stash the framer on the
    # session so the per-call summary at cleanup_session time can
    # surface bytes_in / frames_out / padded_tail_frames /
    # dropped_tail_bytes. The audit says these need measurement
    # before sizing policy; we cannot size without observing.
    session._playback_frame_proc = frame_proc
    first_frame_marked = False
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

            # ponytail: stale-frame drop. Even if a producer thread
            # hasn't yet seen the generation bump (HTTP TTS mid-stream
            # bytes landing here), any item tagged at-or-below
            # cancelled_through is from a barge-in'd turn. Drop silently.
            if generation <= session.cancelled_through:
                continue

            if item_type == "audio":
                if not session.stream_sid:
                    # P0: replaced 5s polling loop with event-based wait.
                    # stream_sid_ready event is set by the Twilio 'start'
                    # handler in STT_Server.py the instant Twilio sends it.
                    await wait_stream_ready(session)
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
                # ponytail: feed the chunk through AudioFrameProcessor.
                # TTS adapters pre-frame to 160-byte items, so this
                # returns one frame per emit. Any non-aligned bytes
                # (e.g. legacy adapters that bypassed framing) get
                # buffered for the next chunk instead of emitted as
                # short packets — that was the source of the <20ms
                # boundary click.
                sent_frames = 0
                for frame in frame_proc.feed(chunk):
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

                    if not first_frame_marked:
                        # ponytail: instrument — first 160-byte frame
                        # crossing the WS boundary is the playback TTFB.
                        timer = getattr(session, "_stage_timer", None)
                        if timer is not None:
                            timer.mark(Stages.FIRST_160_FRAME_SENT)
                        first_frame_marked = True

                    send_start = time.perf_counter()
                    # ponytail: 2026-08-14 audio review — B capture.
                    # Write the exact 160-byte μ-law frame about to be
                    # base64-encoded + sent on the WS. Pair with the A
                    # capture (TTS-adapter output bytes) at
                    # emit_playback_item. Diff A vs B vs the AMR
                    # recording (Twilio / carrier) to localize the
                    # artifact. No-op when TTS_AUDIO_CAPTURE_DIR is
                    # empty (the default). Best-effort: a write
                    # failure logs once and disables capture for
                    # this call, NEVER blocks the WS send.
                    if frame:
                        from STT_server.services.audio_capture import capture_b
                        capture_b(getattr(session, "call_sid", "") or "", frame)
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
                    # ponytail: AUDIO-008 — record pacing drift (how much
                    # wall-clock the send+loop took beyond the desired
                    # 20ms-per-frame budget) as a latency observation.
                    # Operators reading the per-call summary can see
                    # p50/p99 drift; under load this is the signal that
                    # pacing is the bottleneck, not the WS send.
                    drift_ms = elapsed * 1000.0
                    _metrics = getattr(session, "metrics", None)
                    if _metrics is not None:
                        try:
                            _metrics.observe_ms("pacing_drift_ms", drift_ms)
                        except Exception:
                            pass
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
                    # TODO instrumentation:mark_ack — Twilio 'mark' ack handler
                    # lives in STT_Server.py:822, which is outside this agent's
                    # owned file set. When that file gains the
                    # Stages.TWILIO_MARK_ACK instrumentation, mark the timer
                    # via getattr(session, "_stage_timer", None).mark(
                    #     Stages.TWILIO_MARK_ACK,
                    # ) in the mark-ack branch.
                    try:
                        session.mark_counter += 1
                        mark_name = f"gen-{generation}-seg-{session.mark_counter}"
                        session.pending_marks[mark_name] = time.monotonic()
                        # ponytail: AUDIO echo gate — only count this
                        # segment toward the per-generation "still
                        # playing" budget if the segment belongs to
                        # the CURRENTLY ACTIVE generation. Stale
                        # segment_ends from a cancelled turn (drained
                        # by interrupt_current_turn but still race-
                        # processed before the drain landed) must
                        # NOT bump the counter, otherwise the ack
                        # logic below waits forever for a mark that
                        # Twilio will never send.
                        if generation == session.active_generation:
                            session.pending_playback_marks += 1
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