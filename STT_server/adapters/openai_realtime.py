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
    REALTIME_TTS_STREAMING,
    TEXT_SEGMENT_QUEUE_MAXSIZE,
)
from STT_server.domain.language import (
    extract_structured_data,
    get_language_instruction,
)
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_nowait_with_drop
from STT_server.services.credentials_resolver import resolve_for_session


log = logging.getLogger("stt_server")

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"


# ── Helpers ──────────────────────────────────────────────────────────

def _build_instructions(session: CallSession) -> str:
    """Compose the system instructions including dynamic session state.

    ponytail: dropped the hardcoded Tigo/Camila fallback here too
    (same change as build_messages in openai_llm.py). If the agent
    has no system_prompt, raise — the call shouldn't fall back to
    a generic test prompt.
    """
    custom_prompt = getattr(session, 'custom_prompt', None)
    if not custom_prompt or not custom_prompt.strip():
        raise RuntimeError(
            f"Agent {getattr(session, 'agent_id', '<none>')} has no system_prompt "
            f"configured. Set one in the FE (Agents → Edit → System prompt) "
            f"and redeploy."
        )
    log.info("[REALTIME] Using custom_prompt for session=%s (len=%d)", session.session_key, len(custom_prompt))
    parts = [custom_prompt.strip()]

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


def _build_session_update_payload(session: CallSession) -> str:
    """Serialize the session.update JSON once. Same payload across
    every fallback attempt in the connection loop below.
    """
    return json.dumps({
        "type": "session.update",
        "session": {
            # ponytail: OpenAI Realtime API GA schema. The fields
            # that worked in beta no longer all live at the
            # session root. The current accepted top-level keys
            # are: type, output_modalities, instructions, audio,
            # tools, tool_choice, prompt. Anything else returns
            # invalid_request_error code=unknown_parameter and
            # closes the socket.
            #
            # If OpenAI renames again, the server returns
            # unknown_parameter in the WS error event with a clear
            # `param` field — the dispatcher log below captures it.
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": _build_instructions(session),
            "audio": {
                "input": {
                    # ponytail: OpenAI Realtime GA expects MIME
                    # types for audio format. Supported values:
                    # 'audio/pcm' (16-bit linear), 'audio/pcmu'
                    # (mu-law), 'audio/pcma' (A-law). Twilio Media
                    # Streams sends mu-law so we want 'audio/pcmu'.
                    #
                    # sample_rate is NOT accepted at format.* in
                    # GA — it would be unknown_parameter. The
                    # server infers the rate from the type (audio/pcmu
                    # → 8000 Hz). Don't try to send it explicitly.
                    "format": {"type": "audio/pcmu"},
                    "transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                },
            },
        },
    })


# ── Main entry ───────────────────────────────────────────────────────

async def run_realtime_session(session: CallSession) -> None:
    """Connect to OpenAI Realtime and run for the lifetime of the call."""
    user_id = getattr(session, "user_id", None)
    creds = resolve_for_session(session, "stt", "openai")
    api_key = creds.get("api_key")
    if not api_key:
        log.error(
            "[REALTIME] session %s has no OpenAI API key. User must "
            "upload via Settings → API or ModalAgents inline.",
            session.session_key,
        )
        return

    # ponytail: per-agent model only. The agent row's stt_model is the
    # single source of truth — the FE dropdown now lists exactly the
    # three Realtime-capable IDs (gpt-realtime, gpt-4o-realtime-preview,
    # gpt-4o-mini-realtime-preview). If the agent picked something else,
    # fail loud here so the operator sees which row is misconfigured.
    realtime_per_user = creds.get("realtime_model")
    model = (
        realtime_per_user
        or getattr(session, "stt_model", None)
    )
    if not model:
        log.error(
            "[REALTIME] session %s has no realtime model. agent.stt_model "
            "must be one of gpt-realtime / gpt-4o-realtime-preview / "
            "gpt-4o-mini-realtime-preview (set when editing the agent).",
            session.session_key,
        )
        return
    if realtime_per_user and model != getattr(session, "stt_model", None):
        # ponytail: per-user realtime_model override (set via Settings → API
        # / ModalAgents inline) takes precedence over the agent's stt_model.
        pass  # already handled by the chained or above
    # Validate against the known Realtime catalog. If the agent picked
    # something invalid (legacy field, typo), we don't auto-substitute;
    # we let OpenAI 4004 it and the operator sees the model id in the log.
    _VALID_REALTIME_MODELS = {
        "gpt-realtime",
        "gpt-4o-realtime-preview",
        "gpt-4o-mini-realtime-preview",
    }
    if model not in _VALID_REALTIME_MODELS:
        log.warning(
            "[REALTIME] session %s using non-catalog model=%r — OpenAI "
            "will likely reject it. Update the agent's stt_model or the "
            "user's realtime_model field.",
            session.session_key, model,
        )

    # ponytail: OpenAI has been retiring the `*-preview` Realtime models
    # without bumping the FE. If a saved agent row points at one that
    # OpenAI no longer serves, we transparently fall back to `gpt-realtime`
    # (the GA model). The first attempt uses the picked id; on a 4004
    # model_not_found we reopen the WS with the fallback and re-send the
    # session.update. The picked id is preserved in the agent row so a
    # future OpenAI re-release of the same id will resume using it.
    _FALLBACK_MODEL = "gpt-realtime"
    _attempt_chain = [model]
    if model != _FALLBACK_MODEL and model not in _VALID_REALTIME_MODELS:
        # Unknown id — fall back immediately instead of round-tripping
        # an error to OpenAI.
        _attempt_chain = [_FALLBACK_MODEL]
    elif model != _FALLBACK_MODEL:
        # Known id but OpenAI may have retired it. Save the fallback
        # for after a 4004.
        _attempt_chain.append(_FALLBACK_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        # ponytail: removed `OpenAI-Beta: realtime=v1`. OpenAI graduated
        # the Realtime API to GA in 2024-Q4; the beta header now flips
        # the server into a beta path that's been disabled
        # (`invalid_request_error.beta_api_shape_disabled` → ws close
        # 4000). Dropping the header sends the request to the GA
        # endpoint with the same session.update payload it already
        # accepts.
    }

    try:
        # The session.update payload is identical across attempts — we
        # just rebuild the WS connection. Build it once outside the loop.
        _session_update_payload = _build_session_update_payload(session)

        # Run the fallback chain. Each attempt opens a fresh WS,
        # sends session.update, and either succeeds (we break into
        # the live event loop) or fails with a 4004 model_not_found
        # (we close the WS and retry with the next candidate).
        ws = None
        ws_cm = None
        effective_model = None
        last_rc: int | None = None
        for attempt_model in _attempt_chain:
            url = f"{REALTIME_WS_URL}?model={attempt_model}"
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

                ws_cm = ws_connect
                ws = await ws_cm.__aenter__()
            except (OSError, Exception) as exc:
                log.warning(
                    "[REALTIME] session %s WS open failed for model=%r: %s",
                    session.session_key, attempt_model, exc,
                )
                continue

            try:
                # ponytail: OpenAI closes the WS synchronously on schema
                # errors (rc=4004 model_not_found, rc=4001 unsupported
                # protocol, etc). The close code surfaces on the next
                # `send()` or `recv()`. We use a single broad except
                # (Exception) and inspect the connection state via
                # the protocol's close reason — this is more reliable
                # than relying on the exception class hierarchy across
                # websockets versions.
                try:
                    await asyncio.wait_for(
                        ws.send(_session_update_payload),
                        timeout=5.0,
                    )
                except Exception as send_exc:
                    # Try to read the close code off the connection.
                    last_rc = None
                    try:
                        proto = getattr(ws, "protocol", None) or getattr(ws, "writer", None)
                        if proto is not None:
                            # websockets >= 10: ws.protocol.close_code
                            # websockets <  10: ws.close_code
                            last_rc = getattr(proto, "close_code", None)
                            if last_rc is None:
                                last_rc = getattr(ws, "close_code", None)
                    except Exception:
                        pass
                    # Fallback: re-raise's rc attr.
                    if last_rc is None:
                        last_rc = getattr(send_exc, "rc", None)
                    log.warning(
                        "[REALTIME] session %s send to model=%r failed "
                        "(rc=%s, exc=%s). %s",
                        session.session_key, attempt_model, last_rc,
                        type(send_exc).__name__,
                        "Retrying with fallback."
                        if last_rc == 4004 and attempt_model != _attempt_chain[-1]
                        else "Propagating."
                    )
                    try:
                        await ws_cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                    ws = None
                    ws_cm = None
                    if last_rc == 4004 and attempt_model != _attempt_chain[-1]:
                        continue  # try the fallback model
                    raise  # non-4004 — let the outer except handle it
                # Give the server a beat to send any error frame so we
                # can detect a 4004 that arrives AFTER the session.update
                # is accepted (rare but seen).
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass  # No error frame yet — assume accepted.
                except websockets.exceptions.ConnectionClosed as cc:
                    rc = getattr(cc, "rc", None)
                    if rc == 4004 and attempt_model != _attempt_chain[-1]:
                        log.warning(
                            "[REALTIME] session %s model=%r post-send 4004. "
                            "Retrying with fallback %r.",
                            session.session_key, attempt_model,
                            _attempt_chain[-1],
                        )
                        try:
                            await ws_cm.__aexit__(None, None, None)
                        except Exception:
                            pass
                        ws = None
                        ws_cm = None
                        continue
                    raise
                effective_model = attempt_model
                break
            except Exception:
                # Any other failure during send/receive — close this WS
                # and try next. The outer loop will give up after the
                # fallback also fails.
                try:
                    await ws_cm.__aexit__(None, None, None)
                except Exception:
                    pass
                ws = None
                ws_cm = None
                continue

        if ws is None or effective_model is None:
            log.error(
                "[REALTIME] session %s all model attempts failed "
                "(last rc=%s). Giving up.",
                session.session_key, last_rc,
            )
            return

        log.info(
            "OpenAI Realtime connected for %s model=%s user_id=%s",
            session.session_key,
            effective_model,
            user_id,
        )
        # ponytail: surface the effective model in the session dict
        # so downstream code (latency report, billing) sees what was
        # actually used after the fallback chain.
        try:
            session.stt_model = effective_model
        except Exception:
            pass

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
            try:
                if ws_cm is not None:
                    await ws_cm.__aexit__(None, None, None)
            except Exception:
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
                    log.info(
                        "[OPENAI_REALTIME] transcript final session=%s len=%d lang=%s",
                        session.session_key, len(transcript), session.preferred_language,
                    )
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
                    # ponytail: P3 — push to the central transcript_queue so
                    # process_transcripts sees this final. The
                    # `trigger_llm=False` flag stops it from calling
                    # launch_reply_pipeline (OpenAI Realtime handles the LLM
                    # itself on the same WS). Anti-loop, replace-current,
                    # order-escalation, and memory deduplication all run.
                    #
                    # Note: `enqueue_nowait_with_drop` is imported at
                    # module top (line 28) and reused here. An earlier
                    # version of this block did a local `from ... import`
                    # inside the try — Python treats that as an
                    # assignment to a local name and makes the symbol
                    # local to the entire function, so any earlier call
                    # site (lines 447, 755) raised UnboundLocalError. The
                    # bare reference below resolves to the module-level
                    # import.
                    try:
                        enqueue_nowait_with_drop(
                            session.transcript_queue,
                            {
                                "text": transcript,
                                "is_final": True,
                                "speech_final": True,
                                "language": session.preferred_language,
                                "source": "openai_realtime",
                                "trigger_llm": False,
                            },
                            "transcript_queue",
                        )
                    except Exception:
                        log.exception(
                            "Failed to enqueue openai_realtime transcript to "
                            "transcript_queue for session=%s", session.session_key,
                        )
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

            if etype == "response.output_text.delta":
                # ponytail: OpenAI Realtime GA renamed the text output
                # events from "response.text.delta" to
                # "response.output_text.delta". The old event name is
                # never sent in GA — when output_modalities=["text"],
                # every incremental token arrives as
                # "response.output_text.delta". Without matching this
                # event the TTS pipeline never fires after the initial
                # greeting and the caller hears silence even though
                # transcripts are coming in. Accept BOTH names so a
                # regression to the old shape doesn't break us again.
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
                        # P1: drop-newest when full instead of drop-oldest (was
                        # silently losing the opener of an LLM reply). With
                        # maxsize now 64 this is rare, but when it does happen,
                        # losing the NEWEST unconsumed segment is still better
                        # than losing the OLDEST (still pending consumption).
                        try:
                            tq.put_nowait(seg)
                        except asyncio.QueueFull:
                            log.warning("[text_segment_queue] dropped NEWEST item at %d/%d (queue full)",
                                        tq.qsize(), TEXT_SEGMENT_QUEUE_MAXSIZE)
                continue

            if etype == "response.output_text.done":
                tq = session.realtime_text_queue
                if REALTIME_TTS_STREAMING and tq is not None and pending.strip():
                    from STT_server.domain.language import pop_streaming_segments
                    segments, _ = pop_streaming_segments(pending, force=True)
                    for seg in segments:
                        enqueue_nowait_with_drop(tq, seg, "text_segment_queue")
                    pending = ""
                continue

            # ponytail: keep the legacy event names as fallbacks. If
            # OpenAI rolls back to the old shape (some intermediaries
            # still emit "response.text.delta"), keep TTS alive instead
            # of going silent again.
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
                    log.info(
                        "[OPENAI_REALTIME] assistant reply session=%s len=%d",
                        session.session_key, len(reply),
                    )

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
