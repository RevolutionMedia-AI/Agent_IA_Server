import base64
import binascii
import logging
import time

import webrtcvad

from STT_server.config import (
    ASSISTANT_ECHO_IGNORE_MS,
    BARGE_IN_MIN_RMS,
    WEBRTC_VAD_MODE,
    ENABLE_BARGE_IN,
    END_SILENCE_FRAMES,
    FRAME_DURATION_MS,
    MAX_MEDIA_PAYLOAD_BYTES,
    MIN_BARGE_IN_FRAMES,
    MIN_VOICE_RMS,
    PRE_SPEECH_FRAMES,
    SPEECH_FRAMES_MAX,
    SPEECH_START_FRAMES,
    TWILIO_SR,
    VAD_BUFFER_MAX_BYTES,
)
from STT_server.domain.session import CallSession
from STT_server.services import audio_codec
from STT_server.services._instrumentation import StageTimer, Stages
from STT_server.services.common import enqueue_with_drop
from STT_server.services.playback_service import interrupt_current_turn
from STT_server.services.thread_pool import to_thread as _to_thread


log = logging.getLogger("stt_server")
vad = webrtcvad.Vad(WEBRTC_VAD_MODE)
FRAME_SAMPLES = int(TWILIO_SR * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


def get_frame_rms(frame: bytes) -> int:
    return audio_codec.rms(frame, 2)


def _is_probable_voice_sync(frame: bytes) -> tuple[bool, int]:
    rms = get_frame_rms(frame)
    return vad.is_speech(frame, TWILIO_SR) and rms >= MIN_VOICE_RMS, rms


# ponytail: AUDIO-005 — cap gate for speech_frames. A runaway
# utterance (sustained noise, very long monologue) could otherwise
# grow the list unbounded for the duration of a call. Drop the
# tail with a counter + one WARNING per session so the operator
# sees when the cap fires without log spam.
def _append_speech_frame(session: CallSession, frame: bytes) -> None:
    if len(session.speech_frames) >= SPEECH_FRAMES_MAX:
        metrics = getattr(session, "metrics", None)
        if metrics is not None:
            metrics.incr("speech_frames_capped_total", 1)
        if not getattr(session, "_speech_frames_cap_warned", False):
            log.warning(
                "[VAD] speech_frames cap reached (%d) on session=%s; "
                "further frames dropped until next END_SILENCE",
                SPEECH_FRAMES_MAX, session.session_key,
            )
            session._speech_frames_cap_warned = True
        return
    session.speech_frames.append(frame)


async def is_probable_voice(frame: bytes) -> tuple[bool, int]:
    """M4: VAD + RMS are CPU-bound (audio_codec + webrtcvad). At 50 fps per call
    and 10 concurrent calls that's ~1.5 ms of CPU per frame x 500 fps = 750 ms
    of CPU per second on one core, blocking the event loop. Offload to
    the default thread pool so the WebSocket handler stays responsive.
    """
    return await _to_thread(_is_probable_voice_sync, frame)


async def handle_incoming_media(session: CallSession, media_payload: str) -> None:
    # AUDIO-006: bound per-event memory before decoding. b64 inflates
    # ~4/3, so a 4/3-multiple of MAX_MEDIA_PAYLOAD_BYTES catches the
    # ceiling without ever allocating the decoded buffer.
    if len(media_payload) > MAX_MEDIA_PAYLOAD_BYTES * 4 // 3:
        log.warning(
            "[MEDIA] oversized base64 payload: len=%d cap=%d session=%s",
            len(media_payload), MAX_MEDIA_PAYLOAD_BYTES, session.session_key,
        )
        metrics = getattr(session, "metrics", None)
        if metrics is not None:
            metrics.incr("oversized_media_payload_total", 1)
        return
    try:
        raw = base64.b64decode(media_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        log.warning(
            "[MEDIA] invalid base64 payload: %s session=%s",
            exc, session.session_key,
        )
        metrics = getattr(session, "metrics", None)
        if metrics is not None:
            metrics.incr("invalid_base64_total", 1)
        return

    # Log formato de audio recibido
    log.debug(f"[VAD] Audio recibido: len={len(raw)} bytes, sample_rate={TWILIO_SR}, channels=1, frame_dur_ms={FRAME_DURATION_MS}")

    # Route audio to the active STT backend
    # ponytail: per-session routing. session.stt_provider is set in
    # STT_Server.py from the agent config (with USE_OPENAI_REALTIME as
    # the default when unset). CallSession sets the attribute lazily
    # so we guard with getattr for sessions that pre-date the change.
    # Deepgram, Inworld, and any other bidirectional-WS STT share
    # stt_audio_queue and the per-adapter conversion happens inside
    # the adapter (mulaw -> LINEAR16 for Inworld, mulaw passthrough
    # for Deepgram).
    # ponytail: the agent row stores the FE-facing label 'openai' for
    # the OpenAI Realtime STT (the canonical id 'openai_realtime' is
    # only an internal alias). The STT dispatch in STT_Server.py
    # accepts BOTH ('openai_realtime', 'openai'). audio_ingest had a
    # strict equality check that only matched 'openai_realtime' — so
    # every agent with stt_provider='openai' had its audio misrouted
    # to stt_audio_queue (where Deepgram/Inworld adapters consume),
    # NOT to realtime_audio_queue (where the OpenAI Realtime adapter
    # consumes). OpenAI got silence → no transcripts → no LLM
    # response → caller heard silence after the greeting. The fix is
    # to mirror the dispatch check verbatim.
    stt_provider = getattr(session, "stt_provider", "")
    if stt_provider in ("openai_realtime", "openai"):
        target_queue = session.realtime_audio_queue
        queue_name = "realtime_audio_queue"
    else:
        target_queue = session.stt_audio_queue
        queue_name = "stt_audio_queue"

    if target_queue is not None:
        if session.assistant_speaking:
            session.stt_mute_buffer.append(raw)
            # ponytail: AUDIO-002 — observe the mute-buffer high-water
            # mark so the audit's "echo buffer re-injects phantom
            # transcripts after mark" hypothesis can be confirmed with
            # data. Cheap counter, no behavioural change: every chunk
            # appended during assistant_speaking is one entry; if the
            # agent's reply is short and the mark fires quickly, only
            # a handful of chunks ever accumulate. If callers
            # consistently report echo, the gauge + the drop count
            # together pinpoint it.
            _m = getattr(session, "metrics", None)
            if _m is not None:
                try:
                    _m.gauge("mute_buffer_depth", float(len(session.stt_mute_buffer)))
                except Exception:
                    pass
        else:
            if session.stt_mute_buffer:
                # ponytail: AUDIO-002 — count re-injections separately
                # from fresh enqueues so we can distinguish "echo
                # playback" from normal inbound. The previous code
                # drained the buffer with the same enqueue helper as
                # the fresh chunk, which is why nobody could tell.
                _m = getattr(session, "metrics", None)
                if _m is not None:
                    try:
                        _m.incr("mute_buffer_reinjected_chunks", len(session.stt_mute_buffer))
                    except Exception:
                        pass
                for buffered_chunk in session.stt_mute_buffer:
                    await enqueue_with_drop(target_queue, buffered_chunk, queue_name)
                session.stt_mute_buffer.clear()
            await enqueue_with_drop(target_queue, raw, queue_name)

    # Conversión y chequeo de formato
    try:
        pcm16 = audio_codec.ulaw2lin(raw, 2)
        # Verifica que el audio sea mono y 16-bit
        if len(pcm16) % 2 != 0:
            log.warning(f"[VAD] Audio PCM16 no tiene longitud par: {len(pcm16)}")
    except Exception as e:
        log.error(f"[VAD] Error al convertir audio a PCM16: {e}")
        return

    session.vad_buffer.extend(pcm16)
    # AUDIO-005: defensive cap on the per-call vad_buffer. The downstream
    # while-loop drains `FRAME_BYTES`-sized frames and compacts via
    # `del buf[:offset]`, so the buffer should never grow past a couple
    # of seconds of audio. If the loop fails to drain (e.g. an oversized
    # payload arrived and is stuck in VAD), trim the oldest bytes so a
    # misbehaving caller can't grow memory until cleanup_session runs.
    # ponytail: global cap, drop-oldest on overflow, counter for metrics.
    if len(session.vad_buffer) > VAD_BUFFER_MAX_BYTES:
        metrics = getattr(session, "metrics", None)
        if metrics is not None:
            try:
                metrics.incr("vad_buffer_overflow")
            except Exception:
                pass
        del session.vad_buffer[: len(session.vad_buffer) - VAD_BUFFER_MAX_BYTES]

    buf = session.vad_buffer
    offset = 0
    buf_len = len(buf)

    while buf_len - offset >= FRAME_BYTES:
        frame = bytes(buf[offset:offset + FRAME_BYTES])
        offset += FRAME_BYTES

        is_voice, rms = await is_probable_voice(frame)
        log.debug(f"[VAD] Frame: offset={offset}, rms={rms}, is_voice={is_voice}")
        session.pre_speech_frames.append(frame)

        assistant_recently_started = (
            session.assistant_speaking
            and session.assistant_started_at is not None
            and (time.perf_counter() - session.assistant_started_at) * 1000.0 < ASSISTANT_ECHO_IGNORE_MS
        )

        if session.assistant_speaking and not ENABLE_BARGE_IN:
            log.debug(f"[VAD] Ignorando voz porque el asistente está hablando (sin barge-in)")
            session.voice_streak = 0
            session.silence_frames = 0
            session.speech_frames.clear()
            session.speech_frame_count = 0
            continue

        if not session.speech_frames:
            if is_voice:
                session.voice_streak += 1
                log.debug(f"[VAD] Detected voice streak={session.voice_streak}")
                if (
                    ENABLE_BARGE_IN
                    and session.assistant_speaking
                    and not assistant_recently_started
                    and session.voice_streak >= MIN_BARGE_IN_FRAMES
                ):
                    # Compute average RMS across the most recent frames to avoid
                    # single-frame spikes causing false barge-in triggers.
                    try:
                        recent_frames = list(session.pre_speech_frames)[-MIN_BARGE_IN_FRAMES:]
                        if recent_frames:
                            total_r = sum(get_frame_rms(f) for f in recent_frames)
                            avg_rms = int(total_r / len(recent_frames))
                        else:
                            avg_rms = rms
                    except Exception as e:
                        log.debug("[VAD] Error computing avg_rms for barge-in: %s", e)
                        avg_rms = rms

                    if avg_rms >= BARGE_IN_MIN_RMS:
                        if session.assistant_started_at and (time.perf_counter() - session.assistant_started_at) >= 0.6:
                            log.info("Barge-in detectado en %s avg_rms=%s streak=%s", session.session_key, avg_rms, session.voice_streak)
                            await interrupt_current_turn(session)

                if not session.assistant_speaking and session.voice_streak >= SPEECH_START_FRAMES:
                    # session._stage_timer is a runtime-attached per-turn
                    # StageTimer used by the latency dashboard to measure
                    # STT -> LLM -> TTS stage deltas. The previous version
                    # created the timer once-per-session and never reset
                    # it, which meant every turn after the first one
                    # showed ``stt_first_result`` stamped at the FIRST
                    # turn's wall-clock time (e.g. 8383ms across the
                    # entire call), and downstream deltas drifted negative
                    # as monotonic time advanced (e.g.
                    # ``first_160_frame_sent=-41049.339ms`` in 2026-08-14
                    # logs). The fix: build a fresh timer at every new
                    # turn boundary. The OLD timer is flushed to the
                    # log first so its timeline isn't silently dropped.
                    _old_timer = getattr(session, "_stage_timer", None)
                    if _old_timer is not None:
                        try:
                            log.info("[stage_timer] previous turn: %s", _old_timer.to_log_line())
                        except Exception:
                            pass
                    session._stage_timer = StageTimer(
                        call_id=session.session_key,
                        turn_id=session.active_generation,
                        generation=session.active_generation,
                    )
                    session._stage_timer.mark(Stages.STT_FIRST_RESULT)
                    session.last_activity_at = time.monotonic()
                    session.speech_frames.extend(session.pre_speech_frames)
                    session.speech_frame_count = session.voice_streak
                    session.silence_frames = 0
                    log.info(f"[VAD] INICIO DE VOZ: streak={session.voice_streak}, speech_frame_count={session.speech_frame_count}")
                    barge_in_gap = (
                        f" barge_in_gap={(time.monotonic() - session.barge_in_at):.2f}s"
                        if session.barge_in_at is not None
                        else ""
                    )
                    log.info(
                        "[VAD] new user turn: session=%s active_gen=%s%s",
                        session.session_key, session.active_generation, barge_in_gap,
                    )
            else:
                session.voice_streak = 0
            continue

        if is_voice:
            session.voice_streak += 1
            session.silence_frames = 0
            _append_speech_frame(session, frame)
            session.speech_frame_count += 1
            log.debug(f"[VAD] Continuando voz: speech_frame_count={session.speech_frame_count}")
        else:
            session.voice_streak = 0
            _append_speech_frame(session, frame)
            session.silence_frames += 1
            log.debug(f"[VAD] Silencio: silence_frames={session.silence_frames}")

        if session.speech_frames and session.silence_frames >= END_SILENCE_FRAMES:
            log.info(f"[VAD] FIN DE VOZ: speech_frame_count={session.speech_frame_count}, silence_frames={session.silence_frames}")
            session.speech_frames.clear()
            session.pre_speech_frames.clear()
            session.silence_frames = 0
            session.speech_frame_count = 0
            session._speech_frames_cap_warned = False

    # Compact: remove consumed bytes in one operation instead of per-frame
    if offset > 0:
        del buf[:offset]