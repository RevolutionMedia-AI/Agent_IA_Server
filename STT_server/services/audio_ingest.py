import base64
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
    MIN_BARGE_IN_FRAMES,
    MIN_VOICE_RMS,
    PRE_SPEECH_FRAMES,
    SPEECH_START_FRAMES,
    TWILIO_SR,
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


async def is_probable_voice(frame: bytes) -> tuple[bool, int]:
    """M4: VAD + RMS are CPU-bound (audio_codec + webrtcvad). At 50 fps per call
    and 10 concurrent calls that's ~1.5 ms of CPU per frame x 500 fps = 750 ms
    of CPU per second on one core, blocking the event loop. Offload to
    the default thread pool so the WebSocket handler stays responsive.
    """
    return await _to_thread(_is_probable_voice_sync, frame)


async def handle_incoming_media(session: CallSession, media_payload: str) -> None:
    raw = base64.b64decode(media_payload)

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
        else:
            if session.stt_mute_buffer:
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
                    # session._stage_timer is a runtime-attached per-call
                    # StageTimer used by the latency dashboard to measure
                    # STT -> LLM -> TTS stage deltas. It is created lazily
                    # on the first non-empty utterance detection so we
                    # don't pay the cost for calls that never speak, and
                    # the mark() below is guarded so only the FIRST
                    # INICIO DE VOZ event stamps Stages.STT_FIRST_RESULT
                    # (subsequent ones are no-ops, so the timeline keeps
                    # the true first-result timestamp).
                    session._stage_timer = session._stage_timer or StageTimer(
                        call_id=session.session_key,
                        turn_id=0,
                        generation=session.active_generation,
                    )
                    if Stages.STT_FIRST_RESULT not in session._stage_timer._stages:
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
            session.speech_frames.append(frame)
            session.speech_frame_count += 1
            log.debug(f"[VAD] Continuando voz: speech_frame_count={session.speech_frame_count}")
        else:
            session.voice_streak = 0
            session.speech_frames.append(frame)
            session.silence_frames += 1
            log.debug(f"[VAD] Silencio: silence_frames={session.silence_frames}")

        if session.speech_frames and session.silence_frames >= END_SILENCE_FRAMES:
            log.info(f"[VAD] FIN DE VOZ: speech_frame_count={session.speech_frame_count}, silence_frames={session.silence_frames}")
            session.speech_frames.clear()
            session.pre_speech_frames.clear()
            session.silence_frames = 0
            session.speech_frame_count = 0

    # Compact: remove consumed bytes in one operation instead of per-frame
    if offset > 0:
        del buf[:offset]