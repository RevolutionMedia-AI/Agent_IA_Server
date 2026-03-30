import audioop
import base64
import logging
import time

import webrtcvad

from STT_server.config import (
    ASSISTANT_ECHO_IGNORE_MS,
    BARGE_IN_MIN_RMS,
    WEBRTC_VAD_MODE,
    DEEPGRAM_API_KEY,
    ENABLE_BARGE_IN,
    END_SILENCE_FRAMES,
    FRAME_DURATION_MS,
    MIN_BARGE_IN_FRAMES,
    MIN_VOICE_RMS,
    PRE_SPEECH_FRAMES,
    SPEECH_START_FRAMES,
    TWILIO_SR,
    USE_OPENAI_REALTIME,
)
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop
from STT_server.services.playback_service import interrupt_current_turn


log = logging.getLogger("stt_server")
vad = webrtcvad.Vad(WEBRTC_VAD_MODE)
FRAME_SAMPLES = int(TWILIO_SR * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


def get_frame_rms(frame: bytes) -> int:
    return audioop.rms(frame, 2)


def is_probable_voice(frame: bytes) -> tuple[bool, int]:
    rms = get_frame_rms(frame)
    return vad.is_speech(frame, TWILIO_SR) and rms >= MIN_VOICE_RMS, rms


async def handle_incoming_media(session: CallSession, media_payload: str) -> None:
    raw = base64.b64decode(media_payload)

    # Log formato de audio recibido
    log.debug(f"[VAD] Audio recibido: len={len(raw)} bytes, sample_rate={TWILIO_SR}, channels=1, frame_dur_ms={FRAME_DURATION_MS}")

    # Route audio to the active STT backend
    if USE_OPENAI_REALTIME:
        target_queue = session.realtime_audio_queue
        queue_name = "realtime_audio_queue"
    elif DEEPGRAM_API_KEY:
        target_queue = session.stt_audio_queue
        queue_name = "stt_audio_queue"
    else:
        target_queue = None
        queue_name = ""

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
        pcm16 = audioop.ulaw2lin(raw, 2)
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

        is_voice, rms = is_probable_voice(frame)
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
                    session.last_activity_at = time.monotonic()
                    session.speech_frames.extend(session.pre_speech_frames)
                    session.speech_frame_count = session.voice_streak
                    session.silence_frames = 0
                    log.info(f"[VAD] INICIO DE VOZ: streak={session.voice_streak}, speech_frame_count={session.speech_frame_count}")
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