import audioop
import base64
import logging
import time

import webrtcvad

from STT_server.config import (
    ASSISTANT_ECHO_IGNORE_MS,
    BARGE_IN_MIN_RMS,
    DEEPGRAM_API_KEY,
    ENABLE_BARGE_IN,
    END_SILENCE_FRAMES,
    FRAME_DURATION_MS,
    MIN_SPEECH_FRAMES,
    MIN_BARGE_IN_FRAMES,
    MIN_VOICE_RMS,
    PRE_SPEECH_FRAMES,
    SPEECH_START_FRAMES,
    TRIM_TRAILING_SILENCE_FRAMES,
    TWILIO_SR,
)
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop
from STT_server.services.playback_service import interrupt_current_turn


log = logging.getLogger("stt_server")
vad = webrtcvad.Vad(2)
FRAME_SAMPLES = int(TWILIO_SR * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


def get_frame_rms(frame: bytes) -> int:
    return audioop.rms(frame, 2)


def is_probable_voice(frame: bytes) -> tuple[bool, int]:
    rms = get_frame_rms(frame)
    return vad.is_speech(frame, TWILIO_SR) and rms >= MIN_VOICE_RMS, rms


async def handle_incoming_media(session: CallSession, media_payload: str) -> None:
    raw = base64.b64decode(media_payload)
    # Don't feed audio to STT while the agent is speaking — prevents
    # Deepgram from hallucinating on TTS echo / background noise.
    if DEEPGRAM_API_KEY and not session.assistant_speaking:
        await enqueue_with_drop(session.stt_audio_queue, raw, "stt_audio_queue")
    pcm16 = audioop.ulaw2lin(raw, 2)
    session.vad_buffer.extend(pcm16)

    buf = session.vad_buffer
    offset = 0
    buf_len = len(buf)

    while buf_len - offset >= FRAME_BYTES:
        frame = bytes(buf[offset:offset + FRAME_BYTES])
        offset += FRAME_BYTES

        is_voice, rms = is_probable_voice(frame)
        session.pre_speech_frames.append(frame)

        assistant_recently_started = (
            session.assistant_speaking
            and session.assistant_started_at is not None
            and (time.perf_counter() - session.assistant_started_at) * 1000.0 < ASSISTANT_ECHO_IGNORE_MS
        )

        if session.assistant_speaking and not ENABLE_BARGE_IN:
            session.voice_streak = 0
            session.silence_frames = 0
            session.speech_frames.clear()
            session.speech_frame_count = 0
            continue

        if not session.speech_frames:
            if is_voice:
                session.voice_streak += 1
                if (
                    ENABLE_BARGE_IN
                    and session.assistant_speaking
                    and not assistant_recently_started
                    and rms >= BARGE_IN_MIN_RMS
                    and session.voice_streak >= MIN_BARGE_IN_FRAMES
                ):
                    if session.assistant_started_at and (time.perf_counter() - session.assistant_started_at) >= 0.6:
                        log.info("Barge-in detectado en %s rms=%s streak=%s", session.session_key, rms, session.voice_streak)
                        await interrupt_current_turn(session)

                if not session.assistant_speaking and session.voice_streak >= SPEECH_START_FRAMES:
                    session.last_activity_at = time.monotonic()
                    session.speech_frames.extend(session.pre_speech_frames)
                    session.speech_frame_count = session.voice_streak
                    session.silence_frames = 0
            else:
                session.voice_streak = 0
            continue

        if is_voice:
            session.voice_streak += 1
            session.silence_frames = 0
            session.speech_frames.append(frame)
            session.speech_frame_count += 1
        else:
            session.voice_streak = 0
            session.speech_frames.append(frame)
            session.silence_frames += 1

        if session.speech_frames and session.silence_frames >= END_SILENCE_FRAMES:
            trimmed_frames = list(session.speech_frames)
            if TRIM_TRAILING_SILENCE_FRAMES > 0 and len(trimmed_frames) > TRIM_TRAILING_SILENCE_FRAMES:
                trimmed_frames = trimmed_frames[:-TRIM_TRAILING_SILENCE_FRAMES]

            if session.speech_frame_count >= MIN_SPEECH_FRAMES and trimmed_frames:
                session.awaiting_local_final = True
                session.pending_realtime_final = None
                await enqueue_with_drop(
                    session.utterance_queue,
                    (session.active_generation, b"".join(trimmed_frames)),
                    "utterance_queue",
                )

            session.speech_frames.clear()
            session.pre_speech_frames.clear()
            session.silence_frames = 0
            session.speech_frame_count = 0

    # Compact: remove consumed bytes in one operation instead of per-frame
    if offset > 0:
        del buf[:offset]