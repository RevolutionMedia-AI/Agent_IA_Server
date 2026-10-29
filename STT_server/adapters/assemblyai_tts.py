"""AssemblyAI streaming TTS in Twilio's native 8 kHz mu-law format."""
from __future__ import annotations

import asyncio
import logging
import time

import assemblyai as aai

from STT_server.domain.session import CallSession
from STT_server.services._instrumentation import StageTimer, Stages
from STT_server.services.audio_frame_processor import AudioFrameProcessor
from STT_server.services.credentials_resolver import resolve_for_session

log = logging.getLogger("stt_server")
DEFAULT_MODEL = "assemblyai-tts-1"
DEFAULT_VOICE = "aai-kyla"
LANGUAGES = {
    "en": "english", "english": "english",
    "es": "spanish", "spanish": "spanish",
    "fr": "french", "french": "french",
    "de": "german", "german": "german",
    "it": "italian", "italian": "italian",
    "pt": "portuguese", "portuguese": "portuguese",
}


def _config(session: CallSession) -> aai.TTSConfig:
    language = (getattr(session, "preferred_language", None) or "").lower().split("-")[0].strip()
    return aai.TTSConfig(
        voice_id=getattr(session, "voice_id", None) or DEFAULT_VOICE,
        model_id=getattr(session, "tts_model", None) or DEFAULT_MODEL,
        language=LANGUAGES.get(language),
        sample_rate=8000,
        encoding=aai.TTSEncoding.pcm_mulaw,
        container=aai.TTSContainer.raw,
    )


def fetch_preview(
    text: str,
    api_key: str,
    voice_id: str | None = None,
    model_id: str | None = None,
    language: str | None = None,
) -> bytes:
    """Return AssemblyAI's raw mu-law 8 kHz audio for one preview."""
    lang = (language or "").lower().split("-")[0].strip()
    config = aai.TTSConfig(
        voice_id=voice_id or DEFAULT_VOICE,
        model_id=model_id or DEFAULT_MODEL,
        language=LANGUAGES.get(lang),
        sample_rate=8000,
        encoding=aai.TTSEncoding.pcm_mulaw,
        container=aai.TTSContainer.raw,
    )
    with aai.TTSSession(config=config, api_key=api_key) as tts_session:
        return b"".join(tts_session.speak(text))


async def _get_session(session: CallSession, api_key: str, lock: asyncio.Lock):
    tts_session = getattr(session, "_assemblyai_tts_session", None)
    if tts_session is None:
        tts_session = aai.AsyncTTSSession(_config(session), api_key=api_key)
        await tts_session.__aenter__()
        session._assemblyai_tts_session = tts_session

        async def keep_alive() -> None:
            while not session.closed:
                await asyncio.sleep(60)
                async with lock:
                    await tts_session.keep_alive()

        task = asyncio.create_task(keep_alive(), name=f"assemblyai-tts-keepalive-{session.session_key}")
        session._assemblyai_tts_keepalive = task
        session.tasks.append(task)
    return tts_session


async def close_session(session: CallSession) -> None:
    task = getattr(session, "_assemblyai_tts_keepalive", None)
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session._assemblyai_tts_keepalive = None
    tts_session = getattr(session, "_assemblyai_tts_session", None)
    if tts_session:
        session._assemblyai_tts_session = None
        await tts_session.__aexit__(None, None, None)


async def stream_tts_segment(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
    seg_idx: int = 0,
) -> tuple[float | None, float]:
    creds = resolve_for_session(session, "tts", "assemblyai")
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("AssemblyAI API key not configured.")

    started = time.perf_counter()
    ttfb_ms: float | None = None
    frame_proc = AudioFrameProcessor(emit_silence_tail=False)
    lock = getattr(session, "_assemblyai_tts_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        session._assemblyai_tts_lock = lock

    def emit_audio(data: bytes) -> None:
        nonlocal ttfb_ms
        if not data:
            return
        if ttfb_ms is None:
            ttfb_ms = (time.perf_counter() - started) * 1000
            session._stage_timer = session._stage_timer or StageTimer(
                call_id=session.session_key,
                turn_id=0,
                generation=session.active_generation,
            )
            if Stages.TTS_FIRST_BYTE not in session._stage_timer._stages:
                session._stage_timer.mark(Stages.TTS_FIRST_BYTE)
        for frame in frame_proc.feed(data):
            emit_item({"type": "audio", "generation": generation, "data": frame})

    try:
        async with lock:
            tts_session = await _get_session(session, api_key, lock)
            async for chunk in tts_session.speak(text):
                emit_audio(chunk)
    except asyncio.CancelledError:
        tts_session = getattr(session, "_assemblyai_tts_session", None)
        if tts_session is not None:
            try:
                await tts_session.cancel()
            except Exception:
                log.debug("[ASSEMBLYAI_TTS] cancel failed session=%s", session.session_key)
        raise
    except Exception as exc:
        log.exception("[ASSEMBLYAI_TTS] session failed session=%s", session.session_key)
        emit_item({"type": "error", "generation": generation, "message": f"AssemblyAI TTS error: {exc}"})
        await close_session(session)
    finally:
        frame_proc.flush()
        emit_item({"type": "segment_end", "generation": generation})
    return ttfb_ms, (time.perf_counter() - started) * 1000
