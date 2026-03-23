import asyncio
import audioop
import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
import webrtcvad
from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from openai import OpenAI

from fastapi import FastAPI, Request
from fastapi.responses import Response

print("ARCHIVO EN USO:", __file__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stt_server")

PORT = int(os.environ.get("PORT", 8080))

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "entornoLocal.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_STT_PUNCTUATE = os.getenv("DEEPGRAM_STT_PUNCTUATE", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_SMART_FORMAT = os.getenv("DEEPGRAM_STT_SMART_FORMAT", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_DETECT_LANGUAGE = os.getenv("DEEPGRAM_STT_DETECT_LANGUAGE", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_LANGUAGE_HINT = os.getenv("DEEPGRAM_STT_LANGUAGE_HINT", os.getenv("WHISPER_LANGUAGE_HINT", "")).strip().lower() or None
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_TTS_ENCODING = os.getenv("DEEPGRAM_TTS_ENCODING", "mulaw").lower()
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "8000"))

PUBLIC_URL = os.getenv("PUBLIC_URL")

if not PUBLIC_URL:
    raise RuntimeError("Define PUBLIC_URL en las variables de entorno")

if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY no configurada.")

if not DEEPGRAM_API_KEY:
    log.warning("DEEPGRAM_API_KEY no configurada. El STT y el TTS no estaran disponibles.")

if DEEPGRAM_TTS_ENCODING != "mulaw":
    log.warning("Forzando salida TTS en mulaw para Twilio Media Streams.")
    DEEPGRAM_TTS_ENCODING = "mulaw"

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()

TWILIO_SR = 8000
TWILIO_CHANNELS = 1

FRAME_DURATION_MS = 30
FRAME_SAMPLES = int(TWILIO_SR * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2
MIN_UTTERANCE_MS = int(os.getenv("MIN_UTTERANCE_MS", "240"))
MIN_UTTERANCE_BYTES = int(TWILIO_SR * (MIN_UTTERANCE_MS / 1000.0) * 2)
MIN_SPEECH_FRAMES = int(os.getenv("MIN_SPEECH_FRAMES", "5"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "6"))
SPEECH_START_FRAMES = int(os.getenv("SPEECH_START_FRAMES", "2"))
MIN_BARGE_IN_FRAMES = int(os.getenv("MIN_BARGE_IN_FRAMES", "6"))
PRE_SPEECH_FRAMES = int(os.getenv("PRE_SPEECH_FRAMES", "5"))
TRIM_TRAILING_SILENCE_FRAMES = int(os.getenv("TRIM_TRAILING_SILENCE_FRAMES", "4"))
MIN_VOICE_RMS = int(os.getenv("MIN_VOICE_RMS", "260"))
BARGE_IN_MIN_RMS = int(os.getenv("BARGE_IN_MIN_RMS", "700"))
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "false").strip().lower() in {"1", "true", "yes", "on"}
ASSISTANT_ECHO_IGNORE_MS = float(os.getenv("ASSISTANT_ECHO_IGNORE_MS", "1200"))
LOG_TWILIO_PLAYBACK = os.getenv("LOG_TWILIO_PLAYBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
TWILIO_OUTBOUND_CHUNK_BYTES = 160
TWILIO_OUTBOUND_PACING_MS = float(os.getenv("TWILIO_OUTBOUND_PACING_MS", "20"))

STT_TIMEOUT_SEC = float(os.getenv("STT_TIMEOUT_SEC", "0"))
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "5.0"))
TTS_TIMEOUT_SEC = float(os.getenv("TTS_TIMEOUT_SEC", "5.0"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "150"))
DEFAULT_CALL_LANGUAGE = os.getenv("DEFAULT_CALL_LANGUAGE", "en").strip().lower()
SUPPORTED_LANGUAGES = ("en", "es")
SPANISH_LANGUAGE_MARKERS = (
    "hola",
    "gracias",
    "por favor",
    "buenos",
    "buenas",
    "necesito",
    "quiero",
    "puedo",
    "ayuda",
    "como",
    "donde",
    "cuanto",
)
ENGLISH_LANGUAGE_MARKERS = (
    "hello",
    "thanks",
    "thank you",
    "please",
    "help",
    "need",
    "want",
    "where",
    "how",
    "what",
    "today",
)
INITIAL_GREETING_ENABLED = os.getenv("INITIAL_GREETING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
INITIAL_GREETING_TEXT = os.getenv(
    "INITIAL_GREETING_TEXT",
    "Good day. My name is Athenas. Please tell me how I can help you today.",
).strip()

vad = webrtcvad.Vad(2)

SYSTEM_PROMPT = (
    "Eres un asistente de voz telefonico. "
    "Responde en el mismo idioma del usuario. "
    "Da respuestas concretas, naturales y faciles de oir por telefono. "
    "Evita listas, markdown, URLs, texto tecnico y respuestas largas. "
    "Si falta informacion, haz una sola pregunta concreta."
)


@dataclass
class CallSession:
    session_key: str
    call_sid: str | None = None
    stream_sid: str | None = None
    preferred_language: str | None = None
    vad_buffer: bytearray = field(default_factory=bytearray)
    pre_speech_frames: deque[bytes] = field(default_factory=lambda: deque(maxlen=PRE_SPEECH_FRAMES))
    speech_frames: list[bytes] = field(default_factory=list)
    speech_frame_count: int = 0
    voice_streak: int = 0
    silence_frames: int = 0
    active_generation: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    utterance_queue: asyncio.Queue[tuple[int, bytes]] = field(default_factory=asyncio.Queue)
    playback_queue: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    tasks: set[asyncio.Task] = field(default_factory=set)
    pending_marks: set[str] = field(default_factory=set)
    mark_counter: int = 0
    assistant_speaking: bool = False
    assistant_started_at: float | None = None
    closed: bool = False


sessions: dict[str, CallSession] = {}


def normalize_supported_language(lang: str | None) -> str:
    if lang in SUPPORTED_LANGUAGES:
        return lang
    return "es" if lang == "spanish" else DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "en"


def infer_supported_language_from_text(text: str, fallback: str = "en") -> str:
    lowered = text.lower().strip()
    if not lowered:
        return normalize_supported_language(fallback)

    english_hits = sum(marker in lowered for marker in ENGLISH_LANGUAGE_MARKERS)
    spanish_hits = sum(marker in lowered for marker in SPANISH_LANGUAGE_MARKERS)
    has_spanish_chars = any(char in lowered for char in "áéíóúñ¿¡")

    if has_spanish_chars or spanish_hits > english_hits:
        return "es"
    if english_hits > spanish_hits:
        return "en"
    return normalize_supported_language(fallback)


def detect_language(text: str) -> str:
    return infer_supported_language_from_text(text, fallback=DEFAULT_CALL_LANGUAGE)


def get_language_instruction(lang: str) -> str:
    if lang == "en":
        return (
            "Output language is locked to English. "
            "Reply only in English unless the user explicitly switches language."
        )
    return (
        "El idioma de salida esta fijado en espanol. "
        "Responde solo en espanol salvo que el usuario cambie explicitamente de idioma."
    )


def get_tts_model(lang: str) -> str:
    if lang == "en":
        return DEEPGRAM_TTS_MODEL
    return "aura-2-estrella-es"


def normalize_deepgram_language(lang: str | None) -> str | None:
    if not lang:
        return None

    lowered = lang.strip().lower()
    if lowered in {"en", "en-us", "en-gb", "english"}:
        return "en"
    if lowered in {"es", "es-419", "es-es", "spanish"} or lowered.startswith("es-"):
        return "es"
    return None


def trim_history(session: CallSession) -> None:
    if len(session.history) > MAX_HISTORY_MESSAGES:
        session.history[:] = session.history[-MAX_HISTORY_MESSAGES:]


def build_messages(session: CallSession, user_text: str) -> list[dict[str, str]]:
    lang = session.preferred_language or detect_language(user_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": get_language_instruction(lang)},
    ]
    messages.extend(session.history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_text})
    return messages


def split_tts_segments(text: str, max_chars: int = 350) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    segments: list[str] = []
    current: list[str] = []
    count = 0

    for char in stripped:
        current.append(char)
        count += 1
        if char in ".!?" and count >= 40:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            count = 0
        elif count >= max_chars:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            count = 0

    if current:
        segment = "".join(current).strip()
        if segment:
            segments.append(segment)

    return segments


def extract_deepgram_transcript(payload: dict, fallback_language: str | None = None) -> tuple[list[str], str]:
    fallback = normalize_supported_language(fallback_language)
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        return [], fallback

    channel = channels[0] or {}
    alternatives = channel.get("alternatives") or []
    if not alternatives:
        return [], fallback

    alternative = alternatives[0] or {}
    transcript = (alternative.get("transcript") or "").strip()
    detected_language = normalize_deepgram_language(
        alternative.get("detected_language") or channel.get("detected_language")
    )

    if not detected_language:
        languages = alternative.get("languages") or channel.get("languages") or []
        if languages:
            detected_language = normalize_deepgram_language(languages[0])

    if not detected_language and transcript:
        detected_language = infer_supported_language_from_text(transcript, fallback=fallback)

    return ([transcript] if transcript else []), detected_language or fallback


def get_frame_rms(frame: bytes) -> int:
    return audioop.rms(frame, 2)


def is_probable_voice(frame: bytes) -> tuple[bool, int]:
    rms = get_frame_rms(frame)
    return vad.is_speech(frame, TWILIO_SR) and rms >= MIN_VOICE_RMS, rms


def pcm16_to_wav_bytes(pcm16_audio: bytes, sample_rate: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_audio)
    return buffer.getvalue()


def transcribe_sync(pcm16_audio: bytes, language_hint: str | None = None) -> tuple[list[str], str]:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("Deepgram no configurado. Define DEEPGRAM_API_KEY.")
    if not pcm16_audio:
        return [], normalize_supported_language(language_hint or DEFAULT_CALL_LANGUAGE)

    hint = normalize_deepgram_language(language_hint)
    params = {
        "model": DEEPGRAM_STT_MODEL,
        "punctuate": str(DEEPGRAM_STT_PUNCTUATE).lower(),
        "smart_format": str(DEEPGRAM_STT_SMART_FORMAT).lower(),
    }

    if hint:
        params["language"] = hint
    elif DEEPGRAM_STT_DETECT_LANGUAGE:
        params["detect_language"] = "true"

    wav_audio = pcm16_to_wav_bytes(pcm16_audio, sample_rate=TWILIO_SR, channels=TWILIO_CHANNELS)
    url = f"https://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
        "Accept": "application/json",
    }
    timeout = STT_TIMEOUT_SEC if STT_TIMEOUT_SEC > 0 else 45
    request = urllib.request.Request(url, data=wav_audio, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Deepgram STT error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Deepgram STT connection error: {exc}") from exc

    return extract_deepgram_transcript(payload, fallback_language=hint or DEFAULT_CALL_LANGUAGE)


async def transcribe_block(pcm16_audio: bytes, language_hint: str | None = None) -> tuple[list[str], str]:
    return await asyncio.to_thread(transcribe_sync, pcm16_audio, language_hint)


async def call_llm(session: CallSession, user_text: str) -> str:
    if openai_client is None:
        raise RuntimeError("OpenAI no configurada. Define OPENAI_API_KEY.")

    messages = build_messages(session, user_text)

    def sync_call() -> str:
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception:
            log.exception("LLM ERROR")
            return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"

    return await asyncio.to_thread(sync_call)


async def enqueue_playback_clear(session: CallSession) -> None:
    await session.playback_queue.put({"type": "clear", "generation": session.active_generation})


async def interrupt_current_turn(session: CallSession) -> None:
    session.active_generation += 1
    session.assistant_speaking = False
    session.assistant_started_at = None
    session.pending_marks.clear()
    await enqueue_playback_clear(session)


async def stream_tts_segment(session: CallSession, text: str, generation: int) -> tuple[float | None, float]:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("Deepgram no configurado")

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    tts_language = infer_supported_language_from_text(text, fallback=session.preferred_language or DEFAULT_CALL_LANGUAGE)
    model = get_tts_model(tts_language)
    params = urllib.parse.urlencode(
        {
            "model": model,
            "encoding": DEEPGRAM_TTS_ENCODING,
            "sample_rate": DEEPGRAM_TTS_SAMPLE_RATE,
            "container": "none",
        }
    )
    url = f"https://api.deepgram.com/v1/speak?{params}"
    payload = json.dumps({"text": text}).encode("utf-8")
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    def producer() -> None:
        nonlocal ttfb_ms

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break

                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started_at) * 1000

                    loop.call_soon_threadsafe(
                        session.playback_queue.put_nowait,
                        {"type": "audio", "generation": generation, "data": chunk},
                    )
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            loop.call_soon_threadsafe(
                session.playback_queue.put_nowait,
                {
                    "type": "error",
                    "generation": generation,
                    "message": f"Deepgram TTS error {exc.code}: {body}",
                },
            )
        except urllib.error.URLError as exc:
            loop.call_soon_threadsafe(
                session.playback_queue.put_nowait,
                {
                    "type": "error",
                    "generation": generation,
                    "message": f"Deepgram TTS connection error: {exc}",
                },
            )
        finally:
            loop.call_soon_threadsafe(
                session.playback_queue.put_nowait,
                {"type": "segment_end", "generation": generation},
            )

    await asyncio.to_thread(producer)
    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms


async def play_initial_greeting(session: CallSession) -> None:
    if not INITIAL_GREETING_ENABLED or not INITIAL_GREETING_TEXT or not DEEPGRAM_API_KEY:
        return

    session.active_generation += 1
    generation = session.active_generation
    session.history.append({"role": "assistant", "content": INITIAL_GREETING_TEXT})
    trim_history(session)

    try:
        for segment in split_tts_segments(INITIAL_GREETING_TEXT):
            if generation != session.active_generation:
                return
            await asyncio.wait_for(
                stream_tts_segment(session, segment, generation),
                timeout=TTS_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        log.warning("TTS timeout en saludo inicial para %s", session.session_key)
    except Exception:
        log.exception("Error en saludo inicial para %s", session.session_key)


async def send_twilio_media(ws: WebSocket, stream_sid: str, mulaw_audio: bytes) -> None:
    payload = base64.b64encode(mulaw_audio).decode("ascii")
    await ws.send_json(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }
    )


async def send_twilio_mark(ws: WebSocket, stream_sid: str, mark_name: str) -> None:
    await ws.send_json(
        {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": mark_name},
        }
    )


async def send_twilio_clear(ws: WebSocket, stream_sid: str) -> None:
    await ws.send_json({"event": "clear", "streamSid": stream_sid})


async def playback_loop(ws: WebSocket, session: CallSession) -> None:
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

            if item_type == "audio":
                if not session.stream_sid:
                    continue

                if not session.assistant_speaking:
                    session.assistant_started_at = time.perf_counter()
                session.assistant_speaking = True
                chunk = item["data"]
                sent_frames = 0
                for start in range(0, len(chunk), TWILIO_OUTBOUND_CHUNK_BYTES):
                    frame = chunk[start : start + TWILIO_OUTBOUND_CHUNK_BYTES]
                    if frame:
                        await send_twilio_media(ws, session.stream_sid, frame)
                        sent_frames += 1
                        if len(frame) == TWILIO_OUTBOUND_CHUNK_BYTES:
                            await asyncio.sleep(TWILIO_OUTBOUND_PACING_MS / 1000.0)
                if LOG_TWILIO_PLAYBACK and sent_frames:
                    log.info(
                        "Playback audio %s gen=%s bytes=%s frames=%s",
                        session.session_key,
                        generation,
                        len(chunk),
                        sent_frames,
                    )
                continue

            if item_type == "segment_end":
                if not session.stream_sid:
                    session.assistant_speaking = False
                    session.assistant_started_at = None
                    continue

                session.mark_counter += 1
                mark_name = f"gen-{generation}-seg-{session.mark_counter}"
                session.pending_marks.add(mark_name)
                await send_twilio_mark(ws, session.stream_sid, mark_name)
                if LOG_TWILIO_PLAYBACK:
                    log.info("Playback mark enviado %s %s", session.session_key, mark_name)
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


async def handle_incoming_media(session: CallSession, media_payload: str) -> None:
    raw = base64.b64decode(media_payload)
    pcm16 = audioop.ulaw2lin(raw, 2)
    session.vad_buffer.extend(pcm16)

    while len(session.vad_buffer) >= FRAME_BYTES:
        frame = bytes(session.vad_buffer[:FRAME_BYTES])
        del session.vad_buffer[:FRAME_BYTES]

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
            trimmed_frames = session.speech_frames
            if TRIM_TRAILING_SILENCE_FRAMES > 0 and session.silence_frames > 0:
                trailing_trim = min(TRIM_TRAILING_SILENCE_FRAMES, session.silence_frames, len(session.speech_frames))
                candidate_frames = session.speech_frames[:-trailing_trim]
                if candidate_frames:
                    trimmed_frames = candidate_frames

            utterance = b"".join(trimmed_frames)
            session.speech_frames.clear()
            session.pre_speech_frames.clear()
            session.silence_frames = 0
            speech_frame_count = session.speech_frame_count
            session.speech_frame_count = 0

            if len(utterance) < MIN_UTTERANCE_BYTES or speech_frame_count < MIN_SPEECH_FRAMES:
                continue

            session.active_generation += 1
            await session.utterance_queue.put((session.active_generation, utterance))


async def process_utterances(session: CallSession) -> None:
    try:
        while True:
            generation, utterance = await session.utterance_queue.get()
            if generation != session.active_generation:
                continue

            started_at = time.perf_counter()
            language_hint = session.preferred_language or DEEPGRAM_STT_LANGUAGE_HINT

            try:
                stt_started = time.perf_counter()
                if STT_TIMEOUT_SEC > 0:
                    texts, detected_language = await asyncio.wait_for(
                        transcribe_block(utterance, language_hint=language_hint),
                        timeout=STT_TIMEOUT_SEC,
                    )
                else:
                    texts, detected_language = await transcribe_block(utterance, language_hint=language_hint)
                stt_ms = (time.perf_counter() - stt_started) * 1000
            except asyncio.TimeoutError:
                log.warning("STT timeout en %s", session.session_key)
                continue
            except Exception:
                log.exception("STT error en %s", session.session_key)
                continue

            if generation != session.active_generation:
                continue

            full_text = " ".join(text.strip() for text in texts).strip()
            if not full_text:
                continue

            log.info("Usuario (%s): %s", session.session_key, full_text)
            session.preferred_language = normalize_supported_language(
                detected_language or infer_supported_language_from_text(full_text, fallback=session.preferred_language or "en")
            )

            try:
                llm_started = time.perf_counter()
                reply = await asyncio.wait_for(call_llm(session, full_text), timeout=LLM_TIMEOUT_SEC)
                llm_ms = (time.perf_counter() - llm_started) * 1000
            except asyncio.TimeoutError:
                log.warning("LLM timeout en %s", session.session_key)
                reply = "Lo siento, estoy tardando mas de lo normal. Puedes repetirlo?"
                llm_ms = LLM_TIMEOUT_SEC * 1000
            except Exception:
                log.exception("LLM error en %s", session.session_key)
                reply = "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
                llm_ms = 0.0

            if generation != session.active_generation:
                continue

            session.history.extend(
                [
                    {"role": "user", "content": full_text},
                    {"role": "assistant", "content": reply},
                ]
            )
            trim_history(session)

            log.info("Agente (%s): %s", session.session_key, reply)

            tts_metrics: list[tuple[float | None, float]] = []
            for segment in split_tts_segments(reply):
                if generation != session.active_generation:
                    break
                try:
                    metric = await asyncio.wait_for(
                        stream_tts_segment(session, segment, generation),
                        timeout=TTS_TIMEOUT_SEC,
                    )
                    tts_metrics.append(metric)
                except asyncio.TimeoutError:
                    log.warning("TTS timeout en %s", session.session_key)
                    break
                except Exception:
                    log.exception("TTS error en %s", session.session_key)
                    break

            total_ms = (time.perf_counter() - started_at) * 1000
            first_tts_ms = next((metric[0] for metric in tts_metrics if metric[0] is not None), None)
            log.info(
                "Turno %s gen=%s stt_ms=%.1f llm_ms=%.1f tts_ttfb_ms=%s total_ms=%.1f",
                session.session_key,
                generation,
                stt_ms,
                llm_ms,
                f"{first_tts_ms:.1f}" if first_tts_ms is not None else "n/a",
                total_ms,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Error en process_utterances")


async def cleanup_session(session: CallSession, ws: WebSocket) -> None:
    if session.closed:
        return

    session.closed = True

    if session.speech_frames:
        session.speech_frames.clear()

    for task in list(session.tasks):
        task.cancel()

    sessions.pop(session.session_key, None)

    try:
        await asyncio.gather(*session.tasks, return_exceptions=True)
    except Exception:
        pass

    try:
        await ws.close()
    except Exception:
        pass


@app.post("/voice")
async def voice() -> Response:
    ws_url = PUBLIC_URL.rstrip("/")

    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    else:
        ws_url = "wss://" + ws_url

    twiml = f"""
    <Response>
        <Connect>
            <Stream url=\"{ws_url}/media-stream\" />
        </Connect>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()

    session = CallSession(session_key=f"ws-{id(ws)}")

    try:
        session.tasks.add(asyncio.create_task(process_utterances(session)))
        session.tasks.add(asyncio.create_task(playback_loop(ws, session)))

        while True:
            message = await ws.receive_text()
            msg = json.loads(message)
            event = msg.get("event")

            if event == "connected":
                continue

            if event == "start":
                start = msg.get("start", {})
                session.call_sid = start.get("callSid")
                session.stream_sid = start.get("streamSid") or msg.get("streamSid")
                if session.call_sid:
                    session.session_key = session.call_sid
                sessions[session.session_key] = session
                log.info("callSid=%s streamSid=%s", session.call_sid, session.stream_sid)
                session.tasks.add(asyncio.create_task(play_initial_greeting(session)))
                continue

            if event == "media":
                await handle_incoming_media(session, msg["media"]["payload"])
                continue

            if event == "mark":
                mark = msg.get("mark", {}).get("name")
                if mark and mark in session.pending_marks:
                    session.pending_marks.discard(mark)
                if not session.pending_marks:
                    session.assistant_speaking = False
                continue

            if event == "dtmf":
                log.info("DTMF recibido en %s: %s", session.session_key, msg.get("dtmf", {}).get("digit"))
                continue

            if event == "stop":
                log.info("Stream stop para %s", session.session_key)
                break

    except WebSocketDisconnect:
        log.info("WebSocket desconectado para %s", session.session_key)
    except Exception:
        log.exception("Error en media_stream")
    finally:
        await cleanup_session(session, ws)


@app.get("/test-llm-tts")
async def test_llm_tts(q: str = Query(...)) -> dict:
    dummy_session = CallSession(session_key="test")
    dummy_session.preferred_language = detect_language(q)
    reply = await call_llm(dummy_session, q)
    segments = split_tts_segments(reply)
    return {
        "input": q,
        "reply": reply,
        "tts_segments": len(segments),
        "tts_ready": bool(DEEPGRAM_API_KEY),
    }


@app.post("/test-stt")
async def test_stt() -> dict:
    dummy_audio = b"\x00\x00" * TWILIO_SR
    texts, language = await transcribe_block(dummy_audio, language_hint=DEEPGRAM_STT_LANGUAGE_HINT)
    return {
        "text": " ".join(texts).strip(),
        "segments": texts,
        "language": language,
        "stt_ready": bool(DEEPGRAM_API_KEY),
        "model": DEEPGRAM_STT_MODEL,
    }


@app.get("/list-models")
async def list_models() -> dict:
    if openai_client is None:
        return {"error": "OpenAI no configurada"}

    def sync_list() -> dict:
        try:
            models_page = openai_client.models.list()
            if hasattr(models_page, "data"):
                models = [model.id for model in models_page.data]
            else:
                models = [model.id for model in models_page]
            return {"models": models}
        except Exception as exc:
            return {"error": str(exc)}

    return await asyncio.to_thread(sync_list)


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "STT server running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)