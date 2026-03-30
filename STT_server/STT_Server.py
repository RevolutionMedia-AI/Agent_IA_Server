import asyncio
import json
import logging

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from STT_server.adapters.deepgram_stt_realtime import run_realtime_stt
from STT_server.adapters.openai_llm import call_llm, list_models
from STT_server.adapters.openai_realtime import run_realtime_session
from STT_server.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_LANGUAGE_HINT,
    DEEPGRAM_STT_MODEL,
    OPENAI_API_KEY,
    PORT,
    PUBLIC_URL,
    RIME_API_KEY,
    TWILIO_SR,
    USE_OPENAI_REALTIME,
    TWIML_INITIAL_GREETING_ENABLED,
    TWIML_INITIAL_GREETING_LANG,
)
from STT_server.domain.language import detect_language, split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.audio_ingest import handle_incoming_media
from STT_server.services.common import require_debug_endpoints
from STT_server.services.playback_service import play_initial_greeting, playback_loop
from STT_server.services.session_runtime import cleanup_session, monitor_idle_silence, register_session, track_task
from STT_server.services.turn_manager import announce_stt_failure_once, enqueue_transcript_event, process_transcripts


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stt_server")

if not PUBLIC_URL:
    raise RuntimeError("Define PUBLIC_URL en las variables de entorno")

if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY no configurada.")

if not DEEPGRAM_API_KEY:
    log.warning("DEEPGRAM_API_KEY no configurada. El STT no estara disponible.")

if not RIME_API_KEY:
    log.warning("RIME_API_KEY no configurada. El TTS no estara disponible.")


app = FastAPI()

# ── Warm-up TTS en startup ──
@app.on_event("startup")
async def warmup_tts():
    from STT_server.adapters.rime_tts import stream_tts_segment
    from STT_server.domain.session import CallSession
    from STT_server.config import INITIAL_GREETING_TEXT

    def dummy_emit(item):
        # Emisor sincrónico para evitar 'coroutine was never awaited' warnings
        return True

    log.info("[WARMUP] Generando warm-up TTS en inglés (initial greeting)...")
    try:
        session_en = CallSession(session_key="warmup-en")
        session_en.preferred_language = "en"
        await stream_tts_segment(session_en, INITIAL_GREETING_TEXT, 0, dummy_emit)
    except Exception as e:
        log.warning(f"[WARMUP] Error generando warm-up TTS inglés: {e}")


@app.post("/voice")
async def voice() -> Response:
    ws_url = PUBLIC_URL.rstrip("/")

    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    else:
        ws_url = "wss://" + ws_url

    # If configured, play a pre-generated greeting file (hosted on this server)
    # before connecting the media stream. This ensures Twilio plays audio to
    # the caller before opening the websocket stream.
    if TWIML_INITIAL_GREETING_ENABLED:
        play_url = f"{PUBLIC_URL.rstrip('/')}/greeting.wav"
        twiml = f"""
    <Response>
        <Play>{play_url}</Play>
        <Connect>
            <Stream url=\"{ws_url}/media-stream\" />
        </Connect>
    </Response>
    """
    else:
        twiml = f"""
    <Response>
        <Connect>
            <Stream url=\"{ws_url}/media-stream\" />
        </Connect>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")


@app.get("/greeting.wav")
async def greeting_wav() -> Response:
    """Serve the pre-generated warm-up mulaw file as a WAV (mu-law -> PCM16).

    Expects files named `rime_tts_warmup-<lang>_0.mulaw` in the working directory.
    Falls back to English if the requested language file is missing.
    """
    import io
    import wave
    import audioop
    from pathlib import Path

    lang = TWIML_INITIAL_GREETING_LANG or "en"
    fname = Path(f"rime_tts_warmup-{lang}_0.mulaw")
    if not fname.exists():
        # fallback to english
        fname = Path("rime_tts_warmup-en_0.mulaw")
        if not fname.exists():
            return Response(content="", status_code=404)

    try:
        mulaw = fname.read_bytes()
        # Some Python builds expose `mulaw2lin`; others expose `ulaw2lin`.
        # Prefer `mulaw2lin`, fall back to `ulaw2lin` for compatibility.
        if hasattr(audioop, "mulaw2lin"):
            pcm16 = audioop.mulaw2lin(mulaw, 2)
        elif hasattr(audioop, "ulaw2lin"):
            pcm16 = audioop.ulaw2lin(mulaw, 2)
        else:
            raise RuntimeError("audioop lacks mulaw2lin/ulaw2lin")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TWILIO_SR)
            wf.writeframes(pcm16)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        log.exception("Error generating greeting.wav: %s", e)
        return Response(content="", status_code=500)


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    session = CallSession(session_key=f"ws-{id(ws)}")

    try:
        track_task(session, asyncio.create_task(playback_loop(ws, session)))
        track_task(session, asyncio.create_task(process_transcripts(session)))

        while True:
            try:
                message = await ws.receive_text()
            except RuntimeError as e:
                if "WebSocket is not connected" in str(e):
                    log.warning("WebSocket ya no está conectado (probablemente cerrado por timeout o cliente). Saliendo del bucle de media_stream para %s.", session.session_key)
                    break
                else:
                    raise
            except WebSocketDisconnect:
                log.info("WebSocket desconectado para %s", session.session_key)
                break

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
                register_session(session)
                log.info("callSid=%s streamSid=%s", session.call_sid, session.stream_sid)
                if USE_OPENAI_REALTIME:
                    track_task(
                        session,
                        asyncio.create_task(run_realtime_session(session)),
                    )
                else:
                    track_task(
                        session,
                        asyncio.create_task(
                            run_realtime_stt(
                                session,
                                lambda item: enqueue_transcript_event(session, item),
                                announce_stt_failure_once,
                            )
                        ),
                    track_task(session, asyncio.create_task(process_transcripts(session)))
                    from STT_server.adapters.rime_tts import stream_tts_segment
                    from STT_server.config import INITIAL_GREETING_TEXT
                track_task(session, asyncio.create_task(play_initial_greeting(session)))
                track_task(session, asyncio.create_task(monitor_idle_silence(session, ws)))
                continue

            if event == "media":
                    log.info("[WARMUP] Ejecutando warm-up TTS en inglés (initial greeting)...")
                continue

            if event == "mark":
                        await stream_tts_segment(session_en, INITIAL_GREETING_TEXT, 0, dummy_emit)
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

    except Exception:
        log.exception("Error en media_stream (excepción no controlada)")
    finally:
        await cleanup_session(session, ws)


@app.get("/test-llm-tts")
async def test_llm_tts(q: str = Query(...)) -> dict:
    require_debug_endpoints()
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
    require_debug_endpoints()
    from STT_server.adapters.deepgram_stt_batch import transcribe_block
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
async def list_available_models() -> dict:
    require_debug_endpoints()
    return await list_models()


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "STT server running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)