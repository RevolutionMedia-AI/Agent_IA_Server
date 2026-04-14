import asyncio
import json
import logging

import uvicorn
import os
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

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
    ELEVENLABS_API_KEY,
    TWILIO_SR,
    USE_OPENAI_REALTIME,
    TWIML_INITIAL_GREETING_ENABLED,
)
from STT_server.domain.language import detect_language, split_tts_segments, sanitize_tts_text
from STT_server.domain.session import CallSession
from STT_server.services.audio_ingest import handle_incoming_media
from STT_server.services.common import require_debug_endpoints
from STT_server.services.playback_service import playback_loop
from STT_server.services.session_runtime import cleanup_session, monitor_idle_silence, register_session, track_task
from STT_server.services.turn_manager import announce_stt_failure_once, enqueue_transcript_event, process_transcripts


logging.basicConfig(level=logging.WARNING)
# Reduce verbosity of commonly noisy third-party loggers (uvicorn/access, websockets)
for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error", "websockets", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("stt_server")

if not PUBLIC_URL:
    raise RuntimeError("Define PUBLIC_URL en las variables de entorno")

if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY no configurada.")

if not DEEPGRAM_API_KEY:
    log.warning("DEEPGRAM_API_KEY no configurada. El STT no estara disponible.")

if not ELEVENLABS_API_KEY:
    log.warning("ELEVENLABS_API_KEY no configurada. El TTS no estara disponible.")


app = FastAPI()

# Serve static files (e.g. static/greeting.wav) at /static
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Warm-up TTS removed: initial greeting/warm-up generation disabled per request.


@app.post("/voice")
async def voice() -> Response:
    ws_url = PUBLIC_URL.rstrip("/")

    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    else:
        ws_url = "wss://" + ws_url

    # If a static greeting file exists or the TWIML flag is enabled,
    # include a <Play> so Twilio plays the pre-recorded greeting before
    # connecting the media stream. Otherwise connect directly.
    static_local = os.path.join(os.path.dirname(__file__), "static", "greeting.wav")
    if TWIML_INITIAL_GREETING_ENABLED or os.path.exists(static_local):
        play_url = f"{PUBLIC_URL.rstrip('/')}/static/greeting.wav"
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


# Greeting WAV endpoint removed — initial greeting functionality disabled.


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
                    )
                    track_task(session, asyncio.create_task(process_transcripts(session)))
                # Initial greeting removed; do not schedule play_initial_greeting
                track_task(session, asyncio.create_task(monitor_idle_silence(session, ws)))
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
    safe_reply = sanitize_tts_text(reply)
    segments = split_tts_segments(safe_reply)
    return {
        "input": q,
        "reply": reply,
        "sanitized_reply": safe_reply,
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