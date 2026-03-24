import asyncio
import json
import logging

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from STT_server.adapters.deepgram_stt_batch import transcribe_block
from STT_server.adapters.deepgram_stt_realtime import run_realtime_stt
from STT_server.adapters.openai_llm import call_llm, list_models
from STT_server.config import DEEPGRAM_API_KEY, DEEPGRAM_STT_LANGUAGE_HINT, DEEPGRAM_STT_MODEL, DEEPGRAM_TTS_ENCODING, OPENAI_API_KEY, PORT, PUBLIC_URL, TWILIO_SR
from STT_server.domain.language import detect_language, split_tts_segments
from STT_server.domain.session import CallSession
from STT_server.services.audio_ingest import handle_incoming_media
from STT_server.services.common import require_debug_endpoints
from STT_server.services.playback_service import play_initial_greeting, playback_loop
from STT_server.services.session_runtime import cleanup_session, monitor_idle_silence, register_session, track_task
from STT_server.services.turn_manager import announce_stt_failure_once, enqueue_transcript_event, process_local_utterances, process_transcripts


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stt_server")

if not PUBLIC_URL:
    raise RuntimeError("Define PUBLIC_URL en las variables de entorno")

if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY no configurada.")

if not DEEPGRAM_API_KEY:
    log.warning("DEEPGRAM_API_KEY no configurada. El STT y el TTS no estaran disponibles.")

if DEEPGRAM_TTS_ENCODING != "mulaw":
    log.warning("La configuracion TTS debe mantenerse en mulaw para Twilio Media Streams.")


app = FastAPI()


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
        track_task(session, asyncio.create_task(playback_loop(ws, session)))
        track_task(session, asyncio.create_task(process_transcripts(session)))
        track_task(session, asyncio.create_task(process_local_utterances(session)))

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
                register_session(session)
                log.info("callSid=%s streamSid=%s", session.call_sid, session.stream_sid)
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
                track_task(session, asyncio.create_task(play_initial_greeting(session)))
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

    except WebSocketDisconnect:
        log.info("WebSocket desconectado para %s", session.session_key)
    except Exception:
        log.exception("Error en media_stream")
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