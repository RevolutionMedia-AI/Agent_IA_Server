import asyncio
import json
import logging

import uvicorn
import os
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Importar routers
from STT_server.routes.auth import router as auth_router

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
from STT_server.domain.session import CallSession, VALID_TTS_PROVIDERS, VALID_LANGUAGES
from STT_server.domain.tenant import TenantConfig, tenant_store
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

# CORS middleware para permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g. static/greeting.wav) at /static
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Warm-up TTS removed: initial greeting/warm-up generation disabled per request.


@app.post("/voice")
async def voice(tenant_id: str = Query(default=None)) -> Response:
    """Twilio voice webhook. Accepts optional ?tenant_id= to link the call
    to a specific tenant's configuration (prompt, TTS provider, language, etc.).

    When a tenant configures their webhook via /tenants/{id}/configure-webhook,
    the URL includes ?tenant_id=... so incoming calls are automatically linked.
    """
    ws_url = PUBLIC_URL.rstrip("/")

    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    else:
        ws_url = "wss://" + ws_url

    # Build the <Stream> element with optional tenant_id parameter
    stream_params = ""
    if tenant_id:
        stream_params = f'<Parameter name="tenant_id" value="{tenant_id}" />'

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
            <Stream url="{ws_url}/media-stream">{stream_params}</Stream>
        </Connect>
    </Response>
    """
    else:
        twiml = f"""
    <Response>
        <Connect>
            <Stream url="{ws_url}/media-stream">{stream_params}</Stream>
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

                # ── Apply tenant configuration ──
                # Twilio sends custom <Parameter> values in start.customParameters
                custom_params = start.get("customParameters") or {}
                tenant_id = custom_params.get("tenant_id") if isinstance(custom_params, dict) else None
                if tenant_id:
                    tenant = tenant_store.get(tenant_id)
                    if tenant:
                        session.custom_prompt = tenant.custom_prompt
                        session.tts_provider = tenant.tts_provider
                        session.preferred_language = tenant.preferred_language
                        session.tenant_id = tenant_id
                        log.info(
                            "[TENANT] Applied tenant %s config to session %s (prompt=%s, tts=%s, lang=%s)",
                            tenant_id, session.session_key,
                            bool(tenant.custom_prompt), tenant.tts_provider, tenant.preferred_language,
                        )
                    else:
                        log.warning("[TENANT] tenant_id=%s not found, using defaults", tenant_id)

                register_session(session)
                log.info("callSid=%s streamSid=%s tenant_id=%s", session.call_sid, session.stream_sid, tenant_id)
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


# ── Session Configuration API ────────────────────────────────────────
# These endpoints allow the frontend to configure per-session settings
# such as TTS provider, language, and custom system prompt.


@app.get("/config")
async def get_available_config() -> dict:
    """Return available TTS providers and languages."""
    from STT_server.config import DEFAULT_TTS_PROVIDER, DEFAULT_CALL_LANGUAGE
    return {
        "tts_providers": sorted(VALID_TTS_PROVIDERS),
        "default_tts_provider": DEFAULT_TTS_PROVIDER,
        "languages": sorted(VALID_LANGUAGES),
        "default_language": DEFAULT_CALL_LANGUAGE,
    }


@app.get("/sessions")
async def list_sessions() -> dict:
    """List active call sessions with their current configuration."""
    from STT_server.services.session_runtime import sessions
    result = {}
    for key, s in sessions.items():
        result[key] = {
            "call_sid": s.call_sid,
            "preferred_language": s.preferred_language,
            "tts_provider": s.tts_provider,
            "custom_prompt": (s.custom_prompt[:80] + "...") if s.custom_prompt and len(s.custom_prompt) > 80 else s.custom_prompt,
            "assistant_speaking": s.assistant_speaking,
            "closed": s.closed,
        }
    return {"sessions": result, "count": len(result)}


class SessionConfigUpdate(BaseModel):
    """Request body for PATCH /sessions/{session_key}."""
    tts_provider: str | None = None
    preferred_language: str | None = None
    custom_prompt: str | None = None


@app.get("/sessions/{session_key}")
async def get_session_config(session_key: str) -> dict:
    """Get the configuration of a specific session."""
    from STT_server.services.session_runtime import sessions
    session = sessions.get(session_key)
    if not session:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_key}' not found"})
    return {
        "session_key": session.session_key,
        "call_sid": session.call_sid,
        "preferred_language": session.preferred_language,
        "tts_provider": session.tts_provider,
        "custom_prompt": session.custom_prompt,
        "assistant_speaking": session.assistant_speaking,
        "closed": session.closed,
    }


@app.patch("/sessions/{session_key}")
async def update_session_config(session_key: str, body: SessionConfigUpdate = None) -> dict:
    """Update per-session configuration: tts_provider, preferred_language, custom_prompt."""
    from STT_server.services.session_runtime import sessions
    session = sessions.get(session_key)
    if not session:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_key}' not found"})

    if body is None:
        body = SessionConfigUpdate()

    updated = {}

    # Update TTS provider
    if body.tts_provider is not None:
        provider = body.tts_provider.strip().lower()
        if provider not in VALID_TTS_PROVIDERS:
            return JSONResponse(status_code=400, content={"error": f"Invalid tts_provider '{provider}'. Valid: {sorted(VALID_TTS_PROVIDERS)}"})
        session.tts_provider = provider
        updated["tts_provider"] = session.tts_provider
        log.info("[CONFIG] Updated tts_provider for %s: %s", session_key, session.tts_provider)

    # Update preferred language
    if body.preferred_language is not None:
        lang = body.preferred_language.strip().lower()
        if lang not in VALID_LANGUAGES:
            return JSONResponse(status_code=400, content={"error": f"Invalid preferred_language '{lang}'. Valid: {sorted(VALID_LANGUAGES)}"})
        session.preferred_language = lang
        updated["preferred_language"] = session.preferred_language
        log.info("[CONFIG] Updated preferred_language for %s: %s", session_key, session.preferred_language)

    # Update custom prompt
    if body.custom_prompt is not None:
        prompt = body.custom_prompt.strip() if body.custom_prompt else None
        session.custom_prompt = prompt
        updated["custom_prompt"] = session.custom_prompt
        log.info("[CONFIG] Updated custom_prompt for %s (len=%d)", session_key, len(session.custom_prompt) if session.custom_prompt else 0)

    return {
        "session_key": session.session_key,
        "updated": updated,
        "current": {
            "preferred_language": session.preferred_language,
            "tts_provider": session.tts_provider,
            "custom_prompt": session.custom_prompt,
        },
    }


# ── Tenant Management API ────────────────────────────────────────────
# These endpoints allow the frontend to manage tenants (clients) with
# their own Twilio credentials, phone numbers, and agent configuration.


class TenantCreateRequest(BaseModel):
    """Request body for creating/updating a tenant."""
    name: str | None = None
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_phone_number: str | None = None
    custom_prompt: str | None = None
    tts_provider: str | None = None
    preferred_language: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    deepgram_api_key: str | None = None


class OutboundCallRequest(BaseModel):
    """Request body for initiating an outbound call."""
    to_number: str  # E.164 format, e.g. "+15071234567"


@app.get("/tenants")
async def list_tenants() -> dict:
    """List all configured tenants."""
    tenants = tenant_store.list_all()
    return {
        "tenants": [t.to_dict(include_secrets=False) for t in tenants],
        "count": len(tenants),
    }


@app.post("/tenants")
async def create_tenant(body: TenantCreateRequest) -> dict:
    """Create a new tenant with Twilio credentials and agent configuration."""
    import uuid
    tenant_id = f"tenant-{uuid.uuid4().hex[:12]}"

    tenant = TenantConfig(
        tenant_id=tenant_id,
        name=body.name or "",
        twilio_account_sid=body.twilio_account_sid or "",
        twilio_auth_token=body.twilio_auth_token or "",
        twilio_phone_number=body.twilio_phone_number or "",
        custom_prompt=body.custom_prompt,
        tts_provider=body.tts_provider or "elevenlabs",
        preferred_language=body.preferred_language or "es",
        openai_api_key=body.openai_api_key,
        elevenlabs_api_key=body.elevenlabs_api_key,
        elevenlabs_voice_id=body.elevenlabs_voice_id,
        deepgram_api_key=body.deepgram_api_key,
    )

    tenant_store.upsert(tenant)
    log.info("[TENANT] Created tenant %s (%s)", tenant_id, tenant.name)

    return {
        "tenant_id": tenant_id,
        "config": tenant.to_dict(include_secrets=False),
    }


@app.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str) -> dict:
    """Get a tenant's configuration (secrets are masked)."""
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    return tenant.to_dict(include_secrets=False)


@app.patch("/tenants/{tenant_id}")
async def update_tenant(tenant_id: str, body: TenantCreateRequest) -> dict:
    """Update a tenant's configuration. Only provided fields are updated."""
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})

    import time
    updated = {}

    if body.name is not None:
        tenant.name = body.name
        updated["name"] = tenant.name
    if body.twilio_account_sid is not None:
        tenant.twilio_account_sid = body.twilio_account_sid
        updated["twilio_account_sid"] = "updated"
    if body.twilio_auth_token is not None:
        tenant.twilio_auth_token = body.twilio_auth_token
        updated["twilio_auth_token"] = "updated"
    if body.twilio_phone_number is not None:
        tenant.twilio_phone_number = body.twilio_phone_number
        updated["twilio_phone_number"] = tenant.twilio_phone_number
    if body.custom_prompt is not None:
        tenant.custom_prompt = body.custom_prompt.strip() if body.custom_prompt else None
        updated["custom_prompt"] = f"len={len(tenant.custom_prompt)}" if tenant.custom_prompt else "cleared"
    if body.tts_provider is not None:
        provider = body.tts_provider.strip().lower()
        if provider not in VALID_TTS_PROVIDERS:
            return JSONResponse(status_code=400, content={"error": f"Invalid tts_provider '{provider}'. Valid: {sorted(VALID_TTS_PROVIDERS)}"})
        tenant.tts_provider = provider
        updated["tts_provider"] = tenant.tts_provider
    if body.preferred_language is not None:
        lang = body.preferred_language.strip().lower()
        if lang not in VALID_LANGUAGES:
            return JSONResponse(status_code=400, content={"error": f"Invalid preferred_language '{lang}'. Valid: {sorted(VALID_LANGUAGES)}"})
        tenant.preferred_language = lang
        updated["preferred_language"] = tenant.preferred_language
    if body.openai_api_key is not None:
        tenant.openai_api_key = body.openai_api_key
        updated["openai_api_key"] = "updated"
    if body.elevenlabs_api_key is not None:
        tenant.elevenlabs_api_key = body.elevenlabs_api_key
        updated["elevenlabs_api_key"] = "updated"
    if body.elevenlabs_voice_id is not None:
        tenant.elevenlabs_voice_id = body.elevenlabs_voice_id
        updated["elevenlabs_voice_id"] = tenant.elevenlabs_voice_id
    if body.deepgram_api_key is not None:
        tenant.deepgram_api_key = body.deepgram_api_key
        updated["deepgram_api_key"] = "updated"

    tenant.updated_at = time.time()
    tenant_store.upsert(tenant)
    log.info("[TENANT] Updated tenant %s: %s", tenant_id, list(updated.keys()))

    return {
        "tenant_id": tenant_id,
        "updated": updated,
        "current": tenant.to_dict(include_secrets=False),
    }


@app.delete("/tenants/{tenant_id}")
async def delete_tenant(tenant_id: str) -> dict:
    """Delete a tenant configuration."""
    deleted = tenant_store.delete(tenant_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    log.info("[TENANT] Deleted tenant %s", tenant_id)
    return {"deleted": True, "tenant_id": tenant_id}


@app.post("/tenants/{tenant_id}/validate-twilio")
async def validate_tenant_twilio(tenant_id: str) -> dict:
    """Validate a tenant's Twilio credentials."""
    from STT_server.adapters.twilio_api import validate_twilio_credentials
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    if not tenant.has_twilio_credentials:
        return JSONResponse(status_code=400, content={"error": "Tenant does not have Twilio credentials configured"})

    result = await validate_twilio_credentials(tenant.twilio_account_sid, tenant.twilio_auth_token)
    return result


@app.post("/tenants/{tenant_id}/configure-webhook")
async def configure_tenant_webhook(tenant_id: str) -> dict:
    """Automatically configure the Twilio webhook on the tenant's phone number.

    This sets the voice URL to point to our /voice endpoint, so incoming
    calls are routed to this server. The client does NOT need to manually
    configure anything in the Twilio console.
    """
    from STT_server.adapters.twilio_api import configure_voice_webhook
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    if not tenant.has_twilio_credentials:
        return JSONResponse(status_code=400, content={"error": "Tenant does not have Twilio credentials configured"})

    webhook_url = f"{PUBLIC_URL.rstrip('/')}/voice?tenant_id={tenant_id}"
    result = await configure_voice_webhook(
        tenant.twilio_account_sid,
        tenant.twilio_auth_token,
        tenant.twilio_phone_number,
        webhook_url,
    )

    if result.get("success"):
        tenant.webhook_configured = True
        import time
        tenant.updated_at = time.time()
        tenant_store.upsert(tenant)
        log.info("[TENANT] Webhook configured for %s -> %s", tenant_id, webhook_url)

    return result


@app.post("/tenants/{tenant_id}/list-numbers")
async def list_tenant_numbers(tenant_id: str) -> dict:
    """List all phone numbers in the tenant's Twilio account."""
    from STT_server.adapters.twilio_api import list_phone_numbers
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    if not tenant.twilio_account_sid or not tenant.twilio_auth_token:
        return JSONResponse(status_code=400, content={"error": "Tenant does not have Twilio credentials configured"})

    return await list_phone_numbers(tenant.twilio_account_sid, tenant.twilio_auth_token)


@app.post("/tenants/{tenant_id}/call")
async def make_call(tenant_id: str, body: OutboundCallRequest) -> dict:
    """Initiate an outbound call from the tenant's phone number."""
    from STT_server.adapters.twilio_api import make_outbound_call
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    if not tenant.has_twilio_credentials:
        return JSONResponse(status_code=400, content={"error": "Tenant does not have Twilio credentials configured"})

    webhook_url = f"{PUBLIC_URL.rstrip('/')}/voice?tenant_id={tenant_id}"
    result = await make_outbound_call(
        tenant.twilio_account_sid,
        tenant.twilio_auth_token,
        tenant.twilio_phone_number,
        body.to_number,
        webhook_url,
    )

    if result.get("success"):
        log.info("[TENANT] Outbound call from %s: %s -> %s", tenant_id, tenant.twilio_phone_number, body.to_number)

    return result


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "STT server running"}


# ── Incluir routers ─────────────────────────────────────────────────────────
app.include_router(auth_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)