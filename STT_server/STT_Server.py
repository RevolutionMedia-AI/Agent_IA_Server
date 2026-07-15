import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Importar routers
from STT_server.routes.auth import router as auth_router
from STT_server.routes.api import api_router, require_auth

from STT_server.adapters.deepgram_stt_realtime import run_realtime_stt as run_deepgram_realtime_stt
from STT_server.adapters.inworld_stt_realtime import run_realtime_stt as run_inworld_realtime_stt
from STT_server.adapters.assemblyai_stt_realtime import run_realtime_stt as run_assemblyai_realtime_stt
from STT_server.adapters.rime_stt_realtime import run_realtime_stt as run_rime_realtime_stt
from STT_server.adapters.openai_llm import call_llm, list_models
from STT_server.adapters.openai_realtime import run_realtime_session
# ponytail: M8 from the call-flow audit. Fail fast on missing critical
# env vars BEFORE any heavy import. Without this, if a downstream import
# (FastAPI, pydantic, etc.) fails, the operator sees a confusing
# ImportError instead of "PUBLIC_URL is missing". The check runs
# against os.environ directly (not the imported value) so it doesn't
# depend on STT_server.config being importable.
if not os.environ.get("PUBLIC_URL"):
    sys.exit("FATAL: PUBLIC_URL environment variable is required")
from STT_server.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_LANGUAGE_HINT,
    DEEPGRAM_STT_MODEL,
    OPENAI_API_KEY,
    PORT,
    PUBLIC_URL,
    ELEVENLABS_API_KEY,
    TWILIO_AUTH_TOKEN,
    TWILIO_SR,
    USE_OPENAI_REALTIME,
    TWIML_INITIAL_GREETING_ENABLED,
)
from STT_server.domain.language import detect_language, split_tts_segments, sanitize_tts_text
from STT_server.domain.session import CallSession, VALID_TTS_PROVIDERS, VALID_LANGUAGES
from STT_server.domain.tenant import TenantConfig, tenant_store
from STT_server import db_tenants
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
# ponytail: M6 from the call-flow audit. Twilio signs the URL it
# calls, and the verifier rebuilds the URL as PUBLIC_URL + path. A
# path on PUBLIC_URL ("https://host/be") would create a mismatch
# when Twilio hit "/be/voice" — the signature is computed against
# the path Twilio sees, not against the path on PUBLIC_URL. Force
# the operator to set PUBLIC_URL to scheme://host only.
from urllib.parse import urlparse
_parsed = urlparse(PUBLIC_URL)
if _parsed.path and _parsed.path not in ("", "/"):
    log.warning(
        "[VOICE] PUBLIC_URL has a path component (%r). Twilio signature "
        "verification will fail unless the proxy preserves that exact "
        "path on the way to this service. Recommended: set PUBLIC_URL "
        "to scheme://host (no path).",
        _parsed.path,
    )

# ponytail: removed the OPENAI_API_KEY / DEEPGRAM_API_KEY /
# ELEVENLABS_API_KEY boot-time logs entirely. The platform is
# a multi-tenant provider: every user configures their own keys
# via Settings → API (persisted in tools_integrations), and the
# env vars are an internal config knob, not something the operator
# needs to see on every boot. The actual error path (both env
# var and per-user key missing at the call site) is already
# surfaced clearly by credentials_resolver.



# ponytail: startup hook. On Postgres deployments, backfill any tenants
# that exist in the local JSON file but not yet in the DB (the in-memory
# tenant_store was ephemeral, so on a greenfield this is a no-op; the
# moment a JSON file appears the rows get picked up automatically).
#
# Deliberately NOT calling db_call_sessions.list_open_sessions() here:
# session_runtime.py still uses its in-memory dict and never writes to
# the call_sessions table, so a recovery sweep would always return [].
# Add that hook in the same PR that migrates session_runtime to
# db_call_sessions — keeping the two coupled avoids the false signal
# of "we have crash recovery" when we don't.
@asynccontextmanager
async def lifespan(app: FastAPI):
    db_tenants.backfill_from_json()
    # ponytail: migrate any per-user API keys that the legacy JSON path
    # had persisted before /settings/api-keys moved to Postgres. One-shot
    # migration; subsequent restarts see nothing in the JSON file and
    # backfill is a no-op.
    try:
        from STT_server import db_tools
        db_tools.backfill_from_json()
    except Exception as exc:
        log.warning("[startup] db_tools backfill failed (non-fatal): %s", exc)
    yield


app = FastAPI(lifespan=lifespan)

# CORS allowlist — comma-separated env var, defaults to local dev origins.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ponytail: log the active CORS allowlist at startup so a missing
# origin shows up immediately in the deploy logs instead of as a
# cryptic "No Access-Control-Allow-Origin" error in the browser.
log.info("[CORS] allowed origins: %s", ALLOWED_ORIGINS)

# Serve static files (e.g. static/greeting.wav) at /static
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Warm-up TTS removed: initial greeting/warm-up generation disabled per request.


@app.post("/voice")
async def voice(
    tenant_id: str = Query(default=None),
    request: Request = None,
) -> Response:
    """Twilio voice webhook. Accepts optional ?tenant_id= to link the call
    to a specific tenant's configuration (prompt, TTS provider, language, etc.).

    Also reads the form-encoded `To` field (the called number) and
    looks up the matching phone number in phone_numbers.json. If it
    has an agent linked, the agent_id is added to the stream's
    custom parameters so media_stream can pick it up at call start.

    When TWILIO_AUTH_TOKEN is set in the environment, we verify the
    X-Twilio-Signature header against the incoming form params. Any
    request without a valid signature is rejected with 403. This
    blocks attackers who could otherwise POST audio frames to /voice
    and consume our TTS/LLM quota. Set TWILIO_AUTH_TOKEN to enable;
    leaving it unset keeps the endpoint open for local dev.
    """
    # ponytail: read the form once. Twilio sends it as
    # application/x-www-form-urlencoded; the signature verifier and
    # the agent-lookup both need the same dict. Caching it here
    # means the second consumer (the agent lookup later in this
    # function) just reuses form_dict instead of re-parsing the
    # body. FastAPI caches the body, so the second .form() would
    # work, but it's wasteful and confusing.
    form_dict: dict = {}
    if request is not None:
        try:
            form = await request.form()
            # Convert to plain dict (form items are multi-dict-aware
            # but Twilio sends each key once). Convert values to str so
            # the signature helper sees the same shape Twilio signed.
            form_dict = {k: str(v) if v is not None else "" for k, v in form.items()}
        except Exception:
            form_dict = {}

    # ponytail: per-number Twilio auth token. Twilio signs each webhook
    # with the auth token of the Twilio account that owns the phone
    # number that's calling - not the deployer's account. We resolve
    # the called number first, then verify against THAT account's
    # auth token. The TWILIO_AUTH_TOKEN env var is only a fallback for
    # deployments where the operator and the phone-number owner are
    # the same entity (e.g. internal use).
    if form_dict:
        from STT_server.adapters.twilio_api import validate_twilio_signature
        from STT_server.db_phone_numbers import find_by_number as _find_num_for_sig
        per_number_token = None
        try:
            called_to = form_dict.get("To") or form_dict.get("to")
            if called_to:
                row = _find_num_for_sig(called_to)
                if row:
                    per_number_token = row.get("twilio_auth_token") or None
        except Exception as exc:
            log.warning("[VOICE] could not look up per-number auth token: %s", exc)
        token_to_check = per_number_token or TWILIO_AUTH_TOKEN
        # ponytail: C2 from the call-flow audit. The previous
        # `if token_to_check:` silently accepted the webhook when
        # no token was configured — a free DoS / cost-amplification
        # path. Fail loudly with 503 so a misconfigured deploy is
        # impossible to miss in the logs.
        if not token_to_check:
            log.error(
                "[VOICE] No Twilio auth token configured (TWILIO_AUTH_TOKEN env "
                "+ per-number auth_token both empty). Rejecting inbound call. "
                "Set TWILIO_AUTH_TOKEN or configure twilio_auth_token on the "
                "phone number row."
            )
            return Response(
                content="Twilio signature verification not configured",
                status_code=503,
            )
        # Twilio always signs webhooks. A missing signature with a
        # configured token is suspicious (proxy stripping the header,
        # or someone bypassing the check).
        sig = request.headers.get("X-Twilio-Signature", "")
        if not sig:
            return Response(content="missing signature", status_code=403)
        # Twilio signs the URL the request hit. When the
        # request goes through a reverse proxy (Railway, Cloudflare)
        # the Host header reflects the public hostname but the path
        # the signature was computed against is the original
        # PUBLIC_URL path. We rebuild the URL from PUBLIC_URL +
        # request.url.path so signature verification works behind a
        # proxy.
        signature_url = f"{PUBLIC_URL.rstrip('/')}{request.url.path}"
        if request.url.query:
            signature_url += f"?{request.url.query}"
        if not validate_twilio_signature(
            token_to_check, signature_url, sig, form_dict
        ):
            log.warning(
                "[VOICE] invalid Twilio signature from %s (per_number=%s)",
                request.client.host if request.client else "?",
                bool(per_number_token),
            )
            return Response(content="invalid signature", status_code=403)

    ws_url = PUBLIC_URL.rstrip("/")

    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    else:
        ws_url = "wss://" + ws_url

    # Resolve agent_id from the called number (if any). Twilio POSTs
    # the called number as form-encoded 'To' (E.164, like +15551234567).
    # We look it up in phone_numbers.json and pass agent_id to the
    # stream so media_stream can read it from customParameters and
    # pull the per-agent prompt + welcome_message.
    stream_params = []
    if tenant_id:
        stream_params.append(f'<Parameter name="tenant_id" value="{tenant_id}" />')

    agent_id = None
    try:
        # M5: reuse form_dict instead of re-parsing the body.
        called_to = form_dict.get("To") or form_dict.get("to")
        if called_to:
            e164 = str(called_to).strip()
            from STT_server.db_phone_numbers import find_by_number as find_num
            num_row = find_num(e164)
            if num_row:
                agent_id = num_row.get("agent") or None
                # ponytail: log at WARNING so it shows up in the same
                # stream the user reads for errors. INFO was being
                # filtered out and the operator couldn't tell whether
                # the lookup matched or just returned an empty row.
                log.warning(
                    "[VOICE] matched phone %s -> agent_id=%s (db=%s, row_id=%s)",
                    e164, agent_id, "postgres" if is_postgres() else "json",
                    num_row.get("id"),
                )
            else:
                # ponytail: loud "no match" log. Before this, a None
                # return silently dropped the call into the no-agent
                # branch and the user had no way to know whether the
                # lookup ran or failed. Same WARNING level so it sits
                # next to the AGENT/STT warnings they'll grep for.
                log.warning(
                    "[VOICE] no phone_numbers row matched 'To=%s' (digits=%s). "
                    "Either the number isn't in the table or its stored value "
                    "isn't comparable to the E.164 Twilio sent.",
                    called_to, re.sub(r"\D", "", e164),
                )
    except Exception as exc:
        log.warning("[VOICE] failed to look up phone number for agent: %s", exc)

    if agent_id:
        stream_params.append(f'<Parameter name="agent_id" value="{agent_id}" />')

    stream_params_str = ''.join(stream_params)

    # If a static greeting file exists or the TWIML flag is enabled,
    # include a <Play> so Twilio plays the pre-recorded greeting before
    # connecting the media stream. Otherwise connect directly.
    static_local = os.path.join(os.path.dirname(__file__), "static", "greeting.wav")
    # ponytail: bug history. This template used {stream_params} (a
    # Python list), which rendered as ['<Parameter .../>', ...] inside
    # the <Stream> element. Twilio's TwiML parser treated that literal
    # text as a child text node, NOT as <Parameter> elements, and
    # silently dropped them. Net effect: the WebSocket 'start' event
    # arrived with customParameters={} and the call ran without an
    # agent_id (see [AGENT] has no agent_id in customParameters log).
    # stream_params_str was already joined above for exactly this
    # purpose; use it.
    if TWIML_INITIAL_GREETING_ENABLED or os.path.exists(static_local):
        play_url = f"{PUBLIC_URL.rstrip('/')}/static/greeting.wav"
        twiml = f"""
    <Response>
        <Play>{play_url}</Play>
        <Connect>
            <Stream url="{ws_url}/media-stream">{stream_params_str}</Stream>
        </Connect>
    </Response>
    """
    else:
        twiml = f"""
    <Response>
        <Connect>
            <Stream url="{ws_url}/media-stream">{stream_params_str}</Stream>
        </Connect>
    </Response>
    """

    return Response(content=twiml, media_type="application/xml")


async def _watchdog_assistant_speaking(session: CallSession) -> None:
    """H5 from the call-flow audit: force-reset assistant_speaking if
    it's been True too long without Twilio sending a mark event.

    Twilio sends a `mark` event when audio playback completes. If that
    never arrives (network blip, WS timeout, mark lost in transit,
    user mute, anything that drops the event but keeps the WS open),
    assistant_speaking stays True forever and STT barge-in stops
    working because the VAD treats the stuck state as "agent is
    talking, ignore user input". This watchdog catches that case
    and resets the flag after MAX_SPEAKING_SEC.
    """
    MAX_SPEAKING_SEC = 30
    POLL_SEC = 5
    while not session.closed:
        try:
            await asyncio.sleep(POLL_SEC)
        except asyncio.CancelledError:
            return
        if not session.assistant_speaking:
            continue
        if session.assistant_started_at is None:
            continue
        elapsed = time.perf_counter() - session.assistant_started_at
        if elapsed > MAX_SPEAKING_SEC:
            log.warning(
                "[WATCHDOG] assistant_speaking stuck for %.1fs in %s, forcing False",
                elapsed, session.session_key,
            )
            session.assistant_speaking = False
            session.assistant_started_at = None
            session.pending_marks.clear()


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    session = CallSession(session_key=f"ws-{id(ws)}")

    # ponytail: track errors that happen INSIDE the `start` event
    # handler (the bulk of the call setup) so the call can be torn
    # down cleanly. The user flagged: if a RuntimeError fires before
    # the WebSocket starts streaming, Twilio keeps the call open
    # charging minutes in a silent limbo. The except block below
    # catches every exception, plays a short error audio (if a TTS
    # provider is configured), and closes the WS. Twilio sees the
    # close frame and tears the call down — never a silent limbo.
    _call_setup_failed: list[Exception] = []

    try:
        # ponytail: process_transcripts is started exactly once, inside
        # the `start` event handler after the STT engine is wired up.
        # Previously it was also kicked off here unconditionally, which
        # caused two consumers to race on session.transcript_queue
        # whenever the session's STT engine dispatched into the same
        # queue (Deepgram, Inworld). The race produced inconsistent
        # transcript ordering - both consumers pulled from the same
        # queue and one could process partial finals while the other
        # held back. Single consumer only.
        track_task(session, asyncio.create_task(playback_loop(ws, session)))

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
                # ponytail: usage tracking. wall-clock start so duration
                # is real seconds, not monotonic. agent_id is set
                # below from customParameters — initialise here so it's
                # always defined even if no agent is configured.
                session.started_at = time.time()
                session.agent_id = None

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
                        # tenant.user_id was added in 003_tenant_user_link.sql.
                        # Falls back to None for tenants created before that
                        # migration — they keep using system env-var defaults.
                        session.user_id = getattr(tenant, "user_id", None)
                        log.info(
                            "[TENANT] Applied tenant %s config to session %s (prompt=%s, tts=%s, lang=%s, user_id=%s)",
                            tenant_id, session.session_key,
                            bool(tenant.custom_prompt), tenant.tts_provider, tenant.preferred_language,
                            session.user_id,
                        )
                    else:
                        log.warning("[TENANT] tenant_id=%s not found, using defaults", tenant_id)

                await register_session(session)
                log.info("callSid=%s streamSid=%s tenant_id=%s", session.call_sid, session.stream_sid, tenant_id)

                # ── Apply agent config (system prompt + welcome message) ──
                # ponytail: look up the agent via the agent_id we passed
                # in customParameters from the voice() webhook. If found,
                # the agent's prompt overrides the tenant's, and its
                # welcome_message is stored on the session for the
                # initial greeting scheduled below.
                if isinstance(start.get('customParameters'), dict):
                    agent_id_from_params = start['customParameters'].get('agent_id')
                else:
                    agent_id_from_params = None

                agent_cfg = None
                if agent_id_from_params:
                    # ponytail: read the agent from Postgres (with JSON
                    # fallback for local dev). db_agents.get_agent
                    # returns the same shape the route layer sees, so
                    # the fields below don't need to change.
                    try:
                        from STT_server.db_agents import get_agent as _get_agent_db
                        agent_cfg = _get_agent_db(agent_id_from_params)
                    except Exception as exc:
                        log.warning("[AGENT] db lookup failed for %s: %s", agent_id_from_params, exc)
                else:
                    # ponytail: this is the exact failure mode the user
                    # hit — phone was linked, TwiML included agent_id,
                    # but the WebSocket 'start' event didn't carry it
                    # (or the lookup silently failed). Make it loud so
                    # the next time this happens the operator sees the
                    # exact gap without having to grep the TwiML.
                    log.warning(
                        "[AGENT] session %s has no agent_id in customParameters "
                        "(tenant_id=%s, stream_sid=%s). The phone number may not "
                        "be linked to an agent in the agents table.",
                        session.session_key, tenant_id, session.stream_sid,
                    )
                    # returns the same shape the route layer sees, so
                    # the fields below don't need to change.
                    try:
                        from STT_server.db_agents import get_agent as _get_agent_db
                        agent_cfg = _get_agent_db(agent_id_from_params)
                    except Exception as exc:
                        log.warning("[AGENT] db lookup failed for %s: %s", agent_id_from_params, exc)

                if agent_cfg:
                    # Agent's system_prompt overrides the tenant's.
                    if agent_cfg.get('prompt'):
                        session.custom_prompt = agent_cfg['prompt']
                        log.info("[AGENT] Overrode tenant prompt with agent %s prompt (len=%d)",
                                 agent_id_from_params, len(agent_cfg['prompt']))
                    # Stash the welcome_message for play_initial_greeting.
                    if agent_cfg.get('welcome_message'):
                        session.welcome_message = agent_cfg['welcome_message']
                        log.info("[AGENT] Stored welcome_message for agent %s (len=%d)",
                                 agent_id_from_params, len(agent_cfg['welcome_message']))
                    # ponytail: usage record needs to know which agent
                    # took this call so the per-agent totals are right.
                    session.agent_id = agent_cfg.get('id') or agent_id_from_params
                    # user_id flows from the agent row (FK to users.id).
                    # Used downstream by resolve_provider() to pick the
                    # right per-user credential.
                    session.user_id = agent_cfg.get('user_id') or session.user_id
                    # ponytail: provider resolution is per-agent > per-user
                    # auto-detect. The user explicitly asked for "no model
                    # or provider por defecto" — if the agent row has a
                    # provider pinned, we use that; otherwise we scan
                    # the user's per-user credentials and pick the first
                    # available one for the category. If neither, we
                    # fail loud (no env-var fallback) — the call adapter
                    # surfaces a clear error and the operator sees it.
                    from STT_server.services.credentials_resolver import find_first_configured_provider
                    _cfg_stt = (agent_cfg.get('stt_provider') or '').strip().lower() or None
                    _cfg_tts = (agent_cfg.get('tts_provider') or '').strip().lower() or None
                    _cfg_llm = (agent_cfg.get('llm_provider') or '').strip().lower() or None
                    session.stt_provider = _cfg_stt or find_first_configured_provider(session.user_id, 'stt') or ''
                    session.tts_provider = _cfg_tts or find_first_configured_provider(session.user_id, 'tts') or ''
                    session.llm_provider = _cfg_llm or find_first_configured_provider(session.user_id, 'llm') or 'openai'
                    session.stt_model = (agent_cfg.get('stt_model') or None)
                    session.tts_model = (agent_cfg.get('tts_model') or None)
                    session.llm_model = (agent_cfg.get('llm_model') or None)
                    session.voice_id = agent_cfg.get('voice_id') or None
                    session.tts_voice = agent_cfg.get('voice') or None
                    log.info("[AGENT] session %s stt=%s model=%s llm=%s/%s tts=%s/%s voice=%s",
                             session.session_key,
                             session.stt_provider or '-', session.stt_model or '-',
                             session.llm_provider, session.llm_model or '-',
                             session.tts_provider or '-', session.tts_model or '-',
                             session.tts_voice or '-')
                    if session.user_id:
                        log.info("[AGENT] session %s user_id=%s (per-user keys)",
                                 session.session_key, session.user_id)
                if session.stt_provider in ('openai_realtime', 'openai'):
                    # ponytail: 'openai' is the user-facing name in the
                    # FE dropdown (matches the OpenAI STT option the FE
                    # shows); 'openai_realtime' is the historical path
                    # name. They route to the same adapter — the
                    # agent's stt_model field (gpt-4o-transcribe, etc.)
                    # is what tells the OpenAI Realtime API which model
                    # to use. If you want a real OpenAI batch STT path
                    # (REST /v1/audio/transcriptions), that's a separate
                    # adapter that doesn't exist yet — TODO.
                    if session.stt_provider == 'openai':
                        log.info(
                            "[STT] session %s using 'openai' alias for 'openai_realtime'",
                            session.session_key,
                        )
                    track_task(
                        session,
                        asyncio.create_task(run_realtime_session(session)),
                    )
                elif session.stt_provider == 'inworld':
                    track_task(
                        session,
                        asyncio.create_task(
                            run_inworld_realtime_stt(
                                session,
                                lambda item: enqueue_transcript_event(session, item),
                                announce_stt_failure_once,
                            )
                        ),
                    )
                    track_task(session, asyncio.create_task(process_transcripts(session)))
                elif session.stt_provider == 'deepgram':
                    track_task(
                        session,
                        asyncio.create_task(
                            run_deepgram_realtime_stt(
                                session,
                                lambda item: enqueue_transcript_event(session, item),
                                announce_stt_failure_once,
                            )
                        ),
                    )
                    track_task(session, asyncio.create_task(process_transcripts(session)))
                elif session.stt_provider == 'assemblyai':
                    track_task(
                        session,
                        asyncio.create_task(
                            run_assemblyai_realtime_stt(
                                session,
                                lambda item: enqueue_transcript_event(session, item),
                                announce_stt_failure_once,
                            )
                        ),
                    )
                    track_task(session, asyncio.create_task(process_transcripts(session)))
                elif session.stt_provider == 'rime':
                    track_task(
                        session,
                        asyncio.create_task(
                            run_rime_realtime_stt(
                                session,
                                lambda item: enqueue_transcript_event(session, item),
                                announce_stt_failure_once,
                            )
                        ),
                    )
                    track_task(session, asyncio.create_task(process_transcripts(session)))
                else:
                    # No provider configured → no env-var fallback (the
                    # user explicitly asked to drop defaults). The call's
                    # STT path is a no-op without it; the user hears
                    # silence and we log the reason clearly. The most
                    # common cause after my Postgres migration is a
                    # broken agent→phone link (user_id=None cascades
                    # into "no per-user key resolved"), so lead with
                    # that — "Settings → API" alone was misleading
                    # because the agent modal also persists keys now.
                    log.error(
                        "[STT] session %s has no STT provider (agent=%s, user_id=%s). "
                        "Either link this phone number to an agent, set stt_provider on the "
                        "agent, or upload a per-user STT key (ModalAgents inline field or "
                        "Settings → API).",
                        session.session_key, session.agent_id, session.user_id,
                    )
                # ponytail: if the agent has a welcome_message, schedule
                # play_initial_greeting so the TTS speaks first. It's a
                # no-op for agents without one (preserves the previous
                # silent-start behavior).
                if getattr(session, 'welcome_message', None):
                    from STT_server.services.playback_service import play_initial_greeting
                    track_task(
                        session,
                        asyncio.create_task(play_initial_greeting(session))
                    )
                track_task(session, asyncio.create_task(monitor_idle_silence(session, ws)))
                # ponytail: H5 from the call-flow audit. Force-reset
                # assistant_speaking if Twilio never sends the mark
                # event (network blip, WS timeout, mark lost in
                # transit). Without this, assistant_speaking stays
                # True forever and barge-in stops working.
                track_task(
                    session,
                    asyncio.create_task(_watchdog_assistant_speaking(session))
                )
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

    except Exception as exc:
        log.exception("Error en media_stream (excepción no controlada)")
        _call_setup_failed.append(exc)
    finally:
        if _call_setup_failed and not session.closed:
            # ponytail: don't drop the call silently. Play a short
            # error TTS (when possible) so the caller hears "something
            # went wrong" instead of dead air, then close the WS so
            # Twilio tears the call down. play_error_and_hangup is
            # best-effort: it never raises, it just logs and closes.
            err = _call_setup_failed[0]
            try:
                from STT_server.services.playback_service import play_error_and_hangup
                # Map common RuntimeErrors to a short caller-facing
                # message. Everything else gets a generic "config
                # problem" so the user isn't left guessing.
                err_text = str(err)
                if "system_prompt" in err_text:
                    message = "Lo sentimos, este agente no tiene prompt configurado. Adios."
                elif "no STT provider" in err_text or "no TTS provider" in err_text or "no LLM provider" in err_text:
                    message = "Lo sentimos, falta una API key para esta llamada. Adios."
                else:
                    message = "Lo sentimos, hubo un problema de configuracion. Adios."
                await play_error_and_hangup(session, ws, message=message)
            except Exception as play_exc:
                # Last-ditch fallback: just close the WS so Twilio
                # hangs up. The call ends, no silent limbo.
                log.exception("play_error_and_hangup failed: %s", play_exc)
                try:
                    await ws.close()
                except Exception:
                    pass
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


@app.get("/sessions", dependencies=[Depends(require_auth)])
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


@app.get("/sessions/{session_key}", dependencies=[Depends(require_auth)])
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


@app.patch("/sessions/{session_key}", dependencies=[Depends(require_auth)])
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


@app.get("/tenants", dependencies=[Depends(require_auth)])
async def list_tenants() -> dict:
    """List all configured tenants."""
    tenants = tenant_store.list_all()
    return {
        "tenants": [t.to_dict(include_secrets=False) for t in tenants],
        "count": len(tenants),
    }


@app.post("/tenants", dependencies=[Depends(require_auth)])
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


@app.get("/tenants/{tenant_id}", dependencies=[Depends(require_auth)])
async def get_tenant(tenant_id: str) -> dict:
    """Get a tenant's configuration (secrets are masked)."""
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    return tenant.to_dict(include_secrets=False)


@app.patch("/tenants/{tenant_id}", dependencies=[Depends(require_auth)])
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


@app.delete("/tenants/{tenant_id}", dependencies=[Depends(require_auth)])
async def delete_tenant(tenant_id: str) -> dict:
    """Delete a tenant configuration."""
    deleted = tenant_store.delete(tenant_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    log.info("[TENANT] Deleted tenant %s", tenant_id)
    return {"deleted": True, "tenant_id": tenant_id}


@app.post("/tenants/{tenant_id}/validate-twilio", dependencies=[Depends(require_auth)])
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


@app.post("/tenants/{tenant_id}/configure-webhook", dependencies=[Depends(require_auth)])
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


@app.post("/tenants/{tenant_id}/list-numbers", dependencies=[Depends(require_auth)])
async def list_tenant_numbers(tenant_id: str) -> dict:
    """List all phone numbers in the tenant's Twilio account."""
    from STT_server.adapters.twilio_api import list_phone_numbers
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    if not tenant.twilio_account_sid or not tenant.twilio_auth_token:
        return JSONResponse(status_code=400, content={"error": "Tenant does not have Twilio credentials configured"})

    return await list_phone_numbers(tenant.twilio_account_sid, tenant.twilio_auth_token)


@app.post("/tenants/{tenant_id}/call", dependencies=[Depends(require_auth)])
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
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)