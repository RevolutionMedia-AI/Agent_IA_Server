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
    DEEPGRAM_STT_LANGUAGE_HINT,
    PORT,
    PUBLIC_URL,
    TWILIO_SR,
)
from STT_server.domain.language import detect_language, split_tts_segments, sanitize_tts_text
from STT_server.domain.session import CallSession, VALID_TTS_PROVIDERS, VALID_LANGUAGES
from STT_server.domain.tenant import TenantConfig, tenant_store
from STT_server import db_tenants
from STT_server.db import is_postgres
from STT_server.services.audio_ingest import handle_incoming_media
from STT_server.services.common import require_debug_endpoints
from STT_server.services._instrumentation import Stages
from STT_server.services.reconnect import BackoffPolicy, with_backoff
from STT_server.services.playback_service import playback_loop
from STT_server.services.session_runtime import cleanup_session, monitor_idle_silence, monitor_max_call_duration, register_session, track_task
from STT_server.services.turn_manager import announce_stt_failure_once, enqueue_transcript_event, process_transcripts
from STT_server.services.wait_signals import set_stream_ready


# ponytail: was WARNING — the user reported "no more logs after matched
# phone" because every key step after that was at INFO and got
# suppressed. Bumped to INFO so the call lifecycle (WS connect,
# STT dispatch, agent config load, session.update, transcripts,
# LLM responses) is visible. Production log volume is fine on Railway
# at INFO; the third-party noise is still quieted below.
logging.basicConfig(level=logging.INFO)
# Reduce verbosity of commonly noisy third-party loggers (uvicorn/access, websockets)
for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error", "websockets", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("stt_server")
_WS_BACKOFF = BackoffPolicy(base_ms=250, max_ms=8000, factor=2.0)
# ponytail: P2 round-2 — the with_backoff helper is now imported but not
# yet wired around the Twilio WS accept. Activate it when prod logs show
# transient disconnect storms (call_handlers with `asyncio.CancelledError`
# or `WebSocketDisconnect`).

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


def _safe_float(v) -> float | None:
    """Coerce agent-config JSON values to float, returning None on
    empty / non-numeric / out-of-band inputs. The agent row may store
    these as null (no override), a number, or — if the FE ever sends
    a string — something we can't trust. The DB CHECK constraints are
    the real safety net; this is just belt + suspenders."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int | None:
    """See _safe_float. Same coercion for integers (llm_max_tokens)."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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

    # ponytail: 2026-08-14 — pre-generate the static greeting WAVs at
    # boot so the first call after deploy doesn't pay the TTS TTFB
    # (the operator reported "el saludo llega después de 28 Segundos
    # de iniciada la llamada" without this — most of those 28 s are
    # Twilio WS handshake + buffering, all of which we skip when
    # /voice returns TwiML ``<Play>`` of a pre-existing file).
    # Best-effort: if the TTS provider is unreachable at boot the
    # task logs and continues; the /voice handler will lazily
    # generate the file on the next call.
    import asyncio
    from pathlib import Path
    from STT_server.services.greeting import pregenerate_greeting_at_startup

    static_dir = Path(os.path.dirname(__file__)) / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    try:
        from STT_server.config import (
            DEFAULT_TTS_PROVIDER,
            INITIAL_GREETING_TEXT_EN,
            INITIAL_GREETING_TEXT_ES,
        )

        def _resolve_api_key_for_provider(provider: str) -> str:
            """Resolve the TTS API key for boot-time pre-generation.

            Tries the per-user resolver first (canonical path) and
            falls back to the env-var lookup. Returns '' if neither
            yields a key — the pre-generation then skips this
            provider and the in-band greeting fires as a fallback.
            """
            try:
                from STT_server.services.credentials_resolver import resolve_provider
                creds = resolve_provider(None, provider)
                key = (creds.get("api_key") or "").strip()
                if key:
                    return key
            except Exception:
                pass
            return os.getenv(f"{provider.upper()}_API_KEY", "").strip()

        pregenerate_greeting_at_startup(
            static_dir=static_dir,
            texts={
                "en": INITIAL_GREETING_TEXT_EN,
                "es": INITIAL_GREETING_TEXT_ES,
            },
            voice_ids={"en": "", "es": ""},
            api_key_resolver=lambda: _resolve_api_key_for_provider(
                DEFAULT_TTS_PROVIDER
            ),
            tts_provider=DEFAULT_TTS_PROVIDER,
        )
    except Exception as exc:
        log.warning(
            "[startup] greeting pre-generation bootstrap failed: %s", exc,
        )

    # ponytail: emit a heartbeat log every 60s so Railway's container
    # cycles (if any) are visible — every "container alive Ns" line is a
    # proof the worker is up. If you see two of these in a row without
    # any other logs in between, the container just restarted.
    import asyncio

    async def _heartbeat():
        counter = {"n": 0}
        while True:
            # found in audit; preserve unless you know why — 60s is the
            # cadence for the "[heartbeat] container alive" log; not a
            # first-turn wait.
            await asyncio.sleep(60)
            counter["n"] += 1
            try:
                log.info("[heartbeat] container alive t+%ds", counter["n"] * 60)
            except Exception:
                # Log handler itself blew up — don't let the heartbeat
                # itself crash the worker.
                pass

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        yield
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

# CORS allowlist — comma-separated env var. Defaults include both
# the local dev origins AND the production Railway frontend
# (agentiafrontend-production.up.railway.app) so the deploy works
# out of the box without an explicit env var. Operators can override
# the list by setting ALLOWED_ORIGINS to a comma-separated list.
# ponytail: when this list doesn't include the FE origin, the
# browser reports a CORS error on every response (4xx and 5xx) even
# though the BE itself is fine — the symptom looks like a 500 even
# when it's just 401. Defaulting to the live Railway origin is the
# practical fix; security still rests on bearer-token auth.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,"
    "http://localhost:3000,"
    "http://127.0.0.1:5173,"
    "https://agentiafrontend-production.up.railway.app"
)
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS",
        _DEFAULT_CORS_ORIGINS,
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
log.debug("[CORS] allowed origins: %s", ALLOWED_ORIGINS)


# ponytail: bearer-token → request.state.user middleware so ownership
# checks (`request.state.user["user_id"]`) work in route handlers that
# declare ``dependencies=[Depends(require_auth)]`` (which discards the
# dep's return value). The middleware is best-effort: a missing /
# expired / invalid token simply leaves request.state.user unset;
# require_auth still raises 401 for protected routes. Public routes
# (e.g. /health, /voice webhook) ignore the state. Per-request lookup
# is cheap (in-process dict read) and runs only when a Bearer header
# is present.
@app.middleware("http")
async def _attach_user_state(request: Request, call_next):
    try:
        authz = request.headers.get("authorization") or request.headers.get("Authorization")
        if authz and authz.lower().startswith("bearer "):
            token = authz[7:].strip()
            if token:
                # Lazy import — keeps STT_Server.py import cheap and
                # avoids a circular reference (api.py imports from
                # here too).
                from STT_server.routes.api import _parse_expires_at, _resolve_session_entry
                entry = _resolve_session_entry(token)
                if entry is not None:
                    try:
                        expires_at = _parse_expires_at(entry["expires_at"])
                    except Exception:
                        entry = None
                    else:
                        from datetime import datetime, timezone as _tz
                        if datetime.now(_tz.utc) > expires_at:
                            entry = None
                if entry is not None:
                    request.state.user = dict(entry)
    except Exception:
        # Never let the middleware fail the request — require_auth
        # still has the dependency-injection gate for protected routes.
        log.exception("[auth] user-state middleware failed (continuing without)")
    return await call_next(request)


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
    X-Twilio-Signature header against the per-number Twilio auth token.
    Reject any request whose signature doesn't match. No env fallback —
    the user enters their Twilio subaccount credentials when they
    connect a number (ModalConnectNumber → phone_numbers.twilio_auth_token),
    and that's the only source of truth.
    """
    # ponytail: read the form once. Twilio sends it as
    # application/x-www-form-urlencoded; the signature verifier and
    # the agent-lookup both need the same dict. Caching it here
    # means the second consumer (the agent lookup later in this
    # function) just reuses form_dict instead of re-parsing the
    # body. FastAPI caches the body, so the second .form() would
    # work, but it's wasteful and confusing.
    form_dict: dict = {}
    parse_exc: Exception | None = None
    if request is not None:
        try:
            form = await request.form()
            # Convert to plain dict (form items are multi-dict-aware
            # but Twilio sends each key once). Convert values to str so
            # the signature helper sees the same shape Twilio signed.
            form_dict = {k: str(v) if v is not None else "" for k, v in form.items()}
        except Exception as exc:
            parse_exc = exc
            form_dict = {}
    # ponytail: always log the inbound request at WARNING level so the
    # next "no agent_id in customParameters" regression doesn't go
    # silent. The earlier INFO log was being filtered out and the
    # operator couldn't tell whether /voice was even being called.
    log.warning(
        "[VOICE] hit form_keys=%s parse_ok=%s to=%s ct=%s parse_exc=%s",
        sorted(form_dict.keys()),
        parse_exc is None,
        form_dict.get("To") or form_dict.get("to") or "(missing)",
        request.headers.get("content-type", "(missing)") if request else "(no-request)",
        f"{type(parse_exc).__name__}: {parse_exc}" if parse_exc else "(none)",
    )

    # ponytail: FAIL-CLOSED on parse failure. The previous version
    # only ran the signature check `if form_dict:` — a missing or
    # unparseable form body silently bypassed authentication and
    # fell through to TwiML generation. A bot hitting /voice with
    # `Content-Length: 0` got the agent's TwiML for free. Refuse the
    # request before TwiML is generated; the operator sees the
    # warning and Twilio sees a non-2xx so it doesn't keep retrying.
    if parse_exc is not None or not form_dict:
        log.error(
            "[VOICE] refusing /voice: empty or unparseable form body "
            "from %s (parse_exc=%s). The previous code returned 200 "
            "TwiML anyway — that was a fail-open.",
            request.client.host if request and request.client else "?",
            parse_exc,
        )
        return Response(
            content="Bad Request: empty form body",
            status_code=400,
        )

    # ponytail: per-number Twilio auth token. Twilio signs each webhook
    # with the auth token of the Twilio account that owns the phone
    # number that's calling. The user enters that token via
    # ModalConnectNumber; if missing, REJECT the call — there's no
    # fallback to a global credential (removed per the spec).
    per_number_token = None
    per_number_row_id = None
    from STT_server.adapters.twilio_api import validate_twilio_signature
    from STT_server.db_phone_numbers import find_by_number as _find_num_for_sig
    try:
        called_to = form_dict.get("To") or form_dict.get("to")
        if called_to:
            row = _find_num_for_sig(called_to)
            if row:
                per_number_row_id = row.get("id")
                per_number_token = row.get("twilio_auth_token") or None
    except Exception as exc:
        log.warning("[VOICE] could not look up per-number auth token: %s", exc)
    token_to_check = per_number_token
    # ponytail: env fallback gone. If the phone number row has no
    # twilio_auth_token, we refuse the call with 503 — the operator
    # must edit the number and save the Twilio credentials before
    # the webhook can route. No silent global-key acceptance.
    if not token_to_check:
        log.error(
            "[VOICE] phone row %s has no twilio_auth_token — refusing "
            "inbound call to To=%s. The user must save the Twilio "
            "subaccount credentials via the Edit-number modal before "
            "the number can route.",
            per_number_row_id or "(unresolved)",
            called_to or "(missing)",
        )
        return Response(
            content="Phone number has no Twilio auth token configured. "
                   "Edit the number and save the Twilio credentials.",
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
    # ponytail: log the ingredients so the next "invalid signature"
    # doesn't need a redeploy cycle to triage. Twilio signs the
    # URL exactly as it sent the webhook; if PUBLIC_URL diverges
    # (typo in env, trailing slash, http vs https) the signature
    # mismatch is silent.
    if not validate_twilio_signature(
        token_to_check, signature_url, sig, form_dict
    ):
        log.warning(
            "[VOICE] invalid Twilio signature from %s (per_number=%s) "
            "public_url=%r sig_url=%r tok_prefix=%s... recv_sig_prefix=%s...",
            request.client.host if request.client else "?",
            bool(per_number_token),
            PUBLIC_URL,
            signature_url,
            token_to_check[:8],
            sig[:8],
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

    # ponytail: 2026-08-14 — TwiML ``<Play>`` of a pre-generated static
    # greeting is OPT-IN (default OFF). The previous default-permissive
    # behaviour caused the operator to see 30+ seconds of dead silence
    # on every call: when the boot pre-generation had failed (no TTS
    # API key was resolvable at user_id=None), the static file did
    # NOT exist, the URL was a 404, and Twilio waited for the
    # configured 404 timeout before opening the WebSocket. The WS
    # audio was lost during that wait and the user heard nothing.
    # With the opt-in default below, /voice returns plain
    # ``<Connect>`` TwiML unless the pre-generation is known to have
    # succeeded. The in-band greeting (which already has
    # INITIAL_GREETING_MAX_CHARS + MIN_UTTERANCE_VOICE_FRAMES
    # protecting it) fires after the WS opens — same behaviour the
    # operator had before commit fc4da91.
    from STT_server.config import INITIAL_GREETING_TWIML_PLAY
    play_section = ""
    if INITIAL_GREETING_TWIML_PLAY:
        try:
            from pathlib import Path
            from STT_server.services.greeting import static_greeting_path
            static_dir = Path(os.path.dirname(__file__)) / "static"
            static_dir.mkdir(parents=True, exist_ok=True)
            # The /voice webhook doesn't know which language the call
            # is yet — the agent's preferred_language is set later in
            # the start event. Default to ES (the operator's primary
            # market).
            lang = (os.getenv("DEFAULT_CALL_LANGUAGE", "es") or "es").strip().lower()
            if not lang.startswith("en"):
                lang = "es"
            greeting_path = static_greeting_path(static_dir, lang)
            if greeting_path.exists() and greeting_path.stat().st_size > 44:
                play_section = (
                    f'<Play>{PUBLIC_URL.rstrip("/")}/static/{greeting_path.name}</Play>'
                )
                log.info(
                    "[VOICE] including TwiML <Play> for greeting %s (%.1f KB)",
                    greeting_path.name,
                    greeting_path.stat().st_size / 1024,
                )
        except Exception as exc:
            log.warning("[VOICE] could not build TwiML <Play>: %s", exc)

    # ponytail: bug history. This template used {stream_params} (a
    # Python list), which rendered as ['<Parameter .../>', ...] inside
    # the <Stream> element. Twilio's TwiML parser treated that literal
    # text as a child text node, NOT as <Parameter> elements, and
    # silently dropped them. Net effect: the WebSocket 'start' event
    # arrived with customParameters={} and the call ran without an
    # agent_id (see [AGENT] has no agent_id in customParameters log).
    # stream_params_str was already joined above for exactly this
    # purpose; use it.
    if play_section:
        twiml = f"""
    <Response>
        {play_section}
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
            # found in audit; preserve unless you know why — 5s cadence
            # for the H5 assistant_speaking watchdog; not a first-turn wait.
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


async def _pump_realtime_transcripts_to_central_queue(session: CallSession) -> None:
    """ponytail: P3 — OpenAI Realtime provider bypassed process_transcripts
    (memory updates, anti-loop, replace-current). This pump forwards
    realtime_text_queue events to session.transcript_queue so the central
    pipeline sees them."""
    from STT_server.services.common import enqueue_nowait_with_drop
    q = getattr(session, "realtime_text_queue", None)
    if q is None:
        return
    while not session.closed:
        try:
            ev = await q.get()
        except asyncio.CancelledError:
            return
        if ev is None:
            return
        # ponytail: realtime text event shape varies (depends on the
        # adapter); forward verbatim. process_transcripts recognises
        # dict events with at least 'text' and 'is_final' fields. Add
        # turn_id for correlation.
        if not isinstance(ev, dict):
            continue
        ev.setdefault("turn_id", getattr(session, "turn_counter", 0))
        # ponytail: only forward final transcripts to the central
        # pipeline; partials are handled in the adapter or ignored.
        if not ev.get("is_final"):
            continue
        if not enqueue_nowait_with_drop(
            session.transcript_queue,
            ev,
            "transcript_queue",
        ):
            log.warning("[REALTIME->TRANSCRIPT] queue full; dropped final transcript for session=%s",
                        session.session_key)


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    # TODO reconnect: wrap call accept with with_backoff when transient failures appear in prod logs
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
                # ponytail: AUDIO-006 — Twilio's contract guarantees
                # PCMU/8000/1 today, but we capture the actual envelope
                # so a future change is loud not silent. Validation runs
                # before register_session; metrics is None here so we
                # use getattr and increment if attach_metrics has run
                # (e.g. on a re-start event).
                media_format_in = start.get("mediaFormat")
                if isinstance(media_format_in, dict):
                    _mf_encoding = media_format_in.get("encoding")
                    _mf_sample_rate = media_format_in.get("sampleRate")
                    _mf_channels = media_format_in.get("channels")
                else:
                    _mf_encoding = _mf_sample_rate = _mf_channels = None
                _mf_valid = (
                    isinstance(_mf_encoding, str)
                    and (_mf_encoding.startswith("audio/x-mulaw") or _mf_encoding.startswith("audio/pcmu"))
                    and _mf_sample_rate == 8000
                    and _mf_channels == 1
                )
                if _mf_valid:
                    session.media_format = {
                        "encoding": _mf_encoding,
                        "sample_rate": _mf_sample_rate,
                        "channels": _mf_channels,
                    }
                    session.media_format_mismatch = False
                else:
                    session.media_format_mismatch = True
                    log.warning(
                        "[MEDIA] unexpected mediaFormat: encoding=%r sampleRate=%r channels=%r "
                        "(expected audio/x-mulaw or audio/pcmu @ 8000 Hz, mono) session=%s",
                        _mf_encoding, _mf_sample_rate, _mf_channels, session.session_key,
                    )
                    _mf_metrics = getattr(session, "metrics", None)
                    if _mf_metrics is not None:
                        _mf_metrics.incr("invalid_media_format_total", 1)
                if session.call_sid:
                    session.session_key = session.call_sid
                # ponytail: P0 from the call-flow audit. Now that stream_sid
                # is settled, wake any code that was waiting on the event
                # (the playback loop will switch from polling to waiting
                # in the next refactor). Idempotent: re-set is a no-op.
                set_stream_ready(session)
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
                    # ponytail: agent-level runtime knobs. None means "use
                    # the adapter default" — we don't substitute 0.2 / 150
                    # here so the adapter's existing fallback chain keeps
                    # working unchanged for legacy agents.
                    session.llm_temperature = _safe_float(agent_cfg.get('llm_temperature'))
                    # ponytail: hotfix for production bug where agent rows
                    # stored llm_temperature=-1 (a FE sentinel for "use
                    # default") were passed verbatim to OpenAI/Anthropic/
                    # Gemini/MiniMax. The adapters' `is not None` check
                    # let -1.0 through, which made providers default to ~1.0
                    # (their internal fallback), producing over-eager
                    # interpretations like "goodbye on first turn" when
                    # the user said only one short word. Treat any negative
                    # value as "no override" so the adapter default (0.2)
                    # applies. This is the ONLY conditional reset — the
                    # fields below (llm_max_tokens, tts_speed, TTS hint,
                    # tool loading) are independent and must ALWAYS apply
                    # when an agent row is present, regardless of the
                    # temperature value.
                    if session.llm_temperature is not None and session.llm_temperature < 0:
                        session.llm_temperature = None
                    # ponytail: llm_max_tokens + tts_speed are
                    # independent of llm_temperature. Apply them whenever
                    # the agent row is present (the previous code gated
                    # them on llm_temperature<0, so a legitimate
                    # temperature=0.7 agent row with
                    # llm_max_tokens=200 was silently ignored).
                    session.llm_max_tokens = _safe_int(agent_cfg.get('llm_max_tokens'))
                    session.tts_speed = _safe_float(agent_cfg.get('tts_speed'))
                    # ponytail: per-agent idle / silence detection
                    # (008_agent_idle_settings.sql). None on every field
                    # keeps the legacy global IDLE_SILENCE_TIMEOUT_SEC
                    # behaviour so existing agents are unaffected. The
                    # monitor in session_runtime.monitor_idle_silence
                    # reads these on every poll tick.
                    session.idle_enabled = agent_cfg.get('idle_enabled')
                    session.idle_first_timeout_sec = _safe_int(agent_cfg.get('idle_first_timeout_sec'))
                    session.idle_first_message = agent_cfg.get('idle_first_message')
                    session.idle_subsequent_timeout_sec = _safe_int(agent_cfg.get('idle_subsequent_timeout_sec'))
                    session.idle_final_message = agent_cfg.get('idle_final_message')
                    session.idle_disconnect_timeout_sec = _safe_int(agent_cfg.get('idle_disconnect_timeout_sec'))
                    session.idle_max_attempts = _safe_int(agent_cfg.get('idle_max_attempts'))
                    # ponytail: Twilio subaccount auth for the call_transfer
                    # tool executor. We look up the phone_numbers row that
                    # owns this agent (most recently created if multiple),
                    # copy the sid + token onto the session, and the
                    # executor reads from there at tool-call time. Missing
                    # either field is fine — the executor surfaces a clear
                    # error if the LLM tries to invoke a call_transfer tool
                    # on a session without Twilio auth.
                    if session.user_id and session.agent_id:
                        try:
                            from STT_server.db_phone_numbers import find_for_agent as _find_num_for_agent
                            num_row = _find_num_for_agent(session.user_id, session.agent_id)
                            if num_row:
                                session.twilio_account_sid = num_row.get("twilio_account_sid") or None
                                session.twilio_auth_token = num_row.get("twilio_auth_token") or None
                                if session.twilio_account_sid and session.twilio_auth_token:
                                    log.info(
                                        "[TRANSFER] Twilio auth denormalized for %s (sid=%s...)",
                                        session.session_key,
                                        session.twilio_account_sid[:6] or "?",
                                    )
                                else:
                                    log.warning(
                                        "[TRANSFER] phone row %s has no twilio_account_sid/auth_token; "
                                        "call_transfer tools will be unavailable for session %s",
                                        num_row.get("id"), session.session_key,
                                    )
                        except Exception as exc:
                            log.warning(
                                "[TRANSFER] phone lookup failed for session=%s agent=%s: %s",
                                session.session_key, session.agent_id, exc,
                            )
                    # ponytail: credential-source toggle per slot
                    # (009_agent_use_own_key.sql). Denormalized alongside
                    # the rest of the agent_cfg so the TTS/STT/LLM
                    # adapters can pick the right resolver mode in one
                    # read. False = resolver may fall back to platform
                    # env; True = resolver ignores platform env.
                    session.stt_use_own_key = bool(agent_cfg.get("stt_use_own_key"))
                    session.llm_use_own_key = bool(agent_cfg.get("llm_use_own_key"))
                    session.tts_use_own_key = bool(agent_cfg.get("tts_use_own_key"))
                    # ponytail: prepend a per-TTS-provider hint to the
                    # system prompt so the LLM emits the right inline
                    # non-verbal tags (e.g. Inworld's steering). The
                    # previous version appended at the end; gpt-4o-mini
                    # treated end-of-prompt additions as low-priority
                    # context and rarely emitted the tags — the caller
                    # heard a flat delivery. Prepending puts the voice
                    # direction at the top of the system message where
                    # the LLM pays attention. Idempotent: a double
                    # prepend on WS reconnect is prevented by the
                    # has_tts_hint() check. Only runs when tts_provider
                    # is set and the agent has a non-empty custom_prompt;
                    # the in-memory session.custom_prompt is mutated
                    # in-place (not persisted to DB).
                    if session.custom_prompt and session.tts_provider:
                        from STT_server.domain.tts_hints import get_tts_hint, has_tts_hint
                        if not has_tts_hint(session.custom_prompt):
                            hint = get_tts_hint(session.tts_provider)
                            if hint:
                                # NOTE: the order is `hint + custom_prompt`,
                                # not `custom_prompt + hint`. The hint
                                # teaches the LLM that the voice
                                # supports steering and gives a worked
                                # example; placing it at the top of the
                                # system message makes the model
                                # consistently emit the tags. The
                                # agent's instructions stay in the
                                # dominant middle position.
                                session.custom_prompt = hint + session.custom_prompt
                                log.info(
                                    "[AGENT] Prepended TTS hint to custom_prompt for session %s "
                                    "(provider=%s, hint_len=%d, total_len=%d)",
                                    session.session_key, session.tts_provider,
                                    len(hint), len(session.custom_prompt),
                                )
                                # ponytail: dump the first 200 chars of the
                                # final custom_prompt so the operator can
                                # confirm in production that the hint is
                                # at the top of the system message (not
                                # buried after 10KB of Tessa's prompt).
                                # The "gesture vocabulary" line is the
                                # canonical marker; if you grep the
                                # session init log and don't see it in the
                                # first 200 chars, the prepend didn't run
                                # (likely tts_provider != 'inworld' or
                                # has_tts_hint() detected a double-prepend).
                                log.info(
                                    "[AGENT] custom_prompt HEAD session=%s len=%d preview=%r",
                                    session.session_key,
                                    len(session.custom_prompt),
                                    session.custom_prompt[:200],
                                )
                    log.info("[AGENT] session %s stt=%s model=%s llm=%s/%s T=%.2f MT=%s tts=%s/%s voice=%s speed=%.2f",
                             session.session_key,
                             session.stt_provider or '-', session.stt_model or '-',
                             session.llm_provider, session.llm_model or '-',
                             session.llm_temperature if session.llm_temperature is not None else -1.0,
                             session.llm_max_tokens if session.llm_max_tokens is not None else -1,
                             session.tts_provider or '-', session.tts_model or '-',
                             session.tts_voice or '-',
                             session.tts_speed if session.tts_speed is not None else -1.0)
                    if session.user_id:
                        log.info("[AGENT] session %s user_id=%s (per-user keys)",
                                 session.session_key, session.user_id)
                    # Load agent tools for function calling
                    from STT_server.services.session_runtime import _load_agent_tools
                    # ponytail: pass user_id so shared tools owned by
                    # this user are also loaded (the marketplace
                    # pattern: shared tools live with agent_id=
                    # "__shared__" in the same JSON file).
                    session.agent_tools = _load_agent_tools(session.agent_id, session.user_id)
                    if session.agent_tools:
                        log.info("[TOOLS] Loaded %d tools for agent %s", len(session.agent_tools), session.agent_id)
                # ponytail: helper that closes over `session` so the
                # caller can `await _enqueue_transcript(item)` instead
                # of `await lambda item: enqueue_transcript_event(session, item)`.
                # The async-def makes it self-documenting (the IDE flags
                # missing await) and survives a refactor that drops the
                # outer `await` — the previous lambda silently lost the
                # transcript in that case.
                async def _enqueue_transcript(item: dict) -> None:
                    await enqueue_transcript_event(session, item)
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
                    # ponytail: P3 — start the relay pump so realtime
                    # transcripts flow through process_transcripts
                    # (memory / anti-loop / replace-current / order
                    # escalation). Belt-and-suspenders: existing
                    # realtime dispatch path is untouched.
                    pump_task = asyncio.create_task(_pump_realtime_transcripts_to_central_queue(session))
                    session.tasks.add(pump_task)
                elif session.stt_provider == 'inworld':
                    track_task(
                        session,
                        asyncio.create_task(
                            run_inworld_realtime_stt(
                                session,
                                _enqueue_transcript,
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
                                _enqueue_transcript,
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
                                _enqueue_transcript,
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
                # ponytail: 2026-08-14 — fire the initial greeting as soon as the
                # call connects, so the caller hears the agent greet
                # them within ~300 ms (TTS TTFB) instead of dead-air
                # silence. The greeting text is resolved inside
                # ``play_initial_greeting`` with priority:
                #   1. session.welcome_message (per-agent override)
                #   2. INITIAL_GREETING_TEXT_ES / _EN (platform fallback
                #      picked by session.preferred_language)
                # If neither resolves, the function is a no-op and
                # the call starts silent (preserves opt-out for
                # agents that explicitly disable via
                # INITIAL_GREETING_ENABLED=false).
                from STT_server.services.playback_service import play_initial_greeting
                track_task(
                    session,
                    asyncio.create_task(play_initial_greeting(session))
                )
                track_task(session, asyncio.create_task(monitor_idle_silence(session, ws)))
                # Guardrail: hard-timeout to prevent phantom calls.
                # If the call runs longer than MAX_CALL_DURATION_SEC, we
                # force-close even if audio is still flowing.
                track_task(session, asyncio.create_task(monitor_max_call_duration(session, ws)))
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
                # ponytail: P2 round-2 — wire the sequence-number tracker from
                # twilio_media.py. Counts gaps, duplicates, reorders per stream_sid.
                try:
                    from STT_server.adapters.twilio_media import track_twilio_sequence
                    track_twilio_sequence(session, msg)
                except Exception:
                    log.exception("track_twilio_sequence failed")
                await handle_incoming_media(session, msg["media"]["payload"])
                continue

            if event == "mark":
                mark = msg.get("mark", {}).get("name") if isinstance(msg.get("mark"), dict) else msg.get("mark")
                pending = getattr(session, "pending_marks", None)
                if isinstance(pending, dict) and mark and mark in pending:
                    # ponytail: AUDIO echo gate — parse the generation
                    # out of the mark name so we can decrement the
                    # per-generation counter only when the mark belongs
                    # to the ACTIVE generation. The previous code
                    # compared ``not pending`` after every pop, which
                    # fired on every segment of a multi-segment reply
                    # (one at a time) and let VAD grab echo between
                    # segments. Counter only drops when ALL segments
                    # of the current generation have been acked.
                    _gen_match = re.match(r"^gen-(\d+)-seg-\d+$", mark)
                    _ack_generation = int(_gen_match.group(1)) if _gen_match else None
                    sent_at = pending.pop(mark, None)
                    if sent_at is not None:
                        rtt_ms = (time.monotonic() - sent_at) * 1000.0
                        timer = getattr(session, "stage_timer", None)
                        if timer is not None:
                            try:
                                timer.mark(Stages.TWILIO_MARK_ACK)
                            except Exception:
                                pass
                        metrics = getattr(session, "metrics", None)
                        if metrics is not None:
                            try:
                                metrics.observe_ms("mark_ack_rtt_ms", rtt_ms)
                                metrics.incr("mark_acks")
                            except Exception:
                                pass
                        log.info("[MARK_ACK] session=%s mark=%s gen=%s rtt_ms=%.1f",
                                 session.session_key, mark, _ack_generation, rtt_ms)
                    # Decrement counter only when the mark belongs to
                    # the currently active generation; otherwise the
                    # counter is stale (we're in a new turn, the old
                    # turn's marks are racing in). Floor at zero so a
                    # leaked ack can never push the metric negative.
                    if _ack_generation is not None and _ack_generation == session.active_generation:
                        session.pending_playback_marks = max(0, session.pending_playback_marks - 1)
                        # The whole turn has finished playing when
                        # both conditions hold: the counter is zero
                        # AND we were actually speaking (so we don't
                        # fire on a stray ack from a generation we
                        # never sent audio for).
                        if session.pending_playback_marks == 0 and session.assistant_speaking:
                            session.assistant_speaking = False
                            session.assistant_started_at = None
                            log.info(
                                "[MARK_ACK] generation=%s fully played — assistant_speaking -> False for session=%s",
                                _ack_generation, session.session_key,
                            )
                continue

            if event == "dtmf":
                log.info("DTMF recibido en %s: %s", session.session_key, msg.get("dtmf", {}).get("digit"))
                continue

            if event == "stop":
                log.info("Stream stop para %s", session.session_key)
                # ponytail: P2 round-3 — emit Twilio sequence-number summary
                # (gaps/dupes/reorders counted per stream). Idempotent if
                # session.metrics is missing.
                try:
                    from STT_server.adapters.twilio_media import summarize_twilio_sequence
                    summarize_twilio_sequence(session)
                except Exception:
                    log.exception("summarize_twilio_sequence failed")
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
        # ponytail: env fallback for tts_ready gone; this debug endpoint
        # used to confirm "system has TTS configured". Without env, the
        # caller must supply a key as a query param (caller is also
        # required_debug_endpoints so this isn't a public surface).
        "tts_ready": bool(getattr(dummy_session, "user_id", None)),
    }


@app.post("/test-stt")
async def test_stt(api_key: str = "") -> dict:
    require_debug_endpoints()
    from STT_server.adapters.deepgram_stt_batch import transcribe_block
    dummy_audio = b"\x00\x00" * TWILIO_SR
    # ponytail: deepgram key now comes from the query param or, if
    # absent, raises — env fallback gone.
    if not api_key:
        return {"error": "api_key query param is required (env fallback removed)"}
    texts, language = await transcribe_block(dummy_audio, api_key=api_key, language_hint=DEEPGRAM_STT_LANGUAGE_HINT)
    return {
        "text": " ".join(texts).strip(),
        "segments": texts,
        "language": language,
        "stt_ready": True,
        "model": "user-supplied",
    }


@app.get("/list-models")
async def list_available_models(api_key: str = "") -> dict:
    require_debug_endpoints()
    return await list_models(api_key=api_key or None)


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
async def list_sessions(request: Request) -> dict:
    """List active call sessions owned by the authenticated user.

    ponytail: the previous version returned every active session for
    every caller — any admin token could see (and PATCH) another
    admin's live calls. Now we filter by the session's resolved user
    (set from tenant_id at call start), with admin tokens able to
    see sessions whose owner is unset (legacy).
    """
    from STT_server.services.session_runtime import sessions as _sessions
    user = getattr(request.state, "user", None) or {}
    caller_id = user.get("user_id")
    caller_role = (user.get("role") or "").strip().lower()
    result = {}
    for key, s in _sessions.items():
        owner = getattr(s, "user_id", None)
        if owner and owner != caller_id and caller_role != "admin":
            continue
        if not owner and caller_role != "admin":
            continue
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


def _require_session_owner(request: Request, session) -> "JSONResponse | None":
    user = getattr(request.state, "user", None) or {}
    caller_id = user.get("user_id")
    caller_role = (user.get("role") or "").strip().lower()
    owner = getattr(session, "user_id", None)
    if owner and owner == caller_id:
        return None
    if not owner and caller_role == "admin":
        return None
    return JSONResponse(
        status_code=404,
        content={"error": f"Session '{session.session_key}' not found"},
    )


@app.get("/sessions/{session_key}", dependencies=[Depends(require_auth)])
async def get_session_config(session_key: str, request: Request) -> dict:
    """Get the configuration of a specific session. 404 on owner mismatch."""
    from STT_server.services.session_runtime import sessions as _sessions
    session = _sessions.get(session_key)
    if not session:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_key}' not found"})
    forbidden = _require_session_owner(request, session)
    if forbidden is not None:
        return forbidden
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
async def update_session_config(session_key: str, request: Request, body: SessionConfigUpdate = None) -> dict:
    """Update per-session configuration. 404 on owner mismatch."""
    from STT_server.services.session_runtime import sessions as _sessions
    session = _sessions.get(session_key)
    if not session:
        return JSONResponse(status_code=404, content={"error": f"Session '{session_key}' not found"})
    forbidden = _require_session_owner(request, session)
    if forbidden is not None:
        return forbidden

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
async def list_tenants(request: Request) -> dict:
    """List tenants owned by the authenticated user.

    ponytail: the previous version returned every tenant in the store
    regardless of who was calling. Any admin token could see every
    other admin's Twilio subaccount, phone numbers, and agent config.
    Now we filter by ``tenant.user_id == request.state.user['user_id']``
    so a user only sees their own. Tenants created before this field
    existed (no user_id) are only visible to admin tokens (role=admin).
    """
    user = getattr(request.state, "user", None) or {}
    caller_id = user.get("user_id")
    caller_role = (user.get("role") or "").strip().lower()
    tenants = tenant_store.list_all()
    visible = []
    for t in tenants:
        owner = getattr(t, "user_id", None)
        if owner and owner == caller_id:
            visible.append(t)
        elif not owner and caller_role == "admin":
            visible.append(t)
    return {
        "tenants": [t.to_dict(include_secrets=False) for t in visible],
        "count": len(visible),
    }


def _require_tenant_owner(request: Request, tenant: "TenantConfig"):
    """Raise 404 (not 403, to avoid leaking existence) when the caller
    doesn't own the tenant. Returns None on success."""
    user = getattr(request.state, "user", None) or {}
    caller_id = user.get("user_id")
    caller_role = (user.get("role") or "").strip().lower()
    owner = getattr(tenant, "user_id", None)
    if owner and owner == caller_id:
        return None
    if not owner and caller_role == "admin":
        return None
    return JSONResponse(
        status_code=404,
        content={"error": f"Tenant '{tenant.tenant_id}' not found"},
    )


@app.post("/tenants", dependencies=[Depends(require_auth)])
async def create_tenant(request: Request, body: TenantCreateRequest) -> dict:
    """Create a new tenant with Twilio credentials and agent configuration.

    ponytail: stamp the creating user's id onto the tenant so subsequent
    list/get/patch/delete can scope to owner. Without this, the tenant
    becomes orphaned and is only visible to admin role tokens.
    """
    import uuid
    tenant_id = f"tenant-{uuid.uuid4().hex[:12]}"
    user = getattr(request.state, "user", None) or {}
    caller_id = user.get("user_id")

    tenant = TenantConfig(
        tenant_id=tenant_id,
        user_id=caller_id,
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
    log.info("[TENANT] Created tenant %s (%s) owner=%s", tenant_id, body.name, caller_id or "(none)")

    return {
        "tenant_id": tenant_id,
        "config": tenant.to_dict(include_secrets=False),
    }


@app.get("/tenants/{tenant_id}", dependencies=[Depends(require_auth)])
async def get_tenant(tenant_id: str, request: Request) -> dict:
    """Get a tenant's configuration (secrets are masked). 404 on owner mismatch."""
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    forbidden = _require_tenant_owner(request, tenant)
    if forbidden is not None:
        return forbidden
    return tenant.to_dict(include_secrets=False)


@app.patch("/tenants/{tenant_id}", dependencies=[Depends(require_auth)])
async def update_tenant(tenant_id: str, body: TenantCreateRequest, request: Request) -> dict:
    """Update a tenant's configuration. Only provided fields are updated.

    ponytail: ownership check added. PATCH on a tenant you don't own
    returns 404 (same shape as not-found) so we don't leak the existence
    of tenants belonging to other users.
    """
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    forbidden = _require_tenant_owner(request, tenant)
    if forbidden is not None:
        return forbidden

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
async def delete_tenant(tenant_id: str, request: Request) -> dict:
    """Delete a tenant configuration. 404 on owner mismatch."""
    tenant = tenant_store.get(tenant_id)
    if tenant is not None:
        forbidden = _require_tenant_owner(request, tenant)
        if forbidden is not None:
            return forbidden
    deleted = tenant_store.delete(tenant_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    log.info("[TENANT] Deleted tenant %s", tenant_id)
    return {"deleted": True, "tenant_id": tenant_id}


@app.post("/tenants/{tenant_id}/validate-twilio", dependencies=[Depends(require_auth)])
async def validate_tenant_twilio(tenant_id: str, request: Request) -> dict:
    """Validate a tenant's Twilio credentials. 404 on owner mismatch."""
    from STT_server.adapters.twilio_api import validate_twilio_credentials
    tenant = tenant_store.get(tenant_id)
    if not tenant:
        return JSONResponse(status_code=404, content={"error": f"Tenant '{tenant_id}' not found"})
    forbidden = _require_tenant_owner(request, tenant)
    if forbidden is not None:
        return forbidden
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