"""
Admin API routes for the frontend app: dashboard stats, agents, phone numbers,
tools/integrations, settings, and /me alias. All endpoints (except where noted)
require a valid Bearer token; auth helper reads STT_server/data/sessions.json.
Persistence is JSON-file based so the user can swap to SQLite/Postgres later
without breaking the route contracts.
"""
import os
import json
import uuid
import re
import hashlib
import logging
import threading
import time
import urllib.parse
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

# ponytail: log was referenced in 8 places (lines 223, 228, 610, 615, 793,
# 1001, 1021, ...) but never defined. Every endpoint that touched
# one of those lines crashed with NameError -> 500 -> no CORS
# headers. Defined once here, used by all handlers.
log = logging.getLogger("stt_server.routes.api")

from STT_server.security.credentials import (
    encrypt_credentials, decrypt_credentials, decrypt_value,
    encrypt_value,
)
from STT_server.services.credentials_resolver import (
    PROVIDER_CATALOG,
    get_provider_spec,
    is_provider_configured,
    list_provider_models,
    resolve_provider,
    test_provider,
    validate_credentials,
    _build_categorized_models,
)
from STT_server.db import is_postgres
from STT_server.db_agents import (
    list_agents as db_list_agents,
    create_agent as db_create_agent,
    update_agent as db_update_agent,
    delete_agent as db_delete_agent,
)
from STT_server.db_phone_numbers import (
    list_numbers as db_list_numbers,
    create_number as db_create_number,
    update_number as db_update_number,
    delete_number as db_delete_number,
)
from STT_server.db_tools import (
    list_tools as db_list_tools,
    get_tool as db_get_tool,
    create_tool as db_create_tool,
    update_tool as db_update_tool,
    delete_tool as db_delete_tool,
    add_assignment as db_add_assignment,
    remove_assignment as db_remove_assignment,
)
from STT_server.db_settings import (
    get_settings as db_get_settings,
    upsert_settings as db_upsert_settings,
)
from STT_server.db_campaigns import list_campaigns as db_list_campaigns

VALID_MODEL_SERVICES = {"stt", "tts", "llm"}


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
AGENTS_FILE = os.path.join(DATA_DIR, "agents.json")
NUMBERS_FILE = os.path.join(DATA_DIR, "phone_numbers.json")
SETTINGS_DIR = os.path.join(DATA_DIR, "settings")


# ponytail: single lock around the data dir's read-modify-write sequences.
# Without it, two concurrent creates (e.g. user double-clicks + background reload)
# can race _load → mutate → _save and lose one write. This lock makes every
# RMW atomic. Fine for MVP volume; drop the lock when you swap to SQLite.
_data_io_lock = threading.Lock()


@contextmanager
def _data_lock():
    with _data_io_lock:
        yield


def _load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_password(pwd: str) -> str:
    # ponytail: salted PBKDF2-HMAC-SHA256 with 600k iters — matches
    # the format ``routes/auth.hash_password()`` writes for new
    # registrations and the format ``routes/auth.verify_password()``
    # accepts on login. The password-change endpoint below used to
    # write a bare SHA-256 hash here, which the verify function still
    # accepted (backwards-compat) but the next login would
    # transparently upgrade. Now new passwords are PBKDF2 from the
    # start. Same algorithm as auth.py so a user changing their
    # password and a user registering look identical on disk.
    from STT_server.routes.auth import hash_password
    return hash_password(pwd)


# ponytail: in-memory cache of valid (token -> entry). require_auth first checks
# here, then falls back to the file on miss. Cache is invalidated by
# `auth.py` logout/password-change via `invalidate_session` (no such call exists
# in auth.py yet — see the W7 todo in routes/auth.py). Stale entries are
# lazy-evicted on access when expired.
_session_cache = {}  # token -> {"entry": {...}, "expires_at": datetime}


def _parse_expires_at(raw) -> datetime:
    """Parse a stored expires_at value, tolerating both offset-aware and
    naive ISO strings. Sessions written before the auth refactor don't
    carry a timezone suffix, so we attach UTC defensively — the on-disk
    clock was always UTC. Returns an aware datetime.
    """
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def invalidate_session(token: str) -> None:
    _session_cache.pop(token, None)


def invalidate_user_sessions(user_id: str) -> None:
    for tok, payload in list(_session_cache.items()):
        if payload["entry"].get("user_id") == user_id:
            _session_cache.pop(tok, None)


def require_auth(authorization: str = Header(None)) -> dict:
    """Resolve Bearer token; raise 401 on failure.

    ponytail: usa la misma shim que /auth/login y /me. Antes leia
    STT_server/data/sessions.json mientras el login guardaba en Postgres,
    asi /me devolvia 200 pero /agents y /dashboard/stats (que usan
    require_auth) devolvian 401. El setOnUnauthorized del AuthContext
    se disparaba y el usuario quedaba kicked out aunque el login
    habia sido exitoso.

    La shim load_sessions() de STT_server.db_users decide sola
    entre Postgres (si DATABASE_URL esta seteado) y JSON (fallback),
    asi require_auth queda consistente con el resto del flow.
    """
    entry = resolve_bearer(authorization, raise_on_missing=True)
    return entry  # type: ignore[return-value]


# ponyy: the OAuth start endpoint is the one route that needs to
# handle BOTH the Bearer header (direct API call) and the
# `?token=` query param (window.location navigation from the FE).
# When the operator's session token is expired/missing AND they
# clicked Connect, we want a friendly redirect to /login (not a
# 401 page that the browser renders as HTML). require_auth would
# raise 401 immediately and bypass the route's redirect logic, so
# the route uses this non-raising variant and decides what to do
# when there's no auth at all.
def require_auth_optional(authorization: str = Header(None)) -> dict | None:
    """Like require_auth but returns None instead of raising 401.
    Use this on routes that have a graceful fallback (e.g. the
    OAuth start endpoint redirects to /login when auth is missing
    AND a `?token=` was supplied; with no auth at all the route
    surfaces its own 401 with a useful message)."""
    return resolve_bearer(authorization, raise_on_missing=False)


# ponytail: integrations — admin gate. Only user ids listed in
# ADMIN_USER_IDS (comma-separated env var) can pass `_skip_preflight`
# in the body to /integrations. Empty default = nobody can skip, so
# a missing env var is fail-safe. The route handler pulls
# `_skip_preflight` out of the body before validation; this dep
# just enforces the admin check.
_ADMIN_USER_IDS = frozenset(
    s.strip() for s in os.environ.get("ADMIN_USER_IDS", "").split(",") if s.strip()
)


def require_admin(auth: dict = Depends(require_auth)) -> dict:
    """Gate admin-only actions behind ADMIN_USER_IDS.

    Ponytail: a single-user permission system costs nothing here.
    We don't have a roles table; the env var is the source of
    truth. If we ever add a real role system, this dep is the
    one place to swap.
    """
    if auth.get("user_id") not in _ADMIN_USER_IDS:
        raise HTTPException(
            status_code=403,
            detail="admin only (set ADMIN_USER_IDS to grant access)",
        )
    return auth


# ponytail: integrations — service-to-service auth for n8n. The
# internal endpoint that hands the decrypted credentials to n8n is
# the only place that knows the long-lived INTEGRATIONS_N8N_TOKEN.
# Compared against the env var with hmac.compare_digest so timing
# analysis can't brute-force the token. Empty env var = no one can
# authenticate → every internal call returns 401 until the operator
# sets it. This is the right default (fail closed) — n8n won't
# accidentally start pulling credentials without explicit config.
def _expected_service_token() -> str:
    return os.environ.get("INTEGRATIONS_N8N_TOKEN", "").strip()


def require_service_token(authorization: str = Header(None)) -> dict:
    """Validate the shared bearer token used by n8n to call internal
    endpoints. Returns a synthetic context dict (no user_id — the
    caller is n8n, not a logged-in user)."""
    import hmac
    expected = _expected_service_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="INTEGRATIONS_N8N_TOKEN is not configured on the server",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    presented = authorization[len("Bearer "):].strip()
    # Constant-time compare. hmac.compare_digest returns False on
    # length mismatch without leaking length to a remote attacker
    # (Python's `==` short-circuits on the first non-equal byte).
    if not hmac.compare_digest(presented, expected):
        log.warning(
            "[internal] rejected service-token call from %s",
            # request.client.host is added in the dependency for logging.
            "<unknown>",
        )
        raise HTTPException(status_code=401, detail="Invalid service token")
    return {"caller": "n8n"}


def resolve_bearer(authorization, *, raise_on_missing: bool = False):
    """Resolve a Bearer token to its session entry (or None).

    Public helper so other modules (e.g. STT_Server's bearer-state
    middleware) can reuse the same lookup without re-implementing it.
    `raise_on_missing` selects between the dependency-injection 401
    contract (True) and the best-effort middleware contract (False,
    returns None).
    """
    from STT_server.db_users import load_sessions, save_sessions
    if not authorization or not authorization.startswith("Bearer "):
        if raise_on_missing:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return None
    token = authorization[len("Bearer "):]
    sessions = load_sessions()
    entry = sessions.get(token)
    if not entry:
        if raise_on_missing:
            raise HTTPException(status_code=401, detail="Invalid token")
        return None
    try:
        expires_at = _parse_expires_at(entry["expires_at"])
    except ValueError:
        if raise_on_missing:
            raise HTTPException(status_code=401, detail="Token corrupted")
        return None
    if datetime.now(timezone.utc) > expires_at:
        # Best-effort delete; on Postgres the JSON fallback is a no-op.
        try:
            sessions.pop(token, None)
            save_sessions(sessions)
        except Exception:
            pass
        if raise_on_missing:
            raise HTTPException(status_code=401, detail="Token expired")
        return None
    return entry


# Backwards-compatible alias used by the bearer-state middleware.
def _resolve_session_entry(token: str):
    return resolve_bearer(f"Bearer {token}", raise_on_missing=False)


def _get_user(user_id: str) -> Optional[dict]:
    users = _load(USERS_FILE, [])
    return next((u for u in users if u.get("id") == user_id), None)


api_router = APIRouter()


# ---------- /call-status (Twilio status callback) ----------
# ----------------------------------------------------------------------------
# Twilio's `configure_voice_webhook` sets this URL as the call's
# statusCallback. Twilio POSTs here with form-encoded fields like
# CallSid, CallStatus (initiated/ringing/answered/completed/busy/etc),
# To, From, Direction, Duration. We just record the latest status
# per call_sid in memory; the per-call record already has duration
# at cleanup time so this is mostly observability.
# ----------------------------------------------------------------------------

@api_router.post("/call-status")
async def call_status(request: Request) -> dict:
    """Twilio status callback. Twilio POSTs form-encoded data here as
    the call progresses. We accept any status and return 200 so Twilio
    doesn't retry.
    ponytail: same signature model as /voice — Twilio signs every webhook
    with the per-number Twilio subaccount auth token, looked up via the
    called number's row. Reject (403) anything that doesn't match.
    """
    try:
        form = await request.form()
        form_dict = {k: str(v) if v is not None else "" for k, v in form.items()}
    except Exception as exc:
        log.warning("[call-status] could not read form: %s", exc)
        return {"ok": True}
    # Per-number Twilio auth token (same lookup /voice uses).
    called_to = form_dict.get("To") or form_dict.get("to")
    per_number_token = None
    if called_to:
        try:
            from STT_server.adapters.twilio_api import validate_twilio_signature
            from STT_server.db_phone_numbers import find_by_number as _find_num_for_sig
            row = _find_num_for_sig(called_to) or {}
            per_number_token = row.get("twilio_auth_token") or None
        except Exception as exc:
            log.warning("[call-status] per-number token lookup failed: %s", exc)
    sig = request.headers.get("X-Twilio-Signature", "")
    if per_number_token and sig:
        from STT_server.config import PUBLIC_URL as _PUB_URL
        signature_url = f"{_PUB_URL.rstrip('/')}{request.url.path}"
        if request.url.query:
            signature_url += f"?{request.url.query}"
        if not validate_twilio_signature(per_number_token, signature_url, sig, form_dict):
            log.warning("[call-status] invalid Twilio signature from %s to=%s",
                        request.client.host if request.client else "?",
                        called_to or "(missing)")
            raise HTTPException(status_code=403, detail="invalid signature")
    elif per_number_token and not sig:
        # Token is configured but no signature arrived — likely a proxy
        # stripped the header or someone is probing. Refuse.
        log.warning("[call-status] missing X-Twilio-Signature (to=%s)", called_to or "(missing)")
        raise HTTPException(status_code=403, detail="missing signature")
    call_sid = form.get("CallSid")
    call_status_val = form.get("CallStatus")
    duration = form.get("Duration")
    log.info("[call-status] call_sid=%s status=%s duration=%s",
             call_sid, call_status_val, duration)
    return {"ok": True}


# ---------- /health (call-path readiness) ----------
# ----------------------------------------------------------------------------
# Lightweight health endpoint that checks the dependencies the call
# path actually needs (DB reachable, PUBLIC_URL set, at least one
# provider key configured). Railway uses this to know the service is
# really ready to handle traffic, not just that the process started.
# ----------------------------------------------------------------------------

@api_router.get("/health")
def health() -> dict:
    out = {
        "ok": True,
        "checks": {},
    }
    # 1. DB reachable?
    try:
        from STT_server.db import is_postgres, get_conn
        if is_postgres():
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            out["checks"]["postgres"] = "ok"
        else:
            out["checks"]["postgres"] = "skipped (JSON backend)"
    except Exception as exc:
        log.exception("[/health] postgres check failed: %s", exc)
        out["ok"] = False
        out["checks"]["postgres"] = "FAIL: database unavailable"
    # 2. PUBLIC_URL set?
    from STT_server.config import PUBLIC_URL
    if PUBLIC_URL:
        out["checks"]["public_url"] = "ok"
    else:
        out["ok"] = False
        out["checks"]["public_url"] = "FAIL: PUBLIC_URL env not set"
    # 3. ponytail: env-fallback TTS keys removed. Each user brings
    # their own via Settings → API or ModalAgents inline. The healthcheck
    # now reports only the system infrastructure (PUBLIC_URL, DATABASE_URL,
    # CREDENTIAL_ENCRYPTION_KEY), not provider credentials.
    out["checks"]["tts_env_keys"] = "removed — per-user only"
    return out


# ---------- Pydantic schemas ----------

class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1)
    voice: Optional[str] = None
    # ponytail: separate from `voice` — `voice` is the high-level agent
    # voice descriptor; `voice_id` is the actual provider-side id
    # (e.g. Inworld's "Dennis", ElevenLabs' voice id). The TTS
    # dispatcher reads session.voice_id at runtime. Without this
    # field on the schema, Pydantic drops it and the runtime falls
    # back to the provider's hardcoded default on every save.
    voice_id: Optional[str] = None
    language: Optional[str] = "English"
    campaign: Optional[str] = None
    status: Optional[str] = "Active"
    description: Optional[str] = None
    tone: Optional[str] = None
    # System prompt - injected into the LLM context at call time so the
    # agent knows its role, rules, and customer-specific data.
    prompt: Optional[str] = None
    # Welcome message - first thing the agent says when the call
    # connects. Empty = the agent starts silently.
    welcome_message: Optional[str] = None
    # Per-service provider/model selection (New Agent flow).
    # All optional — omitting them leaves the agent with the user's
    # default provider config at call time.
    stt_provider: Optional[str] = None
    stt_model: Optional[str] = None
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # ponytail: per-agent runtime overrides (006_agent_runtime_params.sql).
    # 2026-08-26 regression: the FE was sending these on every save
    # but the Pydantic schema didn't declare them, so Pydantic
    # silently dropped the fields. The values never reached the DB
    # row and the operator saw "I set 0.2 but it resets to blank
    # every save". All three are Optional with None = "inherit the
    # platform default" so a brand-new agent without explicit knobs
    # is fine.
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    tts_speed: Optional[float] = None
    # Idle / silence detection (008_agent_idle_settings.sql). All optional —
    # None on every field = fall back to the global IDLE_SILENCE_TIMEOUT_SEC.
    # When idle_enabled=True the monitor plays the prompt messages at the
    # configured intervals, then closes the websocket after idle_max_attempts
    # of continued silence + idle_disconnect_timeout_sec.
    idle_enabled: Optional[bool] = None
    idle_first_timeout_sec: Optional[int] = None
    idle_first_message: Optional[str] = None
    idle_subsequent_timeout_sec: Optional[int] = None
    idle_final_message: Optional[str] = None
    idle_disconnect_timeout_sec: Optional[int] = None
    idle_max_attempts: Optional[int] = None
    # ponytail: per-slot credential source (009_agent_use_own_key.sql).
    # false (default) means the resolver may fall back to platform env
    # vars when no per-user key is configured. true means the agent
    # must use ONLY the per-user / per-agent credential — useful when
    # the operator wants strict cost isolation or needs to bypass
    # the platform key for compliance reasons.
    stt_use_own_key: Optional[bool] = None
    llm_use_own_key: Optional[bool] = None
    tts_use_own_key: Optional[bool] = None


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    voice: Optional[str] = None
    # ponytail: was missing from the schema — the FE was sending
    # payload.voice_id but Pydantic was silently dropping the field
    # because the schema didn't declare it. Result: agents with
    # custom TTS voices (e.g. Inworld's voice catalog) always
    # reverted to "Dennis" on save because voice_id never made it to
    # the DB row. Added here so the field actually round-trips.
    voice_id: Optional[str] = None
    language: Optional[str] = None
    campaign: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    tone: Optional[str] = None
    # System prompt — gets injected into the LLM context at call time
    # so the agent knows its role, rules, and customer-specific data.
    prompt: Optional[str] = None
    # Welcome message — first thing the agent says when the call
    # connects. Empty = the agent starts silently (caller speaks first).
    welcome_message: Optional[str] = None
    # Per-service provider/model selection (New Agent flow)
    stt_provider: Optional[str] = None
    stt_model: Optional[str] = None
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # ponytail: per-agent runtime overrides (006_agent_runtime_params.sql).
    # 2026-08-26 regression: the FE was sending these on every save
    # but the Pydantic schema didn't declare them, so Pydantic
    # silently dropped the fields. The values never reached the DB
    # row and the operator saw "I set 0.2 but it resets to blank
    # every save". Same as AgentCreate: Optional with None = "inherit
    # the platform default" so the modal can send `null` to clear
    # a knob and the BE drops it (preserve previous value? no — the
    # FE explicitly sends `null` only when the user clears the input,
    # so the user expects the DB to be cleared too). The exclude_none
    # in update_agent means a null is treated as "don't touch the
    # column"; the FE must include the field if it wants to clear it.
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    tts_speed: Optional[float] = None
    # Idle / silence detection — see AgentCreate above.
    idle_enabled: Optional[bool] = None
    idle_first_timeout_sec: Optional[int] = None
    idle_first_message: Optional[str] = None
    idle_subsequent_timeout_sec: Optional[int] = None
    idle_final_message: Optional[str] = None
    idle_disconnect_timeout_sec: Optional[int] = None
    idle_max_attempts: Optional[int] = None
    # ponytail: credential source toggle — see AgentCreate above.
    stt_use_own_key: Optional[bool] = None
    llm_use_own_key: Optional[bool] = None
    tts_use_own_key: Optional[bool] = None


class PhoneNumberCreate(BaseModel):
    provider: str = "twilio"
    country: str = "+1"
    number: str
    # ponytail: the display label is derived from the assigned agent
    # at render time. No free-text `name` on the number — operators
    # kept giving the line and the agent different labels and the
    # list drifted out of sync. The phone_numbers.label column is
    # still accepted (an explicit override) but the FE never sets it
    # today; the FE asks agentsApiV2.list() and shows agent.name.
    label: Optional[str] = None
    # ponytail: campaign routing. Same options the agent modal exposes
    # so a single call flow (call comes in → matches this number →
    # tagged with this campaign) lines up end-to-end with the agent's
    # campaign config. Optional, defaults to no campaign on the BE.
    campaign: Optional[str] = None
    agent: Optional[str] = None
    # ponytail: credenciales de Twilio opcionales. Si el provider es
    # 'twilio' o 'sip' y se pasan, se validan y guardan en el record
    # del phone number (asi cada number puede usar credenciales distintas
    # si el user lo necesita). Si no se pasan, el record queda sin
    # credenciales y el runtime tendra que caer al global.
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    # SIP trunk fields
    sip_host: Optional[str] = None
    sip_username: Optional[str] = None
    sip_password: Optional[str] = None
    # WhatsApp Business API fields. WhatsApp needs a Meta-assigned
    # phone_number_id (NOT the E.164 number) and a long-lived access
    # token. Webhook URL + verify token are app-level config, not per
    # number, so we don't accept them here.
    whatsapp_phone_number_id: Optional[str] = None
    whatsapp_access_token: Optional[str] = None


class PhoneNumberUpdate(BaseModel):
    agent: Optional[str] = None
    status: Optional[str] = None
    label: Optional[str] = None
    campaign: Optional[str] = None
    # ponytail: bug history. This schema only declared the routing
    # fields (agent/status/label/campaign) so Pydantic silently
    # dropped every credential the FE sent on edit. Operators kept
    # seeing "old creds still in use" because the original CREATE
    # path stored the values (PhoneNumberCreate has these fields)
    # but every subsequent PUT stripped them. Mirror the CREATE
    # schema's credential fields here so PUT can actually update
    # them. The DB-layer allowed set in db_phone_numbers.update_number
    # also has to include these — both layers have to agree.
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    sip_host: Optional[str] = None
    sip_username: Optional[str] = None
    sip_password: Optional[str] = None
    whatsapp_phone_number_id: Optional[str] = None
    whatsapp_access_token: Optional[str] = None


class SettingsUpdate(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    timezone: Optional[str] = None
    notifications: Optional[dict] = None
    # ponytail: which OpenAI model the Test button uses to generate
    # test data. Empty / None falls back to the DB default
    # gpt-4o-mini. Validated loosely server-side - the
    # OpenAI client just hands the string through and 400s if the
    # model does not exist for the user account.
    test_data_model: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class ApiKeyUpdate(BaseModel):
    """Body for PUT /settings/api-keys/{service}. credentials is an
    opaque dict whose shape is defined per-service in API_KEY_SERVICES.
    """
    credentials: dict | None = None


# ---------- /me (alias) ----------

@api_router.get("/me")
def me_alias(auth: dict = Depends(require_auth)):
    user = _get_user(auth["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "created_at": user.get("created_at", ""),
    }


# ---------- /dashboard/stats ----------

@api_router.get("/dashboard/stats")
def dashboard_stats(auth: dict = Depends(require_auth)):
    agents = _load(AGENTS_FILE, [])
    numbers = _load(NUMBERS_FILE, [])
    user_agents = [a for a in agents if a.get("user_id") == auth["user_id"]]
    user_numbers = [n for n in numbers if n.get("user_id") == auth["user_id"]]
    active_agents = sum(1 for a in user_agents if a.get("status") == "Active")
    total_calls = 0
    for a in user_agents:
        c = str(a.get("calls", "0")).replace(",", "")
        try:
            total_calls += int(c)
        except ValueError:
            pass
    avg_qa = 0
    if user_agents:
        avg_qa = sum(int(a.get("perf", 0)) for a in user_agents) / len(user_agents)
    return {
        "active_agents": active_agents,
        "calls_today": total_calls,
        "avg_qa_score": f"{int(avg_qa)}%",
        "recent_agents": user_agents[:5],
        "numbers_count": len(user_numbers),
    }


# ---------- /usage (call-minutes + cost ledger) ----------

@api_router.get("/usage")
def usage_summary(auth: dict = Depends(require_auth)):
    """Aggregated call-minutes + cost for the authenticated user.

    Each call is recorded by session_runtime.cleanup_session with
    duration, agent_id, the providers it used, and a flag for whether
    it fell back to the platform key for any of those providers. Cost
    is `duration_seconds / 60 * rate`, where the rate depends on the
    own-vs-platform split (config.py).

    Returns totals, per-agent breakdown, and the 50 most recent calls.
    """
    from STT_server.services.usage_store import aggregate_usage

    # ponytail: build a name lookup once so the FE doesn't have to
    # reconcile agent_id → name itself on every render.
    agents = _load(AGENTS_FILE, [])
    name_lookup = {
        a.get("id"): a.get("name") or a.get("id")
        for a in agents
        if isinstance(a, dict) and a.get("id")
    }
    return aggregate_usage(
        user_id=auth["user_id"],
        agent_name_lookup=name_lookup,
    )


# ---------- /agents CRUD ----------
# ponytail: backed by Postgres when DATABASE_URL is set, otherwise the
# legacy JSON file (same shape). Switching is invisible to the FE.

@api_router.get("/agents")
def list_agents(auth: dict = Depends(require_auth)):
    return db_list_agents(auth["user_id"])


@api_router.post("/agents")
def create_agent(data: AgentCreate, auth: dict = Depends(require_auth)):
    # ponytail: validate provider ids BEFORE the agent hits disk so a
    # typo'd "openai" doesn't quietly lie in the store forever.
    for k in ("stt_provider", "tts_provider", "llm_provider"):
        v = getattr(data, k)
        if v and not get_provider_spec(v):
            raise HTTPException(status_code=400, detail=f"Unknown provider '{v}'")
    return db_create_agent(auth["user_id"], data.dict())


@api_router.put("/agents/{agent_id}")
def update_agent(agent_id: str, data: AgentUpdate, auth: dict = Depends(require_auth)):
    # ponytail: validate provider ids before they reach the store. Cheap
    # defense — no live API call, just catalog lookup.
    for k in ("stt_provider", "tts_provider", "llm_provider"):
        v = getattr(data, k)
        if v and not get_provider_spec(v):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider '{v}'",
            )
    payload = data.dict(exclude_none=True)
    if not is_postgres():
        # JSON path still needs the lock for RMW atomicity.
        with _data_lock():
            agents = _load(AGENTS_FILE, [])
            for a in agents:
                if a["id"] == agent_id and a.get("user_id") == auth["user_id"]:
                    for k, v in payload.items():
                        a[k] = v
                    _save(AGENTS_FILE, agents)
                    return a
        raise HTTPException(status_code=404, detail="Agent not found")
    updated = db_update_agent(agent_id, auth["user_id"], payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not found")
    return updated


@api_router.delete("/agents/{agent_id}")
def delete_agent(agent_id: str, auth: dict = Depends(require_auth)):
    if not is_postgres():
        with _data_lock():
            agents = _load(AGENTS_FILE, [])
            before = len(agents)
            agents = [a for a in agents if not (a["id"] == agent_id and a.get("user_id") == auth["user_id"])]
            if len(agents) == before:
                raise HTTPException(status_code=404, detail="Agent not found")
            _save(AGENTS_FILE, agents)
        return {"success": True}
    if not db_delete_agent(agent_id, auth["user_id"]):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": True}


# ---------- /agents/{agent_id}/tools CRUD ----------
#
# ponytail: 010_agent_tools.sql moved every read / write through
# STT_server.db_tools (Postgres). The STT_server/data/agent_tools.json
# file is now ONLY used as a one-time backfill source on first boot
# — never read by the route layer again. The two helpers below
# (build_tool_payload + _test_tool_row) are the only custom logic
# the route handlers need on top of db_tools: Pydantic + AgentTool
# validation that the storage layer shouldn't have to know about.

# ponytail: marker for tools that any agent of the same owner can
# invoke. Shared tools get explicit per-agent assignments stored
# in agent_tools.assignments (JSONB array of agent_ids).
SHARED_TOOL_AGENT_ID = "__shared__"


def _build_tool_payload(agent_id: str, data: "ToolCreate", user_id: str | None = None) -> dict:
    """Shape the FE payload into the dict db_tools.create_tool
    expects. Centralised so the per-agent and shared-tool creation
    endpoints don't drift apart on validation, kind defaults, or
    field handling.

    ponytail: 016 — when `integration_id` is present, validate:
      * the integration exists and belongs to user_id,
      * the shared/private matrix holds
        (private tool → shared or same-private integration only),
      * the `action` is a registered action for that provider
        (or any well-formed id for generic_webhook).
    The matrix and action validation live here (not in AgentTool)
    because they need the integrations_catalog + db_integrations
    lookups, which AgentTool deliberately avoids.
    """
    from STT_server.domain.tool import AgentTool, VALID_TOOL_KINDS
    if data.kind is not None and data.kind not in VALID_TOOL_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{data.kind}'. Expected one of: {sorted(VALID_TOOL_KINDS)}",
        )
    tool = AgentTool(
        agent_id=agent_id,
        name=data.name,
        description=data.description,
        webhook_url=data.webhook_url or "",
        filler_phrase=data.filler_phrase,
        parameters=data.parameters,
        kind=data.kind,
        destination=data.destination,
        # ponytail: 016 — integration binding is optional on the
        # payload. Existing tests / callers that don't know about
        # the new fields still pass a Body-shaped object whose
        # attributes are just name/description/webhook_url/etc.
        # getattr defaults to None so they don't crash.
        integration_id=getattr(data, "integration_id", None),
        action=getattr(data, "action", None),
    )
    errors = tool.validate()
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"Validation errors: {', '.join(errors)}",
        )
    from STT_server.domain.tool import validate_json_schema
    is_valid, err = validate_json_schema(data.parameters)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON Schema: {err}",
        )
    # ponytail: 016 — integration binding checks. Done after
    # AgentTool.validate() so the basic kind/format checks have
    # already run; we add the catalog-level checks on top.
    # getattr with default so legacy callers (e.g. tests passing a
    # hand-built Body stub) don't crash on attribute lookup.
    if getattr(data, "integration_id", None) and user_id is not None:
        from STT_server.db_integrations import get_integration
        from STT_server.services.integrations_catalog import (
            get_integration_provider_spec,
            is_valid_action,
        )
        integ = get_integration(data.integration_id, user_id)
        if not integ:
            raise HTTPException(
                status_code=404,
                detail=f"Integration '{data.integration_id}' not found",
            )
        # shared/private matrix:
        #   private tool → shared or same-private integration: ok
        #   private tool → other agent's private integration: blocked
        #   shared tool   → shared integration: ok
        #   shared tool   → private integration: blocked
        if integ["agent_id"] not in ("__shared__", agent_id):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Tool '{agent_id}' cannot use integration '{integ['id']}' "
                    f"(integration is scoped to agent '{integ['agent_id']}')"
                ),
            )
        # Action must be registered for the provider (or any well-formed
        # id for generic_webhook).
        if not is_valid_action(integ["provider"], (data.action or "").strip()):
            spec = get_integration_provider_spec(integ["provider"])
            if spec and spec.actions:
                allowed = ", ".join(sorted(a.id for a in spec.actions))
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Action '{data.action}' is not valid for provider "
                        f"'{integ['provider']}'. Allowed: {allowed}"
                    ),
                )
            raise HTTPException(
                status_code=422,
                detail=f"Action '{data.action}' has invalid format (^[a-z0-9_]+$)",
            )
    payload = tool.to_dict()
    # ponytail: forward test_data_model into the payload so
    # db_create_tool / db_update_tool can persist it on the row.
    # AgentTool doesn't carry the field (it's a runtime LLM knob,
    # not a tool definition) so we copy it here once. Empty / None
    # falls back to gpt-4o-mini for parity with the legacy
    # api_keys upsert path.
    tdm = (data.test_data_model or "").strip()
    payload["test_data_model"] = tdm or "gpt-4o-mini"
    return payload


async def _test_tool_row(tool: dict, user_id: str) -> dict:
    """Execute one tool's webhook with realistic args. Shared by both
    the per-agent test endpoint and the shared-tool test endpoint so
    a future change (e.g. async wrapping, retries) only needs to
    happen in one place.

    ponytail: consults the LLM-driven test_data_generator on every
    click so the n8n workflow executes against plausible values,
    not the legacy sample_<paramname> placeholders that n8n
    typically rejects as unprocessable. The generator uses the
    tool's name + description + parameters schema on its own, so
    no per-tool prompt is required.
    """
    from STT_server.services.tool_executor import execute_tool, ToolExecutionError, record_tool_result
    from STT_server.services.test_data_generator import (
        TestDataUnavailable,
        generate_test_payload,
    )
    tool_id = tool.get("id")
    name = tool.get("name", "")
    # ponytail: Test always asks the LLM. The previous version of
    # this function gated the generator on a non-empty test_prompt
    # on the row and fell back to sample_<param> placeholders
    # otherwise — n8n rejected those as unprocessable, so every Test
    # click was a 4xx. test_prompt is now deprecated (see
    # services/test_data_generator.py: "ignored by this generator"),
    # so we consult the LLM unconditionally. test_data_model still
    # defaults to gpt-4o-mini when the operator hasn't picked one.
    test_data_model = (tool.get("test_data_model") or "").strip() or "gpt-4o-mini"
    try:
        sample_args = generate_test_payload(tool, user_id, model=test_data_model)
    except TestDataUnavailable as exc:
        return {"success": False, "error": str(exc)}
    try:
        result = await execute_tool(tool["webhook_url"], sample_args, name)
        record_tool_result(tool_id, True, "test")
        # ponytail: include the generated payload in the response so
        # the FE tooltip / debug view shows what we actually sent —
        # the operator can verify the LLM produced sensible data
        # before they trust the workflow result.
        return {
            "success": True,
            "result": result,
            "sent_payload": sample_args,
        }
    except ToolExecutionError as exc:
        record_tool_result(tool_id, False, "test", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "sent_payload": sample_args,
        }
    except Exception as exc:
        record_tool_result(
            tool_id, False, "test",
            error=f"{type(exc).__name__}: {exc}",
        )
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "sent_payload": sample_args,
        }


@api_router.get("/agents/{agent_id}/tools")
def list_agent_tools(agent_id: str, auth: dict = Depends(require_auth)):
    """List all tools for an agent.

    Returns per-agent tools (``agent_id == agent_id``) AND shared tools
    owned by the same user that have been explicitly assigned to this
    agent. Backed by Postgres via db_tools.list_tools(agent_id=...) —
    the JSONB `?|` operator handles the "shared row whose
    assignments array contains this agent_id" branch in a single
    query.

    ponytail: provider-credential rows (Settings → API saves each
    provider's key into the same ``agent_tools`` table with
    ``agent_id='__shared__'``) are filtered out — they have neither
    a ``webhook_url`` nor a ``destination``, the two fields every real
    tool carries per ``AgentTool.validate()``. Without this filter the
    Edit Agent modal's "Assigned shared" / "Available shared" sections
    would surface OpenAI / Inworld / ElevenLabs alongside the operator's
    actual n8n tools, which is misleading.
    """
    return [
        t for t in db_list_tools(auth["user_id"], agent_id=agent_id)
        if _is_real_tool(t)
    ]


class ToolCreate(BaseModel):
    name: str
    description: str
    # ponytail: webhook_url is required for kind="webhook" (legacy) but
    # irrelevant for kind="call_transfer" — the destination field carries
    # the E.164 number we redirect the live call to. Pydantic can't
    # express "required when X else optional" cleanly, so we accept
    # Optional[str] here and let AgentTool.validate() enforce the rule
    # downstream (single source of truth for the contract).
    webhook_url: Optional[str] = ""
    filler_phrase: str = "Let me check the system..."
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    kind: Optional[str] = None  # "webhook" (default) | "call_transfer"
    destination: Optional[str] = None  # E.164 for call_transfer
    # ponytail: which OpenAI model the BE test_data_generator uses
    # when the operator hits Test on this tool. Pydantic v2's default
    # extra="ignore" would drop this from the body if it stayed out
    # of the model, so we declare it here. Empty / None falls back
    # to gpt-4o-mini in _build_tool_payload.
    test_data_model: Optional[str] = None
    # ponytail: 016 — integration binding. When set, the tool
    # inherits its webhook URL + credentials from the integration
    # row, and the `action` is server-injected into the n8n body
    # (LLM never controls action). Validation against the catalog
    # + shared/private matrix happens in _build_tool_payload.
    integration_id: Optional[str] = None
    action: Optional[str] = None


@api_router.post("/agents/{agent_id}/tools")
def create_agent_tool(agent_id: str, data: ToolCreate, auth: dict = Depends(require_auth)):
    """Create a new tool for an agent."""
    payload = _build_tool_payload(agent_id, data, user_id=auth["user_id"])
    return db_create_tool(auth["user_id"], payload)


@api_router.put("/agents/{agent_id}/tools/{tool_id}")
def update_agent_tool(agent_id: str, tool_id: str, data: ToolCreate, auth: dict = Depends(require_auth)):
    """Update an existing tool.

    kind-aware: a webhook → call_transfer flip (or vice-versa) can't
    sneak past without the matching required field being present
    because we re-run AgentTool.from_dict + .validate() before
    persisting the patch.
    """
    payload = _build_tool_payload(agent_id, data, user_id=auth["user_id"])
    # ponytail: db_update_tool patches with the payload keys it
    # receives — passing the full dict keeps the row's id intact.
    # The user_id check on the WHERE clause prevents an operator from
    # patching another user's row.
    updated = db_update_tool(tool_id, auth["user_id"], payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Tool not found")
    return updated


@api_router.delete("/agents/{agent_id}/tools/{tool_id}")
def delete_agent_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Delete a tool from an agent."""
    if not db_delete_tool(tool_id, auth["user_id"]):
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"success": True}


# ponytail: explicit per-agent tool assignment. Shared tools
# (agent_id="__shared__") used to auto-include for every agent of the
# same owner, which left the operator no way to opt out. These two
# endpoints let the operator pick exactly which agents can invoke
# each shared tool. Per-agent tools (agent_id == agent_id) are
# ignored here — they're implicitly available to their own agent and
# can't be unassigned (delete the tool instead).
@api_router.post("/agents/{agent_id}/tools/{tool_id}/assign")
def assign_shared_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Assign a shared tool to an agent. Idempotent."""
    # ponytail: confirm the agent belongs to this user. The agent
    # row is the source of truth for ownership; without this check a
    # operator could fish another user's agent id and widen a shared
    # tool's blast radius.
    from STT_server.db_agents import get_agent as _get_agent
    if not _get_agent(agent_id, auth["user_id"]):
        raise HTTPException(status_code=404, detail="Agent not found")
    # ponytail: load the tool first so we can short-circuit when
    # the operator tries to assign a per-agent tool (only shared
    # tools are assignable). db_add_assignment is JSONB-level
    # idempotent on its own, but the 400 message needs the kind
    # check to give the FE a clear error.
    tool = db_get_tool(tool_id, auth["user_id"])
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.get("agent_id") != SHARED_TOOL_AGENT_ID:
        raise HTTPException(
            status_code=400,
            detail="Only shared tools can be assigned. Per-agent tools are already available to their agent.",
        )
    # ponytail: reject provider-credential rows. They share the
    # `__shared__` agent_id but lack a webhook_url/destination, so
    # `_is_real_tool()` filters them out. Without this guard an
    # operator who somehow targeted a credential row would see the
    # assignment "succeed" (idempotent no-op on the empty
    # assignments array) but then have a non-functional row in the
    # agent modal — confusing on top of being wrong.
    if not _is_real_tool(tool):
        raise HTTPException(
            status_code=400,
            detail="Provider credentials are not assignable tools. Configure the provider in Settings → API instead.",
        )
    try:
        return db_add_assignment(tool_id, auth["user_id"], agent_id) or tool
    except Exception as exc:
        log.exception("assign tool %s to %s failed", tool_id, agent_id)
        raise HTTPException(status_code=500, detail=f"assign failed: {exc}")


@api_router.delete("/agents/{agent_id}/tools/{tool_id}/assign")
def unassign_shared_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Remove a shared tool assignment from an agent. Idempotent."""
    tool = db_get_tool(tool_id, auth["user_id"])
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.get("agent_id") != SHARED_TOOL_AGENT_ID:
        raise HTTPException(
            status_code=400,
            detail="Only shared tools can be unassigned. Delete per-agent tools instead.",
        )
    try:
        return db_remove_assignment(tool_id, auth["user_id"], agent_id) or tool
    except Exception as exc:
        log.exception("unassign tool %s from %s failed", tool_id, agent_id)
        raise HTTPException(status_code=500, detail=f"unassign failed: {exc}")


# ponytail: integrations assignment - mirrors tools assignment but for integrations.
# Shared integrations (agent_id=="__shared__") are assigned to specific agents via
# integrations.assignments JSONB array. Private integrations (agent_id==agent_id) are
# implicitly available and not assignable.
@api_router.post("/agents/{agent_id}/integrations/{integration_id}/assign")
def assign_shared_integration(agent_id: str, integration_id: str, auth: dict = Depends(require_auth)):
    """Assign a shared integration to an agent. Idempotent."""
    from STT_server.db_agents import get_agent as _get_agent
    if not _get_agent(agent_id, auth["user_id"]):
        raise HTTPException(status_code=404, detail="Agent not found")
    from STT_server.db_integrations import get_integration as _get_integ
    from STT_server.db_integrations import add_integration_assignment as _add_assign
    integ = _get_integ(integration_id, auth["user_id"])
    if not integ:
        raise HTTPException(status_code=404, detail="Integration not found")
    if integ.get("agent_id") != "__shared__":
        raise HTTPException(status_code=400, detail="Only shared integrations can be assigned.")
    try:
        return _add_assign(integration_id, auth["user_id"], agent_id) or integ
    except Exception as exc:
        log.exception("assign integration %s to %s failed", integration_id, agent_id)
        raise HTTPException(status_code=500, detail=f"assign failed: {exc}")


@api_router.delete("/agents/{agent_id}/integrations/{integration_id}/assign")
def unassign_shared_integration(agent_id: str, integration_id: str, auth: dict = Depends(require_auth)):
    """Remove a shared integration assignment from an agent. Idempotent."""
    from STT_server.db_integrations import get_integration as _get_integ
    from STT_server.db_integrations import remove_integration_assignment as _remove_assign
    integ = _get_integ(integration_id, auth["user_id"])
    if not integ:
        raise HTTPException(status_code=404, detail="Integration not found")
    if integ.get("agent_id") != "__shared__":
        raise HTTPException(status_code=400, detail="Only shared integrations can be unassigned.")
    try:
        return _remove_assign(integration_id, auth["user_id"], agent_id) or integ
    except Exception as exc:
        log.exception("unassign integration %s from %s failed", integration_id, agent_id)
        raise HTTPException(status_code=500, detail=f"unassign failed: {exc}")


@api_router.post("/agents/{agent_id}/tools/{tool_id}/test")
async def test_agent_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Test a tool by executing its webhook with sample data."""
    tool = db_get_tool(tool_id, auth["user_id"])
    if not tool or tool.get("agent_id") != agent_id:
        raise HTTPException(status_code=404, detail="Tool not found")
    return await _test_tool_row(tool, auth["user_id"])


# ponytail: global n8n tools marketplace. Tools with agent_id="__shared__"
# in Postgres are visible to any of the owner's agents (loaded by
# _load_agent_tools when called with user_id). The endpoints below
# let the operator create / edit / delete shared tools independently
# of any specific agent — the FE mounts them on the /integrations
# page (navbar entry).
def _is_real_tool(row: dict) -> bool:
    """True when the agent_tools row is a real operator-defined tool.

    Provider credentials (Settings → API keys) also live in the same
    table with ``agent_id='__shared__'`` but they have an empty
    ``webhook_url`` AND a null ``destination`` — real tools always
    have one or the other (enforced by ``AgentTool.validate()``). The
    modal filters "Assigned shared" / "Available shared" on this
    predicate so credential rows don't show up next to the operator's
    n8n tools.
    """
    if not row:
        return False
    return bool(row.get("webhook_url") or row.get("destination"))


@api_router.get("/tools")
def list_shared_tools(auth: dict = Depends(require_auth)):
    """List all shared n8n tools owned by the current user.

    Excludes provider-credential rows (Settings → API saves each
    provider's key into the same ``agent_tools`` table with
    ``agent_id='__shared__'``). They have neither ``webhook_url`` nor
    ``destination``, the two fields every real tool carries per
    ``AgentTool.validate()``. Surfacing them next to actual n8n tools
    in the agent modal's marketplace was misleading — operators were
    trying to assign OpenAI as a callable tool.
    """
    return [
        t for t in db_list_tools(auth["user_id"])
        if t.get("agent_id") == SHARED_TOOL_AGENT_ID and _is_real_tool(t)
    ]


@api_router.post("/tools")
def create_shared_tool(data: ToolCreate, auth: dict = Depends(require_auth)):
    """Create a new shared n8n tool owned by the current user."""
    payload = _build_tool_payload(SHARED_TOOL_AGENT_ID, data, user_id=auth["user_id"])
    return db_create_tool(auth["user_id"], payload)


@api_router.put("/tools/{tool_id}")
def update_shared_tool(tool_id: str, data: ToolCreate, auth: dict = Depends(require_auth)):
    """Update an existing shared n8n tool owned by the current user."""
    payload = _build_tool_payload(SHARED_TOOL_AGENT_ID, data, user_id=auth["user_id"])
    updated = db_update_tool(tool_id, auth["user_id"], payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Tool not found")
    return updated


@api_router.delete("/tools/{tool_id}")
def delete_shared_tool(tool_id: str, auth: dict = Depends(require_auth)):
    """Delete a shared n8n tool owned by the current user."""
    if not db_delete_tool(tool_id, auth["user_id"]):
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"success": True}


@api_router.post("/tools/{tool_id}/test")
async def test_shared_tool(tool_id: str, auth: dict = Depends(require_auth)):
    """Smoke-test a shared n8n tool by hitting its webhook with sample args."""
    tool = db_get_tool(tool_id, auth["user_id"])
    if not tool or tool.get("agent_id") != SHARED_TOOL_AGENT_ID:
        raise HTTPException(status_code=404, detail="Tool not found")
    return await _test_tool_row(tool, auth["user_id"])


# ---------- /campaigns (global suggestions catalog) ----------

@api_router.get("/campaigns")
def list_campaigns(auth: dict = Depends(require_auth)):
    """Returns the union of the curated CAMPAIGN_OPTIONS plus every
    campaign any user has typed into any agent or phone-number record.

    Used by the FE to populate the <datalist> suggestions under the
    campaign text input. The list is read-only from the FE's point of
    view — campaigns get registered server-side when an agent or
    phone-number is saved with a new campaign string.
    """
    return {"campaigns": db_list_campaigns()}


# ---------- /phone-numbers CRUD (and /numbers alias) ----------

@api_router.get("/phone-numbers")
def list_phone_numbers(auth: dict = Depends(require_auth)):
    return db_list_numbers(auth["user_id"])


# ponytail: per-number Twilio validation. The FE's "Test Twilio"
# button in ModalConnectNumber.jsx calls this. It validates the
# account_sid + auth_token pair against the Twilio API
# (fetching the account object) and returns a structured result
# the FE can render. Sid is optional — if omitted, we validate
# the stored key on the row.
@api_router.post("/phone-numbers/validate-twilio")
async def validate_phone_number_twilio(
    data: PhoneNumberCreate,
    auth: dict = Depends(require_auth),
):
    sid = (data.twilio_account_sid or "").strip()
    tok = (data.twilio_auth_token or "").strip()
    if not sid or not tok:
        raise HTTPException(
            status_code=400,
            detail="twilio_account_sid and twilio_auth_token are required to validate",
        )
    from STT_server.adapters.twilio_api import validate_twilio_credentials
    res = await validate_twilio_credentials(sid, tok)
    log.info(
        "[phone-numbers] twilio validate sid=%s... valid=%s msg=%s user_id=%s",
        sid[:6], res.get("valid"), res.get("message", "")[:200], auth["user_id"],
    )
    # If the user supplied a number, also confirm it's owned by this
    # sub-account. 404 means the credentials are valid for the
    # SUB-account but don't own the line — the operator has to either
    # move the number to this sub-account or use creds from the
    # sub-account that owns it.
    owned_by = None
    if data.number:
        from STT_server.adapters.twilio_api import _get_twilio_client
        client = _get_twilio_client(sid, tok)
        try:
            import asyncio as _aio
            def _lookup():
                target = data.number.strip().replace(" ", "")
                if not target.startswith("+"):
                    target = "+" + target
                numbers = client.incoming_phone_numbers.list(phone_number=target)
                return numbers
            numbers = await _aio.to_thread(_lookup)
            owned_by = bool(numbers)
        except Exception as exc:
            log.warning("[phone-numbers] twilio number lookup failed: %s", exc)
            owned_by = None
    return {
        "valid": res.get("valid", False),
        "message": res.get("message", ""),
        "account_status": res.get("account_status"),
        "owned_by_subaccount": owned_by,
    }


@api_router.post("/phone-numbers")
async def create_phone_number(data: PhoneNumberCreate, auth: dict = Depends(require_auth)):
    # ponytail: validate Twilio creds server-side too. Client should
    # validate but never trust it — el regex del FE es solo UX hint.
    if data.provider in ("twilio", "sip"):
        if data.twilio_account_sid is not None and data.twilio_account_sid != "":
            if not re.match(r"^AC[0-9a-fA-F]{32}$", data.twilio_account_sid):
                raise HTTPException(
                    status_code=400,
                    detail="Twilio Account SID must start with 'AC' followed by 32 hex characters",
                )
        if data.twilio_auth_token is not None and data.twilio_auth_token != "":
            if len(data.twilio_auth_token) < 32 or len(data.twilio_auth_token) > 64:
                raise HTTPException(
                    status_code=400,
                    detail="Twilio Auth Token must be 32-64 characters",
                )
    new_number = db_create_number(auth["user_id"], data.dict())

    # ponytail: configure the Twilio webhook so calls actually route
    # to us. Without this the number lives in our DB but Twilio has no
    # idea where to send inbound audio. We use the per-number Twilio
    # creds the user just submitted (with fallback to the env var
    # TWILIO_AUTH_TOKEN). On failure we surface the Twilio error to the
    # FE but keep the row - the user can retry via PUT.
    #
    # The previous version called `asyncio.run(configure_voice_webhook(...))`
    # inside an async endpoint. asyncio.run() cannot be called from a
    # running event loop, so the route crashed with 500 on every Twilio
    # number creation. The fix is to make the endpoint async and await
    # the coroutine directly - configure_voice_webhook already returns
    # one.
    if data.provider == "twilio" and data.twilio_account_sid and data.twilio_auth_token:
        try:
            from STT_server.adapters.twilio_api import configure_voice_webhook
            from STT_server.config import PUBLIC_URL
            webhook_url = f"{PUBLIC_URL.rstrip('/')}/voice"
            result = await configure_voice_webhook(
                data.twilio_account_sid,
                data.twilio_auth_token,
                data.number,
                webhook_url,
            )
            if not result.get("success"):
                log.warning("[phone-numbers] webhook config failed: %s", result.get("error"))
                new_number["webhook_warning"] = result.get("error", "unknown")
            else:
                new_number["webhook_configured"] = True
        except Exception as exc:
            log.warning("[phone-numbers] webhook config exception: %s", exc)
            new_number["webhook_warning"] = str(exc)
    return new_number


@api_router.put("/phone-numbers/{number_id}")
def update_phone_number(number_id: str, data: PhoneNumberUpdate, auth: dict = Depends(require_auth)):
    payload = data.dict(exclude_none=True)
    if not is_postgres():
        with _data_lock():
            numbers = _load(NUMBERS_FILE, [])
            for n in numbers:
                if n["id"] == number_id and n.get("user_id") == auth["user_id"]:
                    n.update(payload)
                    _save(NUMBERS_FILE, numbers)
                    return n
        raise HTTPException(status_code=404, detail="Phone number not found")
    updated = db_update_number(number_id, auth["user_id"], payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Phone number not found")
    return updated


@api_router.delete("/phone-numbers/{number_id}")
def delete_phone_number(number_id: str, auth: dict = Depends(require_auth)):
    if not is_postgres():
        with _data_lock():
            numbers = _load(NUMBERS_FILE, [])
            before = len(numbers)
            numbers = [n for n in numbers if not (n["id"] == number_id and n.get("user_id") == auth["user_id"])]
            if len(numbers) == before:
                raise HTTPException(status_code=404, detail="Phone number not found")
            _save(NUMBERS_FILE, numbers)
        return {"success": True}
    if not db_delete_number(number_id, auth["user_id"]):
        raise HTTPException(status_code=404, detail="Phone number not found")
    return {"success": True}


# /numbers — alias used by some FE code paths
@api_router.get("/numbers")
def list_numbers_alias(auth: dict = Depends(require_auth)):
    return list_phone_numbers(auth)


@api_router.post("/numbers")
def create_number_alias(data: PhoneNumberCreate, auth: dict = Depends(require_auth)):
    return create_phone_number(data, auth)


# ---------- /settings ----------

def _settings_path(user_id: str) -> str:
    # ponytail: sanitize_id enforces a conservative charset on user_id
    # so a value like "../../etc/passwd" can't escape SETTINGS_DIR.
    # The auth layer already constrains user_id to a server-issued
    # value, but the file-inclusion scanner flags this pattern and
    # the helper is the cheapest place to neutralise it.
    from STT_server.utils.safe_path import sanitize_id
    safe_id = sanitize_id(user_id, field="user_id")
    return os.path.join(SETTINGS_DIR, f"{safe_id}.json")


@api_router.get("/settings")
def get_settings(auth: dict = Depends(require_auth)):
    user = _get_user(auth["user_id"])
    defaults = {
        "name": user.get("name", "") if user else "",
        "email": user.get("email", auth.get("email", "")) if user else auth.get("email", ""),
        "company": "",
        "timezone": "America/Mexico_City",
        "notifications": {
            "calls": True,
            "qa": True,
            "weekly": False,
            "marketing": False,
        },
    }
    # ponytail: route through db_settings on Postgres so GET and PUT
    # read/write the same store. The JSON backend keeps _load() since
    # db_get_settings already falls back to settings/<user_id>.json.
    stored = db_get_settings(auth["user_id"]) or {}
    return {**defaults, **stored}


@api_router.get("/settings/llm-options")
def get_settings_llm_options(auth: dict = Depends(require_auth)):
    """LLM provider picker for the Integrations → 'Modelo LLM para los
    test' section.

    Returns every LLM-capable provider from PROVIDER_CATALOG
    (regardless of connection status) with the hardcoded model
    catalog and a `connected` flag derived from the user's
    credentials. The FE filters to connected providers; the rest
    render as '(not connected — go to Settings → API)' hints.

    The current selection comes from settings.test_data_model (set
    via PUT /settings). Empty / unset defaults to "gpt-4o-mini" so
    the FE can show "currently using: gpt-4o-mini" out of the box
    for operators who never touched the picker.
    """
    from STT_server.services.credentials_resolver import get_llm_options
    stored = db_get_settings(auth["user_id"]) or {}
    current_model = (stored.get("test_data_model") or "").strip() or "gpt-4o-mini"
    return get_llm_options(auth["user_id"], current_model)


@api_router.put("/settings")
def update_settings(data: SettingsUpdate, auth: dict = Depends(require_auth)):
    payload = data.dict(exclude_none=True)
    if not is_postgres():
        path = _settings_path(auth["user_id"])
        with _data_lock():
            current = _load(path, {})
            for k, v in payload.items():
                current[k] = v
            _save(path, current)
            if "name" in payload:
                users = _load(USERS_FILE, [])
                for u in users:
                    if u.get("id") == auth["user_id"]:
                        u["name"] = payload["name"]
                        _save(USERS_FILE, users)
                        break
        return current
    stored = db_upsert_settings(auth["user_id"], payload)
    # ponytail: when on Postgres, mirror name into the users row so
    # login /me sees it. db_users helpers stay out of api.py to keep
    # the auth file as the only consumer.
    if "name" in payload and stored is not None:
        try:
            from STT_server.db_users import update_user_name
            update_user_name(auth["user_id"], payload["name"])
        except Exception as exc:
            log.warning("[settings] could not mirror name into Postgres users: %s", exc)
    return stored


@api_router.put("/settings/password")
def change_password(data: PasswordChange, auth: dict = Depends(require_auth)):
    with _data_lock():
        users = _load(USERS_FILE, [])
        user = _get_user(auth["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if _hash_password(data.current_password) != user["password"]:
            raise HTTPException(status_code=401, detail="Current password incorrect")
        if len(data.new_password) < 8:
            raise HTTPException(status_code=400, detail="New password too short")
        user["password"] = _hash_password(data.new_password)
        user["updated_at"] = _now_iso()
        _save(USERS_FILE, users)
        # Invalidate all sessions for this user (file + cache)
        sessions = _load(SESSIONS_FILE, {})
        sessions = {k: v for k, v in sessions.items() if v.get("user_id") != auth["user_id"]}
        _save(SESSIONS_FILE, sessions)
    invalidate_user_sessions(auth["user_id"])
    return {"success": True, "message": "Password updated. Please log in again."}


# ---------- /agents/{id}/pricing-overrides (Enterprise / custom rates) ---
# ----------------------------------------------------------------------------
# Per-agent per-model price overrides for cases the public catalog
# doesn't cover (Enterprise contracts, custom negotiated rates). The FE
# fills these in when the operator picks a tier the catalog flags as
# null; the runtime cost summary merges them with the public catalog at
# resolve time. agent_id is required (no user-level overrides) so that
# the same Enterprise rate can differ across agents under one tenant.
@api_router.get("/agents/{agent_id}/pricing-overrides")
def list_pricing_overrides(agent_id: str, auth: dict = Depends(require_auth)):
    from STT_server import db_agents
    from STT_server import db_pricing_overrides
    agent = db_agents.get_agent(agent_id, auth["user_id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return db_pricing_overrides.list_overrides(agent_id)


class PricingOverrideUpsert(BaseModel):
    unit: Optional[str] = "minute"
    price: Optional[float] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    source: Optional[str] = "manual"


@api_router.put("/agents/{agent_id}/pricing-overrides/{service}/{provider}/{model_id}")
def upsert_pricing_override(
    agent_id: str, service: str, provider: str, model_id: str,
    data: PricingOverrideUpsert, auth: dict = Depends(require_auth),
):
    from STT_server import db_agents
    from STT_server import db_pricing_overrides
    agent = db_agents.get_agent(agent_id, auth["user_id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if service not in ("stt", "tts", "llm"):
        raise HTTPException(status_code=400, detail="service must be stt|tts|llm")
    if data.price is None and data.input_price is None and data.output_price is None:
        raise HTTPException(
            status_code=400,
            detail="at least one of price, input_price, output_price must be set",
        )
    return db_pricing_overrides.upsert_override(
        agent_id, service, provider, model_id, data.model_dump()
    )


@api_router.delete("/agents/{agent_id}/pricing-overrides/{service}/{provider}/{model_id}")
def delete_pricing_override(
    agent_id: str, service: str, provider: str, model_id: str,
    auth: dict = Depends(require_auth),
):
    from STT_server import db_agents
    from STT_server import db_pricing_overrides
    agent = db_agents.get_agent(agent_id, auth["user_id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    ok = db_pricing_overrides.delete_override(agent_id, service, provider, model_id)
    return {"success": ok}


# ---------- /settings/api-keys (per-user provider credentials) ----------
# ----------------------------------------------------------------------------
# El deployer pone OPENAI_API_KEY / ELEVENLABS_API_KEY / etc. en Railway como
# defaults del sistema. Cada user final puede subir sus propias keys desde
# Settings → API. La BE lee primero del storage del user y, si no hay,
# hace fallback al env var. Asi un admin puede usar su key de OpenAI y
# otro user del mismo deploy puede tener la suya sin tocarse.
#
# El catalogo de proveedores vive en
# STT_server.services.credentials_resolver.PROVIDER_CATALOG y es la unica
# fuente de verdad. Anade ahi un nuevo ProviderSpec y FE + BE lo recogen.
# ----------------------------------------------------------------------------

def _serialize_provider(spec, user_id: str) -> dict:
    """Render a ProviderSpec for the FE modal. Drops the test_fn path."""
    fields = [
        {
            "name": f.name,
            "label": f.label,
            "type": f.type,
            "placeholder": f.placeholder,
            "required": f.required,
            "pattern": f.pattern,
            "min_length": f.min_length,
            "max_length": f.max_length,
            "help": f.help,
        }
        for f in spec.fields
    ]
    # ponytail: a provider can serve multiple slots (OpenAI = llm + stt).
    # Always emit `categories` as a list so the FE badge filter doesn't
    # have to special-case single-vs-multi. Falls back to (category,)
    # when the spec didn't opt in.
    categories = list(spec.categories) if spec.categories else [spec.category]
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "categories": categories,
        "fields": fields,
        # True if the user has a saved per-user key for this provider.
        "connected": is_provider_configured(user_id, spec.id)
                     and bool(resolve_provider(user_id, spec.id).get("api_key")
                              or resolve_provider(user_id, spec.id).get("account_sid")),
        "supports_test": bool(spec.test_fn),
    }


@api_router.get("/settings/api-keys")
def list_api_keys(auth: dict = Depends(require_auth)):
    """Returns the catalog of available services with the user's connection status.

    Does NOT return the actual credentials — only whether each one is
    connected. The user can fetch the values explicitly via
    GET /settings/api-keys/{service_id}/value.

    ponytail: `connected` is now derived from the per-user resolver
    (Postgres-backed via db_tools), not from a JSON file. The previous
    override `item["connected"] = spec.id in configured_ids` used to
    shadow the resolver's view with a stale JSON snapshot — which was
    right after a save and wrong after every redeploy on Railway.
    """
    user_id = auth["user_id"]
    services = []
    for spec in PROVIDER_CATALOG:
        services.append(_serialize_provider(spec, user_id))
    return {"services": services}


@api_router.put("/settings/api-keys/{service_id}")
def upsert_api_key(service_id: str, body: ApiKeyUpdate, auth: dict = Depends(require_auth)):
    """Stores/updates the user's credentials for a service.

    Values are validated against the per-field regex/length in the
    provider catalog before encryption, so a typo returns a clean 400
    instead of a confusing 401 from the upstream provider on the next call.

    ponytail: storage moved from data/tools_integrations.json (volatile
    on Railway) to the `tools_integrations` Postgres table via
    db_upsert_tool. The encrypt-then-store pattern is preserved so a
    leaked DB row still doesn't leak plaintext keys.
    """
    spec = get_provider_spec(service_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service_id}'")
    if not body.credentials or not isinstance(body.credentials, dict):
        raise HTTPException(status_code=400, detail="credentials object is required")

    cleaned, errors = validate_credentials(service_id, body.credentials)
    if errors:
        log.warning(
            "[api-keys] upsert rejected service=%s user_id=%s errors=%s",
            service_id, auth["user_id"], errors,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation failed", "errors": errors},
        )
    log.info(
        "[api-keys] upsert accepted service=%s user_id=%s fields=%s",
        service_id, auth["user_id"], list(cleaned.keys()),
    )

    encrypted = encrypt_credentials(cleaned)
    # ponytail: storage shape. Per-user service credentials live as
    # an agent_tools row keyed by `(user_id, id=service_id)` with
    # `agent_id='__shared__'` and the encrypted dict under the
    # `credentials` JSONB column (migration 014 added the column).
    # The function_name column is reused as the canonical service id
    # so the runtime has a stable handle; webhook_url / kind /
    # parameters / etc. stay at their tool-shape defaults and the
    # resolver ignores them. `connected` is a boolean flag the FE
    # reads via /settings/api-keys — kept off the table shape since
    # it's the inverse of "credentials is null".
    from STT_server.db_tools import (
        get_tool as db_get_tool,
        create_tool as db_create_tool,
        update_tool as db_update_tool,
    )
    existing = db_get_tool(service_id, auth["user_id"])
    payload = {
        "agent_id": "__shared__",
        "name": spec.name,
        "description": "",
        "webhook_url": "",
        "filler_phrase": "Let me check the system...",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "kind": "webhook",
        "destination": None,
        "assignments": [],
        "function_name": service_id,
        "test_data_model": "gpt-4o-mini",
        "credentials": encrypted,
    }
    if existing:
        # db_update_tool builds the SET clause dynamically from the
        # payload (skipping DB-managed cols); passing credentials as
        # a dict lets the JSONB cast in the loop handle it correctly.
        # `connected` is recomputed server-side in list_api_keys
        # from the credentials column, so we don't have to write it.
        db_update_tool(service_id, auth["user_id"], payload)
    else:
        db_create_tool(auth["user_id"], payload, tool_id=service_id)
    return {"success": True}


@api_router.delete("/settings/api-keys/{service_id}")
def delete_api_key(service_id: str, auth: dict = Depends(require_auth)):
    """Disconnects a service for the user (falls back to env var default)."""
    if not db_delete_tool(auth["user_id"], service_id):
        raise HTTPException(status_code=404, detail="Key not configured")
    return {"success": True}


@api_router.get("/settings/api-keys/{service_id}/value")
def reveal_api_key(service_id: str, auth: dict = Depends(require_auth)):
    """Returns the user's decrypted credentials for a service.

    This is the only endpoint that exposes plaintext values. It's only
    accessible to the user who owns the credentials (auth check), and
    is intended to be called explicitly when the user clicks a "show"
    toggle in the FE. The list endpoint never returns the values.
    """
    if get_provider_spec(service_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service_id}'")
    # ponytail: read directly from per-user Postgres storage (NOT the
    # resolver — we don't want to leak system env-var values through
    # the reveal endpoint).
    row = db_get_tool(auth["user_id"], service_id)
    if not row or not row.get("connected") or not row.get("credentials"):
        raise HTTPException(status_code=404, detail="Key not configured")
    creds = decrypt_credentials(row["credentials"])
    return {"service": service_id, "credentials": creds}


@api_router.post("/settings/api-keys/{service_id}/test")
async def test_api_key(
    service_id: str,
    body: ApiKeyUpdate | None = None,
    auth: dict = Depends(require_auth),
):
    """Live-validates the credentials for a service.

    The New Agent modal passes the key the user just typed via
    ``body.credentials.api_key``. Settings → API passes nothing and
    we fall back to the per-user stored credential (or env-var).

    Returns ``{valid, message, source}`` where ``source`` is
    'inline' (caller-supplied) | 'user' | 'env' | 'none'.
    """
    import logging
    log = logging.getLogger("stt_server.test_endpoint")
    has_inline = body is not None and isinstance(body.credentials, dict)
    inline_key = body.credentials.get("api_key") if has_inline else None
    # ponytail: forward optional base_url alongside the api_key so the
    # MiniMax validator can hit a token-plan / coding-plan endpoint
    # that isn't in its hardcoded candidate list. The Settings modal
    # surfaces this as a free-text field under the API Key.
    inline_base_url = body.credentials.get("base_url") if has_inline else None
    log.warning(
        "/test hit: service=%s has_body=%s body_type=%s has_key=%s key_len=%s key_preview=%s has_base_url=%s",
        service_id,
        body is not None,
        type(body.credentials).__name__ if has_inline else "n/a",
        inline_key is not None,
        len(inline_key) if inline_key else 0,
        (inline_key[:4] + "…" + inline_key[-4:]) if inline_key and len(inline_key) > 8 else "(too short or empty)",
        bool(inline_base_url and inline_base_url.strip()),
    )
    if get_provider_spec(service_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service_id}'")
    import asyncio as _aio
    result = await _aio.to_thread(
        test_provider,
        auth["user_id"],
        service_id,
        inline_key,
        (inline_base_url or "").strip() or None,
    )
    log.warning(
        "/test result: service=%s source=%s valid=%s msg=%s",
        service_id,
        result.get("source"),
        result.get("valid"),
        (result.get("message") or "(none)")[:200],
    )
    return result


class ListModelsRequest(BaseModel):
    """Body for POST /providers/models. The FE passes the API key the
    user just typed in the New Agent modal. If absent, falls back to
    the user's stored credential (per-user → env var).
    """
    service: str  # "stt" | "tts" | "llm"
    provider: str
    api_key: str | None = None


@api_router.post("/providers/models")
async def list_models(body: ListModelsRequest, auth: dict = Depends(require_auth)):
    """Returns the catalog of models/voices for a (service, provider) pair.

    Used by the New Agent modal — after the user picks a provider and
    enters their API key, the FE calls this to populate the secondary
    dropdown with that provider's actual models. Live fetches are
    done server-side so we can swallow CORS / network failures and
    return a graceful fallback.
    """
    if body.service not in VALID_MODEL_SERVICES:
        raise HTTPException(
            status_code=400,
            detail=f"service must be one of {sorted(VALID_MODEL_SERVICES)}",
        )
    if get_provider_spec(body.provider) is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{body.provider}'")
    import asyncio as _aio
    # ponytail: pass the authenticated user_id so list_provider_models
    # can resolve the per-user stored credential from tools_integrations
    # before falling back to the system env-var. Without this the Edit
    # modal's catalog lookup uses the deployer's key (often a lower
    # plan with fewer voices) instead of the operator's saved key.
    result = await _aio.to_thread(
        list_provider_models,
        body.service, body.provider, body.api_key,
        auth.get("user_id"),
    )
    return result


class CategorizedModelsRequest(BaseModel):
    """Body for POST /providers/models/categorized. Same auth_key
    fallback as ListModelsRequest but returns all three service
    buckets in one shot so the FE doesn't have to know how each
    provider names its TTS / STT / LLM families internally.
    """
    provider: str
    api_key: str | None = None


@api_router.post("/providers/models/categorized")
async def list_categorized_models(
    body: CategorizedModelsRequest, auth: dict = Depends(require_auth),
):
    """Return one provider's full model catalog bucketed by service
    (llm / stt / tts). The FE uses this to render the agent
    picker's three dropdowns in one network round-trip; the contract
    is identical for every provider so the FE doesn't need a
    provider-specific mapping (OpenAI's `tts-1` vs Inworld's
    `voiceId` vs Deepgram's `model`).
    """
    if get_provider_spec(body.provider) is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider '{body.provider}'",
        )
    import asyncio as _aio
    return await _aio.to_thread(
        _build_categorized_models,
        body.provider, body.api_key, auth.get("user_id"),
    )


class TtsPreviewRequest(BaseModel):
    """Body for POST /tts/preview. The FE uses this to let the user
    preview a TTS voice/model before saving the agent config."""
    provider: str  # "elevenlabs" | "rime"
    voice_id: str | None = None
    model_id: str | None = None
    text: str = "Hello, this is a preview of how I will sound on your calls."
    api_key: str | None = None


@api_router.post("/tts/preview")
async def tts_preview(body: TtsPreviewRequest, auth: dict = Depends(require_auth)):
    """Generate a short TTS sample and return mu-law 8 kHz audio bytes
    (audio/mulaw content type, same format Twilio consumes). The FE plays
    this in a regular <audio> element via a Blob URL.

    Same code path the live call uses, so what the user hears in the
    preview is what callers will hear in production.

    Per-user API key is read from the user's stored credentials (falls
    back to env var); if the FE just typed a new key in the modal, it
    can pass it via `api_key` to override.
    """
    from STT_server.adapters.tts_preview import preview_tts
    import asyncio as _aio
    tts_log = logging.getLogger("stt_server.tts_preview")
    creds = resolve_provider(auth["user_id"], body.provider) if auth["user_id"] else {}
    user_key = (body.api_key or creds.get("api_key") or "").strip() or None
    if not user_key:
        raise HTTPException(
            status_code=400,
            detail=f"{body.provider} API key not configured. Add it in Settings -> API or pass api_key in the request body.",
        )
    try:
        audio_bytes = await preview_tts(
            user_id=auth["user_id"],
            provider=body.provider,
            text=body.text,
            voice_id=body.voice_id,
            model_id=body.model_id,
            api_key=body.api_key,
        )
    except Exception as exc:
        tts_log.exception("tts_preview failed for provider=%s voice=%s: %s",
                          body.provider, body.voice_id, exc)
        raise HTTPException(status_code=502, detail="tts provider error")
    if not audio_bytes:
        raise HTTPException(
            status_code=502,
            detail="TTS provider returned no audio. Check that the voice_id is valid for the selected provider.",
        )
    return Response(content=audio_bytes, media_type="audio/wav")

# ============================================================================
# /integrations — third-party CRM / contact-center connections
# ----------------------------------------------------------------------------
# These endpoints own the Integration entity (one row per third-party
# service the user connects the agent to). Tools (in /tools and
# /agents/{id}/tools) reference an integration by integration_id; the
# executor resolves the webhook URL from the integration row + env
# config, and the internal endpoint hands n8n the credentials at call
# time.
#
# Security model:
#   * /integrations* require a user bearer token (require_auth).
#   * /internal/integrations/{id}/credentials requires the shared
#     service token (require_service_token). User tokens do NOT work
#     here — even the integration's owner can't fetch the plaintext
#     credentials back; they can only Replace them.
#   * Credentials are NEVER returned by any /integrations endpoint,
#     not even masked. The FE has to Replace, not Show.
# ============================================================================


# ── Body schemas ─────────────────────────────────────────────────────────────

class IntegrationPreflight(BaseModel):
    """Body for POST /integrations/preflight. NO persistence — runs
    the provider's test_fn against the supplied credentials and
    returns {valid, message}."""
    provider: str
    configuration: dict = Field(default_factory=dict)
    credentials: dict = Field(default_factory=dict)


class IntegrationCreate(BaseModel):
    """Body for POST /integrations. The `skip_preflight` field is admin-
    only and pulled out before validation; without it, the create call
    re-runs the provider's test_fn and refuses to persist a failing
    connection (422)."""
    provider: str
    name: str
    agent_id: str = "__shared__"  # "__shared__" | "agent-<uuid8>"
    configuration: dict = Field(default_factory=dict)
    credentials: dict = Field(default_factory=dict)
    # ponytail: Pydantic v2 strips fields whose name starts with `_`
    # as if they were private — so the admin override is named
    # `skip_preflight` (no underscore) on the wire even though it's
    # only honoured for admin users. Production FE never sets this;
    # admin scripts and tests do.
    skip_preflight: bool = False


class IntegrationUpdate(BaseModel):
    """Body for PUT /integrations/{id}. Empty / missing credential fields
    are treated as "keep existing" per the no-reveal contract — the FE
    has to send a non-empty value to Replace a credential. Empty
    configuration fields keep their stored value too."""
    name: Optional[str] = None
    agent_id: Optional[str] = None
    configuration: Optional[dict] = None
    credentials: Optional[dict] = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _strip_integration_for_wire(row: dict) -> dict:
    """Drop credential + OAuth-internal fields before returning to the FE.

    Credentials: never returned (no masked variant — Replace only).
    oauth_state_hash / oauth_state_expires_at: internal-only — only
    the OAuth callback handler reads them via
    get_integration_by_oauth_state. Leaking them to the FE would
    expose the lookup key for an active flow (a DB row leak doesn't
    expose the real state, but the FE running with a leaked FE
    session could replay it).
    """
    if not row:
        return row
    out = dict(row)
    out.pop("credentials_encrypted", None)
    out.pop("credentials_cipher", None)
    out.pop("oauth_state_hash", None)
    out.pop("oauth_state_expires_at", None)
    return out


def _merge_credentials(
    existing_encrypted: Optional[bytes],
    new_credentials: Optional[dict],
    cipher_key: str = "fernet-v1",
) -> Optional[bytes]:
    """Merge the operator-submitted credentials dict with the stored
    encrypted blob.

    Rules (per the agreed contract):
      * missing field → keep existing
      * "" (empty string) → keep existing (NOT clear; clearing needs
        an explicit revoke op we haven't built yet)
      * non-empty string → replace encrypted value

    Returns the resulting encrypted BYTEA blob (or None when the
    result has no fields — happens when the user clears all
    password fields on update, in which case the integration loses
    its credential binding; rare but supported).
    """
    from STT_server.security.credentials import (
        encrypt_credentials,
        decrypt_credentials,
        encrypt_value,
    )
    existing_plain: dict = {}
    if existing_encrypted:
        try:
            existing_plain = decrypt_credentials(existing_encrypted) if cipher_key == "fernet-v1" else {}
            # ponytail: decrypt_credentials already handled the
            # fernet key; if the cipher differs we leave existing_plain
            # empty so the merge falls back to "set what was sent".
        except Exception as exc:
            log.warning(
                "[integrations] decrypt failed during merge; treating as empty: %s",
                exc,
            )
            existing_plain = {}
    if not new_credentials:
        return existing_encrypted  # nothing to do
    merged = dict(existing_plain)
    for k, v in new_credentials.items():
        if v is None or (isinstance(v, str) and v == ""):
            # missing/empty: keep existing (don't change merged[k])
            continue
        if not isinstance(v, str):
            # skip non-string values silently — the catalog validator
            # would have rejected them on the create path; we don't
            # have catalog validation here, so we just don't write.
            continue
        merged[k] = v.strip()
    if not merged:
        return None
    return encrypt_credentials(merged)


def _run_preflight(
    provider: str,
    configuration: dict,
    credentials: dict,
) -> tuple[bool, str]:
    """Run the provider's test_fn. Returns (valid, message).
    Wraps integrations_tester so the route layer doesn't import it
    twice; centralises the "test_fn=None → not implemented" path."""
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
        validate_integration_fields,
    )
    spec = get_integration_provider_spec(provider)
    if spec is None:
        return False, f"Unknown provider '{provider}'"
    cleaned_config, cleaned_creds, errors = validate_integration_fields(
        provider, configuration, credentials,
    )
    if errors:
        # First error is enough for the preflight banner.
        return False, errors[0].get("message", "invalid credentials")
    if not spec.test_fn:
        return False, f"Test not yet implemented for {provider}"
    from STT_server.services.integrations_tester import run_integration_test
    return run_integration_test(spec.test_fn, cleaned_config, cleaned_creds)


# ── /integrations/providers (catalog) ───────────────────────────────────────

@api_router.get("/integrations/providers")
def list_integration_providers(auth: dict = Depends(require_auth)):
    """Catalog endpoint. Single source of truth for the FE — no parallel
    frontend catalog. Returns the field + action specs the FE renders
    on the Add Integration form + the action dropdown on the Add Tool
    form.
    """
    from STT_server.services.integrations_catalog import (
        INTEGRATION_PROVIDERS,
        IntegrationFieldSpec,
        IntegrationProviderSpec,
        ActionSpec,
    )
    def field_to_wire(f: IntegrationFieldSpec) -> dict:
        return {
            "name": f.name,
            "label": f.label,
            "type": f.type,
            "placeholder": f.placeholder,
            "required": f.required,
            "pattern": f.pattern,
            "min_length": f.min_length,
            "max_length": f.max_length,
            "help": f.help,
            "options": list(getattr(f, "options", ()) or ()),
        }
    def action_to_wire(a: ActionSpec) -> dict:
        return {
            "id": a.id,
            "name": a.name,
            "description": a.description,
"parameters_schema": a.parameters_schema,
        }

    def spec_to_wire(s: IntegrationProviderSpec) -> dict:
        return {
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "description": s.description,
            "has_test": bool(s.test_fn),
            # ponytail: auth_type drives the FE form. "oauth" providers
            # render a single Connect button + Name field; "static"
            # renders the existing fields-based form.
            "auth_type": getattr(s, "auth_type", "static"),
            "oauth_label": getattr(s, "oauth_label", ""),
            "oauth_default_scopes": list(getattr(s, "oauth_default_scopes", ())),
            "fields": [field_to_wire(f) for f in s.fields],
            "actions": [action_to_wire(a) for a in s.actions],
            "prompt_snippet": getattr(s, "prompt_snippet", ""),
        }
    return {
        "providers": [spec_to_wire(s) for s in INTEGRATION_PROVIDERS],
    }


# ── /integrations/preflight ─────────────────────────────────────────────────

@api_router.post("/integrations/preflight")
def preflight_integration(body: IntegrationPreflight, auth: dict = Depends(require_auth)):
    """Run the provider's test_fn against the supplied credentials.
    No persistence. Returns {valid, message, connection_status}.
    Same code path the create endpoint runs internally — exposing it
    separately lets the FE render the green/red banner on the form
    before the operator clicks Save."""
    valid, message = _run_preflight(body.provider, body.configuration, body.credentials)
    return {
        "valid": valid,
        "message": message,
        "connection_status": "connected" if valid else "failed",
    }


# ── /integrations CRUD ──────────────────────────────────────────────────────

@api_router.get("/integrations")
def list_integrations_endpoint(
    agent_id: Optional[str] = None,
    auth: dict = Depends(require_auth),
):
    """List the user's integrations. Optional ?agent_id filter (matches
    db_integrations.list_integrations). NEVER returns credentials."""
    from STT_server.db_integrations import list_integrations as db_list_integrations
    rows = db_list_integrations(auth["user_id"], agent_id=agent_id)
    return {"integrations": [_strip_integration_for_wire(r) for r in rows]}


@api_router.post("/integrations")
def create_integration_endpoint(body: IntegrationCreate, auth: dict = Depends(require_auth)):
    """Create a new integration. Preflight runs automatically; if it
    fails, returns 422 with the test message. Admin-only skip_preflight
    field bypasses the preflight (use only for tests / migrations).

    OAuth providers (Salesforce) skip preflight entirely — the OAuth
    dance IS the test. We persist the row with `connection_status='pending'`
    and the FE follows up with a redirect to /oauth/start. Static
    providers keep the existing preflight + encrypt-and-store flow.
    """
    # Pull skip_preflight out before any validation so a non-admin
    # sending the field can't even cause a log line about admin checks.
    skip_preflight = body.skip_preflight
    if skip_preflight:
        # re-authorize as admin — require_admin raises 403 on miss
        require_admin(auth)
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
        validate_integration_fields,
    )
    spec = get_integration_provider_spec(body.provider)
    if spec is None:
        raise HTTPException(status_code=422, detail=f"Unknown provider '{body.provider}'")
    auth_type = getattr(spec, "auth_type", "static")
    if auth_type == "oauth":
        # OAuth create: just persist the row with name + provider +
        # agent_id + an empty configuration. The OAuth callback
        # writes the actual config (instance_url) + encrypted
        # credentials + flips connection_status to 'connected'.
        # No preflight (the OAuth redirect IS the verification).
        cleaned_config = {}
    else:
        cleaned_config, cleaned_creds, errors = validate_integration_fields(
            body.provider, body.configuration, body.credentials,
        )
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        # preflight unless admin-skipped
        if not skip_preflight:
            valid, message = _run_preflight(body.provider, body.configuration, body.credentials)
            if not valid:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "preflight": {"valid": False, "message": message},
                        "errors": [{"field": "_preflight", "message": message}],
                    },
                )
    from STT_server.db_integrations import create_integration as db_create_integration
    from STT_server.security.credentials import encrypt_credentials
    if auth_type == "oauth":
        encrypted = None
    else:
        encrypted = encrypt_credentials(cleaned_creds) if cleaned_creds else None
    payload = {
        "provider": body.provider,
        "name": body.name,
        "agent_id": body.agent_id,
        "configuration": cleaned_config,
    }
    # ponytail: idempotency guard — if the FE double-submits (double-click before
    # disabled propagates, StrictMode double-invoke, retry), don't create a second row.
    # Same provider+name+agent for same user within the debounce window is a duplicate.
    # We check for a recent row instead of adding a DB unique constraint because legitimate
    # duplicates with same name but different intent should still be allowed after the window.
    from STT_server.db_integrations import list_integrations as db_list_integrations
    existing = [r for r in db_list_integrations(auth["user_id"]) if r["provider"] == body.provider and r["name"] == body.name and r["agent_id"] == body.agent_id]
    # If a row with same key was created within last 5s, treat as dedup and return it (200 not 201)
    if existing:
        from datetime import datetime, timezone as _tz
        for cand in existing:
            try:
                created = datetime.fromisoformat(cand["created_at"].replace("Z", "+00:00"))
                if (datetime.now(_tz.utc) - created).total_seconds() < 5:
                    return _strip_integration_for_wire(cand)
            except Exception:
                pass

    row = db_create_integration(
        auth["user_id"],
        payload,
        credentials_encrypted=encrypted,
        cipher="fernet-v1",
    )
    if auth_type == "oauth":
        # Mark pending so the FE immediately shows the OAuth card as pending.
        # No second insert — the row above is the one.
        from STT_server.db_integrations import mark_integration_status as db_mark
        db_mark(row["id"], auth["user_id"], "pending")
        # Re-read to return the updated status without creating a duplicate row
        from STT_server.db_integrations import get_integration as db_get_integration
        row = db_get_integration(row["id"], auth["user_id"]) or row
    elif not skip_preflight:
        from STT_server.db_integrations import update_integration as db_update_integration
        row = db_update_integration(
            row["id"], auth["user_id"], {},
        ) or row

    # ponytail: auto-create Google Calendar tool for zero-config provider - one click creates integration + tool
    if body.provider == "google_calendar":
        try:
            from STT_server.db_tools import list_tools as db_list_tools, create_tool as db_create_tool
            from STT_server.domain.tool import AgentTool
            # Check if tool already exists for this integration to keep idempotency guard
            try:
                existing_tools = db_list_tools(auth["user_id"], agent_id=row["agent_id"] if row["agent_id"] != "__shared__" else "__shared__")
            except Exception:
                existing_tools = []
            already = any(
                isinstance(t, dict) and t.get("integration_id") == row["id"] and t.get("action") == "agendar_cita_dinamica"
                for t in (existing_tools or [])
            )
            if not already:
                tool = AgentTool(
                    agent_id=row["agent_id"],
                    name="Agendar Cita",
                    description="Agenda una cita en Google Calendar y envía correo de confirmación",
                    integration_id=row["id"],
                    action="agendar_cita_dinamica",
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Nombre completo del asistente"},
                            "email": {"type": "string", "description": "Email del asistente"},
                            "datetime": {"type": "string", "description": "Fecha y hora ISO 8601, ej: 2026-09-04T15:00:00-06:00"},
                            "duration_minutes": {"type": "integer", "description": "Duración en minutos, por defecto 30"},
                            "host_email": {"type": "string", "description": "Email calendario destino (opcional)"},
                        },
                        "required": ["name", "email", "datetime"],
                    },
                )
                db_create_tool(auth["user_id"], tool.to_dict())
                log.info("[integrations] auto-created google_calendar tool for integration %s", row["id"])
        except Exception as exc:
            log.warning("[integrations] auto-create google_calendar tool failed: %s", exc)

    return _strip_integration_for_wire(row)


@api_router.get("/integrations/{integration_id}")
def get_integration_endpoint(integration_id: str, auth: dict = Depends(require_auth)):
    """Detail view. NEVER returns credentials."""
    from STT_server.db_integrations import get_integration as db_get_integration
    row = db_get_integration(integration_id, auth["user_id"])
    if not row:
        raise HTTPException(status_code=404, detail="Integration not found")
    return _strip_integration_for_wire(row)


@api_router.put("/integrations/{integration_id}")
def update_integration_endpoint(
    integration_id: str,
    body: IntegrationUpdate,
    auth: dict = Depends(require_auth),
):
    """Update name / agent_id / configuration / credentials.

    Credentials merge contract: missing or empty string = keep existing.
    No way to clear a credential short of deleting the integration —
    explicit revoke is a future endpoint.
    """
    from STT_server.db_integrations import (
        get_integration as db_get_integration,
        update_integration as db_update_integration,
    )
    existing = db_get_integration(integration_id, auth["user_id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Integration not found")
    patch: dict = {}
    if body.name is not None:
        patch["name"] = body.name
    if body.agent_id is not None:
        patch["agent_id"] = body.agent_id
    if body.configuration is not None:
        from STT_server.services.integrations_catalog import validate_integration_fields
        cleaned_config, _, errors = validate_integration_fields(
            existing["provider"], body.configuration, {},
        )
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        patch["configuration"] = cleaned_config
    new_encrypted_blob = None
    if body.credentials is not None:
        # Merge with existing — empty/missing = keep.
        new_encrypted_blob = _merge_credentials(
            existing.get("credentials_encrypted"),
            body.credentials,
            cipher_key=existing.get("credentials_cipher", "fernet-v1"),
        )
    updated = db_update_integration(
        integration_id, auth["user_id"], patch,
        credentials_encrypted=new_encrypted_blob,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Integration not found")
    return _strip_integration_for_wire(updated)


@api_router.delete("/integrations/{integration_id}")
def delete_integration_endpoint(integration_id: str, auth: dict = Depends(require_auth)):
    """Delete an integration. Returns 409 if any tools still depend
    on it (the count is computed transactionally so a concurrent
    tool create can't sneak in between the count and the delete)."""
    from STT_server.db_integrations import delete_integration as db_delete_integration
    ok, err = db_delete_integration(integration_id, auth["user_id"])
    if err:
        # The error message is the human-readable count: "5 tools depend..."
        # Surface as 409 Conflict with a structured body so the FE can
        # render the count + a "View tools" link.
        from STT_server.db_integrations import count_dependent_tools
        n = count_dependent_tools(integration_id, auth["user_id"])
        raise HTTPException(
            status_code=409,
            detail={"message": err, "tool_count": n},
        )
    if not ok:
        raise HTTPException(status_code=404, detail="Integration not found")
    return {"success": True}


@api_router.post("/integrations/{integration_id}/test")
def test_integration_endpoint(integration_id: str, auth: dict = Depends(require_auth)):
    """Run the provider's test_fn against the stored credentials and
    persist the result (connection_status + last_test_message)."""
    from STT_server.db_integrations import (
        get_integration as db_get_integration,
        update_integration as db_update_integration,
    )
    from STT_server.services.integrations_catalog import get_integration_provider_spec
    from STT_server.security.credentials import decrypt_credentials
    row = db_get_integration(integration_id, auth["user_id"])
    if not row:
        raise HTTPException(status_code=404, detail="Integration not found")
    spec = get_integration_provider_spec(row["provider"])
    if spec is None:
        raise HTTPException(status_code=422, detail=f"Unknown provider '{row['provider']}'")
    if not spec.test_fn:
        # same shape as a real failure so the FE handles it uniformly
        valid, message = False, f"Test not yet implemented for {row['provider']}"
    else:
        try:
            creds_plain = (
                decrypt_credentials(row["credentials_encrypted"])
                if row.get("credentials_encrypted")
                else {}
            )
        except Exception as exc:
            log.warning(
                "[integrations.test] decrypt failed integration_id=%s err=%s",
                integration_id, exc,
            )
            creds_plain = {}
        from STT_server.services.integrations_tester import run_integration_test
        valid, message = run_integration_test(
            spec.test_fn, row.get("configuration") or {}, creds_plain,
        )
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    db_update_integration(
        integration_id, auth["user_id"], {
            "connection_status": "connected" if valid else "failed",
            "last_tested_at": now_iso,
            "last_test_message": message[:500] if message else None,
        },
    )
    return {
        "valid": valid,
        "message": message,
        "connection_status": "connected" if valid else "failed",
        "last_tested_at": now_iso,
    }


# ── /integrations/{id}/oauth/start + /oauth/callback + disconnect ────────────

@api_router.get("/integrations/{integration_id}/oauth/start")
def oauth_start_endpoint(
    integration_id: str,
    token: Optional[str] = None,
    auth: Optional[dict] = Depends(require_auth_optional),
):
    """Generate a fresh OAuth state, persist its hash, redirect the
    operator to the provider's authorize URL.

    The state hash is stored on the integration row (TTL 10 min). The
    callback hashes the incoming state and looks the row up by hash.
    No DB write happens to the secret state — a leaked DB row can't
    be used to ride an active flow.

    The provider redirects back to
    `${SALESFORCE_REDIRECT_URI}` (which we control) — that's where
    the code is exchanged.

    Auth: require_auth (Bearer header) when called from the FE's
    `request()` helper. The FE's OAuth start uses `window.location`
    to follow the redirect (no fetch — see integrationsApi.oauthStart
    in the FE), which can't send an Authorization header. For that
    path the FE passes the JWT in `?token=` as a fallback. The
    fallback is only honoured here, only when the Bearer header is
    absent, and the token is consumed in this single request — no
    server-side logging of the value.
    """
    from STT_server.db_integrations import (
        get_integration as db_get_integration,
        start_oauth_flow,
    )
    from STT_server.services.oauth_providers import (
        generate_state, generate_pkce, build_authorize_url, get_oauth_config, validate_oauth_env,
    )
    # ponytail: query-param token fallback so the FE's full-page
    # redirect (window.location) can auth without losing the
    # operator's session. Bearer header wins when present so a
    # direct API call doesn't need to think about it.
    if auth is None and token:
        from STT_server.routes.api import resolve_bearer
        try:
            auth = resolve_bearer(f"Bearer {token}", raise_on_missing=False)
        except Exception:
            auth = None
    if auth is None:
        # ponytail: window.location navigations can't carry an
        # Authorization header — they go through `?token=`. If
        # that token is missing / expired / malformed, returning
        # 401 leaves the operator stuck on a blank error page
        # (the browser renders the body as HTML). Redirect to
        # the FE's /login with a `?reason=` so the login form can
        # show a "session expired" snackbar; the operator
        # re-authenticates and retries.
        frontend_origin = os.environ.get("FRONTEND_ORIGIN", "").strip().rstrip("/")
        if not frontend_origin:
            frontend_origin = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
        if not frontend_origin:
            allowed = os.environ.get("ALLOWED_ORIGINS", "")
            if allowed:
                frontend_origin = allowed.split(",")[0].strip().rstrip("/")
        if not frontend_origin:
            env_label = os.environ.get("ENVIRONMENT", "production").strip().lower()
            frontend_origin = "http://localhost:5173" if env_label in ("development", "dev", "local", "test") else "https://agentiafrontend-production.up.railway.app"
        if token:
            # Operator sent a token but it didn't authenticate.
            # Most common cause: session expired (the token in
            # localStorage was issued > 7 days ago). Send them
            # to /login with a clear reason so the form can show
            # a snackbar. Without a token, fall through to 401 —
            # no way to recover, the operator probably typed the
            # URL by hand.
            sep = "&" if "?" in frontend_origin else "?"
            return RedirectResponse(
                url=f"{frontend_origin}/login?reason=session_expired&next=/integrations/{integration_id}",
                status_code=302,
            )
        raise HTTPException(
            status_code=401,
            detail="Not authenticated (Bearer header or ?token= required)",
        )
    integ = db_get_integration(integration_id, auth["user_id"])
    if not integ:
        raise HTTPException(status_code=404, detail="Integration not found")
    if integ["provider"] not in ("salesforce",):  # future OAuth providers added here
        raise HTTPException(
            status_code=422,
            detail=f"Provider '{integ['provider']}' is not an OAuth integration",
        )
    # ponyy: env check AT CALL TIME. The boot no longer fails
    # closed, so a deploy that doesn't use Salesforce starts clean.
    # If the operator opens /oauth/start on a misconfigured deploy,
    # they get a 503 with the missing env var named. The
    # configuration entry is then written when they save the
    # integration, so the next deploy with env set picks it up.
    env_ok, missing = validate_oauth_env(integ["provider"])
    if not env_ok:
        log.warning(
            "[oauth.start] env missing provider=%s missing=%s integration_id=%s",
            integ["provider"], missing, integration_id,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "oauth_not_configured",
                "message": (
                    f"Cannot start {integ['provider']} OAuth — backend env vars missing: "
                    f"{', '.join(missing)}. Set them on the backend service and restart."
                ),
                "provider": integ["provider"],
                "integration_id": integration_id,
                "missing_env_vars": list(missing),
            },
        )
    state, state_hash = generate_state()
    # ponyy: PKCE. Generate a verifier + challenge for the
    # authorize URL. Encrypt the verifier at rest — the row is
    # fetched by the callback handler to send the original back
    # to Salesforce in the token exchange.
    #
    # Use `encrypt_value` (not `encrypt_credentials`!) — the latter
    # returns a dict, but the column is BYTEA and psycopg2 can't
    # adapt a dict to bytes. The verifier is a single string; we
    # store it raw, not wrapped in a one-key dict.
    code_verifier, code_challenge = generate_pkce()
    code_verifier_encrypted = encrypt_value(code_verifier)
    updated = start_oauth_flow(
        integration_id, auth["user_id"], state_hash,
        code_verifier_encrypted=code_verifier_encrypted,
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to start OAuth flow")
    cfg = get_oauth_config(integ["provider"])
    authorize_url = build_authorize_url(cfg, state, code_verifier=code_verifier)
    return RedirectResponse(url=authorize_url, status_code=302)


@api_router.get("/integrations/salesforce/oauth/callback")
def oauth_salesforce_callback_endpoint(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    """Salesforce OAuth callback. No auth required — the state hash
    is the binding; the BE resolves the integration row without
    needing the operator to be logged in to the FE.

    On success: exchange code → tokens + instance_url → encrypt +
    persist + clear state → redirect to FE with ?connected=<id>.
    On provider error (?error=access_denied, etc.): redirect to
    FE with ?error=oauth_<code>.
    On our validation failure: redirect to FE with a generic
    ?error=oauth_invalid.
    """
    frontend_origin = os.environ.get("FRONTEND_ORIGIN", "").strip().rstrip("/")
    if not frontend_origin:
        frontend_origin = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
    if not frontend_origin:
        allowed = os.environ.get("ALLOWED_ORIGINS", "")
        if allowed:
            frontend_origin = allowed.split(",")[0].strip().rstrip("/")
    if not frontend_origin:
        env_label = os.environ.get("ENVIRONMENT", "production").strip().lower()
        frontend_origin = "http://localhost:5173" if env_label in ("development", "dev", "local", "test") else "https://agentiafrontend-production.up.railway.app"
    if error:
        log.warning("[oauth.callback] provider error=%s desc=%s", error, error_description)
        sep = "&" if "?" in frontend_origin else "?"
        return RedirectResponse(
            url=f"{frontend_origin}/integrations?error=oauth_{error}{sep}error_description={urllib.parse.quote(error_description or '')}",
            status_code=302,
        )
    if not code or not state:
        return RedirectResponse(
            url=f"{frontend_origin}/integrations?error=oauth_invalid",
            status_code=302,
        )
    from STT_server.db_integrations import (
        consume_oauth_state,
        complete_oauth_flow,
        get_integration_by_id,
    )
    from STT_server.security.credentials import encrypt_credentials, decrypt_credentials
    from STT_server.services.oauth_providers import (
        hash_state, get_oauth_config,
        exchange_code_for_tokens, OAuthError, RefreshTokenRevoked, now_plus_seconds,
    )
    from STT_server.db import get_conn, is_postgres
    state_hash = hash_state(state)
    # ponyy: ATOMIC consume BEFORE exchange. We use the WHERE-filter
    # UPDATE ... RETURNING so a concurrent / replayed callback can't
    # double-exchange the same code. If 0 rows come back, the state
    # was tampered with, expired, or already used — redirect with
    # the same code (no leak about which).
    if is_postgres():
        with get_conn() as conn:
            with conn.cursor() as cur:
                integ = consume_oauth_state(state_hash, cur=cur)
    else:
        integ = consume_oauth_state(state_hash)
    if not integ:
        return RedirectResponse(
            url=f"{frontend_origin}/integrations?error=oauth_invalid_state",
            status_code=302,
        )
    # At this point the state has been consumed. The remaining work
    # (token exchange + persist) uses the SAME cursor (Postgres)
    # so a single transaction wraps the whole callback.
    integration_id = integ["id"]
    user_id = integ["user_id"]
    provider = integ["provider"]
    # ponyy: PKCE. The verifier was returned from consume_oauth_state
    # as a key on the row (`_oauth_code_verifier_encrypted`).
    # Decrypt it now so we can send the original to Salesforce in
    # the token-exchange POST. Missing verifier (legacy row from
    # before migration 018) → exchange without PKCE — falls back
    # to the no-PKCE path, which works on non-PKCE Connected Apps.
    code_verifier_plain = None
    verifier_encrypted = integ.pop("_oauth_code_verifier_encrypted", None)
    if verifier_encrypted:
        try:
              code_verifier_plain = decrypt_value(verifier_encrypted)
        except Exception as exc:
            log.warning("[oauth.callback] verifier decrypt failed integration_id=%s err=%s",
                        integration_id, exc)
    if is_postgres():
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Exchange the code for tokens. Network call to
                # Salesforce — done inside the transaction so the
                # callback is one round trip from "code received"
                # to "tokens persisted".
                try:
                    cfg = get_oauth_config(provider)
                    tokens = exchange_code_for_tokens(cfg, code, code_verifier=code_verifier_plain)
                except RefreshTokenRevoked as exc:
                    log.warning(
                        "[oauth.callback] exchange revoked integration_id=%s err=%s",
                        integration_id, exc,
                    )
                    mark_integration_status_failed(
                        cur, integration_id, user_id, str(exc),
                    )
                    conn.commit()
                    return RedirectResponse(
                        url=f"{frontend_origin}/integrations?error=oauth_code_rejected",
                        status_code=302,
                    )
                except OAuthError as exc:
                    log.exception(
                        "[oauth.callback] exchange error integration_id=%s err=%s",
                        integration_id, exc,
                    )
                    # State was already consumed. Mark failed so the
                    # operator sees the row is broken in the UI; they
                    # can re-trigger OAuth from /oauth/start.
                    mark_integration_status_failed(
                        cur, integration_id, user_id, str(exc),
                    )
                    conn.commit()
                    return RedirectResponse(
                        url=f"{frontend_origin}/integrations?error=oauth_exchange_failed",
                        status_code=302,
                    )
                creds = {
                    "access_token": tokens.access_token,
                    "refresh_token": tokens.refresh_token,
                }
                if tokens.expires_in is not None:
                    creds["expires_at"] = now_plus_seconds(tokens.expires_in)
                if tokens.scope:
                    creds["scope"] = tokens.scope
                encrypted = encrypt_credentials(creds)
                configuration = dict(integ.get("configuration") or {})
                if tokens.instance_url:
                    configuration["instance_url"] = tokens.instance_url
                saved = complete_oauth_flow(
                    integration_id, user_id,
                    credentials_encrypted=encrypted,
                    configuration=configuration,
                    scope=tokens.scope,
                    connection_status="connected",
                    cur=cur,
                )
                if not saved:
                    conn.rollback()
                    log.error(
                        "[oauth.callback] complete_oauth_flow returned None integration_id=%s",
                        integration_id,
                    )
                    return RedirectResponse(
                        url=f"{frontend_origin}/integrations?error=oauth_internal",
                        status_code=302,
                    )
                log.info(
                    "[oauth.callback] connected integration_id=%s provider=%s user_id=%s",
                    saved["id"], saved["provider"], user_id,
                )
                return RedirectResponse(
                    url=f"{frontend_origin}/integrations?connected={saved['id']}",
                    status_code=302,
                )
    # JSON-file fallback: same shape, separate connection per call.
    try:
        cfg = get_oauth_config(provider)
        tokens = exchange_code_for_tokens(cfg, code, code_verifier=code_verifier_plain)
    except RefreshTokenRevoked as exc:
        log.warning(
            "[oauth.callback] exchange revoked integration_id=%s err=%s",
            integration_id, exc,
        )
        from STT_server.db_integrations import clear_oauth_state
        clear_oauth_state(integration_id, user_id)
        return RedirectResponse(
            url=f"{frontend_origin}/integrations?error=oauth_code_rejected",
            status_code=302,
        )
    except OAuthError as exc:
        log.exception(
            "[oauth.callback] exchange error integration_id=%s err=%s",
            integration_id, exc,
        )
        from STT_server.db_integrations import clear_oauth_state
        clear_oauth_state(integration_id, user_id)
        return RedirectResponse(
            url=f"{frontend_origin}/integrations?error=oauth_exchange_failed",
            status_code=302,
        )
    creds = {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
    }
    if tokens.expires_in is not None:
        creds["expires_at"] = now_plus_seconds(tokens.expires_in)
    if tokens.scope:
        creds["scope"] = tokens.scope
    encrypted = encrypt_credentials(creds)
    configuration = dict(integ.get("configuration") or {})
    if tokens.instance_url:
        configuration["instance_url"] = tokens.instance_url
    saved = complete_oauth_flow(
        integration_id, user_id,
        credentials_encrypted=encrypted,
        configuration=configuration,
        scope=tokens.scope,
        connection_status="connected",
    )
    if not saved:
        log.error(
            "[oauth.callback] complete_oauth_flow returned None integration_id=%s",
            integration_id,
        )
        return RedirectResponse(
            url=f"{frontend_origin}/integrations?error=oauth_internal",
            status_code=302,
        )
    log.info(
        "[oauth.callback] connected integration_id=%s provider=%s user_id=%s",
        saved["id"], saved["provider"], user_id,
    )
    return RedirectResponse(
        url=f"{frontend_origin}/integrations?connected={saved['id']}",
        status_code=302,
    )


def mark_integration_status_failed(cur, integration_id, user_id, message: str) -> None:
    """Tiny helper used by the OAuth callback when the token
    exchange fails AFTER the state was consumed. The state is gone
    (consumed), the integration needs reconnect — flip status to
    'failed' and stamp the last error message. The operator sees
    the [Reconnect] button on the detail view."""
    from STT_server.db_integrations import mark_integration_status as _mark
    _mark(
        integration_id, user_id, "failed",
        last_test_message=(message or "oauth_exchange_failed")[:500],
        cur=cur,
    )


@api_router.post("/integrations/{integration_id}/disconnect")
def disconnect_integration_endpoint(integration_id: str, auth: dict = Depends(require_auth)):
    """Disconnect an OAuth integration: best-effort revoke at the
    provider, then clear credentials locally.

    Refuses with 409 if any tools still depend on the integration —
    the operator must remove or reassign those tools first (same
    pattern as DELETE /integrations/{id}).

    After disconnect: connection_status='disconnected' (NOT
    'pending' — pending means "OAuth dance hasn't happened yet",
    disconnected means "was connected, operator chose to unlink").

    ponyy: the body is wrapped in a top-level try/except. The
    inner best-effort revoke already catches its own failures,
    but anything else (DB error, unexpected exception in the
    route layer, etc.) used to propagate as an unhandled 500 with
    no CORS headers — the browser then reported a generic "CORS
    error" that masked the real cause. With this guard, the
    operator always gets a structured 200 with `connection_status`
    and a `reason` field, so the FE can surface a snackbar with
    the actual error.
    """
    try:
        from STT_server.db_integrations import (
            get_integration as db_get_integration,
            count_dependent_tools,
            disconnect_integration as db_disconnect,
        )
        from STT_server.services.oauth_providers import (
            get_oauth_config, revoke_token,
        )
        from STT_server.security.credentials import decrypt_credentials
        integ = db_get_integration(integration_id, auth["user_id"])
        if not integ:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "connection_status": "unknown",
                    "reason": "integration_not_found",
                    "message": "Integration not found",
                },
            )
        deps = count_dependent_tools(integration_id, auth["user_id"])
        if deps > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "connection_status": integ.get("connection_status"),
                    "reason": "dependent_tools",
                    "message": f"{deps} tool{'s' if deps != 1 else ''} depend on this integration. Remove or reassign them first.",
                    "tool_count": deps,
                },
            )
        # Best-effort revoke at the provider. We do this BEFORE wiping
        # local credentials so the revoke call still has the access
        # token. Failures are logged but don't block the local wipe —
        # the operator wants out, we get them out.
        if integ.get("credentials_encrypted"):
            try:
                cfg = get_oauth_config(integ["provider"])
                creds_plain = decrypt_credentials(integ["credentials_encrypted"])
                access_token = creds_plain.get("access_token")
                if access_token:
                    revoke_token(cfg, access_token)
            except Exception as exc:
                # ponyy: this includes the case where SALESFORCE_*
                # env vars are missing (the operator has a stale
                # connection row from before the env was set). We
                # skip the revoke, wipe local creds, and surface a
                # warning — the operator's local state is what
                # matters for "disconnect".
                log.warning(
                    "[oauth.disconnect] remote revoke failed integration_id=%s err=%s",
                    integration_id, exc,
                )
        ok, err = db_disconnect(integration_id, auth["user_id"])
        if err:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "connection_status": integ.get("connection_status"),
                    "reason": "dependent_tools",
                    "message": err,
                },
            )
        if not ok:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "connection_status": "unknown",
                    "reason": "integration_not_found",
                    "message": "Integration not found",
                },
            )
        log.info(
            "[oauth.disconnect] disconnected integration_id=%s user_id=%s",
            integration_id, auth["user_id"],
        )
        return {"success": True, "connection_status": "disconnected"}
    except HTTPException:
        raise
    except Exception as exc:
        # ponyy: never let an unhandled 5xx out without CORS
        # headers. The browser reports those as a generic "CORS
        # error" and the operator can't tell what actually broke.
        # Return 200 with a structured body so the FE surfaces a
        # useful snackbar and we keep an audit trail in the logs.
        log.exception(
            "[oauth.disconnect] unhandled error integration_id=%s err=%s",
            integration_id, exc,
        )
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "connection_status": "unknown",
                "reason": "internal_error",
                "message": f"Disconnect failed: {exc}",
            },
        )


# ── /internal/integrations/{id}/credentials (service-token only) ────────────

@api_router.post("/internal/integrations/{integration_id}/credentials")
def internal_get_integration_credentials(
    integration_id: str,
    request: Request,
    _service: dict = Depends(require_service_token),
):
    """Ponytail: server-to-server endpoint used by n8n at call time.

    n8n calls this with the shared INTEGRATIONS_N8N_TOKEN to fetch
    the decrypted credentials for the integration backing a tool call.
    The tool executor posts `{integration_id, provider, action,
    arguments}` to n8n; n8n then hits this endpoint to grab the
    config + creds it needs to call Salesforce / Zendesk / etc.

    Auth: requires_service_token. User bearer tokens don't work —
    even the integration's owner can't fetch the plaintext creds
    back; they can only Replace them via PUT /integrations/{id}.

    Audit: every hit is logged with provider + integration_id + ip.
    The credentials themselves are NEVER logged (this is the line
    the operator would be paged on if it ever showed up in a log).

    Refresh-on-read: if the provider is OAuth and the access token
    is within 60s of expiry (or missing expires_at), we hold a
    pg_advisory_xact_lock keyed on the integration id and call
    refresh_access_token. Concurrent requests for the same
    integration serialize on the lock — the first one refreshes,
    the rest pick up the freshly-persisted row. If refresh fails
    (revoked/expired), we mark status='failed' and return
    `{credentials: null, reason: "refresh_failed"}` so n8n gets a
    deterministic 401 and the operator sees "Reconnect" in the UI.
    """
    from STT_server.db_integrations import (
        get_integration_by_id,
        acquire_advisory_xact_lock,
        update_integration_credentials,
        mark_integration_status,
    )
    from STT_server.security.credentials import (
        decrypt_credentials, encrypt_credentials,
    )
    from STT_server.services.oauth_providers import (
        get_oauth_config, refresh_access_token, is_token_expiring, now_plus_seconds,
        RefreshTokenRevoked, OAuthError,
    )
    from STT_server.db import get_conn
    # ponytail: internal endpoint uses the unscoped lookup so n8n
    # (which carries only the service token, not a user token) can
    # resolve any integration by id. Cross-user access is intentional
    # and scoped: the service token grants access to every row.
    row = get_integration_by_id(integration_id)
    if not row:
        log.warning(
            "[internal.creds] 404 integration_id=%s ip=%s",
            integration_id, request.client.host if request.client else "?",
        )
        raise HTTPException(status_code=404, detail="Integration not found")
    provider = row.get("provider")
    cipher = row.get("credentials_cipher") or "fernet-v1"
    encrypted = row.get("credentials_encrypted")
    if not encrypted:
        log.warning(
            "[internal.creds] no credentials stored integration_id=%s ip=%s",
            integration_id, request.client.host if request.client else "?",
        )
        return {
            "integration_id": row["id"],
            "provider": provider,
            "configuration": row.get("configuration") or {},
            "credentials": None,
            "reason": "no_credentials",
        }
    try:
        creds_plain = decrypt_credentials(encrypted) if cipher == "fernet-v1" else {}
    except Exception as exc:
        log.warning(
            "[internal.creds] decrypt failed integration_id=%s err=%s",
            integration_id, exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Credentials unavailable — decryption failed",
        )
    # Refresh-on-read only applies to OAuth providers. Static providers
    # never expire (until the operator Replaces them).
    from STT_server.services.integrations_catalog import get_integration_provider_spec
    spec = get_integration_provider_spec(provider)
    is_oauth = bool(spec and getattr(spec, "auth_type", "static") == "oauth")
    from STT_server.db import is_postgres
    if is_oauth and is_token_expiring(creds_plain.get("expires_at")) and is_postgres():
        # ponyy: refresh-on-read runs in ONE connection and ONE
        # transaction. acquire lock → re-read inside the lock →
        # (optional) refresh → persist creds + status → commit.
        # All queries share the same cursor; the helper UPDATE
        # functions accept `cur=` to skip their own connection
        # pool hop. Two concurrent calls for the same integration
        # serialize on pg_advisory_xact_lock and the second one
        # sees the freshly-persisted (fresh) tokens inside the lock.
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    acquire_advisory_xact_lock(cur, integration_id)
                    # Re-read inside the lock so we observe the
                    # latest committed row. The previous lock holder
                    # may have just persisted refreshed tokens.
                    from STT_server.db_integrations import get_integration_by_id as _reload
                    fresh = _reload(integration_id)
                    if fresh and fresh.get("credentials_encrypted"):
                        try:
                            creds_plain = (
                                decrypt_credentials(fresh["credentials_encrypted"])
                                if (fresh.get("credentials_cipher") or "fernet-v1") == "fernet-v1"
                                else creds_plain
                            )
                        except Exception:
                            pass
                    if not is_token_expiring(creds_plain.get("expires_at")):
                        # The previous lock holder already refreshed.
                        # Fall through to return.
                        pass
                    else:
                        refresh_token = creds_plain.get("refresh_token")
                        if not refresh_token:
                            # No refresh token — can't recover. Mark
                            # failed inside the lock and raise.
                            mark_integration_status(
                                integration_id, fresh["user_id"], "failed",
                                last_test_message="no refresh token",
                                cur=cur,
                            )
                            conn.commit()
                            raise HTTPException(
                                status_code=503,
                                detail={
                                    "error": "oauth_refresh_failed",
                                    "message": "Refresh token missing — reconnect required",
                                    "provider": provider,
                                    "integration_id": integration_id,
                                },
                            )
                        try:
                            cfg = get_oauth_config(provider)
                            new_tokens = refresh_access_token(cfg, refresh_token)
                        except RefreshTokenRevoked as exc:
                            mark_integration_status(
                                integration_id, fresh["user_id"], "failed",
                                last_test_message=str(exc),
                                cur=cur,
                            )
                            conn.commit()
                            log.warning(
                                "[internal.creds] refresh revoked integration_id=%s err=%s",
                                integration_id, exc,
                            )
                            raise HTTPException(
                                status_code=503,
                                detail={
                                    "error": "oauth_refresh_failed",
                                    "message": "Refresh token revoked — reconnect required",
                                    "provider": provider,
                                    "integration_id": integration_id,
                                },
                            )
                        except OAuthError as exc:
                            # Transient (network / 5xx from provider).
                            # Mark failed so the operator sees the
                            # error in the UI; raise 503 so n8n
                            # doesn't try to use a stale token.
                            mark_integration_status(
                                integration_id, fresh["user_id"], "failed",
                                last_test_message=str(exc),
                                cur=cur,
                            )
                            conn.commit()
                            log.warning(
                                "[internal.creds] refresh transient error integration_id=%s err=%s",
                                integration_id, exc,
                            )
                            raise HTTPException(
                                status_code=503,
                                detail={
                                    "error": "oauth_refresh_failed",
                                    "message": f"Refresh transient error: {exc}",
                                    "provider": provider,
                                    "integration_id": integration_id,
                                },
                            )
                        merged = dict(creds_plain)
                        merged["access_token"] = new_tokens.access_token
                        # ponyy: Salesforce only re-emits refresh_token
                        # when the operator's session policy rotates
                        # them. Keep the existing refresh_token if the
                        # provider didn't return a fresh one.
                        if new_tokens.refresh_token:
                            merged["refresh_token"] = new_tokens.refresh_token
                        if new_tokens.expires_in is not None:
                            merged["expires_at"] = now_plus_seconds(new_tokens.expires_in)
                        if new_tokens.scope:
                            merged["scope"] = new_tokens.scope
                        update_integration_credentials(
                            integration_id, fresh["user_id"],
                            encrypt_credentials(merged),
                            cur=cur,
                        )
                        mark_integration_status(
                            integration_id, fresh["user_id"], "connected",
                            last_test_message="token refreshed",
                            cur=cur,
                        )
                        creds_plain = merged
                        log.info(
                            "[internal.creds] refreshed token integration_id=%s provider=%s",
                            integration_id, provider,
                        )
                    # Single commit at the end of the with-block. The
                    # transaction was opened by get_conn()'s
                    # context manager (auto-commits on success).
        except HTTPException:
            raise
        except Exception as exc:
            log.warning(
                "[internal.creds] refresh path failed integration_id=%s err=%s",
                integration_id, exc,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "oauth_refresh_failed",
                    "message": f"Refresh path error: {exc}",
                    "provider": provider,
                    "integration_id": integration_id,
                },
            )
    log.info(
        "[internal.creds] ok integration_id=%s provider=%s ip=%s",
        integration_id, row.get("provider"), request.client.host if request.client else "?",
    )
    # ponytail: n8n should never see refresh_token. The BE keeps the
    # full token set (access + refresh + scope) encrypted at rest and
    # uses refresh_token internally when the access token expires.
    # The n8n workflow only needs the bearer for Salesforce API calls.
    # Filter here, at the edge, so even if the internal shape changes,
    # the wire shape stays minimal. Easy to extend for future providers
    # (add an `elif provider == "dynamics":` branch).
    def _credentials_for_n8n(provider: str, credentials: dict) -> dict:
        if provider == "salesforce":
            return {"access_token": credentials.get("access_token")}
        # Unknown provider: return nothing (fail closed). The n8n
        # workflow will error on missing access_token and the operator
        # will see it in the execution log.
        return {}
    return {
        "integration_id": row["id"],
        "provider": provider,
        "configuration": row.get("configuration") or {},
        "credentials": _credentials_for_n8n(provider, creds_plain),
    }
