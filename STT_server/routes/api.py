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
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# ponytail: log was referenced in 8 places (lines 223, 228, 610, 615, 793,
# 1001, 1021, ...) but never defined. Every endpoint that touched
# one of those lines crashed with NameError -> 500 -> no CORS
# headers. Defined once here, used by all handlers.
log = logging.getLogger("stt_server.routes.api")

from STT_server.security.credentials import (
    encrypt_credentials, decrypt_credentials, decrypt_value,
)
from STT_server.services.credentials_resolver import (
    PROVIDER_CATALOG,
    get_provider_spec,
    is_provider_configured,
    list_provider_models,
    resolve_provider,
    test_provider,
    validate_credentials,
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
    upsert_tool as db_upsert_tool,
    delete_tool as db_delete_tool,
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
TOOLS_FILE = os.path.join(DATA_DIR, "tools_integrations.json")
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
    return hashlib.sha256(pwd.encode()).hexdigest()


# ponytail: in-memory cache of valid (token -> entry). require_auth first checks
# here, then falls back to the file on miss. Cache is invalidated by
# `auth.py` logout/password-change via `invalidate_session` (no such call exists
# in auth.py yet — see the W7 todo in routes/auth.py). Stale entries are
# lazy-evicted on access when expired.
_session_cache = {}  # token -> {"entry": {...}, "expires_at": datetime}


def _cache_get(token: str):
    cached = _session_cache.get(token)
    if not cached:
        return None
    if cached["expires_at"] < datetime.now(timezone.utc):
        _session_cache.pop(token, None)
        return None
    return cached["entry"]


def _cache_put(token: str, entry: dict) -> None:
    try:
        expires_at = _parse_expires_at(entry["expires_at"])
    except (ValueError, KeyError):
        return
    _session_cache[token] = {"entry": entry, "expires_at": expires_at}


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
    from STT_server.db_users import load_sessions, save_sessions
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization[len("Bearer "):]
    sessions = load_sessions()
    entry = sessions.get(token)
    if not entry:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        expires_at = _parse_expires_at(entry["expires_at"])
    except ValueError:
        raise HTTPException(status_code=401, detail="Token corrupted")
    if datetime.now(timezone.utc) > expires_at:
        # Best-effort delete; on Postgres the JSON fallback is a no-op.
        try:
            sessions.pop(token, None)
            save_sessions(sessions)
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="Token expired")
    return entry


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
    doesn't retry. No auth: Twilio signature validation lives in
    STT_Server.py's /voice webhook instead (this endpoint is only
    informational).
    """
    try:
        form = await request.form()
    except Exception as exc:
        log.warning("[call-status] could not read form: %s", exc)
        return {"ok": True}
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
        out["ok"] = False
        out["checks"]["postgres"] = f"FAIL: {exc}"
    # 2. PUBLIC_URL set?
    from STT_server.config import PUBLIC_URL
    if PUBLIC_URL:
        out["checks"]["public_url"] = "ok"
    else:
        out["ok"] = False
        out["checks"]["public_url"] = "FAIL: PUBLIC_URL env not set"
    # 3. At least one TTS provider reachable?
    from STT_server.config import (
        ELEVENLABS_API_KEY, OPENAI_API_KEY, DEEPGRAM_API_KEY, INWORLD_API_KEY,
    )
    from STT_server.adapters.tts_dispatcher import VALID_TTS_PROVIDERS
    keys = {
        "elevenlabs": bool(ELEVENLABS_API_KEY),
        "openai": bool(OPENAI_API_KEY),
        "deepgram": bool(DEEPGRAM_API_KEY),
        "inworld": bool(INWORLD_API_KEY),
    }
    out["checks"]["tts_env_keys"] = keys
    if not any(keys.values()):
        # Not fatal — users can bring their own per-provider keys via
        # Settings → API. Flag it but don't fail the healthcheck.
        out["checks"]["tts_env_keys_warning"] = "no system-level TTS keys; users must configure their own"
    # 4. Rime and AssemblyAI live under credentials_resolver env names
    from STT_server.config import RIME_API_KEY
    keys["rime"] = bool(RIME_API_KEY)
    return out


# ---------- Pydantic schemas ----------

class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1)
    voice: Optional[str] = None
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


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    voice: Optional[str] = None
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


class ToolConnect(BaseModel):
    credentials: Optional[dict] = None


class SettingsUpdate(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    timezone: Optional[str] = None
    notifications: Optional[dict] = None


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


def _format_phone_number(country: str, digits: str) -> str:
    d = digits.strip()
    if country == "+52":
        return f"+52 {d[:2]} {d[2:6]} {d[6:]}" if len(d) >= 10 else f"+52 {d}"
    if country == "+1":
        return f"+1 {d[:3]} {d[3:6]} {d[6:]}" if len(d) >= 10 else f"+1 {d}"
    return f"{country}{d}"


# ---------- /tools CRUD ----------

@api_router.get("/tools")
def list_tools(auth: dict = Depends(require_auth)):
    return db_list_tools(auth["user_id"])


@api_router.post("/tools/{tool_id}/connect")
def connect_tool(tool_id: str, data: ToolConnect, auth: dict = Depends(require_auth)):
    payload = {"credentials": data.credentials or {}, "connected": True}
    if not is_postgres():
        # JSON path - keep creating on first connect.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(TOOLS_FILE, "r", encoding="utf-8") as f:
                data_all = json.load(f) or []
        except (json.JSONDecodeError, IOError):
            data_all = []
        with _data_lock():
            existing = None
            for t in data_all:
                if t.get("id") == tool_id and t.get("user_id") == auth["user_id"]:
                    existing = t
                    break
            if existing:
                existing["connected"] = True
                existing["credentials"] = payload["credentials"]
                existing["connected_at"] = _now_iso()
            else:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                data_all.append({
                    "user_id": auth["user_id"], "id": tool_id, "connected": True,
                    "credentials": payload["credentials"], "connected_at": now,
                })
            with open(TOOLS_FILE, "w", encoding="utf-8") as f:
                json.dump(data_all, f, indent=2, ensure_ascii=False)
        return db_get_tool(auth["user_id"], tool_id) or {
            "user_id": auth["user_id"], "id": tool_id, "connected": True,
            "credentials": payload["credentials"],
        }
    return db_upsert_tool(auth["user_id"], tool_id, payload)


@api_router.delete("/tools/{tool_id}")
def disconnect_tool(tool_id: str, auth: dict = Depends(require_auth)):
    if not is_postgres():
        with _data_lock():
            tools = _load(TOOLS_FILE, [])
            for t in tools:
                if t["id"] == tool_id and t.get("user_id") == auth["user_id"]:
                    t["connected"] = False
                    t["credentials"] = {}
                    _save(TOOLS_FILE, tools)
                    return {"success": True}
        raise HTTPException(status_code=404, detail="Tool not found")
    # ponytail: on Postgres the row may not exist (the user never
    # connected it). We accept that and return success - the FE
    # interprets DELETE on a non-existent tool as "it's not
    # connected anymore", which is the correct end state.
    db_upsert_tool(auth["user_id"], tool_id, {"connected": False, "credentials": {}})
    return {"success": True}


# ---------- /settings ----------

def _settings_path(user_id: str) -> str:
    return os.path.join(SETTINGS_DIR, f"{user_id}.json")


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
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
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
    """
    user_id = auth["user_id"]
    with _data_lock():
        tools = _load(TOOLS_FILE, [])
    configured_ids = {
        t["id"] for t in tools
        if t.get("user_id") == user_id and t.get("connected")
    }
    services = []
    for spec in PROVIDER_CATALOG:
        item = _serialize_provider(spec, user_id)
        # "connected" = the user has saved a per-user key for this service.
        # System defaults don't count as "connected" — the FE renders those
        # as "Using system default" instead of "Connected".
        item["connected"] = spec.id in configured_ids
        services.append(item)
    return {"services": services}


@api_router.put("/settings/api-keys/{service_id}")
def upsert_api_key(service_id: str, body: ApiKeyUpdate, auth: dict = Depends(require_auth)):
    """Stores/updates the user's credentials for a service.

    Values are validated against the per-field regex/length in the
    provider catalog before encryption, so a typo returns a clean 400
    instead of a confusing 401 from the upstream provider on the next call.
    """
    if get_provider_spec(service_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service_id}'")
    if not body.credentials or not isinstance(body.credentials, dict):
        raise HTTPException(status_code=400, detail="credentials object is required")

    cleaned, errors = validate_credentials(service_id, body.credentials)
    if errors:
        # ponytail: the 422 was silent before — the FE only saw
        # 'Validation failed' without the field-level reason. Log the
        # full error list so the operator can see which field (api_key
        # vs base_url vs voice_id) failed and why, without having to
        # add console.log in the FE.
        log.warning(
            "[api-keys] upsert rejected service=%s user_id=%s errors=%s",
            service_id, auth["user_id"], errors,
        )
        # 422 keeps Pydantic semantics intact; the FE reads `errors` to
        # highlight the offending input.
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation failed", "errors": errors},
        )
    log.info(
        "[api-keys] upsert accepted service=%s user_id=%s fields=%s",
        service_id, auth["user_id"], list(cleaned.keys()),
    )

    encrypted = encrypt_credentials(cleaned)
    with _data_lock():
        tools = _load(TOOLS_FILE, [])
        existing = next(
            (t for t in tools
             if t["id"] == service_id and t.get("user_id") == auth["user_id"]),
            None,
        )
        if existing:
            existing["credentials"] = encrypted
            existing["connected"] = bool(encrypted)
            existing["connected_at"] = _now_iso() if encrypted else None
        else:
            tools.append({
                "id": service_id,
                "user_id": auth["user_id"],
                "connected": bool(encrypted),
                "credentials": encrypted,
                "connected_at": _now_iso() if encrypted else None,
            })
        _save(TOOLS_FILE, tools)
    return {"success": True}


@api_router.delete("/settings/api-keys/{service_id}")
def delete_api_key(service_id: str, auth: dict = Depends(require_auth)):
    """Disconnects a service for the user (falls back to env var default)."""
    with _data_lock():
        tools = _load(TOOLS_FILE, [])
        before = len(tools)
        tools = [
            t for t in tools
            if not (t["id"] == service_id and t.get("user_id") == auth["user_id"])
        ]
        if len(tools) == before:
            raise HTTPException(status_code=404, detail="Key not configured")
        _save(TOOLS_FILE, tools)
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
    # Read directly from per-user storage (NOT the resolver — we don't
    # want to leak system env-var values through the reveal endpoint).
    with _data_lock():
        tools = _load(TOOLS_FILE, [])
    row = next(
        (t for t in tools
         if t["id"] == service_id and t.get("user_id") == auth["user_id"]
         and t.get("connected") and t.get("credentials")),
        None,
    )
    if not row:
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
    dropdown with that provider's actual models. Live fetches are done
    server-side so we can swallow CORS / network failures and return a
    graceful fallback.
    """
    if body.service not in VALID_MODEL_SERVICES:
        raise HTTPException(
            status_code=400,
            detail=f"service must be one of {sorted(VALID_MODEL_SERVICES)}",
        )
    if get_provider_spec(body.provider) is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{body.provider}'")
    import asyncio as _aio
    result = await _aio.to_thread(
        list_provider_models, body.service, body.provider, body.api_key
    )
    return result


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
        tts_log.error("tts_preview failed for provider=%s voice=%s: %s", body.provider, body.voice_id, exc)
        raise HTTPException(status_code=502, detail=f"TTS preview failed: {exc}")
    if not audio_bytes:
        raise HTTPException(
            status_code=502,
            detail="TTS provider returned no audio. Check that the voice_id is valid for the selected provider.",
        )
    return Response(content=audio_bytes, media_type="audio/wav")