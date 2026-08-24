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
from typing import Optional

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
    upsert_tool as db_upsert_tool,
    delete_tool as db_delete_tool,
    get_tool as db_get_tool,
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

# ponytail: this path used to be defined as a stray indented
# assignment that ran at module load time and was easy to miss
# when scanning the file. It's now a normal module-level
# constant alongside the helper functions that use it.
TOOLS_FILE = os.path.join(DATA_DIR, "agent_tools.json")


def _load_tools():
    # ponytail: on Railway the data/ volume is ephemeral so the
    # agent_tools.json file frequently doesn't exist (every fresh
    # deploy or restart). The previous IOError catch returned an
    # empty list silently — but any other exception class (custom
    # subclasses, OSError subclasses we didn't enumerate) would
    # propagate as a 500 with no log. We now catch the broader
    # Exception family and log the stack so the deploy logs surface
    # the real cause the next time we get a 500 here.
    try:
        with open(TOOLS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, IOError, OSError) as exc:
        log.warning("[tools] load failed (%s): %s", type(exc).__name__, exc)
        return []
    except Exception as exc:
        log.exception("[tools] unexpected load error: %s", exc)
        return []


def _save_tools(tools):
    # ponytail: ensure the data dir exists before opening for write.
    # On Railway the STT_server/data/ directory only ships with a
    # .gitkeep — if no agent has been created yet (the only other
    # code path that touches this directory), open(..., 'w') raises
    # FileNotFoundError, which propagates BEFORE FastAPI's CORS
    # middleware can attach headers. The browser then shows a
    # misleading "No Access-Control-Allow-Origin header" error
    # even though the BE itself is the problem. Same defensive
    # pattern as _save() below (line ~101) — consistent style.
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        # ponytail: surface the real cause to the operator. A bare
        # 500 with FastAPI's "Internal Server Error" body hides
        # the underlying issue (disk full, read-only volume, etc.).
        # Raising HTTPException keeps the CORS middleware in the
        # loop so the browser doesn't mask the failure as a CORS
        # error.
        raise HTTPException(
            status_code=500,
            detail=f"Could not persist agent_tools.json: {exc}",
        )


@api_router.get("/agents/{agent_id}/tools")
def list_agent_tools(agent_id: str, auth: dict = Depends(require_auth)):
    """List all tools for an agent.

    Returns per-agent tools (``agent_id == agent_id``) AND shared tools
    owned by the same user that have been explicitly assigned to this
    agent (``agent_id == "__shared__" and agent_id in assignments``).

    Without the shared branch the FE re-fetch after an assign looked
    identical to the pre-assign state — the shared tool stayed in
    "Available shared" because the response never echoed it back. The
    assignment itself succeeded; only the list refresh lied.
    """
    tools = _load_tools()
    out = []
    for t in tools:
        if t.get("user_id") != auth["user_id"]:
            continue
        if t.get("agent_id") == agent_id:
            out.append(t)
        elif t.get("agent_id") == SHARED_TOOL_AGENT_ID and agent_id in (t.get("assignments") or []):
            out.append(t)
    return out


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


@api_router.post("/agents/{agent_id}/tools")
def create_agent_tool(agent_id: str, data: ToolCreate, auth: dict = Depends(require_auth)):
    """Create a new tool for an agent."""
    from STT_server.domain.tool import AgentTool, validate_json_schema, VALID_TOOL_KINDS
    # ponytail: guard the kind field at the API boundary so an unknown
    # value from the FE never reaches the executor's branch table. The
    # AgentTool constructor also coerces, but failing fast here gives
    # the FE a clearer 400 than a silent fall-through to "webhook".
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
    )
    errors = tool.validate()
    if errors:
        raise HTTPException(status_code=400, detail=f"Validation errors: {', '.join(errors)}")
    is_valid, err = validate_json_schema(data.parameters)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid JSON Schema: {err}")
    tools = _load_tools()
    tool_dict = tool.to_dict()
    tool_dict["user_id"] = auth["user_id"]
    tools.append(tool_dict)
    _save_tools(tools)
    return tool_dict


@api_router.put("/agents/{agent_id}/tools/{tool_id}")
def update_agent_tool(agent_id: str, tool_id: str, data: ToolCreate, auth: dict = Depends(require_auth)):
    """Update an existing tool."""
    from STT_server.domain.tool import AgentTool, validate_json_schema, VALID_TOOL_KINDS
    if data.kind is not None and data.kind not in VALID_TOOL_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{data.kind}'. Expected one of: {sorted(VALID_TOOL_KINDS)}",
        )
    tools = _load_tools()
    for t in tools:
        if t.get("id") == tool_id and t.get("agent_id") == agent_id and t.get("user_id") == auth["user_id"]:
            t["name"] = data.name
            t["description"] = data.description
            t["webhook_url"] = data.webhook_url or ""
            t["filler_phrase"] = data.filler_phrase
            t["parameters"] = data.parameters
            # ponytail: persist the new fields on update. Same kind-aware
            # validation runs at the end so a webhook → call_transfer flip
            # (or vice-versa) can't sneak past without the matching
            # required field being present.
            t["kind"] = data.kind or t.get("kind", "webhook")
            t["destination"] = data.destination
            from datetime import datetime, timezone
            t["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            rebuilt = AgentTool.from_dict(t)
            errors = rebuilt.validate()
            if errors:
                raise HTTPException(status_code=400, detail=f"Validation errors: {', '.join(errors)}")
            is_valid, err = validate_json_schema(data.parameters)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid JSON Schema: {err}")
            _save_tools(tools)
            return t
    raise HTTPException(status_code=404, detail="Tool not found")


@api_router.delete("/agents/{agent_id}/tools/{tool_id}")
def delete_agent_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Delete a tool from an agent."""
    tools = _load_tools()
    before = len(tools)
    tools = [t for t in tools if not (t.get("id") == tool_id and t.get("agent_id") == agent_id and t.get("user_id") == auth["user_id"])]
    if len(tools) == before:
        raise HTTPException(status_code=404, detail="Tool not found")
    _save_tools(tools)
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
    """Assign a shared tool to an agent.

    Adds ``agent_id`` to the tool's ``assignments`` list. Idempotent:
    assigning an already-assigned agent is a no-op so the FE can
    freely click the same button twice.
    """
    tools = _load_tools()
    tool = next(
        (t for t in tools
         if t.get("id") == tool_id and t.get("user_id") == auth["user_id"]),
        None,
    )
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.get("agent_id") != SHARED_TOOL_AGENT_ID:
        raise HTTPException(
            status_code=400,
            detail="Only shared tools can be assigned. Per-agent tools are already available to their agent.",
        )
    # ponytail: confirm the agent belongs to this user. The agent row
    # is the source of truth for ownership; without this check a
    # operator could fish another user's agent id and widen a shared
    # tool's blast radius.
    from STT_server.db_agents import get_agent as _get_agent
    agent = _get_agent(agent_id, auth["user_id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    assignments = tool.get("assignments") or []
    if agent_id in assignments:
        return tool  # idempotent
    assignments.append(agent_id)
    tool["assignments"] = assignments
    from datetime import datetime, timezone
    tool["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _save_tools(tools)
    return tool


@api_router.delete("/agents/{agent_id}/tools/{tool_id}/assign")
def unassign_shared_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Remove a shared tool assignment from an agent.

    Drops ``agent_id`` from the tool's ``assignments`` list. Idempotent
    — unassigning an agent that isn't currently assigned is a no-op.
    """
    tools = _load_tools()
    tool = next(
        (t for t in tools
         if t.get("id") == tool_id and t.get("user_id") == auth["user_id"]),
        None,
    )
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.get("agent_id") != SHARED_TOOL_AGENT_ID:
        raise HTTPException(
            status_code=400,
            detail="Only shared tools can be unassigned. Delete per-agent tools instead.",
        )
    assignments = tool.get("assignments") or []
    if agent_id not in assignments:
        return tool  # idempotent
    assignments = [a for a in assignments if a != agent_id]
    tool["assignments"] = assignments
    from datetime import datetime, timezone
    tool["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _save_tools(tools)
    return tool


@api_router.post("/agents/{agent_id}/tools/test")
async def test_agent_tool(agent_id: str, tool_id: str, auth: dict = Depends(require_auth)):
    """Test a tool by executing its webhook with sample data."""
    tools = _load_tools()
    tool = next((t for t in tools if t.get("id") == tool_id and t.get("agent_id") == agent_id and t.get("user_id") == auth["user_id"]), None)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    from STT_server.services.tool_executor import execute_tool, ToolExecutionError, record_tool_result
    try:
        sample_args = {}
        for param_name in tool.get("parameters", {}).get("required", []):
            sample_args[param_name] = f"sample_{param_name}"
        result = await execute_tool(tool["webhook_url"], sample_args, tool["name"])
        record_tool_result(tool["id"], True, "test")
        return {"success": True, "result": result}
    except ToolExecutionError as exc:
        record_tool_result(tool["id"], False, "test")
        return {"success": False, "error": str(exc)}


# ponytail: global n8n tools marketplace. Tools with agent_id="__shared__"
# in the same agent_tools.json file are visible to any of the owner's
# agents (loaded by _load_agent_tools when called with user_id). The
# endpoints below let the operator create / edit / delete shared tools
# independently of any specific agent — the FE mounts them on the
# /integrations page (navbar entry).
SHARED_TOOL_AGENT_ID = "__shared__"


@api_router.get("/tools")
def list_shared_tools(auth: dict = Depends(require_auth)):
    """List all shared n8n tools owned by the current user."""
    tools = _load_tools()
    return [t for t in tools if isinstance(t, dict)
            and t.get("agent_id") == SHARED_TOOL_AGENT_ID
            and t.get("user_id") == auth["user_id"]]


@api_router.post("/tools")
def create_shared_tool(data: ToolCreate, auth: dict = Depends(require_auth)):
    """Create a new shared n8n tool owned by the current user."""
    from STT_server.domain.tool import AgentTool, validate_json_schema, VALID_TOOL_KINDS
    if data.kind is not None and data.kind not in VALID_TOOL_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{data.kind}'. Expected one of: {sorted(VALID_TOOL_KINDS)}",
        )
    tool = AgentTool(
        agent_id=SHARED_TOOL_AGENT_ID,
        name=data.name,
        description=data.description,
        webhook_url=data.webhook_url or "",
        filler_phrase=data.filler_phrase,
        parameters=data.parameters,
        kind=data.kind,
        destination=data.destination,
    )
    errors = tool.validate()
    if errors:
        raise HTTPException(status_code=400, detail=f"Validation errors: {', '.join(errors)}")
    is_valid, err = validate_json_schema(data.parameters)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid JSON Schema: {err}")
    tools = _load_tools()
    tool_dict = tool.to_dict()
    tool_dict["user_id"] = auth["user_id"]
    tools.append(tool_dict)
    _save_tools(tools)
    return tool_dict


@api_router.put("/tools/{tool_id}")
def update_shared_tool(tool_id: str, data: ToolCreate, auth: dict = Depends(require_auth)):
    """Update an existing shared n8n tool owned by the current user."""
    from STT_server.domain.tool import AgentTool, validate_json_schema, VALID_TOOL_KINDS
    if data.kind is not None and data.kind not in VALID_TOOL_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{data.kind}'. Expected one of: {sorted(VALID_TOOL_KINDS)}",
        )
    tools = _load_tools()
    for t in tools:
        if (t.get("id") == tool_id
                and t.get("agent_id") == SHARED_TOOL_AGENT_ID
                and t.get("user_id") == auth["user_id"]):
            t["name"] = data.name
            t["description"] = data.description
            t["webhook_url"] = data.webhook_url or ""
            t["filler_phrase"] = data.filler_phrase
            t["parameters"] = data.parameters
            t["kind"] = data.kind or t.get("kind", "webhook")
            t["destination"] = data.destination
            from datetime import datetime, timezone
            t["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            rebuilt = AgentTool.from_dict(t)
            errors = rebuilt.validate()
            if errors:
                raise HTTPException(status_code=400, detail=f"Validation errors: {', '.join(errors)}")
            is_valid, err = validate_json_schema(data.parameters)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid JSON Schema: {err}")
            _save_tools(tools)
            return t
    raise HTTPException(status_code=404, detail="Tool not found")


@api_router.delete("/tools/{tool_id}")
def delete_shared_tool(tool_id: str, auth: dict = Depends(require_auth)):
    """Delete a shared n8n tool owned by the current user."""
    tools = _load_tools()
    before = len(tools)
    tools = [t for t in tools
             if not (t.get("id") == tool_id
                     and t.get("agent_id") == SHARED_TOOL_AGENT_ID
                     and t.get("user_id") == auth["user_id"])]
    if len(tools) == before:
        raise HTTPException(status_code=404, detail="Tool not found")
    _save_tools(tools)
    return {"success": True}


@api_router.post("/tools/{tool_id}/test")
async def test_shared_tool(tool_id: str, auth: dict = Depends(require_auth)):
    """Smoke-test a shared n8n tool by hitting its webhook with sample args."""
    tools = _load_tools()
    tool = next((t for t in tools
                 if t.get("id") == tool_id
                 and t.get("agent_id") == SHARED_TOOL_AGENT_ID
                 and t.get("user_id") == auth["user_id"]),
                None)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    from STT_server.services.tool_executor import execute_tool, ToolExecutionError, record_tool_result
    try:
        sample_args = {}
        for param_name in tool.get("parameters", {}).get("required", []):
            sample_args[param_name] = f"sample_{param_name}"
        result = await execute_tool(tool["webhook_url"], sample_args, tool["name"])
        record_tool_result(tool["id"], True, "test")
        return {"success": True, "result": result}
    except ToolExecutionError as exc:
        # ponytail: persist the headline so the FE tooltip on the
        # tool card surfaces "HTTP 500: ..." without a Railway
        # log grep. Other exceptions (connection refused, DNS, etc.)
        # are caught by the generic except below.
        record_tool_result(tool["id"], False, "test", error=str(exc))
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        record_tool_result(
            tool["id"], False, "test",
            error=f"{type(exc).__name__}: {exc}",
        )
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


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
    db_upsert_tool(
        auth["user_id"],
        service_id,
        {
            "credentials": encrypted,
            "connected": bool(encrypted),
            "display_name": spec.name,
            "category": spec.category,
        },
    )
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