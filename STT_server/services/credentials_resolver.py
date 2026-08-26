"""Centralized per-user provider credentials resolver.

The deployer puts provider keys in Railway as env vars (system defaults).
Each end user can upload their own keys from Settings → API in the FE.
At runtime, every adapter asks this module for the right key for the
current session — per-user first, env-var fallback. That makes the
keys behave exactly like env vars: any code path that previously read
`os.environ["OPENAI_API_KEY"]` now gets the user's key when they have
one, and the deployer's key when they don't.

`PROVIDER_CATALOG` is the single source of truth for which services
the app supports, what fields they expose, and how each field should
be validated (regex + min length). The /settings/api-keys REST routes
read this catalog to drive the FE modal, and call `validate_credentials`
to reject malformed input before it hits disk.

Add a new provider here and the FE + BE pick it up automatically.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from STT_server.security.credentials import decrypt_credentials
from STT_server.utils.safe_http import UnsafeURLError, validate_public_url

log = logging.getLogger("stt_server.security.resolver")


# ── Provider catalog ────────────────────────────────────────────────────────

# SSRF guard. Every provider base_url read from user-supplied credentials
# is funneled through _safe_base() so a malicious user can't point it at
# loopback, link-local (cloud metadata), or private VPC ranges.
def _safe_base(creds: dict | None, default: str) -> str:
    """Read `creds['base_url']`, fall back to `default`, validate the
    result against the SSRF allowlist, and return the URL stripped of
    trailing slash.

    Raises UnsafeURLError on rejection — caller should map that to a
    friendly HTTP error so the FE can show "blocked unsafe base_url".
    """
    raw = (creds or {}).get("base_url")
    candidate = (raw.strip().rstrip("/") if raw else default.rstrip("/"))
    validate_public_url(candidate)
    return candidate

@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    type: str = "text"            # "text" | "password"
    placeholder: str = ""
    required: bool = False
    pattern: str | None = None    # regex (re.search)
    min_length: int = 0
    max_length: int = 0
    help: str = ""                # shown below the field on the FE


@dataclass(frozen=True)
class ProviderSpec:
    id: str                       # "openai" | "deepgram" | ...
    name: str                     # human label
    description: str
    category: str                 # "llm" | "stt" | "tts" | "telephony" (primary; used for the tools_integrations row)
    fields: tuple[FieldSpec, ...]
    # If set, the /test endpoint runs this function (sync) with the
    # resolved plain dict and returns (ok, message).
    test_fn: Optional[str] = None  # dotted path, lazy-imported by the route
    # ponytail: a provider can serve multiple slots. OpenAI powers LLM
    # AND realtime STT — we don't want two provider rows for one key.
    # `category` stays the primary for DB storage (tools_integrations
    # has a single category column with a CHECK constraint); `categories`
    # is the full set used by the runtime resolver and the FE badge so
    # OpenAI lights up in both the LLM and STT slots. Empty tuple
    # defaults to (category,) — every existing spec keeps working.
    categories: tuple[str, ...] = ()


# Each pattern is intentionally permissive — the provider's own API
# is the source of truth. We only block obvious mistakes (empty, wrong
# prefix) so the user gets a fast "you typed it wrong" instead of a
# cryptic 401 on the first call.

PROVIDER_CATALOG: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        name="OpenAI",
        category="llm",
        # ponytail: OpenAI powers three slots. Realtime API = STT+LLM,
        # /v1/audio/speech = TTS. One DB row, three categories — the
        # resolver scans by `categories` so the same credential lights
        # up all three. Without 'tts' here, OpenAI would be invisible
        # to find_first_configured_provider(user_id, 'tts') even
        # though tts_dispatcher._stream_openai is wired and works.
        categories=("llm", "stt", "tts"),
        description="Powers the language model in voice calls and admin tools. Used for Chat Completions, Realtime STT+LLM, and /v1/audio/speech TTS.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                # ponytail: project keys are ~180 chars of base64 (can
                # contain '+', '/', '='); the previous regex only
                # accepted [A-Za-z0-9_-] so any base64 char broke it.
                # Test endpoint hit OpenAI's real API and passed; save
                # was 422 from this regex. Allow the base64 alphabet.
                pattern=r"^sk-[A-Za-z0-9+/=_\-]{20,}$",
                min_length=20,
                placeholder="sk-...",
                help="Starts with 'sk-' (or 'sk-proj-' for project keys).",
            ),
            # ponytail: tts_model / realtime_model pickers removed
            # from Settings → API. Model selection happens at the
            # agent level (Dashboard → Agents → New/Edit → LLM/TTS
            # dropdowns); Settings is now credentials-only.
        ),
        # ponytail: env_fallbacks removed. Per-user only.
        test_fn="STT_server.services.credentials_resolver._test_openai",
    ),
    ProviderSpec(
        id="anthropic",
        name="Anthropic",
        category="llm",
        description="Anthropic Claude models (claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus) via the Anthropic API.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                pattern=r"^sk-ant-(api\d{2}-)?[A-Za-z0-9_\-]{20,}$",
                min_length=20,
                placeholder="sk-ant-...",
            ),
            FieldSpec(
                name="base_url", label="Base URL", type="text",
                required=False,
                placeholder="https://api.anthropic.com",
                help="Required for custom-tenant / proxy endpoints. Leave empty to use the canonical api.anthropic.com.",
            ),
        ),
        # ponytail: env_fallbacks removed. The user enters their
        # Anthropic key via Settings → API or ModalAgents inline.
        test_fn="STT_server.services.credentials_resolver._test_anthropic",
    ),
    ProviderSpec(
        id="gemini",
        name="Google Gemini",
        category="llm",
        description="Google Gemini (gemini-1.5-flash, gemini-2.5-flash, gemini-3.1-pro, etc.) via the Gemini API.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                pattern=r"^AIza[0-9A-Za-z_\-]{30,}$",
                min_length=30,
                placeholder="AIza...",
                help="Google AI Studio API key. Starts with 'AIza'.",
            ),
            FieldSpec(
                name="base_url", label="Base URL", type="text",
                required=False,
                placeholder="https://generativelanguage.googleapis.com/v1beta",
                help="Required for custom-tenant / proxy endpoints.",
            ),
        ),
        # ponytail: env_fallbacks removed.
        test_fn="STT_server.services.credentials_resolver._test_gemini",
    ),
    ProviderSpec(
        id="minimax",
        name="MiniMax",
        category="llm",
        description="MiniMax chat completions (MiniMax-M3, MiniMax-M2.7). OpenAI-compatible API at api.minimax.io.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
                placeholder="ey...",
                help="MiniMax API key. Format varies by tier.",
            ),
            # ponytail: optional base URL override. MiniMax has multiple
            # public endpoints depending on the user's plan (token plan
            # vs coding plan vs subscription) and tenant. Without this,
            # validate tries 4 hardcoded candidates and gives up if none
            # match. With it, we hit exactly the URL the user expects.
            FieldSpec(
                name="base_url", label="Base URL", type="text",
                required=False,
                placeholder="https://api.MiniMax.com/v1",
                help="Required for token-plan / coding-plan / custom-tenant endpoints. Leave empty to use the standard candidates.",
            ),
        ),
        # ponytail: env_fallbacks removed.
        test_fn="STT_server.services.credentials_resolver._test_minimax",
    ),
    ProviderSpec(
        id="twilio",
        name="Twilio",
        category="telephony",
        description="Routes inbound and outbound phone calls. All three values are required for live calls.",
        fields=(
            FieldSpec(
                name="account_sid", label="Account SID", type="text",
                required=True,
                pattern=r"^AC[0-9a-fA-F]{32}$",
                min_length=34, max_length=34,
                placeholder="AC...",
            ),
            FieldSpec(
                name="auth_token", label="Auth Token", type="password",
                required=True,
                min_length=32, max_length=64,
            ),
            FieldSpec(
                name="phone_number", label="Phone Number", type="text",
                required=True,
                pattern=r"^\+[1-9]\d{6,14}$",
                placeholder="+15071234567",
                help="E.164 format with leading + and country code.",
            ),
        ),
        # ponytail: env_fallbacks removed. Per-number twilio_auth_token
        # is collected at phone-number create/edit time.
        test_fn="STT_server.services.credentials_resolver._test_twilio",
    ),
    ProviderSpec(
        id="elevenlabs",
        name="ElevenLabs",
        category="tts",
        description="Text-to-speech provider. WebSocket streaming, mu-law 8 kHz output for Twilio.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
                placeholder="...",
            ),
            # ponytail: voice_id / model_id pickers removed. Voice /
            # model selection happens at the agent level (Dashboard
            # → Agents → TTS voice dropdown); Settings is now
            # credentials-only.
        ),
        # ponytail: env_fallbacks removed.
        test_fn="STT_server.services.credentials_resolver._test_elevenlabs",
    ),
    ProviderSpec(
        id="rime",
        name="Rime",
        category="tts",
        # ponytail: Rime STT was removed from the spec; the adapter
        # was deleted in this commit batch. Rime remains TTS-only.
        description="Text-to-speech via Rime's WebSocket API (Astra, Celestino, etc.).",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=10,
                placeholder="...",
            ),
            # ponytail: model_id / speaker_en / speaker_es pickers
            # removed. Voice and model selection happens at the
            # agent level; Settings is now credentials-only.
        ),
        # ponytail: env_fallbacks removed.
        test_fn=None,
    ),
    ProviderSpec(
        id="deepgram",
        name="Deepgram",
        category="stt",
        # ponytail: Deepgram ships both a real-time STT adapter and a
        # TTS adapter (/v1/speak with mulaw/8000). The FE offers it in
        # both dropdowns; the catalog needs to advertise both.
        categories=("stt", "tts"),
        description="Speech-to-text provider. Used for both realtime transcription and the alternative TTS voice.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                pattern=r"^[A-Za-z0-9_\-]{20,}$",
                min_length=20,
                placeholder="...",
            ),
            # ponytail: STT model picker removed. Model selection
            # happens at the agent level (Dashboard → Agents → STT
            # dropdown); Settings is now credentials-only.
        ),
        # ponytail: env_fallbacks removed.
        test_fn="STT_server.services.credentials_resolver._test_deepgram",
    ),
    ProviderSpec(
        id="assemblyai",
        name="AssemblyAI",
        category="stt",
        description="Speech-to-text provider via AssemblyAI's realtime WS. Universal model auto-detects language; per-user model selector overrides the default.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
                help="Raw API key, no Bearer prefix. Shipped verbatim in the Authorization header.",
            ),
            # ponytail: STT model picker removed. Model selection
            # happens at the agent level; Settings is now
            # credentials-only.
        ),
        # ponytail: env_fallbacks removed.
        test_fn=None,
    ),
    ProviderSpec(
        id="inworld",
        name="Inworld",
        category="tts",
        # ponytail: Inworld ships a real-time STT adapter
        # (inworld_stt_realtime.py) in addition to the TTS one. The FE
        # shows it in both dropdowns; advertise both so the resolver
        # finds it when scanning for STT providers.
        categories=("tts", "stt"),
        description="Voice synthesis with character personas. Also powers real-time STT.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
            ),
        ),
        # ponytail: env_fallbacks removed.
        test_fn="STT_server.services.credentials_resolver._test_inworld",
    ),
)


def get_provider_spec(provider_id: str) -> ProviderSpec | None:
    for spec in PROVIDER_CATALOG:
        if spec.id == provider_id:
            return spec
    return None


# ── Validation ──────────────────────────────────────────────────────────────

def validate_credentials(provider_id: str, credentials: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Validate a credentials dict against the provider's FieldSpec rules.

    Returns ``(cleaned, errors)`` where cleaned is a dict with whitespace
    stripped and empty fields removed, and errors is a list of
    ``{"field": "...", "message": "..."}`` objects ready for a 400 response.
    Empty fields are silently dropped so the user can clear a field
    by submitting an empty string.
    """
    spec = get_provider_spec(provider_id)
    errors: list[dict[str, str]] = []
    if spec is None:
        return {}, [{"field": "_provider", "message": f"Unknown service '{provider_id}'"}]

    if not isinstance(credentials, dict):
        return {}, [{"field": "_credentials", "message": "credentials must be an object"}]

    cleaned: dict[str, str] = {}
    for f in spec.fields:
        raw = credentials.get(f.name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            # Field absent or empty: only an error if it's required and
            # the user submitted at least one field at all. They might
            # be clearing a previously-set value; the PUT route treats
            # "no field" as "no change" and "empty field" as "clear".
            continue
        if not isinstance(raw, str):
            errors.append({"field": f.name, "message": f"{f.label} must be a string"})
            continue
        value = raw.strip()
        if f.max_length and len(value) > f.max_length:
            errors.append({"field": f.name, "message": f"{f.label} is too long (max {f.max_length})"})
            continue
        if f.min_length and len(value) < f.min_length:
            errors.append({"field": f.name, "message": f"{f.label} is too short (min {f.min_length})"})
            continue
        if f.pattern and not re.search(f.pattern, value):
            errors.append({
                "field": f.name,
                "message": f"{f.label} doesn't match the expected format. {f.help}".strip(),
            })
            continue
        cleaned[f.name] = value

    # If the user submitted at least one value but no required field is present, error.
    if cleaned and not any(f.name in cleaned for f in spec.fields if f.required):
        for f in spec.fields:
            if f.required:
                errors.append({"field": f.name, "message": f"{f.label} is required"})

    return cleaned, errors


# ── Resolution ──────────────────────────────────────────────────────────────

# ponytail: platform env var names per provider. Reintroduced in
# 009_agent_use_own_key.sql so new agents can ship without forcing
# every operator to paste their own key first. The DEPLOYER sets
# these on Railway (one env per provider — exactly what the old code
# hardcoded before this resolver existed). The resolver reads them
# when the per-user / per-agent row is empty, so a SaaS-style "use
# our keys" UX lights up with a single FE toggle.
PLATFORM_ENV_KEYS: dict[str, dict[str, str]] = {
    "openai":     {"api_key": "OPENAI_API_KEY",       "realtime_model": "OPENAI_REALTIME_MODEL", "tts_model": "OPENAI_TTS_MODEL"},
    "anthropic":  {"api_key": "ANTHROPIC_API_KEY",    "base_url": "ANTHROPIC_BASE_URL"},
    "gemini":     {"api_key": "GEMINI_API_KEY",       "base_url": "GEMINI_BASE_URL"},
    "minimax":    {"api_key": "MINIMAX_API_KEY",      "base_url": "MINIMAX_BASE_URL"},
    "twilio":     {"account_sid": "TWILIO_ACCOUNT_SID", "auth_token": "TWILIO_AUTH_TOKEN", "phone_number": "TWILIO_PHONE_NUMBER"},
    "elevenlabs": {"api_key": "ELEVENLABS_API_KEY",   "voice_id": "ELEVENLABS_VOICE_ID", "model_id": "ELEVENLABS_MODEL_ID"},
    "rime":       {"api_key": "RIME_API_KEY",         "model_id": "RIME_MODEL_ID"},
    "deepgram":   {"api_key": "DEEPGRAM_API_KEY",     "model": "DEEPGRAM_MODEL"},
    "assemblyai": {"api_key": "ASSEMBLYAI_API_KEY",   "model": "ASSEMBLYAI_MODEL"},
    "inworld":    {"api_key": "INWORLD_API_KEY"},
}


def _read_platform(provider_id: str) -> dict[str, str]:
    """Read platform credentials for ``provider_id`` from os.environ.

    Returns an empty dict when no env vars are set for the provider.
    Never raises — a misconfigured deployer just means we fall back
    to the per-user store (which is also empty for new agents, so
    the call fails loud and the operator gets a clear error).
    """
    mapping = PLATFORM_ENV_KEYS.get(provider_id) or {}
    out: dict[str, str] = {}
    for field, env_name in mapping.items():
        v = os.environ.get(env_name, "").strip()
        if v:
            out[field] = v
    return out


def _read_per_user(user_id: str | None, provider_id: str) -> dict[str, str]:
    """Read per-user encrypted credentials and decrypt them. Returns
    empty dict when the user has nothing stored or decryption fails.
    Never raises — the caller falls back to env vars.

    ponytail: storage shape lives in db_tools. Per-user service
    credentials are agent_tools rows keyed by
    `(user_id, id=provider_id, agent_id='__shared__')` with the
    Fernet ciphertext dict under the `credentials` JSONB column.
    The legacy `tools_integrations.connected` boolean column is gone
    — presence of a non-null `credentials` IS the connected flag.

    ponytail: previous version swallowed every exception (DB read
    fail, JSON parse fail, Fernet decrypt fail) and returned {}.
    The caller falls back to env vars but the operator has no way
    to tell why the per-user key is missing. Now we log the actual
    exception so the next time a deploy rotates
    CREDENTIAL_ENCRYPTION_KEY (or the operator's machine restarts
    on ephemeral dev mode), the failure is loud instead of silent.
    """
    if not user_id:
        return {}
    try:
        from STT_server.db_tools import list_tools as db_list_tools
        rows = db_list_tools(user_id)
    except Exception as exc:
        log.warning(
            "[credentials] _read_per_user(%r, %r) list_tools failed: %s "
            "(treating as no per-user key; falling back to env vars)",
            user_id, provider_id, exc,
        )
        return {}
    row = next(
        (r for r in rows
         if r.get("id") == provider_id and r.get("credentials")),
        None,
    )
    if not row or not row.get("credentials"):
        return {}
    try:
        decrypted = decrypt_credentials(row["credentials"]) or {}
    except Exception as exc:
        # ponytail: most common cause is CREDENTIAL_ENCRYPTION_KEY
        # being rotated (or absent — ephemeral dev mode regenerates
        # the key on every container start). The row is still on
        # disk; the operator just needs to re-save the key in
        # Settings → API to encrypt it with the current key.
        log.warning(
            "[credentials] _read_per_user(%r, %r) decrypt failed: %s "
            "(stale ciphertext — re-save the key in Settings → API "
            "to encrypt it with the current CREDENTIAL_ENCRYPTION_KEY)",
            user_id, provider_id, exc,
        )
        return {}
    return {k: v for k, v in decrypted.items() if isinstance(v, str) and v}


def resolve_provider(
    user_id: str | None,
    provider_id: str,
    use_own_key: bool = True,
) -> dict[str, str]:
    """Resolve the active credentials for a provider, for the current user.

    Two source layers, merged with per-user winning on conflict:
      1. Per-user / per-agent credentials (tools_integrations row +
         inline API key state if you wired one up). Read first.
      2. Platform env vars (OPENAI_API_KEY etc., set by the deployer
         on Railway). Used as fallback when the per-user row is
         missing a field — restored in 009_agent_use_own_key.sql so
         a fresh agent works without forcing every operator to paste
         a key first.

    use_own_key=True (default, legacy behaviour): per-user reads
    apply, platform env fills the gaps. A user with their own key
    keeps using it; a user without one falls back to the deployer's
    key automatically.

    use_own_key=False: per-user layer is skipped entirely. Only
    platform env vars apply. Used by callers that branch on the
    agent row's stt/llm/tts_use_own_key column — when the operator
    toggled "use my own key" off, the resolver ignores the stored
    credential even if it exists. (Note: we still never silently
    drop a stored key at the storage layer — this only affects
    resolution at call time. The key stays in tools_integrations
    so flipping the toggle back on restores it.)

    Returns a flat dict of field -> value. Missing fields are absent
    from the dict, not set to None — callers should use ``.get()``.
    """
    spec = get_provider_spec(provider_id)
    if spec is None:
        return {}

    out: dict[str, str] = {}
    platform = _read_platform(provider_id)

    if use_own_key:
        per_user = _read_per_user(user_id, provider_id)
        # per_user wins where set, platform fills the rest
        for k, v in platform.items():
            out[k] = v
        for k, v in per_user.items():
            if v:
                out[k] = v
    else:
        # Toggle off: platform env only
        out.update(platform)

    return out


def resolve_for_session(
    session,                    # CallSession, typed as Any to avoid the
    category: str,              # circular import with domain.session.
    provider_id: str,
) -> dict[str, str]:
    """Session-aware wrapper around resolve_provider.

    Looks up the agent's `use_own_key` toggle for the given category
    (stt / llm / tts) on the session — denormalised at WS start from
    the agent row (009_agent_use_own_key.sql) — and forwards it.
    Callers that already hold a session pass this in instead of
    plumbing the category all the way to resolve_provider().

    Falls back to use_own_key=True when the session attribute is
    missing (legacy sessions predating the migration), so existing
    per-user setups keep working.
    """
    flag_attr = {
        "stt": "stt_use_own_key",
        "llm": "llm_use_own_key",
        "tts": "tts_use_own_key",
    }.get(category)
    use_own = True
    if flag_attr is not None:
        use_own = bool(getattr(session, flag_attr, True))
    user_id = getattr(session, "user_id", None)
    return resolve_provider(user_id, provider_id, use_own_key=use_own)


def is_provider_configured(user_id: str | None, provider_id: str) -> bool:
    """True when the provider has at least one field populated (per-user only)."""
    return bool(resolve_provider(user_id, provider_id))


def find_first_configured_provider(user_id: str | None, category: str) -> str | None:
    """Auto-detect a provider for a service category by scanning the
    per-user credentials the user has actually uploaded.

    Returns the id of the first provider in PROVIDER_CATALOG that
    matches the category and has a populated credential (per-user
    or env). Returns None if the user has nothing configured for
    that category — callers should treat that as a hard error
    (no env-var fallback) and surface a clear message to the FE.

    ponytail: a spec may serve multiple slots (OpenAI = llm + stt).
    The match checks `spec.categories` if non-empty, falling back to
    `(spec.category,)`. The single-category table column on the DB
    row doesn't matter here — only the catalog view does.
    """
    for spec in PROVIDER_CATALOG:
        all_categories = spec.categories if spec.categories else (spec.category,)
        if category not in all_categories:
            continue
        if spec.id == "twilio":  # telephony, not a runtime service
            continue
        if is_provider_configured(user_id, spec.id):
            return spec.id
    return None


# ── LLM picker for the Test data generator ────────────────────────────────
# ponytail: Settings → API stores the per-user API key as a
# credentials row in agent_tools (id=service_id, agent_id='__shared__').
# The /integrations page lists ALL such rows together with real n8n
# tools, which confuses operators — a saved key shouldn't render as
# a "tool" with Test / Edit / Delete buttons. The fix on the FE side
# is to filter credential rows from the tools list and surface them
# in a dedicated "Modelo LLM para los test" section that powers the
# test_data_generator's model picker.
#
# This helper produces the data that section needs: the LLM-capable
# providers (filtered from PROVIDER_CATALOG by category=='llm'),
# each with the hardcoded model catalog and the user's connection
# status. The FE shows only `connected: true` options; the user
# picks a model; the BE stores just the model id (the provider is
# implicit from the model name) in settings.test_data_model.

def _infer_provider_from_model(model: str) -> str | None:
    """Return the provider id whose hardcoded catalog contains `model`.

    The settings store just the model id (e.g. "gpt-4o-mini"); we
    need to know which provider the operator chose for the FE's
    "current selection" badge. Linear scan over the catalogs is fine
    — the model list is tiny.
    """
    if not model:
        return None
    for provider_id, models in _HARDCODED_LLM_MODELS.items():
        if any(m.get("id") == model for m in models):
            return provider_id
    return None


def get_llm_options(user_id: str | None, current_model: str = "") -> dict:
    """Return the LLM provider picker payload for the FE.

    Shape:
      {
        "current": {"provider": "openai" | None, "model": "gpt-4o-mini"},
        "providers": [
          {
            "id": "openai",
            "name": "OpenAI",
            "description": "...",
            "connected": true,
            "models": [{"id": "gpt-4o", "name": "gpt-4o", "description": "..."}],
          },
          ...
        ],
      }

    The list includes every LLM-capable provider from PROVIDER_CATALOG
    (regardless of connection status) so the FE can show "(not
    connected — connect in Settings → API)" for the ones the
    operator hasn't wired up. Connected status is computed from
    _read_per_user for each provider id.
    """
    providers: list[dict] = []
    for spec in PROVIDER_CATALOG:
        all_categories = spec.categories if spec.categories else (spec.category,)
        if "llm" not in all_categories:
            continue
        # ponytail: connected is determined from the per-user row
        # (the agent_tools row with id=spec.id, agent_id='__shared__',
        # credentials is non-null). is_provider_configured already
        # merges per-user + env, so a platform env-var fallback also
        # marks the provider as "connected" — fine for the FE badge.
        per_user = _read_per_user(user_id, spec.id)
        connected = bool(per_user)
        providers.append({
            "id": spec.id,
            "name": spec.name,
            "description": spec.description,
            "connected": connected,
            "categories": list(all_categories),
            "models": list(_HARDCODED_LLM_MODELS.get(spec.id, [])),
        })

    return {
        "current": {
            "provider": _infer_provider_from_model(current_model),
            "model": current_model,
        },
        "providers": providers,
    }


# ── Model discovery per provider ───────────────────────────────────────────
# Each provider returns a static catalog (or a live-fetched one) of
# models/voices. The Frontend calls list_provider_models(service, provider,
# api_key) to populate the secondary dropdown in the New Agent flow.
#
# Where the provider has a public listing endpoint, we fetch live so
# the FE never lists a model that doesn't exist or goes stale when a
# vendor ships a new one. Where the vendor has no public listing
# endpoint, we ship a hardcoded curated list.

# Canonical catalogs for providers without a public listing API. The
# frontend treats these as authoritative — when the vendor DOES add
# a listing endpoint, swap the catalog for a fetcher.

_HARDCODED_TTS_VOICES = {
    "openai": [
        {"id": "alloy",   "name": "Alloy",   "description": "Neutral, balanced"},
        {"id": "echo",    "name": "Echo",    "description": "Warm, conversational"},
        {"id": "fable",   "name": "Fable",   "description": "Expressive, narrative"},
        {"id": "onyx",    "name": "Onyx",    "description": "Deep, authoritative"},
        {"id": "nova",    "name": "Nova",    "description": "Bright, friendly"},
        {"id": "shimmer", "name": "Shimmer", "description": "Soft, gentle"},
    ],
    "rime": [
        {"id": "Astra",     "name": "Astra",     "description": "Female, energetic (English)"},
        {"id": "celestino", "name": "Celestino", "description": "Male, warm (Spanish)"},
        {"id": "Audrey",    "name": "Audrey",    "description": "Female, professional"},
        {"id": "Sierra",    "name": "Sierra",    "description": "Female, calm"},
    ],
    "elevenlabs": [
        {"id": "r8iaJkwUpytwsK5jNHRG",   "name": "Aria",          "description": "Female, warm middle-aged"},
        {"id": "21m00Tcm4TlvDq8ikWAM",   "name": "Rachel",        "description": "Female, calm young"},
        {"id": "AZnzlk1XvdvUeBnXmlld",   "name": "Domi",          "description": "Male, energetic young"},
        {"id": "EXAVITQu4vr4xnSDxMaL",   "name": "Bella",         "description": "Female, expressive young"},
    ],
    "deepgram": [
        {"id": "aura-asteria-en", "name": "Asteria (en)", "description": "Female, conversational"},
        {"id": "aura-luna-en",    "name": "Luna (en)",    "description": "Female, calm"},
        {"id": "aura-orion-en",   "name": "Orion (en)",   "description": "Male, narrative"},
        {"id": "aura-athena-en",  "name": "Athena (en)",  "description": "Female, professional"},
    ],
    # ponytail: no curated Inworld list. The live catalog from
    # GET /voices/v1/voices is authoritative; if the fetch fails,
    # the agent modal returns an empty list + an actionable error so
    # the operator knows to check their API key (see
    # list_provider_models → provider_id == "inworld").
}

_HARDCODED_STT_MODELS = {
    "deepgram": [
        {"id": "nova-3",       "name": "Nova-3",     "description": "Latest, highest accuracy (English + multilingual)"},
        {"id": "nova-3-medical","name": "Nova-3 Medical","description": "Medical-domain variant"},
        {"id": "nova-2",       "name": "Nova-2",     "description": "Previous gen, multilingual"},
        {"id": "nova-2-general","name": "Nova-2 General","description": "Nova-2 general-purpose"},
        {"id": "enhanced",     "name": "Enhanced",   "description": "Original Deepgram enhanced model"},
        {"id": "base",         "name": "Base",       "description": "Original base model"},
    ],
    "assemblyai": [
        {"id": "best",  "name": "Best",  "description": "Highest accuracy, multilingual"},
        {"id": "nano",  "name": "Nano",  "description": "Lowest latency, lower accuracy"},
        # ponytail: Universal-Streaming family. Listed so the FE can
        # attach per-model pricing; these names also match what the
        # AssemblyAI docs publish today.
        {"id": "universal-streaming",                "name": "Universal-Streaming",                "description": "English streaming, balanced"},
        {"id": "universal-streaming-multilingual",   "name": "Universal-Streaming Multilingual",   "description": "Multilingual streaming"},
        {"id": "universal-3.5-pro-realtime",         "name": "Universal-3.5 Pro Realtime",         "description": "Highest accuracy, realtime"},
    ],
    # ponytail: STT catalog now strictly streaming-only. OpenAI's
    # Whisper / gpt-4o-transcribe / gpt-4o-mini-transcribe are
    # batch REST endpoints — they don't satisfy the "low-latency
    # via WebSockets or HTTP chunked" rule in the realtime spec, so
    # they were removed. The agent's stt_model must be one of these
    # Realtime IDs for the openai_realtime adapter to work.
    "openai": [
        {"id": "gpt-realtime",                 "name": "GPT Realtime",                  "description": "OpenAI's latest GA realtime model (audio + text, low latency)"},
        {"id": "gpt-4o-realtime-preview",     "name": "GPT-4o Realtime Preview",       "description": "gpt-4o class audio + text Realtime API (preview)"},
        {"id": "gpt-4o-mini-realtime-preview", "name": "GPT-4o-mini Realtime Preview", "description": "Smaller / cheaper Realtime preview"},
    ],
    # ponytail: Rime STT removed entirely. Out of spec.
    "inworld": [
        # ponytail: real Inworld model id per
        # https://docs.inworld.ai/stt/overview. The placeholder
        # `inworld-default` we used to ship was rejected by the
        # /stt/v1/transcribe:streamBidirectional handshake.
        {"id": "inworld/inworld-stt-1", "name": "Inworld STT-1",
         "description": "Inworld first-party STT, 30 languages, voice profile + turn detection"},
    ],
}

_HARDCODED_LLM_MODELS = {
    "minimax": [
        # ponytail: real model IDs from api.minimax.io/v1/models.
        # The previous catalog (minimax / minimax-v1 / abab*) was
        # placeholder names that the dev had typed while waiting on
        # access to the real MiniMax API. Replaced with the IDs the
        # /v1/models endpoint actually returns, otherwise the FE
        # dropdown would ship model ids the LLM rejects.
        {"id": "MiniMax-M3",   "name": "MiniMax-M3",   "description": "Latest MiniMax model"},
        {"id": "MiniMax-M2.7", "name": "MiniMax-M2.7", "description": "Mid-tier MiniMax model"},
        {"id": "MiniMax-M2.5", "name": "MiniMax-M2.5", "description": "Smaller / cheaper MiniMax model"},
    ],
    # ponytail: Anthropic catalog trimmed to real Anthropic models
    # only (the previous pricing file carried several fictional
    # future-model names like `claude-fable-5`, `claude-mythos-5`,
    # `claude-opus-4-8` that never shipped; the kept entries below
    # are the ones Anthropic actually publishes today).
    "anthropic": [
        {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "description": "Latest balanced model, streaming"},
        {"id": "claude-haiku-3-5",  "name": "Claude Haiku 3.5",  "description": "Fast, lower-cost, streaming"},
        {"id": "claude-haiku-3",    "name": "Claude Haiku 3",    "description": "Smallest / cheapest, streaming"},
    ],
    # ponytail: OpenAI / Gemini / MiniMax fallback catalogs so the
    # dropdown is never empty when the user hasn't validated a key
    # yet (live fetch is the path of least surprise when the key IS
    # valid, but until then we still want the operator to see the
    # canonical model list and pick one). Mirrored from the FE pricing
    # table so the two stay in sync.
    "openai": [
        {"id": "gpt-4o",      "name": "gpt-4o",      "description": "Streaming-optimized flagship, multimodal"},
        {"id": "gpt-4o-mini", "name": "gpt-4o-mini", "description": "Cheap / fast streaming variant"},
        {"id": "o4-mini",     "name": "o4-mini",     "description": "Reasoning, streaming"},
    ],
    "gemini": [
        {"id": "gemini-1-5-flash",       "name": "gemini-1.5-flash",       "description": "Fast / cheap, streaming"},
        {"id": "gemini-2-5-flash",       "name": "gemini-2.5-flash",       "description": "Fast / cheap, streaming"},
        {"id": "gemini-3-1-pro",         "name": "gemini-3-1-pro",         "description": "Flagship ≤200K"},
        {"id": "gemini-3-1-pro-long",    "name": "gemini-3-1-pro-long",    "description": "Flagship >200K"},
        {"id": "gemini-3-5-flash",       "name": "gemini-3-5-flash",       "description": "Standard"},
        {"id": "gemini-3-flash",         "name": "gemini-3-flash",         "description": "Fast, cheap"},
        {"id": "gemini-3-1-flash-lite", "name": "gemini-3-1-flash-lite", "description": "Cheapest fast"},
        {"id": "gemini-2-5-pro",         "name": "gemini-2.5-pro",         "description": "Legacy pro"},
        {"id": "gemini-2-5-pro-long",    "name": "gemini-2.5-pro-long",    "description": "Legacy pro long"},
        {"id": "gemini-2-5-flash-lite",  "name": "gemini-2.5-flash-lite",  "description": "Legacy cheap"},
        {"id": "gemini-embeddings",      "name": "gemini-embeddings",      "description": "Embeddings"},
    ],
}


def _fetch_openai_models(api_key: str) -> list[dict]:
    """GET https://api.openai.com/v1/models — live list, returns id+owned_by+created."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = __import__("json").loads(resp.read().decode("utf-8"))
    # ponytail: coarse filter for clearly-not-a-pickable families
    # (legacy GPT-3 base models, embeddings, image gen, moderation).
    # The per-service picker in list_provider_models does the fine
    # filter (LLM excludes tts/realtime/transcribe; STT excludes
    # non-realtime; TTS includes only tts/realtime). Don't put
    # "tts-" or "realtime" here — the per-service filter is the
    # source of truth, and having the coarse filter eat them would
    # keep tts-1 / gpt-4o-mini-tts out of the TTS dropdown.
    SKIP_PREFIX = ("davinci", "curie", "babbage", "ada", "text-embedding",
                   "whisper-", "dall-e", "gpt-3.5-turbo-instruct")
    keep = []
    for m in payload.get("data", []):
        mid = m.get("id", "")
        if not any(mid.startswith(p) for p in SKIP_PREFIX):
            keep.append({
                "id": mid,
                "name": mid,
                "description": m.get("owned_by", ""),
            })
    return keep[:50]  # safety cap


def _fetch_anthropic_models(api_key: str, base_url: str = "https://api.anthropic.com") -> list[dict]:
    """GET /v1/models from Anthropic. Requires x-api-key + anthropic-version
    headers. Returns the live catalog so the FE dropdown shows models
    Anthropic actually serves today, not a hardcoded snapshot that
    goes stale on every release.

    SSRF: base_url is validated against the public-IP allowlist before
    we hand it to urlopen. A user-supplied override pointing at loopback
    or cloud metadata raises UnsafeURLError and we fall back to [].
    """
    import urllib.error
    import urllib.request
    try:
        try:
            base = _safe_base({"base_url": base_url}, "https://api.anthropic.com")
        except UnsafeURLError as exc:
            log.warning("[anthropic-models] blocked base_url: %s", exc)
            return []
        req = urllib.request.Request(
            f"{base}/v1/models?limit=200",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = __import__("json").loads(resp.read().decode("utf-8"))
        keep = []
        for m in payload.get("data", []):
            mid = m.get("id", "")
            if mid:
                keep.append({
                    "id": mid,
                    "name": m.get("display_name", mid),
                    "description": m.get("type", "Anthropic model"),
                })
        return keep
    except Exception as exc:
        log.info("[anthropic-models] fetch failed: %s", _sanitize_error(str(exc))[:200])
        return []


def _fetch_minimax_models(api_key: str, base_url: str | None = None) -> list[dict]:
    """GET /v1/models from MiniMax. The endpoint accepts both the
    canonical api.minimax.io and a custom base_url (token plan /
    coding plan / subscription) the user picked in Settings → API.
    Returns the live catalog so the FE dropdown reflects whatever
    MiniMax actually serves the user's account.

    SSRF: base_url validated via _safe_base().
    """
    import urllib.error
    import urllib.request
    try:
        try:
            base = _safe_base({"base_url": base_url}, "https://api.minimax.io/v1")
        except UnsafeURLError as exc:
            log.warning("[minimax-models] blocked base_url: %s", exc)
            return []
        req = urllib.request.Request(
            f"{base}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = __import__("json").loads(resp.read().decode("utf-8"))
        keep = []
        for m in payload.get("data", []):
            mid = m.get("id", "")
            if mid:
                keep.append({
                    "id": mid,
                    "name": mid,
                    "description": m.get("owned_by", "MiniMax model"),
                })
        return keep
    except Exception as exc:
        log.info("[minimax-models] fetch failed: %s", _sanitize_error(str(exc))[:200])
        return []


def _fetch_rime_models() -> list[dict]:
    """GET /v1/models from Rime. Rime's REST API is public — no auth
    header required for the model list. Used for both STT and TTS
    catalog fetches (the same endpoint lists both).
    """
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request("https://users-ws.rime.ai/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = __import__("json").loads(resp.read().decode("utf-8"))
        keep = []
        for m in payload.get("data", []) or []:
            mid = m.get("id", "")
            if mid:
                keep.append({
                    "id": mid,
                    "name": m.get("name", mid),
                    "description": m.get("description", "Rime model"),
                })
        return keep
    except Exception as exc:
        log.info("[rime-models] fetch failed: %s", _sanitize_error(str(exc))[:200])
        return []


def _fetch_rime_voices() -> list[dict]:
    """GET /v1/voices from Rime. Same endpoint family as models;
    no auth required for the public list.
    """
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request("https://users-ws.rime.ai/v1/voices")
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = __import__("json").loads(resp.read().decode("utf-8"))
        keep = []
        for v in payload.get("data", []) or []:
            vid = v.get("id", "")
            if vid:
                keep.append({
                    "id": vid,
                    "name": v.get("name", vid),
                    "description": v.get("description", "Rime voice"),
                })
        return keep
    except Exception as exc:
        log.info("[rime-voices] fetch failed: %s", _sanitize_error(str(exc))[:200])
        return []


def _fetch_inworld_voices(api_key: str) -> list[dict]:
    """GET https://api.inworld.ai/voices/v1/voices — Inworld's public
    voice list endpoint. Auth is Basic with the user's Inworld key.

    Unpaginated: Inworld's documented legacy code path returns the
    full voice list (≤2000) in a single response when called with no
    `pageSize` / `pageToken`. Our account sits at ~223 voices, well
    under the cap. No pagination, no `nextPageToken` cursor — one
    round-trip is enough.

    Returns the user's accessible voices (account-scoped) with the
    metadata the FE surfaces in the agent modal:
      id, name, displayName, description,
      gender (male|female|neutral|''),
      languageCode (BCP-47, e.g. "en-US") and langCode (legacy enum
        like "EN_US"),
      categories (e.g. ["companions", "enterprise"]),
      tags, source (SYSTEM|IVC|PVC), ageGroup.

    Language priority (lowest → highest):
      1. legacy langCode ("EN_US") — converted to BCP-47 ("en-US").
      2. promptLanguages[0] (BCP-47 already, list of locales the
         voice can synthesise). Populated for IVC clones and any
         voice the user customises — including ones where the live
         API omitted languageCode. Before this fallback the operator
         saw these clones fall into the FE's "OTHER" bucket.
      3. live languageCode (BCP-47 directly) — used by newer
         entries; kept for forward-compat.
    """
    import urllib.error
    import urllib.request
    base = "https://api.inworld.ai/voices/v1/voices"
    headers = {"Authorization": f"Basic {api_key}"}
    keep: list[dict] = []
    try:
        req = urllib.request.Request(base, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for v in payload.get("voices", []) or []:
            vid = v.get("voiceId") or v.get("id", "")
            if not vid:
                continue
            raw_lang = v.get("langCode") or ""
            live_langcode = v.get("languageCode") or ""
            prompt_langs = v.get("promptLanguages") or []
            canonical = live_langcode
            if not canonical and prompt_langs:
                canonical = prompt_langs[0]
            if not canonical and raw_lang and "_" in raw_lang:
                try:
                    lang_part, region_part = raw_lang.split("_", 1)
                    canonical = f"{lang_part.lower()}-{region_part.upper()}"
                except Exception:
                    canonical = raw_lang
            keep.append({
                "id": vid,
                "name": v.get("displayName") or v.get("name") or vid,
                "displayName": v.get("displayName") or v.get("name") or vid,
                "description": v.get("description", ""),
                "gender": v.get("gender", "") or "",
                "languageCode": canonical,
                "langCode": raw_lang,
                "categories": list(v.get("categories") or []),
                "tags": list(v.get("tags") or []),
                "source": v.get("source", "") or "",
                "ageGroup": v.get("ageGroup", "") or "",
                "promptLanguages": prompt_langs,
            })
        return keep
    except Exception as exc:
        log.warning("[inworld-voices] fetch failed (key prefix %s...): %s",
                    api_key[:6] if api_key else "(empty)",
                    _sanitize_error(str(exc))[:200])
        return []


def _fetch_assemblyai_models(api_key: str) -> list[dict]:
    """GET /v2/models from AssemblyAI. The api_key IS the Authorization
    header value (no "Bearer" prefix — AssemblyAI uses the raw key).
    """
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://api.assemblyai.com/v2/models",
            headers={"Authorization": api_key},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = __import__("json").loads(resp.read().decode("utf-8"))
        keep = []
        for m in payload.get("models", []) or []:
            mid = m.get("id", "")
            if mid and m.get("available", True):
                keep.append({
                    "id": mid,
                    "name": m.get("name", mid),
                    "description": m.get("description", "AssemblyAI model"),
                })
        return keep
    except Exception as exc:
        log.info("[assemblyai-models] fetch failed: %s", _sanitize_error(str(exc))[:200])
        return []


def _fetch_gemini_models(api_key: str) -> list[dict]:
    """GET https://generativelanguage.googleapis.com/v1beta/models?key=..."""
    import urllib.error
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = __import__("json").loads(resp.read().decode("utf-8"))
    keep = []
    for m in payload.get("models", []):
        name = m.get("name", "")
        # Gemini returns names like "models/gemini-1.5-pro" — strip prefix.
        short = name.replace("models/", "")
        if "generateContent" in m.get("supportedGenerationMethods", []):
            keep.append({
                "id": short,
                "name": short,
                "description": m.get("displayName", ""),
            })
    return keep


def _fetch_deepgram_models(api_key: str) -> list[dict]:
    """GET https://api.deepgram.com/v1/models — live STT model list."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        "https://api.deepgram.com/v1/models",
        headers={"Authorization": f"Token {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = __import__("json").loads(resp.read().decode("utf-8"))
    keep = []
    for m in payload.get("stt", []) or []:
        keep.append({
            "id": m.get("canonical_name") or m.get("name"),
            "name": m.get("name", ""),
            "description": m.get("architecture") or "",
        })
    return keep


def list_provider_models(service: str, provider_id: str, api_key: str | None = None,
                          user_id: str | None = None) -> dict:
    """Returns the catalog of models/voices for a provider+service.

    Credential resolution order:
      1. `api_key` passed by the FE (the user's inline value from the
         modal's API-key input, if any).
      2. The user's per-user stored credential, decrypted from
         tools_integrations using `user_id` (the auth context from the
         route). This is the key the operator typed in Settings → API
         for this account.
      3. The system env-var fallback for the provider.

    The docstring previously claimed `api_key=None` falls back to the
    user's stored credential, but the implementation only fell back to
    env-var (`resolve_provider(None, ...)`). That meant the Edit modal's
    catalog lookup used the deployer's key (lower plan, fewer voices)
    instead of the operator's saved key — exactly the regression that
    hid Bruno, Marie, etc. from the dropdown. Now the per-user
    credential is the second fallback.

    Returns:
        {"models": [{"id": "...", "name": "...", "description": "..."}, ...]}
        or {"models": [], "error": "..."} on failure.
    """
    creds = api_key
    if not creds and user_id:
        try:
            from STT_server.services.credentials_resolver import resolve_provider as _rp
            creds = _rp(user_id, provider_id).get("api_key")
        except Exception:
            creds = None
    if not creds:
        try:
            from STT_server.services.credentials_resolver import resolve_provider as _rp
            creds = _rp(None, provider_id).get("api_key")
        except Exception:
            creds = None
    creds = creds or ""

    try:
        if service == "llm":
            if provider_id == "openai":
                # Live fetch when the user has a key. Without a key we
                # fall back to the provider's own hardcoded catalog so
                # the dropdown is never empty *and never shows the wrong
                # provider's models* (the previous Anthropic fallback
                # was the source of the dropdown cross-contamination bug).
                if creds:
                    models = _fetch_openai_models(creds)
                    if models:
                        # ponytail: the LLM picker used to return the
                        # raw fetch — the operator saw 60+ models
                        # including legacy gpt-3.5-turbo, gpt-4,
                        # gpt-4-turbo, dated snapshots
                        # (gpt-4o-2024-05-13, o1-2024-12-17), and
                        # everything else in the catalog. Apply a
                        # current-generation filter:
                        #  1. Drop TTS / STT / embeddings / images /
                        #     moderation / search / realtime.
                        #  2. Keep only the gpt-4o / gpt-4.1 / gpt-5 /
                        #     o1 / o3 / o4 / o5 families.
                        #  3. Drop dated snapshots
                        #     (any name with a -YYYY or -YYYY-MM
                        #     suffix) — those are deprecated.
                        import re
                        _SKIP_TOKENS = (
                            "tts", "whisper-", "transcribe",
                            "embedding", "dall-e", "moderation",
                            "search-", "realtime",
                        )
                        _KEEP_PREFIXES = (
                            "gpt-4o", "gpt-4.1", "gpt-5",
                            "o1", "o3", "o4", "o5",
                        )
                        _DATED_SNAPSHOT = re.compile(r"-\d{4}(-\d{2})?(-preview)?(-light)?$")
                        llm = []
                        for m in models:
                            mid = m["id"]
                            mid_l = mid.lower()
                            if any(s in mid_l for s in _SKIP_TOKENS):
                                continue
                            if not any(mid.startswith(p) for p in _KEEP_PREFIXES):
                                continue
                            if _DATED_SNAPSHOT.search(mid):
                                continue
                            llm.append(m)
                        if llm:
                            return {"models": llm}
                return {"models": _HARDCODED_LLM_MODELS.get("openai", [])}
                if provider_id == "anthropic":
                    if creds:
                        # honor custom base_url for token-plan / custom
                        # tenant endpoints; default to api.anthropic.com.
                        # SSRF: validate via _safe_base before fetch.
                        try:
                            base = _safe_base(creds, "https://api.anthropic.com")
                        except UnsafeURLError as exc:
                            log.warning("[llm-models anthropic] blocked base_url: %s", exc)
                            base = None
                        if base:
                            models = _fetch_anthropic_models(creds, base)
                            if models:
                                return {"models": models}
                    return {"models": _HARDCODED_LLM_MODELS["anthropic"]}
                if provider_id == "gemini":
                    if creds:
                        models = _fetch_gemini_models(creds)
                        if models:
                            return {"models": models}
                    # Empty list when no key - the dropdown shows "No options"
                    # and the FE can prompt the user to validate first. Same
                    # rationale as the OpenAI branch above.
                    return {"models": _HARDCODED_LLM_MODELS.get("gemini", [])}
                if provider_id == "minimax":
                    if creds:
                        # honor custom base_url the user picked in
                        # Settings → API (token plan / coding plan / etc.).
                        # SSRF: validate via _safe_base before fetch.
                        try:
                            base = _safe_base(creds, "https://api.minimax.io/v1")
                        except UnsafeURLError as exc:
                            log.warning("[llm-models minimax] blocked base_url: %s", exc)
                            base = None
                        if base:
                            models = _fetch_minimax_models(creds, base)
                            if models:
                                return {"models": models}
                    return {"models": _HARDCODED_LLM_MODELS["minimax"]}
                return {"models": []}

        if service == "tts":
            # Live for OpenAI if key present, hardcoded catalog otherwise.
            if provider_id == "openai":
                if creds:
                    models = _fetch_openai_models(creds)
                    # ponytail: previous filter was the hardcoded voice
                    # whitelist PLUS "starts with tts-" — that left
                    # gpt-4o-mini-tts (suffix variant) and the realtime
                    # TTS-capable models out, so the operator didn't
                    # see what they were actually allowed to pick.
                    # Open the filter to anything that's a real TTS
                    # model: the hardcoded voices, anything starting
                    # with "tts-", anything containing "-tts" or
                    # "realtime".
                    tts_models = [m for m in models
                                  if m["id"] in {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
                                  or m["id"].startswith("tts-")
                                  or "-tts" in m["id"].lower()
                                  or "realtime" in m["id"].lower()]
                    if tts_models:
                        return {"models": tts_models}
                return {"models": _HARDCODED_TTS_VOICES["openai"]}
            if provider_id == "rime":
                # Rime REST API is public (no auth) — always try the
                # live fetch. Rime's voices endpoint is separate from
                # models, so we run both and dedupe by id.
                try:
                    live = list({v["id"]: v for v in (
                        _fetch_rime_voices() + _fetch_rime_models()
                    )}.values())
                except Exception:
                    live = []
                if live:
                    return {"models": live}
                return {"models": _HARDCODED_TTS_VOICES["rime"]}
            if provider_id == "elevenlabs":
                if creds:
                    try:
                        # ElevenLabs /v1/voices returns the user's available voices.
                        import urllib.error
                        import urllib.request
                        req = urllib.request.Request(
                            "https://api.elevenlabs.io/v1/voices",
                            headers={"xi-api-key": creds},
                        )
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            payload = __import__("json").loads(resp.read().decode("utf-8"))
                            live = [
                                {"id": v["voice_id"], "name": v.get("name", v["voice_id"]),
                                 "description": v.get("category", "")}
                                for v in payload.get("voices", [])
                            ]
                            if live:
                                return {"models": live}
                    except Exception:
                        pass
                return {"models": _HARDCODED_TTS_VOICES["elevenlabs"]}
            if provider_id == "deepgram":
                if creds:
                    try:
                        dg = _fetch_deepgram_models(creds)
                        tts = [m for m in dg if (m.get("description") or "").lower() == "tts"]
                        if tts:
                            return {"models": tts}
                    except Exception:
                        pass
                return {"models": _HARDCODED_TTS_VOICES["deepgram"]}
            if provider_id == "inworld":
                # ponytail: show exactly what Inworld returns. The
                # live catalog (~223 SYSTEM voices + any IVC / PVC
                # clones on the account) is authoritative — never
                # blend it with a stale curated subset. If the fetch
                # fails (no key, auth error, network), return empty
                # + an actionable error so the operator sees the gap
                # and fixes it.
                if creds:
                    live = _fetch_inworld_voices(creds)
                    if live:
                        return {"models": live}
                    # Live fetch returned empty (auth failure, network,
                    # or zero voices on the account). Bubble the empty
                    # state to the FE. The modal's Dropdown degrades to
                    # a free-text voice input when `models` is empty, so
                    # the operator can still pick a voice by id.
                    return {"models": [], "error": "Inworld voice catalog unavailable — check your API key or try again."}
                # No API key provided. The agent can't synthesise TTS
                # without one, so don't fabricate a dropdown from a
                # stale list.
                return {"models": [], "error": "Inworld API key not configured."}
            return {"models": []}

        if service == "stt":
            if provider_id == "deepgram":
                if creds:
                    try:
                        models = _fetch_deepgram_models(creds)
                        stt = [m for m in models if (m.get("description") or "").lower() != "tts"]
                        if stt:
                            return {"models": stt}
                    except Exception:
                        pass
                return {"models": _HARDCODED_STT_MODELS["deepgram"]}
            if provider_id == "assemblyai":
                if creds:
                    models = _fetch_assemblyai_models(creds)
                    if models:
                        return {"models": models}
                return {"models": _HARDCODED_STT_MODELS["assemblyai"]}
            if provider_id == "openai":
                # Realtime-compatible STT. The previous commit (561f6be)
                # tightened this to drop batch transcribe; see its
                # message. The matching LLM / TTS filters for OpenAI
                # live in their own `if service == ...` blocks above.
                if creds:
                    try:
                        models = _fetch_openai_models(creds)
                        stt = [m for m in models
                               if "realtime" in m["id"].lower()
                               and "transcribe" not in m["id"].lower()]
                        if stt:
                            return {"models": stt}
                    except Exception:
                        pass
                return {"models": _HARDCODED_STT_MODELS.get("openai", [])}
            if provider_id == "inworld":
                return {"models": _HARDCODED_STT_MODELS["inworld"]}
            return {"models": []}

        return {"models": [], "error": f"Unknown service '{service}'"}
    except Exception as exc:
        log.exception("[list_provider_models] %s/%s failed: %s", service, provider_id, exc)
        return {"models": [], "error": _sanitize_error(str(exc))[:300]}


# ── Connection test helpers (used by /test endpoint) ────────────────────────

def _import(path: str):
    module_name, _, attr = path.rpartition(".")
    mod = __import__(module_name, fromlist=[attr])
    return getattr(mod, attr)


def _sanitize_error(message: str) -> str:
    """Redact any API-key-like substrings from an error message.

    ponytail: the OpenAI Python SDK (and Anthropic, Google, etc.)
    include the first/last few characters of the key in their error
    responses for debugging. We don't want the user to see that in
    the FE. Match common key shapes and replace with `***`. The
    rest of the message (HTTP code, endpoint hint, retry advice) is
    preserved so the user still gets useful feedback.
    """
    if not message:
        return message
    import re as _re
    # OpenAI: sk-..., sk-proj-..., sk-svcacct-...
    # Match the prefix + at least 8 chars (a real key is 40+; this
    # threshold just avoids false positives on things like 'sk-abc'
    # in error text).
    msg = _re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "sk-***", message)
    # Anthropic: sk-ant-...
    msg = _re.sub(r"\bsk-ant-[A-Za-z0-9_\-]{8,}", "sk-ant-***", msg)
    # Google: AIza... real keys are 39 chars total; allow {15,} after
    # the prefix to catch truncated strings (the original OpenAI /
    # Google message often masks the key leaving only ~19 chars).
    msg = _re.sub(r"\bAIza[A-Za-z0-9_\-]{15,}", "AIza***", msg)
    # Generic Bearer/Basic tokens in URLs or messages
    msg = _re.sub(r"(?i)(bearer|basic)\s+[A-Za-z0-9_\-\.=]{12,}", r"\1 ***", msg)
    # Long hex/alnum blobs that look like raw keys (30+ chars in a row,
    # with a word boundary so we don't redact normal text). Real API
    # keys are 32+ chars; 30 gives a small buffer.
    msg = _re.sub(r"\b[A-Za-z0-9_\-]{30,}\b", "***", msg)
    return msg


def _test_openai(creds: dict[str, str]) -> tuple[bool, str]:
    """Cheapest OpenAI call: list models. Avoids any billing."""
    try:
        from openai import OpenAI
    except Exception as exc:
        return False, f"openai SDK not installed: {exc}"
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    try:
        client = OpenAI(api_key=key, timeout=10.0)
        # List with limit=1 is the cheapest call available.
        page = client.models.list()
        # Touch the iterator so the request actually fires.
        first = next(iter(page), None)
        return True, "ok" if first is not None else "no models returned"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def _test_deepgram(creds: dict[str, str]) -> tuple[bool, str]:
    """Hit Deepgram's /v1/projects which is auth-protected and free."""
    import urllib.error
    import urllib.request
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    try:
        req = urllib.request.Request(
            "https://api.deepgram.com/v1/projects",
            headers={"Authorization": f"Token {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def _test_elevenlabs(creds: dict[str, str]) -> tuple[bool, str]:
    """ElevenLabs /v1/voices — auth-protected, free."""
    import urllib.error
    import urllib.request
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": key},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def _test_gemini(creds: dict[str, str]) -> tuple[bool, str]:
    """Hit Gemini's /v1beta/models?key=.. — auth-protected, free."""
    import urllib.error
    import urllib.request
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=1"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def _test_anthropic(creds: dict[str, str]) -> tuple[bool, str]:
    """Hit Anthropic's /v1/messages with a 1-token reply request. Cheap
    (≤ 50 input tokens) and proves the key can authenticate.

    SSRF: creds["base_url"] is validated via _safe_base() before the
    request is made. A user-supplied override pointing at loopback /
    cloud metadata raises UnsafeURLError and we report that as the
    failure cause instead of attempting the request.
    """
    import urllib.error
    import urllib.request
    import json
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    try:
        base = _safe_base(creds, "https://api.anthropic.com")
    except UnsafeURLError as exc:
        return False, f"base_url rejected by SSRF guard: {exc}"
    try:
        req = urllib.request.Request(
            f"{base}/v1/messages",
            data=json.dumps({
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            body = "(no body)"
        return False, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def _test_minimax(creds: dict[str, str]) -> tuple[bool, str]:
    """Hit MiniMax /v1/models with bearer auth.

    Resolution order:
      1. creds["base_url"]   (user typed it in Settings → API)
      2. MINIMAX_BASE_URL env (server-wide override)
      3. The single canonical candidate: api.minimax.io/v1

    SSRF: every candidate (creds + env + default) is run through
    _safe_base() before the request. Unsafe URLs (loopback / private /
    cloud metadata) are silently dropped from the candidate list and
    reported as "rejected by SSRF guard" so the FE can show a clear
    error to the user.

    DNS failures (`Name or service not known`, errno -2) are reported
    distinctly from HTTP failures so the FE can tell the user "this
    URL is unreachable from our network" instead of misleading them
    with an opaque "HTTP None via ..." message.
    """
    import urllib.error
    import urllib.request
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"

    candidate_bases: list[str] = []
    creds_base = (creds.get("base_url") or "").strip().rstrip("/")
    if creds_base:
        candidate_bases.append(creds_base)
    env_base = os.environ.get("MINIMAX_BASE_URL", "").strip().rstrip("/")
    if env_base and env_base != creds_base:
        candidate_bases.append(env_base)
    candidate_bases.append("https://api.minimax.io/v1")

    # Validate every candidate up-front. Rejected ones are reported and
    # skipped — never used to build a URL.
    safe_candidates: list[str] = []
    rejected: list[str] = []
    for base in candidate_bases:
        try:
            validate_public_url(base)
            safe_candidates.append(base)
        except UnsafeURLError as exc:
            rejected.append(f"{base} ({exc})")
    if not safe_candidates:
        return False, (
            "All base_url candidates were rejected by the SSRF guard: "
            + "; ".join(rejected)
        )
    candidate_bases = safe_candidates

    last_status = None
    last_body = ""
    last_url = ""
    last_kind = "http"  # "http" | "dns" | "network"
    seen = set()
    for base in candidate_bases:
        if base in seen:
            continue
        seen.add(base)
        url = f"{base}/models"
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return True, f"HTTP {resp.status} via {base}"
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            last_url = base
            last_kind = "http"
            try:
                last_body = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                last_body = "(no body)"
            continue
        except Exception as exc:
            last_url = base
            msg = _sanitize_error(str(exc))
            last_body = msg[:200]
            # ponytail: errno -2 (EAI_NONAME) is "Name or service not
            # known" — pure DNS failure, not auth. Report it distinctly
            # so the FE doesn't render "HTTP None via ..." which makes
            # the user think their key is bad when the real problem
            # is that the host doesn't resolve from this container.
            if "Name or service not known" in msg or "No such host" in msg:
                last_kind = "dns"
            else:
                last_kind = "network"
            continue
    if last_kind == "dns":
        return False, f"DNS unreachable: {last_url} (hostname doesn't resolve from this server)"
    if last_kind == "network":
        return False, f"Network error via {last_url}: {last_body}"
    return False, f"HTTP {last_status} via {last_url}: {last_body}"


def _test_twilio(creds: dict[str, str]) -> tuple[bool, str]:
    """Validate the SID + token pair by fetching the account.

    Uses the Twilio SDK's Client.api.accounts(sid).fetch() which is the
    same call the official docs recommend. Returns (True, status) on
    success or (False, message) on auth failure / SDK missing.
    """
    try:
        from twilio.rest import Client
    except Exception as exc:
        return False, f"twilio SDK not installed: {exc}"
    sid = creds.get("account_sid")
    token = creds.get("auth_token")
    if not sid or not token:
        return False, "account_sid and auth_token are required"
    try:
        client = Client(sid, token)
        account = client.api.accounts(sid).fetch()
        return True, f"account status: {account.status}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def _test_inworld(creds: dict[str, str]) -> tuple[bool, str]:
    """Auth check via GET /voices/v1/voices. We don't POST to /tts/v1/voice
    here because that requires picking a valid voice_id from the user's
    workspace, and we don't know which ones they have at validation time.
    If the key authenticates against the voices list endpoint, the key
    is good; if it doesn't, the response body tells us exactly why."""
    import urllib.error
    import urllib.request
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    try:
        req = urllib.request.Request(
            "https://api.inworld.ai/voices/v1/voices?pageSize=1",
            headers={"Authorization": f"Basic {key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        return False, f"Inworld {exc.code}: {err_body or exc.reason}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))[:300]


def test_provider(
    user_id: str | None,
    provider_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Run the catalog-defined test_fn against the resolved credentials.

    Priority order for the key:
      1. ``api_key`` argument — what the FE just typed in the New Agent flow.
      2. Per-user stored credential (``tools_integrations.json``).
      3. System env-var fallback.

    ``base_url`` is the parallel override for providers whose test_fn
    needs to hit a custom endpoint (MiniMax token plan / coding plan).
    Same priority order: caller > per-user > env. Both are forwarded
    into the creds dict that the test_fn consumes.

    Returns a dict with ``valid`` (bool), ``message`` (str) and
    ``source`` ('inline' | 'user' | 'env' | 'none') so the FE can show
    where the tested key came from.
    """
    spec = get_provider_spec(provider_id)
    if spec is None:
        return {"valid": False, "message": f"Unknown service '{provider_id}'", "source": "none"}

    if api_key and api_key.strip():
        # Caller passed an explicit key (New Agent flow). Use it directly,
        # don't read from disk — the key is not necessarily persisted.
        creds = {"api_key": api_key.strip()}
        if base_url and base_url.strip():
            creds["base_url"] = base_url.strip()
        source = "inline"
    else:
        creds = resolve_provider(user_id, provider_id)
        if not creds:
            return {"valid": False, "message": "No credentials configured for this user or system default.", "source": "none"}
        # Did the active value come from per-user storage or env? Inspect raw
        # storage vs env so the FE can show "using your key" vs "using
        # system default".
        per_user_keys = set(_read_per_user(user_id, provider_id).keys())
        has_per_user = any(per_user_keys & creds.keys())
        source = "user" if has_per_user else "env"

    if not spec.test_fn:
        return {"valid": True, "message": "Format is valid. No live test available for this provider.", "source": source}

    try:
        fn = _import(spec.test_fn)
    except Exception as exc:
        return {"valid": False, "message": f"Test harness not available: {exc}", "source": source}

    try:
        ok, message = fn(creds)
    except Exception as exc:
        ok, message = False, _sanitize_error(str(exc))[:300]
    return {"valid": bool(ok), "message": message, "source": source}


# ── Categorized models for the agent picker ──────────────────────────────
# ponytail: the agent modal used to call POST /providers/models
# three times — once per service (llm / stt / tts) — every time
# asking the BE to filter OpenAI's /v1/models live response by a
# different keyword. The FE had to know each provider's naming
# convention and the BE did the same work three times. Replaced
# with a single POST /providers/models/categorized call that
# returns the provider's full catalog bucketed by service, with
# the same {id, label} contract for every provider. The FE
# doesn't need to know that OpenAI names its TTS family `tts-*`
# and Inworld names its catalog by `voiceId`; it just renders
# the three buckets.


def _format_model_label(model_id: str) -> str:
    """Render a human-readable label from a model id.

    Examples (operator-facing labels from the picker mockup):
        "gpt-4.1-mini"            -> "GPT-4.1 Mini"
        "gpt-4o-mini-transcribe"  -> "GPT-4o Mini Transcribe"
        "gpt-4o-mini-tts"          -> "GPT-4o Mini Tts"
        "o3-mini"                 -> "O3 Mini"
        "tts-1"                    -> "Tts 1"
        "nova-2-general"           -> "Nova 2 General"

    Two rules:
      1. "gpt-*" → "GPT-*" (the OpenAI family prefix is always
         uppercase, regardless of what follows — "gpt-4.1",
         "gpt-4o", "gpt-realtime", etc.).
      2. Everything else: split on "-", capitalize the first word
         (first letter only) and title-case subsequent words. The
         picker label is for display; the raw id is the lookup
         key the FE sends back, so the operator can copy paste
         from the dropdown without losing precision.
    """
    if not model_id:
        return ""
    if model_id.lower().startswith("gpt"):
        # ponytail: the OpenAI family prefix is always uppercase.
        # Handles both "gpt-..." (the typical case) and the bare
        # "gpt" id (rare; some operator display strings only show
        # the base family).
        if model_id.lower() == "gpt":
            return "GPT"
        body = model_id[3:]  # strip the "gpt" prefix (3 chars, no dash)
        # body now starts with "-" if model_id had the dash (e.g.
        # "gpt-4.1-mini" -> "-4.1-mini"). Split on "-" and
        # drop the leading empty token so head = "GPT-4.1".
        words = body.split("-")
        if words and words[0] == "":
            words = words[1:]
        if not words or all(w == "" for w in words):
            return "GPT"
        head = "GPT-" + (words[0][:1].upper() + words[0][1:] if words[0] else "")
        # Subsequent words keep their original case (not title-cased)
        # so "gpt-realtime" -> "GPT-realtime", "gpt-4.1-mini" -> "GPT-4.1 Mini".
        # Title-case subsequent words (the user expects
        # "gpt-4.1-mini" -> "GPT-4.1 Mini" with capital M).
        rest = " ".join(w[:1].upper() + w[1:] for w in words[1:] if w)
        joined = (head + " " + rest).strip() if rest else head
        return joined or "GPT"
    words = model_id.split("-")
    if not words or all(w == "" for w in words):
        return ""
    head = words[0][:1].upper() + words[0][1:]
    rest = " ".join(w[:1].upper() + w[1:] for w in words[1:] if w)
    joined = (head + " " + rest).strip() if rest else head
    return joined


def _classify_openai_model(model_id: str) -> str | None:
    """Return "llm" / "stt" / "tts" for an OpenAI model id, or None
    if the id doesn't fit a service.

    Mirrors the operator-supplied classifier (commit message):
      llm: gpt-4*, gpt-5*, o1*, o3*, o4* — and not in the
           excludedLLMTerms set (realtime, audio, transcribe, tts,
           image, embedding, whisper, moderation, search,
           computer-use, deep-research, codex).
      stt: id contains "transcribe" or "whisper".
      tts: id contains "tts" or "speech".
    """
    mid = model_id.lower()
    excluded = (
        "realtime", "audio", "transcribe", "tts", "image",
        "embedding", "whisper", "moderation", "search",
        "computer-use", "deep-research", "codex",
    )
    # STT: anything with transcribe or whisper in the name. Apply
    # this check first because some LLM-prefixed models also include
    # the word "transcribe" (e.g. gpt-4o-transcribe) and we want
    # them in the STT bucket, not LLM.
    if "transcribe" in mid or "whisper" in mid:
        return "stt"
    if "tts" in mid or "speech" in mid:
        return "tts"
    if mid.startswith(("gpt-4", "gpt-5", "o1", "o3", "o4")):
        # ponytail: drop dated snapshots (-YYYY-MM-DD, -YYYY-MM,
        # bare -YYYY) and bare legacy ids ("gpt-4", "gpt-4-turbo",
        # "gpt-3.5-turbo"). The picker only needs the current-gen
        # base ids; the operator can pick a snapshot via the BE-side
        # tools/agent modal override later if they really need one.
        import re as _re
        if _re.search(r"-\d{2,4}(-\d{2}){0,2}(-preview|-light)?\Z", mid):
            return None
        if mid in ("gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
                   "gpt-3.5-turbo-instruct", "gpt-5.6-luna",
                   "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"):
            return None
        if not any(term in mid for term in excluded):
            return "llm"
    return None


def _build_categorized_models(
    provider_id: str, api_key: str | None, user_id: str | None,
) -> dict:
    """Return one provider's full model catalog bucketed by service.

    Shape:
        {
            "provider": "<id>",
            "models": {
                "llm": [{"id": ..., "label": ...}, ...],
                "stt": [{"id": ..., "label": ...}, ...],
                "tts": [{"id": ..., "label": ...}, ...],
            },
        }

    For OpenAI the live /v1/models response is fetched and routed
    through the operator-supplied classifier (gpt-4* / gpt-5* / o1* /
    o3* / o4* with realtime-audio-transcribe-tts-image-... exclusions
    for llm; transcribe / whisper for stt; tts / speech for tts).
    For other providers we route to the existing per-service
    catalogs (TTS for Inworld / ElevenLabs / Rime / Deepgram
    Aura; STT for Deepgram / AssemblyAI; LLM for Anthropic /
    Google Gemini / MiniMax).
    """
    spec = get_provider_spec(provider_id)
    out = {"provider": provider_id, "models": {"llm": [], "stt": [], "tts": []}}

    def _to_entries(items):
        # ponytail: the categorized picker is a uniform contract
        # across every provider, so each entry gets {id, label} at
        # minimum. Inworld voices come from the live catalog with
        # rich per-voice metadata (description, gender, languageCode,
        # categories, tags, source) — we forward those fields so the
        # FE can render the rich 3-line Dropdown with the language
        # grouping. Other providers' curated catalogs don't carry
        # per-model metadata beyond name/description, so only
        # Inworld hits this branch.
        entries = []
        for it in items or []:
            mid = it.get("id") or it.get("voiceId") or ""
            if not mid:
                continue
            entry = {
                "id": mid,
                "label": _format_model_label(mid),
            }
            if "voiceId" in it or "displayName" in it:
                # Inworld voice. Forward the metadata the picker
                # needs for the rich Dropdown + language grouping.
                for f in (
                    "description", "gender", "languageCode",
                    "langCode", "categories", "tags",
                    "source", "ageGroup", "displayName", "name",
                ):
                    if it.get(f) not in (None, "", []):
                        entry[f] = it[f]
            entries.append(entry)
        return entries

    if provider_id == "openai":
        # ponytail: hit /v1/models live with the operator's key
        # (or stored credential / env fallback), then bucket every
        # id via _classify_openai_model. Buckets are sorted by id
        # so the FE picker has a stable order.
        # Same credential resolution as list_provider_models: inline
        # api_key, then per-user stored, then platform env.
        creds = api_key
        if not creds and user_id:
            try:
                from STT_server.services.credentials_resolver import resolve_provider as _rp
                creds = _rp(user_id, provider_id).get("api_key")
            except Exception:
                creds = None
        if not creds:
            try:
                from STT_server.services.credentials_resolver import resolve_provider as _rp
                creds = _rp(None, provider_id).get("api_key")
            except Exception:
                creds = None
        creds = creds or ""
        if not creds:
            return out
        try:
            models = _fetch_openai_models(creds)
        except Exception:
            return out
        buckets = {"llm": [], "stt": [], "tts": []}
        for m in models:
            mid = m.get("id", "")
            bucket = _classify_openai_model(mid)
            if not bucket:
                continue
            buckets[bucket].append(mid)
        for k in buckets:
            buckets[k].sort()
        for k, ids in buckets.items():
            out["models"][k] = _to_entries([{"id": i} for i in ids])
        return out

    # ponytail: every other provider's catalog lives in the curated
    # hardcoded table (openai/rime/elevenlabs/deepgram for TTS,
    # deepgram/assemblyai/openai/inworld for STT, anthropic/gemini/
    # MiniMax for LLM). Inworld's TTS bucket is fetched live — when
    # the fetch fails the bucket is empty + an error is bubbled to
    # the caller. Voice-only providers (Inworld) put their voices
    # in tts so the FE can render them in the TTS picker; the FE
    # doesn't need a separate "voices" bucket because every voice is
    # also a TTS option. This keeps the contract uniform across
    # providers.
    for service in ("stt", "tts", "llm"):
        try:
            items = list_provider_models(service, provider_id, api_key, user_id).get("models", [])
        except Exception:
            items = []
        out["models"][service] = _to_entries(items)
    return out
