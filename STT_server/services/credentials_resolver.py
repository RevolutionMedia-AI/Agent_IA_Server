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
from dataclasses import dataclass, field
from typing import Any, Optional

from STT_server.security.credentials import decrypt_credentials

log = logging.getLogger("stt_server.security.resolver")


# ── Provider catalog ────────────────────────────────────────────────────────

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
    category: str                 # "llm" | "stt" | "tts" | "telephony"
    fields: tuple[FieldSpec, ...]
    # env vars the resolver falls back to when no per-user key is set.
    # Each value is mapped to the field name it should populate.
    env_fallbacks: tuple[tuple[str, str], ...] = ()   # [(env_var, field_name), ...]
    # If set, the /test endpoint runs this function (sync) with the
    # resolved plain dict and returns (ok, message).
    test_fn: Optional[str] = None  # dotted path, lazy-imported by the route


# Each pattern is intentionally permissive — the provider's own API
# is the source of truth. We only block obvious mistakes (empty, wrong
# prefix) so the user gets a fast "you typed it wrong" instead of a
# cryptic 401 on the first call.

PROVIDER_CATALOG: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        name="OpenAI",
        category="llm",
        description="Powers the language model in voice calls and admin tools. Used for both Chat Completions and Realtime STT+LLM.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                pattern=r"^sk-(proj-)?[A-Za-z0-9_\-]{20,}$",
                min_length=20,
                placeholder="sk-...",
                help="Starts with 'sk-' (or 'sk-proj-' for project keys).",
            ),
            FieldSpec(
                name="realtime_model", label="Realtime model", type="text",
                required=False,
                pattern=r"^gpt-4o(-[A-Za-z0-9.\-]+)?-realtime(-preview)?(-[0-9]{4}-\d{2}-\d{2})?$|^gpt-realtime(-[0-9]{4}-\d{2}-\d{2})?$",
                placeholder="gpt-4o-mini-realtime-preview",
                help="Optional. Defaults to OPENAI_REALTIME_MODEL on the backend.",
            ),
        ),
        env_fallbacks=(
            ("OPENAI_API_KEY", "api_key"),
            ("OPENAI_REALTIME_MODEL", "realtime_model"),
        ),
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
        env_fallbacks=(("ANTHROPIC_API_KEY", "api_key"), ("ANTHROPIC_BASE_URL", "base_url")),
        test_fn="STT_server.services.credentials_resolver._test_anthropic",
    ),
    ProviderSpec(
        id="gemini",
        name="Google Gemini",
        category="llm",
        description="Google Gemini (gemini-1.5-pro, gemini-1.5-flash, etc.) via the Gemini API. Picked as the agent's LLM in Settings → New Agent.",
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
        env_fallbacks=(("GEMINI_API_KEY", "api_key"), ("GEMINI_BASE_URL", "base_url")),
        test_fn="STT_server.services.credentials_resolver._test_gemini",
    ),
    ProviderSpec(
        id="minimax",
        name="MiniMax",
        category="llm",
        description="MiniMax chat completions (MiniMax-M3, MiniMax-M2.7, MiniMax-M2.5). OpenAI-compatible API at api.minimax.io.",
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
        env_fallbacks=(("MINIMAX_API_KEY", "api_key"), ("MINIMAX_BASE_URL", "base_url")),
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
        env_fallbacks=(),
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
            FieldSpec(
                name="voice_id", label="Voice ID", type="text",
                required=False,
                pattern=r"^[A-Za-z0-9]{10,40}$",
                placeholder="r8iaJkwUpytwsK5jNHRG",
                help="Optional. Defaults to ELEVENLABS_TTS_VOICE_ID on the backend.",
            ),
            FieldSpec(
                name="model_id", label="Model ID", type="text",
                required=False,
                pattern=r"^eleven_[A-Za-z0-9_]+$",
                placeholder="eleven_flash_v2_5",
                help="Optional. Defaults to ELEVENLABS_TTS_MODEL_ID.",
            ),
        ),
        env_fallbacks=(
            ("ELEVENLABS_API_KEY", "api_key"),
            ("ELEVENLABS_TTS_VOICE_ID", "voice_id"),
            ("ELEVENLABS_TTS_MODEL_ID", "model_id"),
        ),
        test_fn="STT_server.services.credentials_resolver._test_elevenlabs",
    ),
    ProviderSpec(
        id="rime",
        name="Rime",
        category="tts",
        description="Text-to-speech via WebSocket streaming. PCM -> mu-law conversion handled in the adapter.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=10,
                placeholder="...",
            ),
            FieldSpec(
                name="model_id", label="Model ID", type="text",
                required=False,
                pattern=r"^[A-Za-z0-9_\-]{1,40}$",
                placeholder="mist-v2",
                help="Optional. Defaults to RIME_TTS_MODEL_ID.",
            ),
            FieldSpec(
                name="speaker_en", label="English speaker", type="text",
                required=False,
                pattern=r"^[A-Za-z0-9_\-]{1,40}$",
                placeholder="Astra",
            ),
            FieldSpec(
                name="speaker_es", label="Spanish speaker", type="text",
                required=False,
                pattern=r"^[A-Za-z0-9_\-]{1,40}$",
                placeholder="celestino",
            ),
        ),
        env_fallbacks=(
            ("RIME_API_KEY", "api_key"),
            ("RIME_TTS_MODEL_ID", "model_id"),
        ),
        test_fn=None,
    ),
    ProviderSpec(
        id="deepgram",
        name="Deepgram",
        category="stt",
        description="Speech-to-text provider. Used for both realtime transcription and the alternative TTS voice.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                pattern=r"^[A-Za-z0-9_\-]{20,}$",
                min_length=20,
                placeholder="...",
            ),
            FieldSpec(
                name="model", label="STT model", type="text",
                required=False,
                pattern=r"^[a-z0-9\-]{1,40}$",
                placeholder="nova-3",
                help="Optional. Defaults to DEEPGRAM_STT_MODEL on the backend.",
            ),
        ),
        env_fallbacks=(
            ("DEEPGRAM_API_KEY", "api_key"),
            ("DEEPGRAM_STT_MODEL", "model"),
        ),
        test_fn="STT_server.services.credentials_resolver._test_deepgram",
    ),
    ProviderSpec(
        id="assemblyai",
        name="AssemblyAI",
        category="stt",
        description="Speech-to-text provider (alternative to Deepgram). Reserved for future use.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
            ),
        ),
        env_fallbacks=(("ASSEMBLYAI_API_KEY", "api_key"),),
        test_fn=None,
    ),
    ProviderSpec(
        id="inworld",
        name="Inworld",
        category="tts",
        description="Voice synthesis with character personas. Reserved for future use.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
            ),
        ),
        env_fallbacks=(("INWORLD_API_KEY", "api_key"),),
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

def _read_per_user(user_id: str | None, provider_id: str) -> dict[str, str]:
    """Read per-user encrypted credentials and decrypt them. Returns
    empty dict when the user has nothing stored or decryption fails.
    Never raises — the caller falls back to env vars.
    """
    if not user_id:
        return {}
    try:
        from STT_server.routes.api import _load  # local import: avoid cycle
        tools = _load(_tools_path(), [])
    except Exception:
        return {}
    row = next(
        (t for t in tools
         if t.get("id") == provider_id and t.get("user_id") == user_id and t.get("connected")),
        None,
    )
    if not row or not row.get("credentials"):
        return {}
    decrypted = decrypt_credentials(row["credentials"]) or {}
    return {k: v for k, v in decrypted.items() if isinstance(v, str) and v}


def _tools_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # STT_server/
        "data",
        "tools_integrations.json",
    )


def resolve_provider(user_id: str | None, provider_id: str) -> dict[str, str]:
    """Resolve the active credentials for a provider, for the current user.

    Resolution order:
      1. Per-user encrypted storage (Settings → API).
      2. System env-var defaults (Railway).

    Returns a flat dict of field -> value. Missing fields are absent
    from the dict, not set to None — callers should use ``.get()``.
    """
    spec = get_provider_spec(provider_id)
    if spec is None:
        return {}

    out: dict[str, str] = {}

    per_user = _read_per_user(user_id, provider_id)
    for f in spec.fields:
        if f.name in per_user and per_user[f.name]:
            out[f.name] = per_user[f.name]

    for env_var, field_name in spec.env_fallbacks:
        if field_name in out:
            continue
        val = os.environ.get(env_var, "").strip()
        if val:
            out[field_name] = val

    return out


def is_provider_configured(user_id: str | None, provider_id: str) -> bool:
    """True when the provider has at least one field populated (per-user or env)."""
    return bool(resolve_provider(user_id, provider_id))


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
    # ponytail: Inworld voice catalog is fetched live from
    # GET /voices/v1/voices in list_provider_models. This hardcoded
    # entry is only the fallback for when the user has no Inworld key
    # configured (the FE then shows an empty dropdown). The "Dennis"
    # placeholder we shipped before turned out to be an ElevenLabs
    # voice id - not Inworld - and produced HTTP 400 on every request.
    "inworld": [],
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
        {"id": "slam-1", "name": "SLAM-1", "description": "Streaming speech LM"},
        # ponytail: Universal-Streaming family. Listed so the FE can
        # attach per-model pricing; these names also match what the
        # AssemblyAI docs publish today.
        {"id": "universal-streaming",                "name": "Universal-Streaming",                "description": "English streaming, balanced"},
        {"id": "universal-streaming-multilingual",   "name": "Universal-Streaming Multilingual",   "description": "Multilingual streaming"},
        {"id": "universal-3.5-pro-realtime",         "name": "Universal-3.5 Pro Realtime",         "description": "Highest accuracy, realtime"},
    ],
    "openai": [
        {"id": "gpt-4o-transcribe",      "name": "GPT-4o Transcribe",      "description": "Latest OpenAI STT via Realtime API"},
        {"id": "gpt-4o-mini-transcribe", "name": "GPT-4o Mini Transcribe", "description": "Lower-latency variant"},
        {"id": "whisper-1",             "name": "Whisper-1",             "description": "Original Whisper model"},
    ],
    "rime": [
        {"id": "mist-v2", "name": "Mist v2", "description": "Rime STT default"},
    ],
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
    "anthropic": [
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "Latest balanced model"},
        {"id": "claude-3-5-haiku-20241022",  "name": "Claude 3.5 Haiku",  "description": "Fast, lower-cost"},
        {"id": "claude-3-opus-20240229",     "name": "Claude 3 Opus",     "description": "Highest capability, slower"},
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
    # Filter to chat-style models + Realtime. Skip legacy/embedding/audio-only.
    SKIP_PREFIX = ("davinci", "curie", "babbage", "ada", "text-embedding",
                   "whisper-", "tts-", "dall-e", "gpt-3.5-turbo-instruct")
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


def list_provider_models(service: str, provider_id: str, api_key: str | None = None) -> dict:
    """Returns the catalog of models/voices for a provider+service.

    If `api_key` is None, falls back to the user's stored credential for
    the provider. The hardcoded catalogs don't require a key. Live
    fetchers will raise if no key is available, so the caller must
    pass one explicitly.

    Returns:
        {"models": [{"id": "...", "name": "...", "description": "..."}, ...]}
        or {"models": [], "error": "..."} on failure.
    """
    creds = api_key
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
                return {"models": _fetch_openai_models(creds) if creds else _HARDCODED_LLM_MODELS["anthropic"]}
                # NOTE: fallback above is a safety valve — when called for openai LLM
                # without a key, we return Anthropic's catalog so the FE never crashes
                # on an empty response. The FE should never reach this branch in
                # practice; it's only hit if the user picks OpenAI LLM and has no
                # stored key.
            if provider_id == "anthropic":
                return {"models": _HARDCODED_LLM_MODELS["anthropic"]}
            if provider_id == "gemini":
                models = _fetch_gemini_models(creds) if creds else []
                return {"models": models or _HARDCODED_LLM_MODELS["anthropic"]}
                # Same safety-valve fallback when key missing.
            if provider_id == "minimax":
                return {"models": _HARDCODED_LLM_MODELS["minimax"]}
            return {"models": []}

        if service == "tts":
            # Live for OpenAI if key present, hardcoded catalog otherwise.
            if provider_id == "openai":
                if creds:
                    models = _fetch_openai_models(creds)
                    tts_models = [m for m in models
                                  if m["id"] in {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
                                  or m["id"].startswith("tts-")]
                    if tts_models:
                        return {"models": tts_models}
                return {"models": _HARDCODED_TTS_VOICES["openai"]}
            if provider_id == "rime":
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
                # ponytail: live fetch from GET /voices/v1/voices. The
                # hardcoded "Dennis" we shipped before was wrong - it
                # was an ElevenLabs voice id, not Inworld. We now query
                # the workspace's actual voice catalog so the FE
                # dropdown only shows voices that exist. Falls back to
                # an empty list when no key/credentials are available
                # (the FE then shows "No options" and the user can type
                # an id manually).
                if creds:
                    try:
                        req = urllib.request.Request(
                            "https://api.inworld.ai/voices/v1/voices?pageSize=200",
                            headers={"Authorization": f"Basic {creds}"},
                        )
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            payload = json.loads(resp.read().decode("utf-8"))
                        voices_raw = payload.get("voices") or []
                        live = [
                            {
                                "id": v["voiceId"],
                                "name": v.get("displayName") or v["voiceId"],
                                "description": " · ".join(filter(None, [
                                    v.get("description") or "",
                                    f"{v.get('gender', '')} {v.get('ageGroup', '')}".strip(),
                                    f"({v.get('source', '')})" if v.get("source") else "",
                                ])).strip(" ·") or "Inworld voice",
                            }
                            for v in voices_raw
                            if v.get("voiceId")
                        ]
                        log.info(
                            "[inworld-voice-catalog] fetch ok status=200 raw_voices=%d mapped=%d totalSize=%s",
                            len(voices_raw), len(live), payload.get("totalSize"),
                        )
                        if live:
                            return {"models": live}
                        log.warning(
                            "[inworld-voice-catalog] no voices mapped — keys=%s",
                            list((voices_raw[0] or {}).keys()) if voices_raw else "(empty list)",
                        )
                    except urllib.error.HTTPError as exc:
                        err_body = ""
                        try:
                            err_body = exc.read().decode("utf-8", errors="replace").strip()
                        except Exception:
                            pass
                        log.warning(
                            "[inworld-voice-catalog] HTTP %s body=%s",
                            exc.code, err_body[:300],
                        )
                    except Exception as exc:
                        log.warning(
                            "[inworld-voice-catalog] exception type=%s msg=%s",
                            type(exc).__name__, _sanitize_error(str(exc))[:300],
                        )
                return {"models": _HARDCODED_TTS_VOICES["inworld"]}
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
                return {"models": _HARDCODED_STT_MODELS["assemblyai"]}
            if provider_id == "openai":
                # ponytail: fetch /v1/models live y filtrar transcribe/realtime
                # en vez de hardcodear 3 modelos. Cuando OpenAI agregue uno
                # nuevo, aparece automaticamente sin tocar codigo.
                if creds:
                    try:
                        models = _fetch_openai_models(creds)
                        stt = [m for m in models
                               if "transcribe" in m["id"].lower()
                               or "whisper" in m["id"].lower()
                               or m["id"].startswith("gpt-4o-realtime")]
                        if stt:
                            return {"models": stt}
                    except Exception:
                        pass
                return {"models": _HARDCODED_STT_MODELS["openai"]}
            if provider_id == "rime":
                return {"models": _HARDCODED_STT_MODELS["rime"]}
            if provider_id == "inworld":
                return {"models": _HARDCODED_STT_MODELS["inworld"]}
            return {"models": []}

        return {"models": [], "error": f"Unknown service '{service}'"}
    except Exception as exc:
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
    """
    import urllib.error
    import urllib.request
    import json
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"
    base = (creds.get("base_url") or "https://api.anthropic.com").rstrip("/")
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

    ponytail: previous versions carried placeholder hostnames
    (`api.MiniMax.com`, `MiniMax.com/v1/api`, etc.) that never
    resolved from production. The real MiniMax API is at
    api.minimax.io (OpenAI-compatible, all lowercase). If your
    tenant needs a custom endpoint, set the URL explicitly via
    Settings → API or the env var.

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
    """Validate the SID + token pair by fetching the account."""
    try:
        from twilio.rest import Client
    except Exception as exc:
        return False, f"twilio SDK not installed: {exc}"
    sid = creds.get("account_sid")
    token = creds.get("auth_token")


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
    if not sid or not token:
        return False, "account_sid and auth_token are required"
    try:
        client = Client(sid, token)
        account = client.api.accounts(sid).fetch()
        return True, f"account status: {account.status}"
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
