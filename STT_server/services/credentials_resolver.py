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
        description="Anthropic Claude models. Reserved for future use; the voice pipeline currently uses OpenAI.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                pattern=r"^sk-ant-(api\d{2}-)?[A-Za-z0-9_\-]{20,}$",
                min_length=20,
                placeholder="sk-ant-...",
            ),
        ),
        env_fallbacks=(("ANTHROPIC_API_KEY", "api_key"),),
        test_fn=None,
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
        ),
        env_fallbacks=(("GEMINI_API_KEY", "api_key"),),
        test_fn="STT_server.services.credentials_resolver._test_gemini",
    ),
    ProviderSpec(
        id="minimax",
        name="MiniMax (MiniMax)",
        category="llm",
        description="MiniMax (formerly MiniMax AI) chat completions. Picked as the agent's LLM in Settings → New Agent.",
        fields=(
            FieldSpec(
                name="api_key", label="API Key", type="password",
                required=True,
                min_length=20,
                placeholder="ey...",
                help="MiniMax API key. Format varies by tier.",
            ),
        ),
        env_fallbacks=(("MINIMAX_API_KEY", "api_key"),),
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
        test_fn=None,
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
        {"id": "inworld-default", "name": "Inworld Default", "description": "Inworld ASR default model"},
    ],
}

_HARDCODED_LLM_MODELS = {
    "minimax": [
        {"id": "minimax",        "name": "MiniMax",        "description": "Default MiniMax chat model"},
        {"id": "minimax-v1",     "name": "MiniMax v1",     "description": "Stable v1 release"},
        {"id": "abab6.5s-chat",  "name": "abab6.5s-chat",  "description": "MiniMax abab6.5s series"},
        {"id": "abab5.5-chat",   "name": "abab5.5-chat",   "description": "MiniMax abab5.5 series"},
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
                return {"models": _HARDCODED_TTS_VOICES.get("elevenlabs", [])}
                # Inworld has no canonical voice catalog publicly; reuse curated list.
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


def _test_minimax(creds: dict[str, str]) -> tuple[bool, str]:
    """Hit MiniMax /v1/models with bearer auth.

    ponytail: MiniMax tiene varios endpoints publicos segun el tenant
    (api.MiniMax.com, api.MiniMax.chat, MiniMax.com/v1/api/...). Pruebo
    en orden hasta que alguno responda 200. Si ninguno anda, devuelvo
    el body del ultimo response para que el usuario sepa QUE esta
    pasando (el codigo 401 anterior era opaco).

    Override con la env var MINIMAX_BASE_URL si tu tenant usa uno
    personalizado.
    """
    import urllib.error
    import urllib.request
    key = creds.get("api_key")
    if not key:
        return False, "api_key is required"

    candidate_bases = []
    if os.environ.get("MINIMAX_BASE_URL"):
        candidate_bases.append(os.environ["MINIMAX_BASE_URL"].rstrip("/"))
    # Orden de preferencia: el endpoint estandar OpenAI-compatible.
    candidate_bases.extend([
        "https://api.MiniMax.com/v1",
        "https://MiniMax.com/v1/api",
        "https://api.MiniMax.chat/v1",
        "https://MiniMax.com/v1",
    ])

    last_status = None
    last_body = ""
    last_url = ""
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
            try:
                last_body = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                last_body = "(no body)"
            continue
        except Exception as exc:
            last_url = base
            last_body = _sanitize_error(str(exc))[:200]
            continue
    return False, f"HTTP {last_status} via {last_url}: {last_body}"


def _test_twilio(creds: dict[str, str]) -> tuple[bool, str]:
    """Validate the SID + token pair by fetching the account."""
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


def test_provider(
    user_id: str | None,
    provider_id: str,
    api_key: str | None = None,
) -> dict:
    """Run the catalog-defined test_fn against the resolved credentials.

    Priority order for the key:
      1. ``api_key`` argument — what the FE just typed in the New Agent flow.
      2. Per-user stored credential (``tools_integrations.json``).
      3. System env-var fallback.

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
