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


# ── Connection test helpers (used by /test endpoint) ────────────────────────

def _import(path: str):
    module_name, _, attr = path.rpartition(".")
    mod = __import__(module_name, fromlist=[attr])
    return getattr(mod, attr)


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
        return False, str(exc)[:300]


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
        return False, str(exc)[:300]


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
        return False, str(exc)[:300]


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
        return False, str(exc)[:300]


def test_provider(user_id: str | None, provider_id: str) -> dict:
    """Run the catalog-defined test_fn against the resolved credentials.

    Returns a dict with ``valid`` (bool), ``message`` (str) and
    ``source`` ('user' | 'env' | 'none') so the FE can show where the
    tested key came from.
    """
    spec = get_provider_spec(provider_id)
    if spec is None:
        return {"valid": False, "message": f"Unknown service '{provider_id}'", "source": "none"}

    creds = resolve_provider(user_id, provider_id)
    if not creds:
        return {"valid": False, "message": "No credentials configured for this user or system default.", "source": "none"}

    # Did the active value come from per-user storage or env? Inspect raw
    # storage vs env so the FE can show "using your key" vs "using system default".
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
        ok, message = False, str(exc)[:300]
    return {"valid": bool(ok), "message": message, "source": source}
