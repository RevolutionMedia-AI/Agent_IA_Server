"""OpenAI TTS catalog + request validation.

Single source of truth for which OpenAI models this product can run as
the TTS stage of a voice call, and which voices each model accepts.

Why a static catalog instead of hitting /v1/models: OpenAI's model
endpoint does not list voices, and the voice set is a property of the
model, not of the account. `POST /providers/models/categorized` already
buckets OpenAI ids into llm/stt/tts (see _classify_openai_model), so the
model dropdown is live -- but the VOICE dropdown needs this table.

pony tail: pure functions only, no I/O, no SDK import. The dispatcher,
the preview endpoint and the frontend mirror all read from here, and the
tests exercise it without an API key.
"""
from __future__ import annotations

# ponytail: 2026-10-02 -- audio format is fixed by the Speech API and is
# what the pipeline converts FROM. Documented here because the converter
# call site hardcodes 24000 and a mismatch is a silent speed/pitch bug.
OPENAI_PCM_SAMPLE_RATE = 24000
OPENAI_PCM_SAMPLE_WIDTH = 2  # bytes, signed 16-bit little-endian
OPENAI_PCM_CHANNELS = 1

# The Speech API input limit. Truncating instead of erroring keeps a long
# agent reply audible instead of dropping the whole turn on a 400.
MAX_INPUT_CHARS = 2000

# instructions is free-text steering. Bounded so a fat-fingered paste
# cannot blow up the request or the prompt cache.
MAX_INSTRUCTIONS_CHARS = 600


# ── Transport budgets ──────────────────────────────────────────────
# ponytail: tuned for a phone call, not a batch job. A conversational turn
# that has not produced a first byte in 10 s is dead to the caller, who
# has already started hanging up. Once bytes flow the read budget is
# shorter still, because a stalled socket mid-stream means the caller is
# hearing a gap, not a pause.
OPENAI_TTS_FIRST_BYTE_TIMEOUT_SEC = float(
    __import__("os").getenv("OPENAI_TTS_FIRST_BYTE_TIMEOUT_SEC", "10"))
OPENAI_TTS_READ_TIMEOUT_SEC = float(
    __import__("os").getenv("OPENAI_TTS_READ_TIMEOUT_SEC", "5"))
# ponytail: the adapter uses http.client, whose per-connection timeout
# covers connect + send + response headers, and then re-arms the socket
# with the two budgets above. Kept as a named constant because a
# connection that cannot even be established should fail faster than one
# waiting on the provider's first byte.
OPENAI_TTS_CONNECT_TIMEOUT_SEC = float(
    __import__("os").getenv("OPENAI_TTS_CONNECT_TIMEOUT_SEC", "4"))


# ── Models ──────────────────────────────────────────────────────────
# response_format is pinned to "pcm" for EVERY OpenAI TTS model, including
# the legacy tts-1 / tts-1-hd. Reason: the audio pipeline converts
# PCM16 -> mu-law 8 kHz and nothing else. Handing it mp3 bytes would emit
# noise, and existing agents already have tts-1 rows working today only
# because the dispatcher forced response_format=pcm. Introducing an mp3
# branch would be a new code path the pipeline cannot consume.
#
# `supports_instructions` is the real per-model difference: the Speech API
# only honours `instructions` on gpt-4o-mini-tts.
OPENAI_TTS_MODELS: dict[str, dict] = {
    "gpt-4o-mini-tts": {
        "label": "GPT-4o Mini TTS",
        "recommended": True,
        "supports_instructions": True,
        "supports_speed": True,
        "response_format": "pcm",
        # Informational pricing metadata; provider pricing may change.
        "cost_per_1m_chars_usd": 0.60,
        "description": (
            "Speech synthesis with natural-language voice instructions "
            "(accent, tone, pace). Recommended default."
        ),
    },
    "tts-1": {
        "label": "TTS-1",
        "recommended": False,
        "supports_instructions": False,
        "supports_speed": True,
        "response_format": "pcm",
        "cost_per_1m_chars_usd": 15.0,
        "description": "Legacy. Faster, lower quality, no voice instructions.",
    },
    "tts-1-hd": {
        "label": "TTS-1 HD",
        "recommended": False,
        "supports_instructions": False,
        "supports_speed": True,
        "response_format": "pcm",
        "cost_per_1m_chars_usd": 30.0,
        "description": "Legacy HD. Highest cost, no voice instructions.",
    },
}

DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_TTS_VOICE = "marin"


# ── Voices ──────────────────────────────────────────────────────────
# Per-model voice sets. `gpt-4o-mini-tts` carries the full current
# catalog; the legacy tts-* models only accept the original six.
#
# `preferred` marks the two voices OpenAI recommends for quality
# (marin, cedar). They are first in the FE dropdown but NOT forced --
# every voice stays selectable.
_GPT4O_MINI_TTS_VOICES: dict[str, dict] = {
    "marin":    {"label": "Marin",    "preferred": True,  "style": "Warm, confident narration"},
    "cedar":    {"label": "Cedar",    "preferred": True,  "style": "Calm, grounded"},
    "alloy":    {"label": "Alloy",    "preferred": False, "style": "Neutral and balanced"},
    "ash":      {"label": "Ash",      "preferred": False, "style": "Clear, conversational"},
    "ballad":   {"label": "Ballad",   "preferred": False, "style": "Expressive, melodic"},
    "coral":    {"label": "Coral",    "preferred": False, "style": "Bright, friendly"},
    "echo":     {"label": "Echo",     "preferred": False, "style": "Warm and upbeat"},
    "fable":    {"label": "Fable",    "preferred": False, "style": "Dramatic, storytelling"},
    "nova":     {"label": "Nova",     "preferred": False, "style": "Energetic"},
    "onyx":     {"label": "Onyx",     "preferred": False, "style": "Deep, authoritative"},
    "sage":     {"label": "Sage",     "preferred": False, "style": "Calm, measured"},
    "shimmer":  {"label": "Shimmer",  "preferred": False, "style": "Light, airy"},
    "verse":    {"label": "Verse",    "preferred": False, "style": "Even, narration"},
}

_LEGACY_TTS_VOICES: dict[str, dict] = {
    "alloy":   {"label": "Alloy",   "preferred": False, "style": "Neutral and balanced"},
    "echo":    {"label": "Echo",    "preferred": False, "style": "Warm and upbeat"},
    "fable":   {"label": "Fable",   "preferred": False, "style": "Dramatic, storytelling"},
    "onyx":    {"label": "Onyx",    "preferred": False, "style": "Deep, authoritative"},
    "nova":    {"label": "Nova",    "preferred": False, "style": "Energetic"},
    "shimmer": {"label": "Shimmer", "preferred": False, "style": "Light, airy"},
}

VOICES_BY_MODEL: dict[str, dict[str, dict]] = {
    "gpt-4o-mini-tts": _GPT4O_MINI_TTS_VOICES,
    "tts-1": _LEGACY_TTS_VOICES,
    "tts-1-hd": _LEGACY_TTS_VOICES,
}


# ── Resolution / validation ────────────────────────────────────────

def resolve_model(model_id: str | None) -> str:
    """Return a usable OpenAI TTS model id.

    Unknown or missing ids fall back to the recommended default rather
    than raising: a pre-existing agent row with a model we no longer
    offer should still place calls instead of muting every call.
    """
    mid = (model_id or "").strip()
    return mid if mid in OPENAI_TTS_MODELS else DEFAULT_OPENAI_TTS_MODEL


def voices_for_model(model_id: str | None) -> dict[str, dict]:
    """Voice table for *model_id*, ordered preferred-first."""
    table = VOICES_BY_MODEL.get(resolve_model(model_id), {})
    return dict(sorted(
        table.items(),
        key=lambda kv: (not kv[1].get("preferred"), kv[0]),
    ))


def voice_ids_for_model(model_id: str | None) -> list[str]:
    return list(voices_for_model(model_id))


def is_valid_voice(model_id: str | None, voice_id: str | None) -> bool:
    return (voice_id or "").strip() in VOICES_BY_MODEL.get(resolve_model(model_id), {})


def resolve_voice(model_id: str | None, voice_id: str | None) -> str:
    """Return a voice the model actually accepts.

    A stored voice that is invalid for the selected model falls back to
    the model's default rather than letting OpenAI reject the request
    with an opaque 400 mid-call. DEFAULT_OPENAI_TTS_VOICE wins when the
    model accepts it, so marin/cedar being co-preferred does not make
    cedar the silent default via alphabetical tie-break.
    """
    table = VOICES_BY_MODEL.get(resolve_model(model_id), {})
    vid = (voice_id or "").strip()
    if vid in table:
        return vid
    if DEFAULT_OPENAI_TTS_VOICE in table:
        return DEFAULT_OPENAI_TTS_VOICE
    for name, meta in voices_for_model(model_id).items():
        if meta.get("preferred"):
            return name
    return next(iter(table), DEFAULT_OPENAI_TTS_VOICE)


def sanitize_instructions(instructions: str | None, model_id: str | None) -> str:
    """Trim voice instructions and drop them on models that reject them.

    Empty is a valid outcome: TTS must keep working when the operator
    never configured instructions.
    """
    if not OPENAI_TTS_MODELS.get(resolve_model(model_id), {}).get("supports_instructions"):
        return ""
    text = " ".join((instructions or "").split())
    return text[:MAX_INSTRUCTIONS_CHARS]


def sanitize_input(text: str) -> str:
    """Clamp the Speech API input limit. Never returns empty for non-empty input."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= MAX_INPUT_CHARS:
        return cleaned
    return cleaned[:MAX_INPUT_CHARS].rstrip()


def response_format_for(model_id: str | None) -> str:
    return OPENAI_TTS_MODELS.get(resolve_model(model_id), {}).get("response_format", "pcm")


def src_sample_rate(model_id: str | None) -> int:
    """Sample rate the Speech API emits for this model's chosen format."""
    return OPENAI_PCM_SAMPLE_RATE if response_format_for(model_id) == "pcm" else 0


def build_speech_request(
    text: str,
    *,
    model_id: str | None = None,
    voice_id: str | None = None,
    instructions: str | None = None,
    speed: float | None = None,
) -> tuple[dict, int]:
    """Build the /v1/audio/speech JSON body plus the source sample rate.

    Returns ``(body, src_rate)`` so the caller can hand src_rate straight
    to the PCM->mu-law converter instead of hardcoding 24000 at a second
    site.

    Validates on the way in (spec: never trust the frontend): model and
    voice are checked against this module's catalog, instructions are
    trimmed and dropped for models that reject them, speed is clamped to
    the documented 0.25..4.0 range, and text is clamped to the 2000-char
    input limit.
    """
    model = resolve_model(model_id)
    body: dict = {
        "model": model,
        "input": sanitize_input(text),
        "voice": resolve_voice(model, voice_id),
        "response_format": response_format_for(model),
    }
    hint = sanitize_instructions(instructions, model)
    if hint:
        body["instructions"] = hint
    if OPENAI_TTS_MODELS[model].get("supports_speed") and speed is not None:
        body["speed"] = round(max(0.25, min(4.0, float(speed))), 2)
    return body, src_sample_rate(model)


# ── Transport errors ───────────────────────────────────────────────

class OpenAITtsError(RuntimeError):
    """Typed transport/HTTP failure from the OpenAI Speech API.

    `status` is the HTTP code when there was one, else None for
    connect/read/timeout failures. `retryable` tells the caller whether a
    retry could plausibly succeed: 429 and 5xx and transport errors can,
    401/403/404 cannot (they need an operator to fix a key, a model or a
    voice). `stage` distinguishes "first byte" from "mid stream" so the
    latency metrics say which one blew up.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 stage: str = "connect", retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.stage = stage
        self.retryable = retryable


def classify_http_status(status: int, body_snippet: str = "") -> None:
    """Raise OpenAITtsError for a non-2xx Speech API response.

    The body snippet is already truncated by the caller and never
    contains the Authorization header, so it is safe to log.
    """
    if 200 <= status < 300:
        return
    retryable = status == 429 or status >= 500
    if status in (401, 403):
        kind = "invalid or missing API key"
    elif status == 404:
        kind = "unknown model or voice"
    elif status == 429:
        kind = "rate limited"
    elif status >= 500:
        kind = "provider error"
    else:
        kind = "request rejected"
    raise OpenAITtsError(
        f"OpenAI TTS HTTP {status} ({kind}){': ' + body_snippet if body_snippet else ''}",
        status=status,
        stage="first_byte",
        retryable=retryable,
    )


def catalog_payload() -> dict:
    return {
        "models": [
            {
                "id": mid,
                "label": meta["label"],
                "recommended": meta.get("recommended", False),
                "supportsInstructions": meta.get("supports_instructions", False),
                "supportsSpeed": meta.get("supports_speed", False),
                "responseFormat": meta["response_format"],
                "description": meta.get("description", ""),
                "voices": [
                    {
                        "id": vid,
                        "label": vmeta["label"],
                        "preferred": vmeta.get("preferred", False),
                        "hint": vmeta.get("style", ""),
                    }
                    for vid, vmeta in voices_for_model(mid).items()
                ],
            }
            for mid, meta in OPENAI_TTS_MODELS.items()
        ],
        "defaultModel": DEFAULT_OPENAI_TTS_MODEL,
        "defaultVoice": DEFAULT_OPENAI_TTS_VOICE,
        "sampleRate": OPENAI_PCM_SAMPLE_RATE,
    }


# ── Self-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    body, rate = build_speech_request(
        "Hola, ¿en qué puedo ayudarte?",
        model_id="gpt-4o-mini-tts",
        voice_id="marin",
        instructions="Habla en español latinoamericano natural.",
    )
    assert body["model"] == "gpt-4o-mini-tts", body
    assert body["voice"] == "marin", body
    assert body["instructions"].startswith("Habla en español"), body
    assert body["response_format"] == "pcm", body
    assert rate == 24000, rate

    # Legacy model must drop instructions (format stays pcm).
    body, rate = build_speech_request(
        "hola", model_id="tts-1", voice_id="alloy", instructions="ignorado",
    )
    assert "instructions" not in body, body
    assert body["response_format"] == "pcm", body

    # A voice the model does not accept falls back, never 400s.
    body, _ = build_speech_request("hola", model_id="tts-1", voice_id="marin")
    assert body["voice"] in VOICES_BY_MODEL["tts-1"], body
    assert body["voice"] != "marin", body

    # Unknown model -> recommended default, still a valid request.
    body, rate = build_speech_request("hola", model_id="gpt-9-imaginary")
    assert body["model"] == DEFAULT_OPENAI_TTS_MODEL, body
    assert rate == 24000

    assert voice_ids_for_model("gpt-4o-mini-tts")[:2] == ["cedar", "marin"] or \
           set(voice_ids_for_model("gpt-4o-mini-tts")[:2]) == {"marin", "cedar"}
    assert is_valid_voice("tts-1", "shimmer")
    assert not is_valid_voice("tts-1", "verse")

    # Input limit clamp.
    long_text = "a" * 5000
    assert len(build_speech_request(long_text)[0]["input"]) == MAX_INPUT_CHARS

    # Empty instructions stay absent (not sent as "").
    assert "instructions" not in build_speech_request("hola", instructions="")[0]

    print("openai_tts: OK", body["model"], body["voice"], rate)