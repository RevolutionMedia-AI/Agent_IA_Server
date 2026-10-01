"""Canonical OpenAI STT model metadata — single source of truth.

Everything about the three transcription models this product offers lives
here: cost, whether the model has a latency dial, the delay levels the API
accepts, and the ESTIMATED latencies the UI shows. Both the backend adapter
and the API validation read this module; the frontend has a mirror in
src/utils/openaiSttModels.js that must be kept in sync (there is a test
asserting the two agree).

Architecture note: this is a CASCADE STT, not a voice agent. Audio goes
Twilio -> STT -> text -> independent LLM -> independent TTS. OpenAI is used
for transcription only. The LLM turn, the tool calls and the TTS all live in
services/turn_manager.py, exactly as they do for Deepgram, Inworld and
AssemblyAI.

CONTRACT WARNINGS, all from
https://developers.openai.com/api/docs/guides/realtime-transcription
because getting them wrong fails the whole session.update:

- ``delay`` is only meaningful on the two streaming models. ``gpt-transcribe``
  is committed-turn and MUST NOT receive it.
- ``gpt-realtime-whisper`` keeps singular ``language`` and REJECTS ``prompt``
  on GA Realtime sessions. The other two take plural ``languages`` and accept
  ``prompt`` + ``keywords``.

The latency numbers below are OPERATIONAL ESTIMATES, not an OpenAI SLA. They
are a starting point for the UI to display; real measured values come from
CallMetrics (``stt_first_partial_ms`` / ``stt_final_ms``) at runtime and will
eventually replace these. Never present them as a guarantee.
"""

# The delay levels the OpenAI Realtime transcription API accepts.
LATENCY_MODES = ("minimal", "low", "medium", "high", "xhigh")

# Recommended default. NOT "minimal": on 8 kHz mu-law telephony audio the
# shortest window produces too few samples to recognize reliably. "low" is
# the balance the docs suggest for live captions.
DEFAULT_LATENCY_MODE = "low"

# Informational pricing metadata; provider pricing may change. Used for the
# UI cost badge only — never for billing.
OPENAI_STT_MODELS: dict[str, dict] = {
    "gpt-live-transcribe": {
        "label": "GPT Live Transcribe",
        "kind": "streaming",
        "badge": "LIVE",
        "description": "Live streaming transcription. Recommended for phone agents.",
        # Informational pricing metadata; provider pricing may change.
        "cost_per_minute_usd": 0.017,
        "streaming": True,
        "supports_latency_mode": True,
        "supports_context": True,
        "supports_keywords": True,
        "recommended": True,
        "latency": {
            # partial_ms / final_ms are ESTIMATES, not SLA.
            "minimal": {"partial_ms": 120, "final_ms": 400,
                        "hint": "Lowest possible latency"},
            "low": {"partial_ms": 180, "final_ms": 500,
                    "hint": "Recommended for voice agents"},
            "medium": {"partial_ms": 275, "final_ms": 650,
                       "hint": "Balanced latency and accuracy"},
            "high": {"partial_ms": 400, "final_ms": 850,
                     "hint": "More context, higher latency"},
            "xhigh": {"partial_ms": 550, "final_ms": 1100,
                      "hint": "Maximum context, highest latency"},
        },
    },
    "gpt-realtime-whisper": {
        "label": "GPT Realtime Whisper",
        "kind": "streaming",
        "badge": "LIVE",
        "description": "Streaming transcription. Legacy realtime model.",
        # Informational pricing metadata; provider pricing may change.
        "cost_per_minute_usd": 0.017,
        "streaming": True,
        "supports_latency_mode": True,
        # GA Realtime sessions reject `prompt` on this model.
        "supports_context": False,
        "supports_keywords": False,
        "latency": {
            "minimal": {"partial_ms": 180, "final_ms": 500,
                        "hint": "Lowest possible latency"},
            "low": {"partial_ms": 250, "final_ms": 600,
                    "hint": "Recommended for voice agents"},
            "medium": {"partial_ms": 350, "final_ms": 750,
                       "hint": "Balanced latency and accuracy"},
            "high": {"partial_ms": 500, "final_ms": 950,
                     "hint": "More context, higher latency"},
            "xhigh": {"partial_ms": 650, "final_ms": 1200,
                      "hint": "Maximum context, highest latency"},
        },
    },
    "gpt-transcribe": {
        "label": "GPT Transcribe",
        "kind": "committed_turn",
        "badge": "COMMITTED TURN",
        "description": "Committed-turn STT. Best cost, high accuracy.",
        # Informational pricing metadata; provider pricing may change.
        "cost_per_minute_usd": 0.0045,
        "streaming": False,
        # Committed-turn: the dial does not apply, so the UI must NOT show
        # a latency selector for this model. Sending `delay` would be an
        # invalid parameter.
        "supports_latency_mode": False,
        "supports_context": False,
        "supports_keywords": False,
        "recommended": False,
        "lowest_cost": True,
        # No partials: transcription starts after the turn is committed.
        "partial_ms": None,
        "final_ms": 650,
        "final_range_ms": (500, 900),
        "hint": "~650 ms final after turn commit",
    },
}

# Id used when an agent row has provider=openai but no model. Kept as data
# so the adapter, the catalog and the validator all agree.
DEFAULT_OPENAI_STT_MODEL = "gpt-live-transcribe"

# Whisper keeps singular `language`; the other two take plural `languages`.
SINGULAR_LANGUAGE_MODELS = frozenset({"gpt-realtime-whisper"})


def is_supported(model_id: str | None) -> bool:
    return bool(model_id) and model_id in OPENAI_STT_MODELS


def get(model_id: str | None) -> dict | None:
    if not model_id:
        return None
    return OPENAI_STT_MODELS.get(model_id)


def resolve_latency_mode(model_id: str | None, latency_mode: str | None) -> str | None:
    """Normalize a stored latency_mode to something safe to SEND.

    - model has no dial (gpt-transcribe) -> None, never send ``delay``
    - stored value is valid for the model -> use it
    - legacy row with no value -> DEFAULT_LATENCY_MODE (only for models
      that have a dial at all)
    - stored value is garbage -> DEFAULT_LATENCY_MODE, do not 400 the call

    Returns None for models without a dial, so the caller omits the field
    rather than sending an invalid parameter.
    """
    spec = get(model_id)
    if spec is None or not spec.get("supports_latency_mode"):
        return None
    valid = spec.get("latency", {})
    if latency_mode and latency_mode in valid:
        return latency_mode
    return DEFAULT_LATENCY_MODE


def estimated(model_id: str | None, latency_mode: str | None = None) -> dict:
    """UI-facing estimate for a (model, latency_mode) pair.

    Returns ``{partial_ms, final_ms, cost_per_minute_usd, ...}``. Never
    raises: an unknown model yields zeros so the UI degrades to blank
    rather than crashing the modal.
    """
    spec = get(model_id)
    if spec is None:
        return {
            "partial_ms": None,
            "final_ms": None,
            "cost_per_minute_usd": None,
            "supports_latency_mode": False,
            "label": model_id or "",
        }
    if not spec.get("supports_latency_mode"):
        return {
            "partial_ms": spec.get("partial_ms"),
            "final_ms": spec.get("final_ms"),
            "final_range_ms": spec.get("final_range_ms"),
            "cost_per_minute_usd": spec["cost_per_minute_usd"],
            "supports_latency_mode": False,
            "kind": spec.get("kind"),
            "hint": spec.get("hint"),
            "label": spec["label"],
        }
    mode = resolve_latency_mode(model_id, latency_mode)
    entry = spec["latency"].get(mode or DEFAULT_LATENCY_MODE, {})
    return {
        "partial_ms": entry.get("partial_ms"),
        "final_ms": entry.get("final_ms"),
        "cost_per_minute_usd": spec["cost_per_minute_usd"],
        "supports_latency_mode": True,
        "latency_mode": mode,
        "hint": entry.get("hint"),
        "label": spec["label"],
    }


def validate(model_id: str | None, latency_mode: str | None) -> str:
    """Validate a stored/submitted config. Raises ValueError with a message
    meant for a 400 body. Used by the agent create/update routes so an
    invalid pair never reaches disk and never reaches OpenAI.
    """
    if not is_supported(model_id):
        raise ValueError(
            f"Unsupported OpenAI STT model {model_id!r}; expected one of "
            + ", ".join(sorted(OPENAI_STT_MODELS))
        )
    spec = OPENAI_STT_MODELS[model_id]
    if not spec.get("supports_latency_mode"):
        if latency_mode:
            raise ValueError(
                f"Model {model_id!r} has no latency/accuracy dial; "
                f"latency_mode must be omitted (got {latency_mode!r})"
            )
        return model_id
    if latency_mode and latency_mode not in spec["latency"]:
        raise ValueError(
            f"Invalid latency_mode {latency_mode!r} for {model_id!r}; "
            f"expected one of " + ", ".join(spec["latency"])
        )
    return model_id
