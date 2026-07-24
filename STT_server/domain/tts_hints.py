"""TTS-specific instructions to prepend to the LLM system prompt.

Why this exists
---------------
Some TTS providers (notably Inworld) support inline non-verbal tags
in the synthesised text that steer the voice — laugh, sigh, whisper,
angry, etc. Without an explicit instruction, the LLM emits clean
prose and the TTS never gets a chance to vary the voice, so the
caller hears a flat, robotic delivery.

This module holds the per-provider hint blocks. The session bootstrap
PREPENDS the hint for the active TTS provider to the agent's
`custom_prompt` (not appends — earlier experiments showed the LLM
treats end-of-prompt additions as low-priority context and rarely
emits the tags). The hint stays short so the agent's own prompt
remains the dominant behavioural spec; we only add the
voice-direction vocabulary that the TTS consumes.

The hint is only added once per session (idempotent guard via
_HINT_MARKER in STT_Server.py).
"""
from __future__ import annotations


# Emotional / non-verbal steering tags supported by Inworld TTS.
# Inworld interprets inline cues in parentheses and modulates the
# voice accordingly. The model is smart enough to distinguish
# "(laughs)" as a steering tag from a real parenthesis "(like this)"
# because the steering tags are short and don't contain content
# between the parens.
#
# Design notes:
# - The previous version of this hint (appended at the end of the
#   agent's prompt, generic "use when natural" language) was largely
#   ignored by the LLM in production. The customer heard a flat
#   delivery because gpt-4o-mini treats appended instructions as
#   low-priority context once the dominant system prompt is set.
# - This version is PREPENDED (so it's at the top of the system
#   message), uses imperative directives, and includes a worked
#   example the LLM can pattern-match. The example is in a
#   customer-service voice so the LLM carries the pattern into
#   the agent's domain.
# - The example does NOT leak customer-specific data. It uses
#   generic phrases ("let me check that for you", "give me one
#   moment") that any customer-service agent will produce.
INWORLD_STEERING_INSTRUCTIONS = """[TTS Steering — Inworld]
Your spoken output is rendered by Inworld TTS, which interprets inline parenthesised tags as voice direction. You MUST use them so the caller hears a natural, human voice — not a flat, robotic read.

Rules:
- Use one or more tags in every reply. The minimum is ONE tag per turn (a `(sighs)` or `(pause)` works). The caller hangs up if you sound like a TTS demo.
- Pick the tag that matches the emotional content:
    - Empathetic moments: `(sighs)`, `(softly)`, `(exhales)`
    - Hesitation / thinking: `(pause)`, `(breath)`, `(clears throat)`
    - Cheerful moments: `(laughs)`, `(chuckles)`, `(happily)`
    - Confidential / serious: `(whispered)`, `(calmly)`, `(seriously)`
    - Annoyed but professional: `(sighs)`, `(patiently)` (NEVER `(angry)` — that's for the operator, not the customer)
- Place each tag between words, separated by spaces: "Hmm, (pause) let me check that for you." — NOT inside a word ("te(laughs)stimonio" doesn't work).
- Don't over-tag. One or two tags per turn is enough. Three or more starts to feel performative.

Worked example (customer-service voice):
> User: "My package is two weeks late and I'm really frustrated."
> You: "(sighs) I completely understand — let me pull up your order right away. (pause) Can I get your order number, please?"

When you open a call or transfer, end on a calm or thoughtful tag, not on dead air. When the user shares something emotional, mirror it with a sympathetic tag before continuing.

Never tell the user about these tags. Never list them in your visible output. They are silent voice-direction, only.
"""


# Per-provider hint table. Add entries as new providers are onboarded
# with similar inline-tagging features. Keep each entry short — the
# agent's own prompt remains the dominant context; the hint only
# teaches the model that steering is available and how to use it.
TTS_PROVIDER_HINTS: dict[str, str] = {
    "inworld": INWORLD_STEERING_INSTRUCTIONS,
}


# Marker that the bootstrap uses to detect a double-prepend on
# session re-init (e.g. WS reconnect). STT_Server.py checks for this
# before adding the hint.
_HINT_MARKER = "[TTS Steering — Inworld]"


def get_tts_hint(provider_id: str | None) -> str | None:
    """Return the steering instruction block for a TTS provider, or
    None if the provider has no hint registered.

    Case-insensitive match against `TTS_PROVIDER_HINTS` keys.
    """
    if not provider_id:
        return None
    return TTS_PROVIDER_HINTS.get(provider_id.strip().lower())


def has_tts_hint(prompt: str | None) -> bool:
    """True if the prompt already contains a TTS hint block. Used by
    the session bootstrap to avoid prepending the same hint twice on
    WS reconnect.
    """
    if not prompt:
        return False
    return _HINT_MARKER in prompt
