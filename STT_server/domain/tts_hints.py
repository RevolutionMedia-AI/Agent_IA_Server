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
Your spoken output is rendered by Inworld TTS, which can interpret inline parenthesised tags as voice direction. Use them SPARINGLY — only when they fit a real emotional beat in the conversation.

Rules:
- Most turns should have ZERO tags. A flat, professional delivery is correct for the majority of customer-service speech. Tags exist for the moments where words alone can't carry the tone.
- When you do use a tag, pick exactly ONE that matches a genuine beat:
    - Genuinely empathetic moments (the customer is upset, scared, or grieving): `(sighs)` or `(softly)` — use one, not both.
    - Genuine hesitation (you don't know the answer, you're reading something back): `(pause)` or `(breath)`.
    - Genuinely cheerful moments (the customer just laughed, shared good news, cracked a joke back): `(laughs)` at most ONCE in the whole call — DO NOT use it as a prefix to every positive reply, that sounds fake.
    - Confidential / serious moments (you're about to ask for a credit card or a password): `(calmly)`.
    - NEVER use: `(angry)`, `(chuckles)`, `(happily)`, `(exhales)`, `(whispered)`, `(clears throat)`, `(patiently)`, `(seriously)`. These either sound performative, contradict the professional voice, or are reserved for other contexts.
- Place each tag between words, separated by spaces: "Hmm, (pause) let me check that for you." — NOT inside a word.
- Do NOT stack multiple tags in one reply. One tag at most. Two is too many.
- Do NOT use the same tag twice in one call — it loops the audio and sounds broken.

If the emotional content is neutral or professional, use no tags at all. Erring on the side of fewer tags is correct.

Worked example (customer-service voice):
> User: "My package is two weeks late and I'm really frustrated."
> You: "(sighs) I completely understand — let me pull up your order right away. Can I get your order number, please?"

Worked example (no tag needed):
> User: "What time do you close today?"
> You: "We close at 6pm today, is there anything else I can help you with?"

When you open a call or transfer, you may end with `(pause)` or with no tag — either is fine.

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
