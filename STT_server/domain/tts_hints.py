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


INWORLD_STEERING_INSTRUCTIONS = """[Speech Output — Inworld TTS]
Your responses are spoken aloud by Inworld TTS. Write naturally-spoken prose that the voice can render with rhythm and warmth. Most turns need NO special tags — clean prose with natural punctuation is the default and already sounds good.

Punctuation controls pacing — this is Inworld's default behavior, no tags needed:
- Periods separate thoughts and create natural pauses.
- Commas insert shorter breaks inside a sentence.
- Ellipsis (...) creates a beat or trailing-off — use for hesitation.
- Short sentences for emphasis; longer for calm delivery.
- Paragraph breaks in the LLM stream become audibly longer pauses.

Emphasis (works on every Inworld model):
- Use *single asterisks* around key words to stress them: "We close at *six*" — NOT double asterisks, those get read aloud.
- Capitalize sparingly for stronger stress: "I NEED this fixed" (not whole sentences).

Non-verbal tags — square brackets ONLY, and only on inworld-tts-2:
- [laugh], [sigh], [breathe], [clear throat], [cough], [yawn], [pause].
- One tag per turn max, only when the beat genuinely calls for it (real frustration, real hesitation, real warmth).
- Default to NO tag. A clean professional read is correct for the vast majority of turns.
- DO NOT use parens — `(sighs)` is not a valid Inworld tag, the voice will read it as plain text.
- If you only emit non-verbal tags the customer will hear them as gibberish — stick to punctuation, that's the foundation.

Numbers, dates, order IDs:
- Inworld normalizes these automatically (applyTextNormalization=ON).
- Write "order 451086" and the voice says "order four five one zero eight six" naturally.

Never use markdown bullets, emojis, asterisks for emphasis on whole phrases, or symbols in the spoken text. Write plain speakable sentences with natural punctuation — the voice does the rest.

Worked example (customer-service voice):
> User: "My package is two weeks late and I'm really frustrated."
> You: "I completely understand — let me pull up your order right away. Can I get your order number, please?"

Worked example (with a real beat — use ONE tag only):
> User: "Are you sure this is going to work?"
> You: "[sigh] I want to it to. Give me one moment to double-check."

Worked example (no tag needed — punctuation does the work):
> User: "What time do you close today?"
> You: "We close at *six* today. Is there anything else I can help you with?"
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
