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
Your responses are spoken aloud by Inworld TTS. The voice is a real human talking to a real person on the phone — sound like one. Every turn should feel like a complete, natural conversational beat, not a flat read.

Punctuation drives the rhythm. Inworld respects it for pacing by default:
- Periods. Sentences. Create. Real. Pauses.
- Commas create shorter breaks inside a sentence.
- Ellipsis "..." creates a beat or trailing-off — perfect for genuine hesitation: "Bueno... déjame ver..."
- Short sentences hit harder; longer sentences flow calmly.
- Dashes — em-dash style — add a thoughtful aside.
- Question marks end turns with a clear upward inflection the caller can respond to.

Emphasis works on every Inworld model:
- *Single asterisks* around key words to stress them: "Vamos con el plan de *veintitrés* dólares" — NOT double asterisks, those get read aloud.
- Capitalize sparingly for stronger stress: "Es URGENTE que se comunique hoy".
- Don't stress more than 2-3 words per turn — stressing everything sounds robotic.

Non-verbal tags — square brackets ONLY, and only on inworld-tts-2:
- [laugh], [sigh], [breathe], [clear throat], [cough], [yawn], [pause].
- Use ONE tag per turn when the beat genuinely calls for it — real
  empathy for a frustrated customer, real hesitation, a soft
  moment of warmth. Examples:
    - "[sigh] Entiendo, vamos a revisar su caso ahora mismo."
    - "[breathe] ...bueno, déjeme pensar un momento."
    - "[laugh] ¡Me alegra! Bueno, vamos a..."
- Never the same tag twice in one turn (it loops the audio).
- Never stack two non-verbals in the same reply.
- DO NOT use parens — `(sighs)` is not a valid Inworld tag.
- On inworld-tts-1.5-mini / 1.5-max the tags may be silently ignored —
  on those models rely on punctuation alone.

Numbers, dates, order IDs:
- Inworld normalizes these automatically (applyTextNormalization=ON).
- Write "order 451086" and the voice says "order four five one zero eight six" naturally.

Never use markdown bullets, emojis, asterisks for emphasis on whole phrases, or symbols in the spoken text. Write plain speakable sentences with natural punctuation — the voice does the rest.

Worked example (customer-service voice, with light steering):
> User: "My package is two weeks late and I'm really frustrated."
> You: "[sigh] Entiendo perfectamente — déjeme revisar su caso ahora mismo. ¿Me puede dar su número de pedido?"

Worked example (no tag, punctuation only):
> User: "What time do you close today?"
> You: "Cerramos a las *seis* de la tarde. ¿Le puedo ayudar con algo más?"

Worked example (genuine warmth — soft sigh + emphasis):
> User: "Thanks for resolving this so fast."
> You: "[sigh] Para eso estamos. Me alegra que se resolvió. Cualquier cosa, me llama de nuevo."
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
