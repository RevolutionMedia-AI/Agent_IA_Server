"""TTS-specific instructions to prepend to the LLM system prompt.

Why this exists
---------------
Some TTS providers (notably Inworld) support inline markup that
varies the voice output — non-verbal cues like `[breathe]` or
`[sigh]`, steering instructions like `[speak calmly, professionally.]`,
and explicit pauses via `<break time="250ms" />`. Without an
explicit instruction, the LLM emits clean prose and the TTS
never gets a chance to vary the voice, so the caller hears a
flat, robotic delivery.

This module holds the per-provider hint blocks. The session bootstrap
PREPENDS the hint for the active TTS provider to the agent's
`custom_prompt` (not appends — earlier experiments showed the LLM
treats end-of-prompt additions as low-priority context and rarely
emits the tags). The hint stays short so the agent's own prompt
remains the dominant behavioural spec; we only add the
voice-direction vocabulary that the TTS consumes.

The hint is only added once per session (idempotent guard via
_HINT_MARKER in STT_Server.py).

The Inworld hint is tuned for the Laboratorio C.G.O. agent
(customer-service voice — natural but professional, not actor-
on-stage dramatic). A different agent (sales, comedy, support)
would warrant a different baseline.
"""
from __future__ import annotations


INWORLD_STEERING_INSTRUCTIONS = """[TTS Steering — Inworld]
[Speech Output — Inworld TTS-2]
Every response is SPOKEN. Write for the ear, not the eye.

Default voice: calm, warm, professional, conversational.

PAUSES:
For normal multi-sentence responses, use ONE natural <break>.
Very short replies do not need a break.
Never use more than two.
  <break time="200ms" />     between sentences
  <break time="350ms" />     before answering / after a beat

NON-VERBALS — at most ONE per turn:
  [breathe] is rare. Never put [breathe] at the beginning of a
  response. Use it only between clauses or sentences when the
  response is long enough to naturally require a breath.
  Good: Entiendo lo que necesita. [breathe] Le explico las opciones disponibles.
  Bad:  [breathe] Muy bien.
  [sigh]        only when caller is frustrated
NEVER use: [laugh], [cough], [yawn], [clear throat].

STEERING — at most ONE, only when caller's state changes:
  [speak warmly]                    — friendly moments
  [speak softly and reassuringly]   — caller is worried or upset
  [speak slightly slower and clearly] — caller is spelling something
NEVER start every response with a steering tag. It becomes canned.

LENGTH — phone responses are SHORT:
  Normal reply: 10–30 spoken words
  Explanation: up to 50 words
  NEVER read a catalog / list / menu

NUMBERS — for IDs / codes / phones where every digit matters:
  "El código es cinco, seis, cero, uno, ocho, seis."
NEVER use "order 451086" — Inworld may normalize.

NEVER: Markdown emphasis, bullets, headers, emojis, code spans.

EXAMPLE OUTPUTS:

NORMAL:
El examen no requiere ayuno. <break time="250ms" /> ¿Desea agendar una cita?

OCCASIONAL NATURAL BREATH:
Entiendo. [breathe] Déjeme explicárselo con calma.
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
