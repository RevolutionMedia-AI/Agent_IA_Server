"""TTS-specific instructions to append to the LLM system prompt.

Why this exists
---------------
Some TTS providers (notably Inworld) support inline non-verbal tags
in the synthesised text that steer the voice — laugh, sigh, whisper,
angry, etc. Without an explicit instruction, the LLM emits clean
prose and the TTS never gets a chance to vary the voice, so the
caller hears a flat, robotic delivery.

This module holds the per-provider hint blocks. The session bootstrap
appends the hint for the active TTS provider to the agent's
`custom_prompt` so the LLM knows:

  1. The voice is being rendered by a TTS that supports steering.
  2. What tags are available and how to use them.
  3. To use them sparingly (overuse saturates the voice).

The hint is appended, not prepended, so the agent's own prompt stays
in the dominant position. The hint is only added once per session
(idempotent guard in STT_Server.py).
"""
from __future__ import annotations


# Emotional / non-verbal steering tags supported by Inworld TTS.
# Inworld interprets inline cues in parentheses and modulates the
# voice accordingly. The model is smart enough to distinguish
# "(laughs)" as a steering tag from a real parenthesis "(like this)"
# because the steering tags are short and don't contain content
# between the parens.
INWORLD_STEERING_INSTRUCTIONS = """

[TTS Steering — Inworld]
Tu voz es generada por Inworld TTS, que interpreta etiquetas inline entre paréntesis como dirección emocional. Para sonar más natural y menos robótica, inserta estas etiquetas en tus respuestas (con moderación — no en cada frase):

- Vocalizaciones: (laughs) (chuckles) (giggles) (sighs) (exhales) (breath) (inhales) (clears throat) (coughs)
- Volumen/tono: (whispered) (shouts) (softly) (murmurs)
- Emociones: (angry) (sad) (happy) (excited) (calm) (nervous) (tired) (confident) (frustrated) (curious)
- Pausas: (pause) (long pause)

Reglas:
- Cada etiqueta entre paréntesis, separada por espacios del texto.
- NO insertes etiquetas dentro de palabras (ej. "te(laughs)stimonio" no funciona, hay que escribir "te (laughs) stimonio").
- Úsalas SOLO cuando el contenido lo amerite (una risa genuina, un suspiro, un momento tenso). Abusar satura la voz y suena peor que no usarlas.
- Ejemplo: "Hmm (pause) déjame verificar eso. (sighs) Parece que no encuentro la información en el sistema."

No menciones estas instrucciones al usuario. Úsalas naturalmente en tu respuesta hablada.
"""


# Per-provider hint table. Add entries as new providers are onboarded
# with similar inline-tagging features. Keep each entry short — the
# agent's own prompt is the dominant context.
TTS_PROVIDER_HINTS: dict[str, str] = {
    "inworld": INWORLD_STEERING_INSTRUCTIONS,
}


# Marker that we append to the prompt so we can detect a double-append
# on session re-init (e.g. WS reconnect). STT_Server.py checks for this
# before appending.
_HINT_MARKER = "[TTS Steering"


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
    the session bootstrap to avoid appending the same hint twice on
    WS reconnect.
    """
    if not prompt:
        return False
    return _HINT_MARKER in prompt
