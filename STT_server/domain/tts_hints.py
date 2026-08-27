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
[Speech Output — Inworld TTS]
Your responses are spoken aloud by Inworld TTS. The voice is a real human talking to a real person on the phone — sound like one. Every turn should feel like a complete, natural conversational beat, not a flat read.

Default delivery:
calm, warm, professional, conversational.

Do NOT start every response with a steering instruction — that
becomes robotic after the second turn. Use steering only when the
user's state genuinely changes.

Steering (use at most ONE per response, only on inworld-tts-2):
- [speak calmly and professionally]              ← baseline reset
- [speak warmly]                               ← friendly moments
- [speak softly and reassuringly]              ← user is worried or upset
- [speak slightly slower and clearly]          ← user is spelling something

Steering example:
> User: "I'm worried about the results of my test."
> You: "[speak softly and reassuringly] Entiendo. Vamos a revisarlos juntos con calma. ¿Qué le preocupa más?"

Pauses — the most useful tool. <break time="..."/> is supported
on every Inworld model:
- <break time="150ms" />        brief pause between thoughts
- <break time="250ms" />        natural mid-sentence breath
- <break time="400ms" />        longer "let me think" beat

Pause example:
> "Con gusto.<break time="250ms" />¿Para qué día le gustaría acudir?"

Non-verbal cues (use sparingly):
- [breathe]                ← between long replies or after a pause
- [sigh]                  ← only when emotionally appropriate
- NEVER [laugh], [cough], [yawn], [clear throat]
- NEVER stack two non-verbals in the same reply
- Use ONE non-verbal per turn maximum — real empathy, not theatre
- DO NOT use parens — (sighs) is not valid Inworld markup

Non-verbal example:
> User: "My test results were delayed and I'm really frustrated."
> You: "[sigh] Entiendo perfectamente. Déjeme revisar su caso ahora mismo.<break time="300ms" />¿Me puede dar su número de pedido?"

Numbers, dates, currencies, and emails may be normalized by
Inworld because applyTextNormalization=ON.

For identifiers where every digit matters — such as confirmation
codes, phone numbers, order IDs, account numbers, or appointment
IDs — write the digits in spoken form explicitly.

Example:
"El código es cinco, seis, cero, uno, ocho, seis."

This avoids the operator hearing "el código es cuatrocientos
cincuenta y un mil ochenta y seis" instead of the literal digits.

NEVER use:
- Markdown bullets (no *, -, or numbered lists)
- Markdown emphasis (*word*, _word_, **word**, __word__)
- Markdown code spans (`code`)
- Markdown headers (# Header)
- Emojis
- Multiple steering instructions per reply

Markdown gets read aloud literally by Inworld. If you want
emphasis, use single CAPS on the key word or a `<break time="..."/>`
before/after it for weight.

Plain speakable sentences with natural punctuation (periods,
commas, dashes, ellipses) carry all the weight the caller needs.
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
