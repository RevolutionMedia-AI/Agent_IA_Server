from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    DEEPGRAM_TTS_MODEL,
    FILLER_TEXT_EN,
    FILLER_TEXT_ES,
    FILLER_TTS_ENABLED,
    STREAMING_SEGMENT_MAX_CHARS,
    STT_FAILURE_PROMPT_EN,
    STT_FAILURE_PROMPT_ES,
)


SUPPORTED_LANGUAGES = ("en", "es")

SYSTEM_PROMPT = (
    "Eres un asistente de voz telefonico en tiempo real. "
    "Solo puedes responder en ingles o espanol. "
    "Responde siempre en el mismo idioma del usuario y nunca cambies de idioma a menos que el usuario lo haga explicitamente. "
    "Si el idioma no es claro, usa ingles por defecto. "
    "Tu prioridad es responder rapido, claro y de forma natural. "
    "Usa frases cortas, claras y faciles de entender por telefono. "
    "Responde en 1 o 2 frases maximo. "
    "Usa lenguaje simple y directo. "
    "Evita listas, markdown, URLs, lenguaje tecnico y respuestas largas. "
    "Manten una sola idea por respuesta y haz solo una pregunta a la vez. "
    "Guia al usuario paso a paso y pide solo la informacion necesaria para avanzar. "
    "Evita estructuras dificiles de pronunciar y asegurate de que todo suene natural al escucharse. "
    "Si no entiendes algo, pide que repitan de forma breve. "
    "No inventes informacion. "
    "Manten un tono profesional, claro y conversacional, como un agente humano eficiente. "
    "Cada respuesta debe ser facil de procesar en una llamada en tiempo real."
)

SPANISH_LANGUAGE_MARKERS = (
    "hola",
    "gracias",
    "por favor",
    "buenos",
    "buenas",
    "necesito",
    "quiero",
    "puedo",
    "ayuda",
    "como",
    "donde",
    "cuanto",
)

ENGLISH_LANGUAGE_MARKERS = (
    "hello",
    "thanks",
    "thank you",
    "please",
    "help",
    "need",
    "want",
    "where",
    "how",
    "what",
    "today",
)

INCOMPLETE_TRAILING_MARKERS = {
    "a",
    "about",
    "also",
    "an",
    "and",
    "because",
    "been",
    "but",
    "como",
    "con",
    "de",
    "del",
    "el",
    "for",
    "i",
    "if",
    "just",
    "la",
    "like",
    "los",
    "me",
    "my",
    "o",
    "or",
    "para",
    "pero",
    "please",
    "por",
    "porque",
    "que",
    "si",
    "so",
    "sobre",
    "some",
    "than",
    "that",
    "the",
    "then",
    "to",
    "with",
    "y",
    "yo",
}

INCOMPLETE_TRAILING_PHRASES = {
    "and i",
    "and my",
    "because i",
    "can you",
    "could you",
    "de mi",
    "for my",
    "i need",
    "i want",
    "me gustaria",
    "para mi",
    "por que",
    "que me",
    "y mi",
    "y yo",
}


def normalize_supported_language(lang: str | None) -> str:
    if not lang:
        return DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "en"

    lowered = lang.strip().lower()
    if lowered in SUPPORTED_LANGUAGES:
        return lowered
    if lowered in {"english", "en-us", "en-gb"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"spanish", "es-419", "es-es"} or lowered.startswith("es-"):
        return "es"
    return DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "en"


def infer_supported_language_from_text(text: str, fallback: str = "en") -> str:
    lowered = text.lower().strip()
    if not lowered:
        return normalize_supported_language(fallback)

    english_hits = sum(marker in lowered for marker in ENGLISH_LANGUAGE_MARKERS)
    spanish_hits = sum(marker in lowered for marker in SPANISH_LANGUAGE_MARKERS)
    has_spanish_chars = any(char in lowered for char in "áéíóúñ¿¡")

    if has_spanish_chars or spanish_hits > english_hits:
        return "es"
    if english_hits > spanish_hits:
        return "en"
    return normalize_supported_language(fallback)


def detect_language(text: str) -> str:
    return infer_supported_language_from_text(text, fallback=DEFAULT_CALL_LANGUAGE)


def get_language_instruction(lang: str) -> str:
    if normalize_supported_language(lang) == "en":
        return (
            "Output language is locked to English. "
            "Reply only in English unless the user explicitly switches language."
        )
    return (
        "El idioma de salida esta fijado en espanol. "
        "Responde solo en espanol salvo que el usuario cambie explicitamente de idioma."
    )


def get_tts_model(lang: str) -> str:
    if normalize_supported_language(lang) == "en":
        return DEEPGRAM_TTS_MODEL
    return "aura-2-estrella-es"


def get_filler_text(lang: str) -> str:
    if not FILLER_TTS_ENABLED:
        return ""
    return FILLER_TEXT_EN if normalize_supported_language(lang) == "en" else FILLER_TEXT_ES


def get_stt_failure_prompt(lang: str) -> str:
    return STT_FAILURE_PROMPT_EN if normalize_supported_language(lang) == "en" else STT_FAILURE_PROMPT_ES


def normalize_deepgram_language(lang: str | None) -> str | None:
    if not lang:
        return None

    lowered = lang.strip().lower()
    if lowered in {"en", "en-us", "en-gb", "english"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"es", "es-419", "es-es", "spanish"} or lowered.startswith("es-"):
        return "es"
    return None


def looks_like_incomplete_utterance(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if stripped.endswith((",", ";", ":", "-", "(", "/")):
        return True

    if stripped[-1] in ".!?":
        return False

    tokens = stripped.lower().replace("?", "").replace("!", "").replace(".", "").split()
    if not tokens:
        return False

    last_token = tokens[-1]
    if last_token in INCOMPLETE_TRAILING_MARKERS:
        return True

    if len(tokens) >= 2:
        last_phrase = " ".join(tokens[-2:])
        if last_phrase in INCOMPLETE_TRAILING_PHRASES:
            return True

    return False


def split_tts_segments(text: str, max_chars: int = 150) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    segments: list[str] = []
    current: list[str] = []
    count = 0

    for char in stripped:
        current.append(char)
        count += 1
        if char in ".!?" and count >= 40:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            count = 0
        elif count >= max_chars:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            count = 0

    if current:
        segment = "".join(current).strip()
        if segment:
            segments.append(segment)

    return segments


def pop_streaming_segments(buffer: str, force: bool = False) -> tuple[list[str], str]:
    remainder = buffer
    segments: list[str] = []

    while remainder:
        cut_index: int | None = None

        for index, char in enumerate(remainder):
            if char in ".!?\n":
                cut_index = index + 1
                break

        if cut_index is None and len(remainder) >= STREAMING_SEGMENT_MAX_CHARS:
            cut_index = remainder.rfind(" ", 0, STREAMING_SEGMENT_MAX_CHARS)
            if cut_index <= 0:
                cut_index = STREAMING_SEGMENT_MAX_CHARS

        if cut_index is None:
            break

        segment = remainder[:cut_index].strip()
        remainder = remainder[cut_index:].lstrip()
        if segment:
            segments.append(segment)

    if force and remainder.strip():
        segments.append(remainder.strip())
        remainder = ""

    return segments, remainder
