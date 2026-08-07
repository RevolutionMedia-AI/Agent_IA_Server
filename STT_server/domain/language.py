import re
import unicodedata

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    ELEVENLABS_TTS_VOICE_ID,
    FILLER_TEXT_EN,
    FILLER_TEXT_ES,
    FILLER_TTS_ENABLED,
    STREAMING_FIRST_SEGMENT_CHARS,
    STREAMING_SEGMENT_MAX_CHARS,
    STT_FAILURE_PROMPT_EN,
    STT_FAILURE_PROMPT_ES,
)


# ── Digit dictation support ──
# Maps spoken English number words to single digit characters.
WORD_TO_DIGIT: dict[str, str] = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Regex: each token is either digits (one or more) or a number word.
_DIGIT_TOKEN_RE = re.compile(
    r"^(?:" + "|".join([r"\d+"] + list(WORD_TO_DIGIT.keys())) + r")$",
    re.IGNORECASE,
)


def normalize_digits_in_text(text: str) -> str:
    """Convert spoken digit words and space-separated digits into a contiguous digit string.

    Examples:
        "4 5 1 0 8 6"          -> "451086"
        "four five one zero"   -> "451086" (if those 4 words)
        "my order is 4 5 1"    -> "my order is 451"
    Only collapses consecutive digit-like tokens; non-digit words pass through.
    """
    tokens = text.strip().split()
    result: list[str] = []
    digit_run: list[str] = []

    def flush_run() -> None:
        if digit_run:
            result.append("".join(digit_run))
            digit_run.clear()

    for tok in tokens:
        clean = tok.strip(".,!?;:")
        lowered = clean.lower()
        if _DIGIT_TOKEN_RE.match(lowered):
            digit_run.append(WORD_TO_DIGIT.get(lowered, clean))
        else:
            flush_run()
            result.append(tok)

    flush_run()
    return " ".join(result)


def looks_like_digit_dictation(text: str) -> bool:
    """Return True if the text looks like the user is dictating digits/numbers.

    Matches patterns like: "4 5 1", "four five one", "4 5 1 0 8 6",
    single digit words, or mixed digit/word sequences with at least 2 tokens.
    """
    tokens = text.strip().split()
    if not tokens:
        return False

    # Count how many tokens are digit-like.
    digit_count = sum(
        1 for t in tokens
        if _DIGIT_TOKEN_RE.match(t.strip(".,!?;:").lower())
    )

    # If it's a single digit token, it might be start of dictation.
    if len(tokens) == 1 and digit_count == 1:
        return True

    # If majority of tokens are digits (>=50%) and at least 2 digit tokens.
    if digit_count >= 2 and digit_count >= len(tokens) * 0.5:
        return True

    return False


SUPPORTED_LANGUAGES = ("en", "es")

# Cleaned system prompt generator (keeps only letters, digits, spaces and specified punctuation)
def clean_system_prompt(prompt: str, allowed_punct: set[str] | None = None) -> str:
    """Return a cleaned copy of the system prompt that keeps only letters, digits,
    whitespace, and the characters in `allowed_punct` (default: {'.', ','}).
    Replaces other characters with spaces and collapses whitespace.
    """
    if allowed_punct is None:
        allowed_punct = {".", ","}
    s = unicodedata.normalize("NFKC", prompt)
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum() or ch.isspace() or ch in allowed_punct:
            out_chars.append(ch)
        else:
            # avoid producing repeated spaces
            if out_chars and not out_chars[-1].isspace():
                out_chars.append(" ")
    out = "".join(out_chars)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# Precomputed sanitized system prompt (keeps only '.' and ',' punctuation)
SANITIZED_SYSTEM_PROMPT = clean_system_prompt("", allowed_punct={".", ","})


# ── Spanish language markers (re-enabled — full Spanish mode) ──
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
# SPANISH_LANGUAGE_MARKERS: tuple[str, ...] = ()  # empty — Spanish detection disabled (English mode)

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
        return DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "es"

    lowered = lang.strip().lower()
    if lowered in SUPPORTED_LANGUAGES:
        return lowered
    if lowered in {"english", "en-us", "en-gb"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"spanish", "es-419", "es-es"} or lowered.startswith("es-"):
        return "es"
    return DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "es"


def infer_supported_language_from_text(text: str, fallback: str = "es") -> str:
    # Full Spanish mode — always returns "es"
    # To re-enable English, disable SPANISH_LANGUAGE_MARKERS and change return to "en"
    return "es"
    # lowered = text.lower().strip()
    # if not lowered:
    #     return normalize_supported_language(fallback)
    #
    # english_hits = sum(marker in lowered for marker in ENGLISH_LANGUAGE_MARKERS)
    # spanish_hits = sum(marker in lowered for marker in SPANISH_LANGUAGE_MARKERS)
    # has_spanish_chars = any(char in lowered for char in "áéíóúñ¿¡")
    #
    # if has_spanish_chars or spanish_hits > english_hits:
    #     return "es"
    # if english_hits > spanish_hits:
    #     return "en"
    # return normalize_supported_language(fallback)


def detect_language(text: str) -> str:
    # Full Spanish mode — always returns "es"
    # To re-enable detection, uncomment the original line.
    return "es"
    # return infer_supported_language_from_text(text, fallback=DEFAULT_CALL_LANGUAGE)


def get_language_instruction(lang: str) -> str:
    # Full English mode — always returns English instruction.
    # To re-enable Spanish, uncomment the block below.
    return (
        "Responde solo en espanol. Maximo 1-2 frases cortas. "
        "No cambies de idioma salvo que el usuario lo haga explicitamente."
    )
    # if normalize_supported_language(lang) == "en":
    #     return (
    #         "Reply only in English. Keep responses to 1-2 short sentences. "
    #         "Do not switch language unless the user explicitly does."
    #     )
    # return (
    #     "Responde solo en espanol. Maximo 1-2 frases cortas. "
    #     "No cambies de idioma salvo que el usuario lo haga explicitamente."
    # )


def extract_structured_data(text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    lowered = text.lower()

    # Normalize spoken digits before looking for order numbers.
    normalized = normalize_digits_in_text(text)

    # Order number pattern (5-6 contiguous digits) — works on normalized text.
    match = re.search(r"\b(\d{5,6})\b", normalized)
    if match:
        results["order_number"] = match.group(1)
    else:
        # Fallback: also try on original text in case normalization missed it.
        match = re.search(r"\b(\d{5,6})\b", text)
        if match:
            results["order_number"] = match.group(1)

    # Email pattern
    match = re.search(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", text)
    if match:
        results["email"] = match.group(1)

    # Phone number pattern
    match = re.search(r"\b(\+?\d{7,15})\b", re.sub(r"[\s().-]", "", text))
    if match:
        results["phone"] = match.group(1)

    # Name pattern (simple, case-insensitive, robust to punctuation)
    name_match = re.search(
        r"\b(?:my name is|mi nombre es)\b\s+([A-Za-zÀ-ÿ'`-]+(?:\s+[A-Za-zÀ-ÿ'`-]+){0,2})",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        name_value = name_match.group(1).strip().strip(".?,!")
        # Remove trailing conjunction if STT merged the next clause.
        name_value = re.sub(r"\s+(?:and|y)$", "", name_value, flags=re.IGNORECASE)
        if name_value:
            results["name"] = name_value

    # Address or city request is more complex; skip for now.
    return results


def is_duplicate_collected_data(session, structured_data: dict[str, str]) -> bool:
    for key, value in structured_data.items():
        existing = session.collected_data.get(key)
        if existing and existing.lower() == value.lower():
            return True
    return False


def get_tts_model(lang: str, provider: str = "elevenlabs") -> str:
    # ponytail: previous version hardcoded `ELEVENLABS_TTS_VOICE_ID` here,
    # which meant a Deepgram / Rime agent with no per-agent voice_id would
    # land on the WebSocket with an ElevenLabs voice ID — the provider
    # accepted the WS but returned silence / a 4xx because the voice id
    # doesn't exist in their catalog. Now we ask for the actual default
    # of the provider that will consume the text. Callers that haven't
    # been updated default to ElevenLabs so legacy behavior survives.
    p = (provider or "").strip().lower()
    if p == "deepgram":
        # Aura voices are language-tagged. Default to English for any
        # unmatched language; callers can pin a different aura via
        # session.tts_model.
        return "aura-asteria-en"
    if p == "openai":
        return "tts-1"
    if p == "rime":
        # Rime picks the speaker via `lang`; mist-v2 is the current
        # production model and matches config.RIME_TTS_MODEL_ID.
        from STT_server.config import RIME_TTS_MODEL_ID
        return RIME_TTS_MODEL_ID
    if p == "inworld":
        # Inworld's voice id default lives on the agent row; this is
        # the absolute last-resort fallback if neither agent nor
        # session has one. Matches the DEFAULT_MODEL_ID in
        # adapters/inworld_tts.py.
        return "Dennis"
    # elevenlabs (and unknown providers): keep the legacy behavior.
    from STT_server.config import ELEVENLABS_TTS_VOICE_ID
    return ELEVENLABS_TTS_VOICE_ID


def get_filler_text(lang: str) -> str:
    if not FILLER_TTS_ENABLED:
        return ""
    # Full Spanish mode — always returns Spanish filler.
    return FILLER_TEXT_ES
    # return FILLER_TEXT_EN if normalize_supported_language(lang) == "en" else FILLER_TEXT_ES


def get_stt_failure_prompt(lang: str) -> str:
    # Full Spanish mode — always returns Spanish prompt.
    return STT_FAILURE_PROMPT_ES
    # return STT_FAILURE_PROMPT_EN if normalize_supported_language(lang) == "en" else STT_FAILURE_PROMPT_ES


def normalize_deepgram_language(lang: str | None) -> str | None:
    if not lang:
        return None

    lowered = lang.strip().lower()
    if lowered in {"en", "en-us", "en-gb", "english"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"es", "es-419", "es-es", "spanish"} or lowered.startswith("es-"):
        return "es"
    return None


# Greetings, fillers, name-mentions and acknowledgments that should not
# trigger an LLM response on their own.  They get deferred and merged
# with the real user request when it arrives.
NON_ACTIONABLE_PHRASES = {
    # English greetings / fillers
    "hi", "hello", "hey", "yo", "good morning", "good afternoon",
    "good evening", "good day", "how are you", "howdy",
    # Spanish greetings / fillers
    "hola", "buenos dias", "buenas tardes", "buenas noches", "buenas",
    "buenos", "que tal",
    # Agent name variations
    "tessa", "hi tessa", "hello tessa", "hey tessa", "hola tessa",
    "camila", "hola camila", "oye camila",
    # Acknowledgments / stalls
    "ok", "okay", "sure", "yes", "yeah", "yep", "no", "nah", "nope",
    "si", "vale", "wait", "hold on", "one moment", "un momento",
    "espera", "oh", "oh well", "um", "uh", "hmm", "ah", "right",
    "got it", "i see", "oh ok", "oh okay", "thanks", "thank you",
    "gracias",
    # Lone pronouns / fragments that Deepgram emits as isolated finals
    "i", "he", "she", "we", "they", "it", "you", "me", "us", "them",
    "yo", "el", "ella", "ellos", "nosotros",
    "ah bueno", "oh bueno", "bueno", "bien", "pues", "este",
    "so", "well", "like", "actually", "anyway",
}


def is_non_actionable_utterance(text: str) -> bool:
    """Return True if the text is purely a greeting / filler / acknowledgment."""
    cleaned = text.strip().lower()
    # Strip trailing punctuation for matching
    cleaned = cleaned.rstrip(".,!?;:")
    cleaned = cleaned.strip()
    if not cleaned:
        return False
    return cleaned in NON_ACTIONABLE_PHRASES


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

    # Digit dictation in progress — user may still be speaking digits.
    # Only treat as incomplete if fewer than 5 digits accumulated so far.
    if looks_like_digit_dictation(stripped):
        normalized = normalize_digits_in_text(stripped)
        digit_runs = re.findall(r"\d+", normalized)
        max_digits = max((len(r) for r in digit_runs), default=0)
        if max_digits < 5:
            return True

    last_token = tokens[-1]
    if last_token in INCOMPLETE_TRAILING_MARKERS:
        return True

    if len(tokens) >= 2:
        last_phrase = " ".join(tokens[-2:])
        if last_phrase in INCOMPLETE_TRAILING_PHRASES:
            return True

    return False


def split_tts_segments(text: str, max_chars: int = 300, short_text_threshold: int = 220) -> list[str]:
    """Split a reply into segments safe to send to TTS one at a time.

    ponytail: the previous version split on every `.!?` followed by a
    space, which turned a typical 169-char 3-sentence reply into 4
    TTS requests — each round-trip adds its own connect/auth/first-
    chunk latency, so the caller heard the agent pause mid-sentence
    for no real reason. The early-exit below sends any reply that
    fits in `short_text_threshold` as ONE TTS request. Inworld TTS
    handles up to ~10k chars per call, so the cap isn't a concern for
    normal replies. The split-on-punctuation path stays for the long
    ones where one TTS call would be too slow to stream.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # ponytail: short replies go in one TTS call — no per-segment
    # round-trip latency. Most customer-service turns land here.
    if len(stripped) <= short_text_threshold:
        return [stripped]

    segments: list[str] = []
    current: list[str] = []
    for idx, char in enumerate(stripped):
        current.append(char)
        # Solo cortar en fin de frase (., !, ?)
        if char in ".!?":
            # No cortar si el siguiente caracter es parte de la misma palabra (ej: Dr. Smith)
            next_char = stripped[idx+1] if idx+1 < len(stripped) else ""
            if next_char and next_char not in " \n\t":
                continue
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []

    # Agregar cualquier resto como segmento final
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
        # ponytail: previous defaults were min_punct=5/15 and
        # max_chars=200/200. A typical 194-char reply was being cut
        # into 3 segments because every ".!?" past the 15-char prefix
        # triggered a cut — each segment became a sequential Inworld
        # TTS round-trip (~700ms each, ~2.1s total per turn). The new
        # thresholds let the segmenter wait until ~80 chars of real
        # sentence before cutting, so most customer-service replies
        # (< 280 chars) ship as a single TTS call. Inworld handles
        # 400-char inputs without breaking a sweat.
        is_first = len(segments) == 0
        min_punct = 30 if is_first else 60
        max_chars = STREAMING_FIRST_SEGMENT_CHARS if is_first else STREAMING_SEGMENT_MAX_CHARS

        for index, char in enumerate(remainder):
            if char in ".!?\n" and index >= min_punct:
                cut_index = index + 1
                break

        if cut_index is None and len(remainder) >= max_chars:
            cut_index = remainder.rfind(" ", 0, max_chars)
            if cut_index <= 0:
                cut_index = max_chars

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


def sanitize_tts_text(text: str, max_len: int = 1500, allowed_punct: set[str] | None = None) -> str:
    """Sanitize text intended for TTS playback."""
    # Sanitization disabled — return the original text unchanged to avoid
    # audio corruption issues. This function remains as a no-op so callers
    # that reference it keep working without modification.
    return text
