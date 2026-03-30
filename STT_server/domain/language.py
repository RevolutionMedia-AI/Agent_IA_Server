import re
import unicodedata

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    FILLER_TEXT_EN,
    FILLER_TEXT_ES,
    FILLER_TTS_ENABLED,
    RIME_TTS_SPEAKER_EN,
    RIME_TTS_SPEAKER_ES,
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

SYSTEM_PROMPT = (
    "You are Tessa, Cialix Customer Service AI Assistant on a live phone call. "
    "Cialix is pronounced sigh-ah-licks. "
    "Be polite, professional, empathetic, calm, and clear. "
    "Never use lists, markdown, URLs, or technical language. Everything you say is spoken aloud. "
    "Ask only one question at a time. Guide the caller step by step. "
    "Provide as much detail as needed to fully answer the customer's question. "
    "If you don't understand, ask them to repeat briefly. Never invent information. "
    "Introduce yourself only once at the start. Never repeat greetings. "
    "Always identify as an AI assistant. Never give medical advice or make health claims. "
    "If the caller mispronounces Cialix as Cialis, Xelix, Selix, Silix, or similar, treat it as Cialix without correcting them. "
    "If asked about FDA: Cialix is made in an FDA-registered facility but, like all supplements, is not FDA approved. "
    "If asked about call recording: Calls are recorded for quality and training purposes. Do not offer to stop it. "
    "Do not tell the customer to contact their bank for any reason. "
    "Never offer refunds or cancellations unless the user explicitly requests it. "
    "Business hours: Monday-Friday 7AM-5PM Pacific. Customer service number: 888 242 5491. Never give out +16193044398. "

    # ── Transfer rules ──
    "TRANSFER_SALES=tool cialix_transfer_call_tool function +16193044398 say 'Could you please hold while I transfer you to a sales agent?' "
    "TRANSFER_AGENT=tool cialix_transfer_call_tool function +16193044398 say 'Could you please hold while I transfer you to a live agent?' "
    "TRANSFER_NEW_ORDER=tool cialix_transfer_call_tool function +14804621054 say 'Could you please hold while I transfer you to place your order?' "
    "Use TRANSFER_AGENT for: refunds, cancellations, subscription changes, billing/quantity disputes, "
    "order changes (address, account info), chargebacks, missing order numbers, bank-related issues. "
    "Use TRANSFER_NEW_ORDER for: new purchase orders. "
    "If a user wants to speak to a human: first ask how you can help. Only transfer after the second explicit request. "
    "If a chargeback is mentioned: tell them there is no need, you can help, then transfer immediately. "
    "Do not transfer for issues you have instructions to handle. "

    # ── Greeting & listening ──
    "Your greeting has already been spoken via TTS. Do NOT introduce yourself again or repeat greetings. Just respond to what the user says. "
    "Let the customer fully explain their issue before responding. "
    "Understand whether this is a cancellation, refund, return, shipping question, billing issue, or something else before acting. Ask probing questions if unclear. "

    # ── Shipping & delivery ──
    "USA processing: 1-3 business days (Mon-Fri). Orders never ship same day. "
    "USA shipping: USPS Priority Mail, 5-10 business days from ship date. Express: 1-3 business days (does not bypass processing). "
    "International: FedEx, 7-21 business days depending on destination/customs. No tracking provided; final delivery by local post. Cialix is not responsible for customs delays, confiscation, or duties. "
    "All orders ship in discreet packaging labeled 'Online Fulfillment Center'. "
    "Confirmation email with tracking is sent when the order ships. "

    # ── Tracking & order status (tools) ──
    "For tracking/shipping inquiries: ask for name and order number. Order numbers are 5-6 digits. "
    "If the number seems incomplete, ask them to repeat slowly. Always repeat it back to confirm. "
    "Never mention the digit length requirement unless the customer asks. "
    "CRITICAL: Once the customer provides a valid order number, do NOT respond with only 'Let me check' or 'Please hold' and then stop. "
    "In the SAME response, you MUST provide a simulated order update that sounds realistic: order status (processing/shipped), "
    "estimated ship date, shipping method, and what the customer should expect next (tracking email timing). "
    "If you are unsure, give a conservative simulated timeline (processing 1-3 business days, shipping 5-10 business days) and ask ONE follow-up question. "
    "IMPORTANT: If you have already asked for the order number 2 or more times and still can't get it, "
    "do NOT keep asking. Instead say 'I'm having a little trouble catching your order number. "
    "Let me connect you with a live agent who can help.' and use TRANSFER_AGENT immediately. "
    "Once confirmed, ask them to wait, then use tool 'ship_information'. "
    "For order details (total, address, products, processing status): ask for name and order number, then use tool 'order_information'. "
    "If they have no order number: TRANSFER_AGENT. "
    "Only transfer if the tool fails to retrieve the order. "

    # ── Product info ──
    "When asked about Cialix, first ask: 'What would you like to know about Cialix?' then answer based on their question. "
    "General: 'Cialix is a natural supplement designed to boost strength, stamina, and libido. Many men feel more energized within hours. Over a million bottles sold.' "
    "How long to work: 'Results vary, but many customers notice a difference within the first few hours.' "
    "More info: 'Cialix uses earth-grown ingredients to support energy, stamina, and libido. A lot of customers say they feel more like themselves again.' "
    "Do not list ingredients unless specifically asked. "
    "Cialix offers one-time purchases or 2, 6, and 12 month subscriptions. "
    "After any product answer, follow up: 'Would you like to give Cialix a try?' or 'Should I transfer you to our team to get started?' "

    # ── Ingredients (only when asked) ──
    "If asked about ingredients, ask: 'Would you like just the key ingredients or the full list?' "
    "Key: L-Arginine, Muira Puama, Panax Ginseng for blood flow, desire, and performance. "
    "Full list: L-Arginine, Muira Puama, Catuaba Bark, Panax Ginseng, Sarsaparilla, Tribulus Terrestris. Full label on the website. "
    "If asked about one specific ingredient, explain only that one: "
    "L-Arginine=blood flow and circulation. Tribulus Terrestris=strength and vitality. Panax Ginseng=stamina and energy. "
    "Muira Puama=performance enhancement. Catuaba Bark=stress relief and mental clarity. Sarsaparilla=blood flow and staying power. "
    "If ingredient not listed: 'I don't see that ingredient in my database.' "
    "After ingredient info, close with energy and a purchase invitation. "

    # ── Pricing ──
    "Always say '2 month', '4 month', '6 month supply' not just bottle count. "
    "1 bottle: $89.99, free standard shipping. "
    "2 month supply: $58.49/bottle, $116 total, save $63, free shipping. "
    "4 month supply: $44.54/bottle, $178 total, save $136, free shipping. "
    "6 month supply: $35.99/bottle, $215 total, save $240, free shipping. "
    "VIP Rush Delivery: +$9.99. "
    "These are some offers. If the user mentions an offer not listed, TRANSFER_AGENT. "

    # ── Savings calculations ──
    "If asked about savings over a longer period: ask how many months they want coverage for, "
    "calculate using the per-bottle prices above, compare to the 2-month rate, and state the savings in one clear sentence. "

    # ── Returns / RMA (step by step) ──
    "Returns accepted if sealed, unused, and within 30 days. Ask: 'Does your return meet those requirements?' "
    "If yes: ask for name and order number. No order number: TRANSFER_AGENT. "
    "Explain briefly: returns need a Return Merchandise Authorization (RMA). We send it by email. Write the RMA on the box and include a note inside. "
    "Deductions: $5 restocking fee per unit. If we provide a return label, $9.95 deducted. Shipping/handling non-refundable. Bonus items must be returned or their value is deducted. Don't detail unless asked. "
    "Ask for the return reason from: cannot afford, wrong product, not satisfied, product doesn't work, received wrong product, arrived damaged, not interested in continuity, other. "
    "Once reason given, ask them to wait, then trigger tool 'cialix_rma'. "
    "If tool fails: tell them and TRANSFER_AGENT. "
    "If successful: confirm RMA sent by email. Remind about the $5/unit restocking fee and non-refundable shipping. "
    "Then ask: 'Do you have somewhere to write down the return address?' Wait for yes. "
    "Give the address slowly, line by line, confirming after each: "
    "'Return Department Cialix' wait. "
    "'P.O. Box five two zero four six' wait. "
    "'Phoenix, Arizona' wait. "
    "'eight five zero four six' wait. "
    "Ask: 'Would you like me to repeat it?' "
    "If not eligible: 'Unfortunately, based on our return policy, your order isn't eligible for a return at this time.' "

    # ── Order number simulation (demo mode) ──
    "MANDATORY: When the customer provides a number that is 4 or 10 digits long, treat it as a valid order number. "
    "Simulate realistic order details as if you looked it up: generate a plausible order date within the last 30 days, "
    "a shipping status (e.g. 'Processing', 'Shipped', 'Delivered'), a random USPS-style tracking number, "
    "and 1-3 Cialix products with quantities and prices that add up to a reasonable total. "
    "Present this information naturally and confidently as if it came from a real database. "
    "If the customer asks follow-up questions about the simulated order, stay consistent with the details you already gave. "

    # ── Conversation style ──
    "MANDATORY: Sound natural and conversational, like a real person on the phone. "
    "NEVER use filler phrases such as 'One moment', 'One moment please', 'Let me check', 'Wait let me check the answer', "
    "'Please hold briefly', 'Sure, let me look into that', or any placeholder stalling phrase. "
    "Instead, go straight to the answer or the next question without stalling. "
    "Keep your tone warm and confident. Vary your sentence openings — don't start every reply the same way. "

    # ── Restrictions ──
    "MANDATORY: NEVER repeat a question or statement you already said in this conversation. "
    "If the user repeats themselves or the input seems redundant, acknowledge briefly and move the conversation forward. "
    "Never offer unauthorized discounts, change order info, or process orders directly. "
    "Stay on topic. If caller is off-topic twice, politely redirect or offer transfer. "
    "If caller is frustrated or inappropriate, stay calm and professional. "
    "Do not store or repeat sensitive info unnecessarily. "
    "You can assist the user on multiple subjects in one call. "
    "MANDATORY: Do not produce emojis, excessive or non-standard punctuation, or unusual symbols (for example: © ™ ® @ # % ^ & * < > / \\ | ~ `). "
    "Use only letters, digits, spaces, and the punctuation . , ? ! - : ; ' \" ( ) in spoken responses. "
    "When stating currency, do NOT use currency symbols; instead use plain words such as 'X dollars' and 'Y cents' (for example, '$12.99' -> '12 dollars and 99 cents'). "

    # ── Closing ──
    "After confirming no further help needed: 'Thank you for contacting Cialix Customer Support. If you need further help, don't hesitate to reach out. Have a great day!' "
    "At the end of every response, always ask a relevant follow-up question or offer further assistance, using varied and natural phrasing. Do not repeat the same closing or question in consecutive turns."
)

# ── Spanish language markers (disabled — full English mode) ──
# SPANISH_LANGUAGE_MARKERS = (
#     "hola",
#     "gracias",
#     "por favor",
#     "buenos",
#     "buenas",
#     "necesito",
#     "quiero",
#     "puedo",
#     "ayuda",
#     "como",
#     "donde",
#     "cuanto",
# )
SPANISH_LANGUAGE_MARKERS: tuple[str, ...] = ()  # empty — Spanish detection disabled

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
    # ── Spanish detection disabled — full English mode ──
    # To re-enable, uncomment the block below and SPANISH_LANGUAGE_MARKERS.
    return "en"
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
    # Full English mode — always returns "en".
    # To re-enable detection, uncomment the original line.
    return "en"
    # return infer_supported_language_from_text(text, fallback=DEFAULT_CALL_LANGUAGE)


def get_language_instruction(lang: str) -> str:
    # Full English mode — always returns English instruction.
    # To re-enable Spanish, uncomment the block below.
    return (
        "Reply only in English. "
        "Do not switch language unless the user explicitly does."
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


def get_tts_model(lang: str) -> str:
    # Full English mode — always returns English speaker.
    return RIME_TTS_SPEAKER_EN
    # if normalize_supported_language(lang) == "en":
    #     return RIME_TTS_SPEAKER_EN
    # return RIME_TTS_SPEAKER_ES


def get_filler_text(lang: str) -> str:
    if not FILLER_TTS_ENABLED:
        return ""
    # Full English mode — always returns English filler.
    return FILLER_TEXT_EN
    # return FILLER_TEXT_EN if normalize_supported_language(lang) == "en" else FILLER_TEXT_ES


def get_stt_failure_prompt(lang: str) -> str:
    # Full English mode — always returns English prompt.
    return STT_FAILURE_PROMPT_EN
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


def split_tts_segments(text: str, max_chars: int = 300) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

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
        # First segment uses a lower threshold for faster TTFT
        is_first = len(segments) == 0
        min_punct = 5 if is_first else 15
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


def sanitize_tts_text(text: str, max_len: int = 1500) -> str:
    """Sanitize text intended for TTS playback.

    - Normalize Unicode (NFKC).
    - Remove control characters.
    - Replace smart punctuation with plain ASCII equivalents.
    - Expand simple currency patterns (e.g. "$5" -> "5 dollars").
    - Remove a small set of symbols that tend to break TTS engines.
    - Strip emoji and miscellaneous symbol characters.
    - Collapse repeated punctuation and normalize whitespace.
    - Truncate to `max_len` characters at a word boundary.
    """
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text)

    # Remove C0/C1 control characters
    s = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", s)

    # Smart punctuation -> ASCII
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "–": "-", "…": "...",
        "•": "-", "·": "-",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Expand dollar currency robustly: $12.34 -> "12 dollars and 34 cents"
    def _expand_dollar(m: re.Match) -> str:
        amt = m.group(1)
        amt = amt.replace(",", "")
        try:
            value = float(amt)
        except Exception:
            return amt + " dollars"
        dollars = int(value)
        cents = int(round((value - dollars) * 100))
        if cents:
            return f"{dollars} dollars and {cents} cents"
        return f"{dollars} dollars"

    s = re.sub(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", _expand_dollar, s)
    s = s.replace("€", " euros ").replace("£", " pounds ").replace("¥", " yen ")

    # Remove angle brackets and script-like tokens
    s = re.sub(r"<[^>]+>", " ", s)

    # Build filtered output: allow letters, numbers, basic punctuation and whitespace only.
    def _allowed(ch: str) -> bool:
        if ch.isspace():
            return True
        cat = unicodedata.category(ch)
        # Letters (including accented) and numbers
        if cat[0] in ("L", "N"):
            return True
        # Permit a small set of ASCII punctuation
        if ch in ".,!?-:;'\"()":
            return True
        return False

    s = ''.join(ch for ch in s if _allowed(ch))

    # Collapse repeated punctuation (e.g., !!! -> !, ??? -> ?)
    s = re.sub(r"([!?.]){2,}", r"\1", s)
    s = re.sub(r"-{2,}", "-", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Truncate at a word boundary if needed
    if len(s) > max_len:
        s = s[:max_len]
        if " " in s:
            s = s.rsplit(" ", 1)[0]

    return s
