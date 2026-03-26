import re

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    FILLER_TEXT_EN,
    FILLER_TEXT_ES,
    FILLER_TTS_ENABLED,
    RIME_TTS_SPEAKER_EN,
    RIME_TTS_SPEAKER_ES,
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
    # ── Voice behavior (top priority) ──
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

    # ── Closing ──
    "After confirming no further help needed: 'Thank you for contacting Cialix Customer Support. If you need further help, don't hesitate to reach out. Have a great day!'"
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

    # Name pattern (simple)
    if "my name is" in lowered or "mi nombre es" in lowered:
        name_candidate = None
        if "my name is" in lowered:
            name_candidate = text.split("my name is", 1)[1].strip().split(" ")[0:3]
        elif "mi nombre es" in lowered:
            name_candidate = text.split("mi nombre es", 1)[1].strip().split(" ")[0:3]
        if name_candidate:
            results["name"] = " ".join(name_candidate).strip().strip(".?,!")

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
    count = 0

    for char in stripped:
        current.append(char)
        count += 1
        if char in ".!?" and count >= 80:
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
            if char in ".!?\n" and index >= 15:
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
