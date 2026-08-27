"""Tests for the safe speech-text sanitizer.

The previous version was a no-op (passed text straight to Inworld).
The operator's 2026-08-26 production case:

  Hola *mundo*   →  Inworld reads "asterisco mundo asterisco"

We strip Markdown contextually so:
  Hola *mundo*   →  Hola mundo
  ulises_test@gmail.com  →  ulises_test@gmail.com  (unchanged)

The hard requirement: never modify semantic content. Email
addresses, IDs, phone numbers, URLs that happen to contain `_`,
`.`, `*` etc. must round-trip unchanged.
"""
from STT_server.domain.language import sanitize_tts_text


# ── Markdown emphasis stripping ────────────────────────────────────


def test_strips_bold_double_asterisk():
    """`**word**` → `word`. Inworld would otherwise read "asterisco
    asterisco word asterisco asterisco"."""
    assert sanitize_tts_text("Hola **mundo**") == "Hola mundo"
    assert sanitize_tts_text("**hola**") == "hola"


def test_strips_italic_single_asterisk():
    """`*word*` → `word`. The marker is glued to the word with no
    whitespace — that's Markdown italic, not a stray asterisk."""
    assert sanitize_tts_text("Hola *mundo*") == "Hola mundo"


def test_strips_bold_double_underscore():
    """`__word__` → `word`."""
    assert sanitize_tts_text("Hola __mundo__") == "Hola mundo"


def test_strips_italic_single_underscore_with_word_neighbors():
    """`_word_` (underscore glued to letters on both sides) → `word`.
    The regex requires non-word chars on both sides — that's how
    Markdown italics look."""
    assert sanitize_tts_text("Hola _mundo_") == "Hola mundo"


def test_does_not_strip_email_address():
    """`ulises_test@gmail.com` must round-trip unchanged. The
    underscores in the local part have letters on both sides AND
    another `_` later in the address — the regex's `(?<!\w)`
    lookbehind rejects it (data, not Markdown)."""
    assert sanitize_tts_text("ulises_test@gmail.com") == "ulises_test@gmail.com"


def test_does_not_strip_customer_id():
    """`customer_id_123` — single `_` between letters and a digit.
    The regex requires `_` glued to letters AND not part of a wider
    `_word_` pattern. The trailing `_123` isn't a marker boundary."""
    assert sanitize_tts_text("customer_id_123") == "customer_id_123"


def test_does_not_strip_id_with_at_sign():
    """`ABC_123` — underscore between letters and digits, no closing
    marker. Must stay unchanged."""
    assert sanitize_tts_text("ABC_123") == "ABC_123"


def test_does_not_strip_lone_asterisk():
    """`* foo *` (whitespace inside the markers) — NOT Markdown
    italic. The sanitizer leaves it alone because the regex
    requires the marker glued to the content (`*foo*`, no
    whitespace)."""
    # Lone asterisks with whitespace inside stay.
    assert sanitize_tts_text("* hola *") == "* hola *"


def test_strips_markdown_headers_at_line_start():
    """`# Header` and `## Subheader` at line start are headers.
    Mid-string `#` (like `Ticket #1234`) stays."""
    assert sanitize_tts_text("# Bienvenidos") == "Bienvenidos"
    assert sanitize_tts_text("## Subtítulo") == "Subtítulo"
    assert sanitize_tts_text("Ticket #1234 confirmado") == "Ticket #1234 confirmado"


def test_strips_inline_code_backticks():
    """`code` (paired backticks) → `code`."""
    assert sanitize_tts_text("Hola `código` mundo") == "Hola código mundo"


def test_strips_strikethrough():
    """~~strikethrough~~ → strikethrough."""
    assert sanitize_tts_text("~~viejo~~") == "viejo"


# ── Non-verbal allowlist + cap ────────────────────────────────────


def test_allows_breathe_and_sigh():
    """The allowlist keeps `[breathe]` and `[sigh]` in the text.
    These are the only ones the Laboratorio C.G.O. agent should
    use; the prompt guides the model toward them."""
    assert sanitize_tts_text("[breathe] Hola") == "[breathe] Hola"
    assert sanitize_tts_text("Hola [sigh]") == "Hola [sigh]"


def test_drops_other_nonverbals():
    """`[laugh]`, `[cough]`, `[yawn]`, `[clear throat]` — NOT in
    the allowlist. They're dropped silently because the prompt
    tells the LLM never to emit them for a customer-service
    voice. If one slips through, the sanitizer drops it instead
    of letting Inworld read the literal bracket."""
    assert sanitize_tts_text("[laugh] hola [cough]") == " hola "
    assert sanitize_tts_text("[yawn] hola") == " hola"
    assert sanitize_tts_text("[clear throat] hola") == " hola"


def test_caps_nonverbals_at_two():
    """Hard cap of 2 — emergency guard against runaway outputs.
    `[breathe] a [sigh] b [breathe] c` keeps the first two,
    drops the rest."""
    out = sanitize_tts_text("[breathe] a [sigh] b [breathe] c")
    assert out.count("[breathe]") + out.count("[sigh]") == 2, (
        f"cap not enforced — got {out!r}"
    )


# ── Markup preservation ────────────────────────────────────────────


def test_preserves_break_tags():
    """`<break time="300ms" />` must round-trip unchanged. The
    sanitizer doesn't touch `<...>` at all."""
    text = "De acuerdo. <break time=\"300ms\" /> El check-up."
    assert sanitize_tts_text(text) == text


def test_preserves_steering_tags():
    """`[speak calmly, professionally.]` — steering tags are
    allowlist items (in the allowlist we treat them the same way
    as `[breathe]` / `[sigh]`). Round-trip unchanged."""
    text = "[speak calmly, professionally.] Hola."
    assert sanitize_tts_text(text) == text


def test_preserves_combined_markup():
    """Multiple tags + Markdown in one reply. Only the unsafe
    pieces get stripped, the markup stays."""
    text = (
        "**Buenas tardes**."
        "<break time=\"250ms\" />"
        "[breathe] ¿Cuál es el *precio* del estudio?"
    )
    out = sanitize_tts_text(text)
    assert "<break" in out
    assert "[breathe]" in out
    assert "precio" in out
    assert "**" not in out
    assert "*" not in out


# ── Truncation safety net ──────────────────────────────────────────


def test_truncates_overlong_text():
    """Defensive: cap at max_len (default 1500). Long runaway
    responses get truncated to fit Inworld's per-call budget."""
    text = "x" * 2000
    assert len(sanitize_tts_text(text)) == 1500


def test_short_text_passes_through():
    """Default cap is generous — short replies untouched."""
    assert sanitize_tts_text("Hola") == "Hola"
    assert sanitize_tts_text("") == ""
