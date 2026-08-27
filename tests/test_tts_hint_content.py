"""Guard tests for the Inworld TTS hint.

The hint is a plain string prepended to the agent's system prompt.
There's no behavior to test in isolation — the LLM reads the
string and decides what to emit. The tests below just pin the
content so a future refactor can't silently drop critical
guidance (the "[breathe]" mention, the explicit "write digits
in spoken form for IDs" rule, the "never [laugh]/[cough]"
list, etc.).
"""
from STT_server.domain.tts_hints import (
    INWORLD_STEERING_INSTRUCTIONS,
    TTS_PROVIDER_HINTS,
    _HINT_MARKER,
    get_tts_hint,
    has_tts_hint,
)


# ── Structural sanity ────────────────────────────────────────────


def test_inworld_hint_is_registered():
    """The Inworld hint must be reachable via get_tts_hint('inworld').
    Anything else (capitalization, whitespace) silently breaks the
    hint lookup at STT_Server.py bootstrap time."""
    assert get_tts_hint("inworld") is INWORLD_STEERING_INSTRUCTIONS
    # Case-insensitive lookup per the function's contract.
    assert get_tts_hint("Inworld") is INWORLD_STEERING_INSTRUCTIONS
    assert get_tts_hint("INWORLD") is INWORLD_STEERING_INSTRUCTIONS


def test_unknown_provider_returns_none():
    """Adding a hint for a new provider requires updating this test
    — the absence is intentional until that provider lands."""
    assert get_tts_hint("elevenlabs") is None
    assert get_tts_hint("openai") is None
    assert get_tts_hint(None) is None


def test_marker_detects_double_prepend():
    """STT_Server.py uses has_tts_hint() to detect when the bootstrap
    ran twice (WS reconnect) and avoid prepending the hint on top of
    itself. The marker must appear in the hint text."""
    assert _HINT_MARKER in INWORLD_STEERING_INSTRUCTIONS
    assert has_tts_hint(INWORLD_STEERING_INSTRUCTIONS) is True
    assert has_tts_hint("") is False
    assert has_tts_hint(None) is False


# ── Content drift guards ────────────────────────────────────────


def test_hint_lists_supported_pauses():
    """`<break time="..."/>` is the markup for pauses. The hint
    must show the operator which durations are sensible."""
    # At least one break duration must be mentioned. The hint was
    # tightened to 200ms/350ms + 250ms example in the 2026-08-26
    # rewrite — pin that at least one is present so a future refactor
    # can't drop the pause guidance entirely.
    assert "<break" in INWORLD_STEERING_INSTRUCTIONS
    assert "ms" in INWORLD_STEERING_INSTRUCTIONS


def test_hint_prohibits_high_variance_nonverbals():
    """For a customer-service voice, the only allowed non-verbals
    are [breathe] and [sigh]. Anything theatrical ([laugh],
    [cough], [yawn], [clear throat]) must be explicitly
    prohibited."""
    for prohibited in ("[laugh]", "[cough]", "[yawn]", "[clear throat]"):
        # The hint must mention the prohibited tag in a NEVER USE
        # context. We check that the line containing the tag says
        # NEVER (case-insensitive).
        for line in INWORLD_STEERING_INSTRUCTIONS.splitlines():
            if prohibited in line:
                assert "never" in line.lower(), (
                    f"line containing {prohibited!r} must say NEVER: "
                    f"{line!r}"
                )
                break
        else:
            pytest.fail(f"hint doesn't mention prohibited tag {prohibited!r}")


def test_hint_mentions_breathe_and_sigh_as_allowed():
    """The allowed non-verbals for the Laboratorio agent are
    `[breathe]` and `[sigh]`. The hint must list them as allowed."""
    assert "[breathe]" in INWORLD_STEERING_INSTRUCTIONS
    assert "[sigh]" in INWORLD_STEERING_INSTRUCTIONS


def test_hint_limits_steering_to_one_per_response():
    """The whole point of inline steering is that it stays
    invisible. Starting every reply with `[speak calmly,
    professionally]` becomes robotic after the second turn. The
    hint must enforce the "use at most one" rule."""
    assert "ONE" in INWORLD_STEERING_INSTRUCTIONS.upper() and (
        "at most one" in INWORLD_STEERING_INSTRUCTIONS.lower()
        or "at most one per response" in INWORLD_STEERING_INSTRUCTIONS.lower()
    ), (
        "hint must enforce 'at most one steering per response'"
    )


def test_hint_prohibits_starting_every_reply_with_steering():
    """This is the specific failure mode the operator was seeing
    on real calls. The hint must explicitly tell the model NOT to
    lead every reply with a steering tag."""
    low = INWORLD_STEERING_INSTRUCTIONS.lower()
    assert "never start every response" in low or "do not start every response" in low


def test_hint_explicit_ids_digits_spoken_form():
    """This is the operator's correction from 2026-08-26. The previous
    version said "order 451086" reads naturally — wrong, Inworld
    may normalize it. For identifiers where every digit matters,
    the LLM must write digits explicitly."""
    low = INWORLD_STEERING_INSTRUCTIONS.lower()
    assert "spoken form" in low or "cinco, seis" in low or "five" in low, (
        "hint must teach the LLM to spell out digits for IDs"
    )


def test_hint_rejects_markdown():
    """Markdown emphasis gets read aloud literally by Inworld.
    The hint must explicitly say NEVER use Markdown emphasis."""
    assert "Markdown" in INWORLD_STEERING_INSTRUCTIONS
    low = INWORLD_STEERING_INSTRUCTIONS.lower()
    assert "never" in low and "markdown" in low


def test_hint_default_delivery_mentions_calm_and_warm():
    """The Laboratorio agent's baseline is `calm, warm,
    professional, conversational` — set explicitly in the hint so
    the model has a default voice when no steering is needed."""
    assert "calm" in INWORLD_STEERING_INSTRUCTIONS
    assert "warm" in INWORLD_STEERING_INSTRUCTIONS
    assert "professional" in INWORLD_STEERING_INSTRUCTIONS
