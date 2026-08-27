"""Tests for the markup-aware streaming segmenter.

Regression guard for the 2026-08-26 TTS markup bug. The previous
segmenter cut at `.!?\\n` anywhere in the buffer, including inside
Inworld markup tags. With the new TTS hint asking the LLM to emit
`<break ... />` and `[breathe]` inline, every reply risks orphaning
a tag at the boundary:

  - `[sigh.] ¿Cómo está?`    cut at `.` inside the tag → tag
                              orphaned, sigh silently dropped
  - `<break time="300ms">`   cut mid-tag → Inworld rejects the
    malformed XML on the next round-trip

The fix is a state-machine segmenter that PROTECTS markup (punctuation
inside `<...>` and `[...]` is invisible) without auto-cutting at
markup boundaries (which would create artificial segments like
`[sigh.]` alone).
"""
from STT_server.domain.language import pop_streaming_segments


def test_square_tag_with_period_inside_is_not_split():
    """The original bug: `[sigh.]` has a `.` inside square brackets.
    The previous segmenter cut at that period and left the next
    segment starting with `]`. Inworld discarded the broken tag
    silently. The state machine sees the `.` inside INSIDE_SQUARE
    and waits for the `]` before resuming punctuation cuts.

    `force=True` flushes whatever the segmenter couldn't split on
    yet — equivalent to the response.done final-flush in
    openai_realtime.py.
    """
    buffer = "[sigh.] ¿Cómo está?"
    segments, remainder = pop_streaming_segments(buffer, force=True)
    assert remainder == "", f"expected empty remainder, got {remainder!r}"
    assert len(segments) == 1, f"expected 1 segment, got {len(segments)}: {segments!r}"
    assert segments[0] == "[sigh.] ¿Cómo está?", (
        f"tag must stay intact. Got: {segments[0]!r}"
    )


def test_angle_tag_with_period_inside_is_not_split():
    """Same shape for `<break time="300ms"> Hola.` — the `.` after
    `Hola` is a valid cut, but only because by the time we get
    there we're back in NORMAL state."""
    buffer = "<break time=\"300ms\" /> Hola. ¿Cómo? Bien."
    # force=True so the short buffer flushes as one segment (the
    # streaming path waits for more chunks; we don't want the test
    # to wait for those).
    segments, remainder = pop_streaming_segments(buffer, force=True)
    assert remainder == ""
    assert len(segments) == 1, f"expected 1 segment (no cut yet), got: {segments!r}"
    assert segments[0] == "<break time=\"300ms\" /> Hola. ¿Cómo? Bien."


def test_long_reply_with_break_tag_cuts_at_real_punctuation():
    """Long reply (>= min_punct=100) cuts at the FIRST real
    punctuation in NORMAL state, not at the period after `Hola`
    inside the markup. Verifies the state machine resumes NORMAL
    after the closing `>`."""
    buffer = (
        "Claro que sí, con mucho gusto le ayudo. "                    # ~46 chars
        "<break time=\"300ms\" /> "                                  # ~26 chars -> total ~72
        "El estudio requiere entre ocho y doce horas de ayuno. "     # ~55 chars -> total ~127
        "Puede tomar agua natural sin problema."                      # ~38 chars -> total ~165
    )
    # First segment should cut at the `.` after `estudio` (around
    # char 127) because we're back in NORMAL state by then. The break
    # tag stays in the first segment.
    segments, remainder = pop_streaming_segments(buffer)
    assert len(segments) >= 1, f"expected >=1 segment, got: {segments!r}"
    # The first segment must contain the break tag intact.
    assert "<break" in segments[0]
    assert segments[0].endswith("."), (
        f"first cut should land on a real sentence end. Got: {segments[0]!r}"
    )


def test_unclosed_square_bracket_does_not_deadlock():
    """Defensive: malformed LLM output like "texto con [tag incompleto"
    would deadlock the segmenter waiting for `]`. The state machine
    force-exits INSIDE_SQUARE back to NORMAL after `_MARKUP_MAX_CHARS`
    characters, so the segmenter keeps making progress on the rest."""
    buffer = "texto con [tag incompleto más texto sin cierre"
    segments, remainder = pop_streaming_segments(buffer, force=True)
    # The buffer is force-flushed as one segment (no punctuation
    # boundary reachable inside because of the open bracket). The
    # hard requirement is "no hang" — the segmenter must return.
    assert len(segments) == 1 or remainder != "", (
        f"segmenter must return something. Got segments={segments!r} remainder={remainder!r}"
    )
    # Whatever was returned must be a string.
    for seg in segments:
        assert isinstance(seg, str)


def test_unclosed_angle_bracket_does_not_deadlock():
    """Same defensive cap for `<` without a matching `>`. An LLM
    emitting `<break incompleto` (no closing `/>`) used to lock
    the segmenter waiting for `>` that never comes. Now we
    force-exit after `_MARKUP_MAX_CHARS`."""
    buffer = "texto con <break incompleto más texto sin cierre."
    segments, remainder = pop_streaming_segments(buffer, force=True)
    # After force-flush the buffer must come back as one segment
    # (no punctuation boundary reached before force-exit triggered
    # the rescue back to NORMAL). The hard requirement: no hang.
    assert len(segments) == 1, f"expected 1 segment after force-flush, got: {segments!r}"
    assert segments[0].startswith("texto con <break incompleto"), (
        f"force-flushed segment should contain the entire buffer. "
        f"Got: {segments[0]!r}"
    )


def test_steering_with_comma_inside_keeps_comma_protected():
    """TTS-2 steering tags like `[speak calmly, professionally.]`
    contain a comma. The state machine protects the comma inside
    the brackets — punctuation is invisible there. After the `]`
    we're back in NORMAL and the next `.!?` cuts normally."""
    buffer = (
        "[speak calmly, professionally.] "                # 32 chars
        "Claro que sí. "                                  # 15 chars -> total ~47
        "Con mucho gusto le ayudo con eso."               # 31 chars -> total ~78
    )
    # Short buffer — force-flush as one segment.
    segments, remainder = pop_streaming_segments(buffer, force=True)
    assert remainder == ""
    assert len(segments) == 1
    # The comma inside the brackets must NOT have triggered any cut.
    assert segments[0] == buffer
