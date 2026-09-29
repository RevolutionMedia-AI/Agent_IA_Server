"""The post-handoff resume copy must speak the agent's language.

Production complaint: an agent whose prompt, greeting and TTS voice were all
Spanish came back from a failed handoff chain saying "Sorry, nobody answered
the transfer. How else can I help you?" — in English, out of a Spanish call.

Root cause chain, all three parts of which are asserted below:
  1. the agents.language column shipped with DEFAULT 'English' and the API
     schema defaulted to the same, storing display words where every consumer
     wants a canonical 'en'/'es' code;
  2. nothing normalized between the two, so session.preferred_language became
     the string 'english';
  3. the resume copy branched on that string, and 'english' starts with 'en'.
"""
import pytest

import STT_server.STT_Server as srv
from STT_server.routes.api import AgentCreate, AgentUpdate


# ── the copy itself ────────────────────────────────────────────────────────

@pytest.mark.parametrize("lang", ["es", "Spanish", "es-MX", "es-419", None, ""])
def test_resume_copy_is_spanish_for_spanish_and_unset(lang):
    """Unset must be Spanish: the platform default is es, so a missing
    language must never fall through to the English branch."""
    welcome, note = srv._resume_copy(lang)
    assert "nadie contestó" in welcome
    assert "Spanish" not in welcome and "nobody answered" not in welcome
    assert "cliente" in note


@pytest.mark.parametrize("lang", ["en", "English", "en-US", "en-GB"])
def test_resume_copy_is_english_for_english(lang):
    welcome, note = srv._resume_copy(lang)
    assert "nobody answered" in welcome
    assert "caller" in note


def test_resume_copy_pair_is_consistent():
    """Both halves come from the same branch. A caller that hears Spanish
    while the LLM is instructed in English (or vice versa) is the exact
    failure being fixed."""
    for lang in ("en", "es"):
        welcome, note = srv._resume_copy(lang)
        if lang == "es":
            assert "nadie contestó" in welcome and "nadie contestó" in note
        else:
            assert "nobody answered" in welcome and "nobody answered" in note


# ── the write path: the API validator ──────────────────────────────────────
#
# These expectations are DISPLAY WORDS, not codes, and that is the whole
# lesson: agents.language is a word enum guarded by agents_language_check
# (see test_agent_language_column_contract.py). A first attempt asserted
# 'en' / 'es' here and every PUT /agents/{id} 500'd in production. The codes
# are still accepted on input; the column gets the word.

@pytest.mark.parametrize("raw,expected", [
    ("en", "English"), ("English", "English"), ("ENGLISH", "English"),
    ("en-US", "English"), ("en-GB", "English"),
    ("es", "Spanish"), ("Spanish", "Spanish"), ("es-MX", "Spanish"),
    ("es-419", "Spanish"), ("  Spanish  ", "Spanish"),
    (None, None), ("", None),
])
def test_agent_language_is_canonicalized_on_write(raw, expected):
    """Both schemas. Codes and display words both go in; the CHECK's word
    comes out."""
    assert AgentCreate(name="a", language=raw).language == expected
    assert AgentUpdate(language=raw).language == expected


@pytest.mark.parametrize("bad", ["fr", "klingon", "e", "english please"])
def test_agent_language_rejects_junk_on_write(bad):
    """A typo cannot silently become Spanish at 3am."""
    with pytest.raises(ValueError):
        AgentCreate(name="a", language=bad)
    with pytest.raises(ValueError):
        AgentUpdate(language=bad)


def test_bilingual_still_loads_because_the_check_requires_it():
    """Not offered in the UI — one call has one TTS language and the resume
    copy has two branches. But agents_language_check lists it, so a legacy
    row holding it must still save instead of 500ing the whole agent."""
    assert AgentUpdate(language="Bilingual").language == "Bilingual"


def test_agent_create_default_is_a_value_the_check_allows():
    assert AgentCreate(name="a").language == "English"


# ── the read path: normalization onto the session ──────────────────────────

def test_legacy_display_word_normalizes_to_a_code():
    """A row written before the validator still reads as a valid code, so
    Inworld / OpenAI Realtime never receive language='english'."""
    from STT_server.domain.language import normalize_supported_language
    assert normalize_supported_language("English") == "en"
    assert normalize_supported_language("Spanish") == "es"


# ── the operator's own sentence (migration 025) ─────────────────────────────

def test_operator_message_is_spoken_verbatim():
    """The whole point of the field: the operator's words, unchanged."""
    line = "Perdón, ya no hay nadie en la oficina. ¿Le dejo su mensaje?"
    welcome, note = srv._resume_copy("es", line)
    assert welcome == line
    assert line in note


def test_operator_message_reaches_the_note_in_the_right_language():
    """The note is what the live model actually acts on — welcome_message
    is not spoken on this path. An English note with a Spanish sentence
    makes the model paraphrase back into English."""
    line = "Perdón, ya no hay nadie en la oficina."
    _, note_es = srv._resume_copy("es", line)
    _, note_en = srv._resume_copy("en", line)
    assert "exactamente esto" in note_es
    assert "word for word" in note_en
    # both still forbid the re-transfer loop
    assert "NO invoques" in note_es
    assert "Do NOT" in note_en


@pytest.mark.parametrize("blank", [None, "", "   ", "\n\t "])
def test_blank_operator_message_falls_back_to_the_built_in_copy(blank):
    """Empty means 'use the default in my language', not 'say nothing'."""
    assert srv._resume_copy("es", blank)[0].startswith("Disculpa")
    assert srv._resume_copy("en", blank)[0].startswith("Sorry")


def test_the_field_accepts_and_caps_the_message():
    assert AgentCreate(name="a", transfer_unavailable_message="hola").transfer_unavailable_message == "hola"
    assert AgentUpdate(transfer_unavailable_message="hola").transfer_unavailable_message == "hola"
    with pytest.raises(Exception):
        AgentCreate(name="a", transfer_unavailable_message="x" * 1001)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
