"""Tests for deterministic break insertion (Speech Formatter)."""
from STT_server.domain.language import format_for_tts


def test_no_break_for_single_sentence():
    assert format_for_tts("De acuerdo.") == "De acuerdo."
    assert format_for_tts("¿Qué día le gustaría acudir?") == "¿Qué día le gustaría acudir?"
    assert format_for_tts("Hola, ¿cómo estás?") == "Hola, ¿cómo estás?"


def test_inserts_break_for_two_sentences():
    text = "El estudio requiere ayuno. ¿Desea agendarlo?"
    out = format_for_tts(text)
    assert '<break time="250ms" />' in out
    assert out.count("<break") == 1


def test_inserts_break_for_three_sentences():
    text = "Con gusto. El perfil de lípidos mide colesterol. Requiere ayuno."
    out = format_for_tts(text)
    assert out.count("<break") == 1
    # Break after first sentence
    assert out.startswith("Con gusto. <break")


def test_does_not_duplicate_existing_break():
    text = "Hola. <break time=\"200ms\" /> Mundo. ¿Cómo estás?"
    assert format_for_tts(text) == text
    text2 = "Hola <break time=\"300ms\" /> mundo. Otra oración."
    assert format_for_tts(text2) == text2


def test_empty_and_single_word():
    assert format_for_tts("") == ""
    assert format_for_tts("Hola") == "Hola"
    assert format_for_tts("Hola.") == "Hola."


def test_preserves_nonverbals():
    text = "Entiendo. [breathe] Le explico las opciones."
    out = format_for_tts(text)
    assert "[breathe]" in out
    assert "<break" in out
