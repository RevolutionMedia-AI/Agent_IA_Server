"""Cuelgue tras despedida: el agente cierra, la llamada muere por REST.

Sin esto la llamada solo moría por idle-timeout (45 s de aire muerto)
o max-duration (30 min). Solo la respuesta del AGENTE dispara el
cuelgue — un "gracias" suelto no es un cierre.
"""
import time

import pytest

from STT_server.adapters import twilio_api
from STT_server.domain.language import is_farewell_closing
from STT_server.domain.session import CallSession
from STT_server.services.turn_manager import _hangup_after_farewell


@pytest.mark.parametrize("text", [
    "¡Adiós! Gracias por llamar.",
    "Adios, que tengas un excelente dia.",
    "Que tenga un buen día, estamos en contacto.",
    "Fue un placer atenderle. ¡Hasta luego!",
    "Goodbye, thanks for calling.",
    "Bye! Have a great day.",
])
def test_farewell_closing_detects_agent_goodbye(text):
    assert is_farewell_closing(text) is True


@pytest.mark.parametrize("text", [
    "",
    "Gracias.",
    "Buenas tardes, ¿en qué puedo ayudarle?",
    "¿Te gustaría proceder de esa manera?",
    "Necesitaría tu nombre completo y tu correo electrónico.",
    "Baby, cuéntame más del puesto.",  # "bye" dentro de otra palabra no cuelga
])
def test_farewell_closing_ignores_mid_call_text(text):
    assert is_farewell_closing(text) is False


def _session(**over) -> CallSession:
    s = CallSession(session_key="farewell-test", active_generation=1)
    s.twilio_account_sid = "AC123"
    s.twilio_auth_token = "tok"
    s.call_sid = "CA123"
    s.last_activity_at = time.monotonic() - 100
    for k, v in over.items():
        setattr(s, k, v)
    return s


async def test_hangup_after_farewell_calls_rest(monkeypatch):
    calls = []

    async def fake_hangup(sid, tok, call_sid):
        calls.append((sid, tok, call_sid))
        return {"success": True, "call_sid": call_sid}

    monkeypatch.setattr(twilio_api, "hangup_call", fake_hangup)
    await _hangup_after_farewell(_session(), 1)
    assert calls == [("AC123", "tok", "CA123")]


async def test_hangup_after_farewell_aborts_if_user_spoke(monkeypatch):
    async def fake_hangup(*a):
        raise AssertionError("must not hang up while the user is talking")

    monkeypatch.setattr(twilio_api, "hangup_call", fake_hangup)
    s = _session()
    s.last_activity_at = time.monotonic() + 100  # voz posterior a la despedida
    await _hangup_after_farewell(s, 1)


async def test_hangup_after_farewell_aborts_on_new_turn(monkeypatch):
    async def fake_hangup(*a):
        raise AssertionError("must not hang up a new turn")

    monkeypatch.setattr(twilio_api, "hangup_call", fake_hangup)
    await _hangup_after_farewell(_session(active_generation=2), 1)
