"""El transfer por chat-streaming llegaba muerto al executor.

Causa raíz (logs CAcaa3f780580eb0c5f268ab7210f5f1ac, agente AI-first
con humano segundo): el modelo SÍ emitía el call_transfer, pero
stream_llm_reply_sync solo devolvía los tool calls
`if tool_calls and execute_tool_callback` — y turn_manager jamás
pasa ese callback. Resultado: la IA decía "te transfiero ahora
mismo" y nunca marcaba. Y cuando el modelo emitía SOLO el tool call
(sin texto, completion=10), el reply vacío dejaba
assistant_speaking=True clavado ~30 s hasta el watchdog: VAD + idle
congelados, "la llamada nunca le llega a la IA".
"""
from STT_server.adapters.openai_llm import stream_llm_reply_sync
from STT_server.domain.session import CallSession
from STT_server.services import turn_manager


class _FakeFn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _FakeTc:
    def __init__(self, index, id="", fn=None):
        self.index = index
        self.id = id
        self.function = fn


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, delta=None):
        self.usage = None
        self.choices = [_FakeChoice(delta)] if delta is not None else []


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    def create(self, **kw):
        return list(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(chunks)})()


def _tool_chunks():
    return [
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeTc(0, id="call_1", fn=_FakeFn("transferir_rrhh", '{"destino":"+15550001111"}')),
        ])),
    ]


def test_streaming_returns_tool_calls_without_callback():
    reply, err, tool_calls = stream_llm_reply_sync(
        [], lambda: False, lambda s: None, lambda s=None: None, lambda: None,
        _FakeClient(_tool_chunks()),
        provider="openai",
        tools=[{"type": "function", "function": {"name": "transferir_rrhh"}}],
        session=None,  # igual que turn_manager: sin execute_tool_callback
    )
    assert err is None
    assert tool_calls == [{
        "id": "call_1", "name": "transferir_rrhh",
        "arguments": {"destino": "+15550001111"},
    }]


async def test_empty_reply_releases_stuck_speaking_flag(monkeypatch):
    async def fake_stream(session, user_text, generation):
        return ("", 1.0, [], None)

    monkeypatch.setattr(turn_manager, "stream_llm_reply_with_tts", fake_stream)
    session = CallSession(session_key="empty-reply", active_generation=1)
    await turn_manager.handle_agent_reply(session, "hola", 1, trigger="final")
    assert session.assistant_speaking is False
    assert session.assistant_started_at is None


async def test_stale_generation_does_not_touch_new_turn_flag(monkeypatch):
    async def fake_stream(session, user_text, generation):
        return ("", 1.0, [], None)

    monkeypatch.setattr(turn_manager, "stream_llm_reply_with_tts", fake_stream)
    session = CallSession(session_key="stale-gen", active_generation=2)
    session.assistant_speaking = True  # dueño: el turno nuevo (gen 2)
    await turn_manager.handle_agent_reply(session, "hola", 1, trigger="final")
    assert session.assistant_speaking is True
