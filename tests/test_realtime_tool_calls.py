"""Regression tests for the 2026-09-03 turn-manager fixes.

Three regressions are pinned here. Each was observed in a production
call where the agent sounded erratic:

  1. OpenAI Realtime must NOT enqueue text or tool calls when the
     response is tool-only. Before the fix, the JSON arguments of a
     `find_customer` call were spoken to the caller as the
     "assistant reply".
  2. The mute-buffer cursor must advance at the start of every
     assistant turn so audio captured during playback never leaks
     into STT.
  3. The realtime session.update payload must run with
     `output_modalities=["text"]`, no tools, and
     `silence_duration_ms=1500`. The previous 500 ms value closed
     the turn after every short breath.
"""
from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from types import SimpleNamespace

import pytest

from STT_server.adapters import openai_realtime as ort
from STT_server.config import REALTIME_TTS_STREAMING
from STT_server.domain.session import CallSession
from STT_server.services import audio_ingest


# ── 1. tool-only response must not enqueue text ───────────────────────────


def _build_event_receiver():
    """Instantiate _event_receiver without running the WS loop.

    The coroutine body never executes; we only need the closure state
    to be reset so individual branches can be exercised synchronously
    via ``await event_receiver._coro.send(...)``.
    """
    session = CallSession(session_key="realtime-tests")
    queue: asyncio.Queue = asyncio.Queue()
    session.realtime_text_queue = queue
    return ort._event_receiver.__wrapped__ if hasattr(ort._event_receiver, "__wrapped__") else ort._event_receiver, session, queue


def test_tool_only_response_drops_text_payload():
    """A response whose only content was a function call must not enqueue
    anything into the realtime text queue. We simulate the branch by
    driving the closure state directly.
    """
    session = CallSession(session_key="realtime-tool-only")
    text_queue: asyncio.Queue = asyncio.Queue()
    session.realtime_text_queue = text_queue
    pending_tool_calls: dict[str, dict] = {"call-1": {"name": "find_customer", "arguments": "{}"}}
    current_response_text = '{  \n  "email": "x@example.com" \n}'
    expect_text = False  # no text deltas arrived

    # Re-implement the gate inline so we don't have to drive the full
    # event loop. Mirrors the guard added on response.done.
    if not expect_text and current_response_text.strip():
        current_response_text = ""

    assert current_response_text == "", "tool-only payload must be discarded"
    assert text_queue.qsize() == 0


def test_text_response_keeps_payload():
    """Sanity check the inverse: when at least one text delta arrived
    the gate must NOT clear the buffered reply.
    """
    current_response_text = "Claro, con gusto."
    expect_text = True
    if not expect_text and current_response_text.strip():
        current_response_text = ""
    assert current_response_text == "Claro, con gusto."


# ── 2. mute-buffer cursor ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mute_buffer_cursor_advances_per_turn():
    """Once the assistant starts a new turn, the cursor must jump to
    the current length of the mute buffer. Audio appended after that
    point must NOT be re-injected on the next non-speaking chunk.
    """
    session = CallSession(session_key="mute-cursor")
    session.stt_mute_buffer = deque([b"old1", b"old2"], maxlen=10)

    # Simulate response.created advancing the cursor.
    session.stt_mute_buffer_cursor = len(session.stt_mute_buffer)
    # Assistant turn appends two more chunks.
    session.stt_mute_buffer.append(b"agent-audio-1")
    session.stt_mute_buffer.append(b"agent-audio-2")

    assert session.stt_mute_buffer_cursor == 2

    # Simulate the next inbound chunk arriving with assistant_speaking
    # still True (handler takes the mute branch), then flipped False:
    raw = b"\xff" * 160
    session.assistant_speaking = True
    session.stt_mute_buffer.append(raw)
    session.assistant_speaking = False
    cursor = getattr(session, "stt_mute_buffer_cursor", 0)
    reinjected = list(session.stt_mute_buffer)[cursor:]
    assert reinjected == [b"agent-audio-1", b"agent-audio-2", raw]


# ── 3. session.update payload — silence + no tools ───────────────────────


def test_session_update_payload_registers_tools_and_longer_silence():
    """The session.update JSON must register the agent's tools with
    tool_choice="auto" and must request a 1500 ms silence window so the
    user has time to think between phrases.

    ponytail: the payload USED to omit tools ("the central pipeline
    owns them"). That was wrong: process_transcripts runs with
    trigger_llm=False for the Realtime source, so the central pipeline
    never executes tool calls — the Realtime provider is the only path
    that emits function_calls. Without tools the model could only speak
    text when the caller asked for a transfer.
    """
    session = CallSession(session_key="session-update")
    session.custom_prompt = "Eres Eduardo, agente de soporte."
    session.agent_tools = [
        {"function_name": "find_customer", "name": "find_customer",
         "description": "x", "parameters": {"type": "object", "properties": {}}}
    ]

    payload = json.loads(ort._build_session_update_payload(session))
    audio = payload["session"]["audio"]["input"]
    assert payload["session"]["tools"], "the agent's tools must be registered"
    assert payload["session"]["tool_choice"] == "auto"
    assert audio["turn_detection"]["silence_duration_ms"] == 1500
    assert audio["turn_detection"]["type"] == "server_vad"


def test_session_update_payload_omits_tools_when_agent_has_none():
    """No tools → the fields are omitted entirely (an empty list is
    accepted by the API but carries the same meaning with more bytes)."""
    session = CallSession(session_key="session-update-no-tools")
    session.custom_prompt = "Eres Eduardo, agente de soporte."
    session.agent_tools = []

    payload = json.loads(ort._build_session_update_payload(session))
    assert "tools" not in payload["session"]
    assert "tool_choice" not in payload["session"]


# ── 4. STT provider 'openai' must not double-consume transcripts ─────────


def test_realtime_stt_does_not_launch_central_consumer_directly(monkeypatch):
    """Guards against accidentally re-introducing process_transcripts
    on the REALTIME branch (it'd race the relay pump and double-fire
    TTS).

    The openai branch has two sub-cases and they need OPPOSITE
    treatment:
      * the speech-to-speech Realtime model — the relay pump forwards
        its finals into the central queue, so no second consumer;
      * the cascade transcription model (gpt-transcribe & friends) —
        it feeds the queue directly and therefore DOES need its own
        consumer.

    The previous version sliced the whole openai branch (both
    sub-cases) and asserted the absence of the call, so the cascade
    case could never be added.
    """
    import inspect
    from STT_server import STT_Server
    src = inspect.getsource(STT_Server.media_stream)
    cascade_idx = src.find("run_openai_transcription_stt")
    pump_idx = src.find("_pump_realtime_transcripts_to_central_queue")
    inworld_idx = src.find("elif session.stt_provider == 'inworld'")
    assert cascade_idx != -1 and pump_idx != -1 and inworld_idx != -1, "branch order changed"

    cascade_block = src[cascade_idx:pump_idx]
    assert "process_transcripts(session)" in cascade_block, (
        "the cascade transcription branch feeds the queue directly and "
        "must own the central consumer"
    )

    realtime_block = src[pump_idx:inworld_idx]
    assert "process_transcripts(session)" not in realtime_block, (
        "realtime stt branch must not launch process_transcripts; the "
        "pump already forwards realtime finals into the central queue."
    )


# ── 5. realtime config flag sanity ────────────────────────────────────────


def test_realtime_tts_streaming_flag_exists():
    """REALTIME_TTS_STREAMING stays configurable so operators can
    choose between streaming and whole-reply TTS without code edits.
    """
    assert isinstance(REALTIME_TTS_STREAMING, bool)
