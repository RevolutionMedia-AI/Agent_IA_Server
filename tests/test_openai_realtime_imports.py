"""Regression tests for the openai_realtime adapter.

These are static-import / source-level checks. The realtime session
is async + websocket-heavy and would need a full mock harness to
exercise end-to-end; what we're catching here is the kind of bug
that bites silently in production until a session runs long
enough to hit the affected branch:

  UnboundLocalError: cannot access local variable
  'enqueue_nowait_with_drop' where it is not associated with a value

A local `from ... import` inside `_event_receiver` makes the name
local throughout the function via Python's scoping rules, so any
earlier reference (barge-in cleanup, the finally block) raises
UnboundLocalError. The fix was to keep the import at module top —
this test pins that.
"""
import datetime
import inspect
import json
import re
from unittest.mock import patch


def test_enqueue_helper_imported_exactly_once_at_module_top():
    """openai_realtime.py must import enqueue_nowait_with_drop
    exactly once (at module top). A local re-import inside a
    function would shadow the module symbol via Python scoping and
    break _event_receiver with UnboundLocalError. The error is silent
    until a realtime session actually reaches the affected branch
    — which is exactly what happens when the operator talks after
    the initial greeting.
    """
    from STT_server.adapters import openai_realtime

    src = inspect.getsource(openai_realtime)
    needle = "from STT_server.services.common import enqueue_nowait_with_drop"
    count = src.count(needle)
    assert count == 1, (
        f"openai_realtime.py must import enqueue_nowait_with_drop "
        f"exactly once (at module top). Found {count} occurrences — "
        f"a local re-import would shadow the module symbol via "
        f"Python scoping and break _event_receiver with UnboundLocalError "
        f"on the first user turn after the welcome greeting."
    )


def test_event_receiver_has_no_local_enqueue_imports():
    """Pin: there must be NO `from ... import enqueue_nowait_with_drop`
    statements inside _event_receiver's source. The function uses the
    helper at multiple branch points (transcript push, streaming
    segments, end-of-stream, finally cleanup) and a local import would
    make earlier branches fail before the import ever runs."""
    from STT_server.adapters import openai_realtime

    src = inspect.getsource(openai_realtime._event_receiver)
    needle = "from STT_server.services.common import enqueue_nowait_with_drop"
    assert needle not in src, (
        "_event_receiver contains a local `from ... import` of "
        "enqueue_nowait_with_drop. That import statement makes the "
        "symbol local to the whole function via Python scoping, so "
        "any reference before the import runs raises UnboundLocalError."
    )


def test_event_receiver_no_any_local_common_imports():
    """Belt-and-suspenders: scan the whole file for any local
    re-import of the common helpers we depend on at module top. The
    local-import bug class is broader than just enqueue_nowait_with_drop.
    """
    from STT_server.adapters import openai_realtime

    src = inspect.getsource(openai_realtime)
    # Module-level import lines (start of file) are fine. Look for
    # any extra `from STT_server.services.common import` lines that
    # aren't at column 0 (i.e. inside a function).
    lines = src.splitlines()
    in_function_imports = [
        (i + 1, line) for i, line in enumerate(lines)
        if re.match(r"\s+from STT_server\.services\.common import", line)
    ]
    # Find which function owns each offending line.
    offenders = []
    for lineno, line in in_function_imports:
        # Walk backwards to find the enclosing `def` line.
        for j in range(lineno - 2, -1, -1):
            if lines[j].lstrip().startswith(("def ", "async def ", "class ")):
                func_name = lines[j].split("(")[0].split("def ")[-1].strip()
                offenders.append((j + 1, func_name, line.strip()))
                break
    assert not offenders, (
        f"Local `from STT_server.services.common import` inside functions "
        f"is a bug — the imported symbol becomes function-local via "
        f"Python scoping, breaking every earlier reference. Offending "
        f"lines: {offenders}"
    )


# ── Tool-calling plumbing (P4) ──────────────────────────────────────
# The Realtime adapter used to send the model a session.update
# without a `tools` field. The model had no way to call the Google
# Calendar webhook the operator assigned to the agent, so it
# hallucinated the action ("your appointment is scheduled") and
# the calendar stayed empty. These tests pin the contract that
# fixes that — `_build_session_update_payload` MUST include the
# tools in OpenAI's function-call format so the model sees them.


class _StubSession:
    """Minimal CallSession stand-in for the payload builder.

    The builder reads `agent_tools` (for the tools block),
    `session_key` (for the log line), `custom_prompt`,
    `collected_data`, `history`, and `agent_id` (via
    _build_instructions). Everything else we leave None.
    """


def _session_with_tools(tools: list[dict], *, user_id: str | None = None) -> _StubSession:
    s = _StubSession()
    s.agent_tools = tools
    s.session_key = "test-session"
    s.custom_prompt = "You are a helpful assistant."
    s.collected_data = {}
    s.history = []
    s.agent_id = "agent-test"
    s.preferred_language = "en"
    s.user_id = user_id
    return s


def test_session_update_includes_tools_when_assigned():
    """The session.update payload MUST include a `tools` array when
    the agent has any. Without it the model can't emit function_call
    events and the agent hallucinates the action."""
    from STT_server.adapters.openai_realtime import _build_session_update_payload

    session = _session_with_tools([{
        "id": "tool-google-calendar",
        "name": "schedule_meeting",
        "function_name": "schedule_meeting",
        "description": "Schedule a meeting on Google Calendar",
        "webhook_url": "https://n8n.example.com/webhook/cal",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string", "format": "date-time"},
            },
            "required": ["title", "start"],
        },
    }])
    payload = json.loads(_build_session_update_payload(session))

    assert "tools" in payload["session"], (
        "session.update missing `tools` — model has no way to call the "
        "agent's tools and will hallucinate the action."
    )
    tools = payload["session"]["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    # ponytail: OpenAI Realtime uses the FLAT schema
    # {type, name, description, parameters} — NOT the nested
    # {type, function: {name, ...}} shape that chat-completions
    # accepts. The previous version shipped with the nested shape
    # and the server rejected every tool with
    # `missing_required_parameter: session.tools[0].name` (the error
    # is checked at `.tools[0].name`, not `.tools[0].function.name`).
    assert "name" in tools[0], (
        "Realtime tool schema requires `name` at the top level of the "
        "tool object, not nested under `function`."
    )
    assert "function" not in tools[0], (
        "Realtime rejects the chat-completions nested `{type, function: {...}}` "
        "shape — tool definitions must be flat."
    )
    assert tools[0]["name"] == "schedule_meeting"
    assert tools[0]["description"] == "Schedule a meeting on Google Calendar"
    assert tools[0]["parameters"]["properties"]["start"]["format"] == "date-time"


def test_session_update_omits_tools_when_none_assigned():
    """When the agent has no tools, the session.update MUST omit
    the `tools` key (not pass an empty list). Some OpenAI server
    versions reject empty tool arrays as `invalid_request_error`,
    so we test the omit-on-empty contract directly."""
    from STT_server.adapters.openai_realtime import _build_session_update_payload

    session = _session_with_tools([])
    payload = json.loads(_build_session_update_payload(session))
    assert "tools" not in payload["session"], (
        "session.update must omit `tools` when the agent has none — "
        "empty arrays are rejected by some OpenAI server versions."
    )
    assert "tool_choice" not in payload["session"]


def test_session_update_sets_tool_choice_auto_when_tools_present():
    """`tool_choice` defaults to `none` on the Realtime API. Without
    `tool_choice=auto` the model is forbidden from calling tools
    even if they're listed. Force it on whenever tools are present."""
    from STT_server.adapters.openai_realtime import _build_session_update_payload

    session = _session_with_tools([{
        "id": "t1", "name": "do_thing", "function_name": "do_thing",
        "description": "", "webhook_url": "https://x",
        "parameters": {"type": "object", "properties": {}},
    }])
    payload = json.loads(_build_session_update_payload(session))
    assert payload["session"].get("tool_choice") == "auto"


def test_session_update_sanitises_function_name_for_legacy_rows():
    """Legacy rows (pre function_name column) need the display name
    sanitised to satisfy OpenAI's `^[a-zA-Z0-9_-]+$` regex. Otherwise
    the server returns `invalid_request_error` on session.update."""
    from STT_server.adapters.openai_realtime import _build_session_update_payload

    session = _session_with_tools([{
        "id": "t-legacy",
        "name": "Schedule Meeting! 2024",
        # no function_name — fallback path
        "description": "Old-style tool",
        "webhook_url": "https://n8n.example.com/webhook/x",
        "parameters": {"type": "object", "properties": {}},
    }])
    payload = json.loads(_build_session_update_payload(session))
    fn_name = payload["session"]["tools"][0]["name"]
    import re as _re
    assert _re.match(r"^[A-Za-z0-9_-]+$", fn_name), (
        f"function_name '{fn_name}' would be rejected by OpenAI "
        f"(regex violation on session.update)."
    )
    assert " " not in fn_name and "!" not in fn_name


# ── Timezone-aware datetimes (T6) ────────────────────────────────────
# The operator reported the calendar event landed at 4 AM when the
# user requested 11 AM. Root cause: the LLM emitted a naive ISO
# string (`2024-09-04T11:00:00`) — n8n treated it as UTC and Google
# Calendar displayed in Tijuana time (UTC-7) = 4 AM. The fix is to
# inject the operator's IANA timezone + a concrete offset into the
# system prompt so the model emits `2024-09-04T11:00:00-07:00`.
# These tests pin that the directive ships and uses the right
# example offset for common timezones.


class _FakeSettings:
    def __init__(self, tz: str | None):
        self.tz = tz

    def get(self, key, default=None):
        return self.tz if key == "timezone" else default


def _patch_settings(tz: str | None):
    return patch(
        "STT_server.adapters.openai_realtime._resolve_user_timezone",
        return_value=tz,
    )


def test_instructions_include_timezone_directive_for_known_tz():
    """When the user has a timezone in /settings, the system
    instructions must include an explicit "use this offset in any
    datetime" directive. Without it the LLM emits naive ISO
    strings and the calendar event lands at the wrong wall time."""
    from STT_server.adapters.openai_realtime import _build_instructions

    session = _StubSession()
    session.custom_prompt = "You are a helpful assistant."
    session.session_key = "s"
    session.collected_data = {}
    session.history = []
    session.agent_id = "a"
    session.user_id = "user-1"

    with _patch_settings("America/Mexico_City"):
        prompt = _build_instructions(session)

    # The directive must mention the IANA name so the LLM has a
    # ground truth to anchor its reasoning against (not just an
    # offset, which could come from anywhere).
    assert "America/Mexico_City" in prompt, (
        "system prompt missing the operator's IANA timezone — the "
        "LLM has no anchor for the offset."
    )
    # The directive must include a concrete offset example so the
    # LLM doesn't have to compute the offset from the IANA name
    # (which it's famously bad at).
    assert "-07:00" in prompt or "-06:00" in prompt, (
        "system prompt missing a concrete UTC offset example for "
        "Mexico_City (Tijuana is UTC-7 in summer, UTC-6 in winter)."
    )
    # Must explicitly forbid naive datetimes — the exact failure
    # mode that produced the 4 AM event.
    assert "naive" in prompt.lower() and "offset" in prompt.lower(), (
        "system prompt must warn the LLM about naive datetimes and "
        "demand an explicit offset."
    )


def test_instructions_skip_timezone_directive_when_user_has_no_timezone():
    """When the operator hasn't picked a timezone in /settings
    (anonymous call, admin test, legacy row), the directive is
    skipped — we don't want to inject a wrong default that the
    LLM would treat as truth."""
    from STT_server.adapters.openai_realtime import _build_instructions

    session = _StubSession()
    session.custom_prompt = "You are a helpful assistant."
    session.session_key = "s"
    session.collected_data = {}
    session.history = []
    session.agent_id = "a"
    session.user_id = None

    with _patch_settings(None):
        prompt = _build_instructions(session)

    assert "OPERATOR TIMEZONE" not in prompt, (
        "system prompt injected a timezone directive even though "
        "the user has no timezone configured — would train the LLM "
        "to emit the wrong offset."
    )


def test_instructions_offset_for_eastern_timezone():
    """Ponytail: the directive must compute the right offset for
    other IANA zones too, not just hard-code -07:00. Eastern time
    is UTC-4 in summer (EDT) / UTC-5 in winter (EST) — we don't
    hard-code the sign."""
    from STT_server.adapters.openai_realtime import _build_instructions

    session = _StubSession()
    session.custom_prompt = "You are a helpful assistant."
    session.session_key = "s"
    session.collected_data = {}
    session.history = []
    session.agent_id = "a"
    session.user_id = "user-1"

    with _patch_settings("America/New_York"):
        prompt = _build_instructions(session)

    assert "America/New_York" in prompt
    # Eastern is UTC-4 (EDT) in summer, UTC-5 (EST) in winter.
    # Accept either — never the wrong sign.
    assert ("-04:00" in prompt) or ("-05:00" in prompt), (
        f"system prompt missing a valid Eastern offset; got: {prompt!r}"
    )


def test_resolve_user_timezone_returns_settings_value():
    """Direct unit test for _resolve_user_timezone — the helper that
    looks up the user's tz in db_settings."""
    from STT_server.adapters import openai_realtime

    with patch(
        "STT_server.db_settings.get_settings",
        return_value={"timezone": "Asia/Tokyo", "name": "test"},
    ):
        tz = openai_realtime._resolve_user_timezone("user-1")
    assert tz == "Asia/Tokyo"


def test_resolve_user_timezone_returns_none_when_no_settings():
    from STT_server.adapters import openai_realtime

    with patch("STT_server.db_settings.get_settings", return_value=None):
        tz = openai_realtime._resolve_user_timezone("user-1")
    assert tz is None


def test_resolve_user_timezone_returns_none_when_no_user_id():
    from STT_server.adapters import openai_realtime

    # No user_id — admin test endpoint or anonymous session.
    # The directive should be skipped, not crash.
    tz = openai_realtime._resolve_user_timezone(None)
    assert tz is None


def test_resolve_user_timezone_swallows_db_errors():
    """Settings lookup failure must NOT break session start — we
    return None and the operator gets no a directive (better than
    blocking the call). Logged at WARNING so the operator sees
    the underlying cause in BE logs."""
    from STT_server.adapters import openai_realtime

    with patch(
        "STT_server.db_settings.get_settings",
        side_effect=RuntimeError("simulated DB outage"),
    ):
        tz = openai_realtime._resolve_user_timezone("user-1")
    assert tz is None