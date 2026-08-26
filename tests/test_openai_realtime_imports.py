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
import inspect
import re


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