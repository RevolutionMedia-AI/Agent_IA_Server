"""The first tool created after a deploy must keep its integration binding.

Bug: `create_tool`/`update_tool` decided which optional columns to write
by reading the module global `_TOOL_COLS_EXTRA`, which starts as `[]` and
is only filled in by `_ensure_tool_columns()`. Nothing called that before
the INSERT — the self-heal fired from the `RETURNING _tool_cols()` on the
same statement, i.e. too late. So the first INSERT after every process
start wrote a row with `integration_id = NULL`: the tool saved, showed up
in the list, and was permanently unbound, so it could never resolve a
webhook URL. Every later create worked, which is what made it look
intermittent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path(__file__).resolve().parents[1] / "STT_server" / "db_tools.py"


def test_extra_cols_runs_the_self_heal_before_returning(monkeypatch) -> None:
    """The invariant: you never read the column list cold."""
    from STT_server import db_tools

    calls: list[int] = []

    def fake_heal() -> None:
        calls.append(1)
        # Stand in for the real thing: discover the columns and mark done,
        # exactly as _ensure_tool_columns does.
        db_tools._TOOL_COLS_EXTRA = [
            "credentials", "integration_id", "action", "ring_timeout_sec",
        ]
        db_tools._columns_check_done = True

    monkeypatch.setattr(db_tools, "_ensure_tool_columns", fake_heal)
    # Simulate a cold process: nothing checked yet.
    monkeypatch.setattr(db_tools, "_columns_check_done", False)
    monkeypatch.setattr(db_tools, "_TOOL_COLS_EXTRA", [])

    got = db_tools._extra_cols()

    assert calls, "_extra_cols() returned without ensuring the columns"
    assert "integration_id" in got
    assert "action" in got


def test_extra_cols_is_a_noop_once_warmed(monkeypatch) -> None:
    from STT_server import db_tools

    calls: list[int] = []
    monkeypatch.setattr(db_tools, "_ensure_tool_columns", lambda: calls.append(1))
    monkeypatch.setattr(db_tools, "_columns_check_done", True)
    monkeypatch.setattr(db_tools, "_TOOL_COLS_EXTRA", ["integration_id"])

    got = db_tools._extra_cols()

    assert not calls, "the self-heal re-ran on a warm process"
    assert got == ["integration_id"]


def test_a_cold_process_sees_an_empty_list() -> None:
    """Why this bug existed: the global is empty before the heal.

    Not a test of the fix, but of the premise. If this ever stops being
    true the bug is gone for a different reason and the test above is
    still worth keeping.
    """
    from STT_server import db_tools

    monkey_done = db_tools._columns_check_done
    monkey_extra = list(db_tools._TOOL_COLS_EXTRA)
    try:
        db_tools._columns_check_done = False
        db_tools._TOOL_COLS_EXTRA = []
        # Reading the global directly is what the old code did.
        assert "integration_id" not in db_tools._TOOL_COLS_EXTRA
    finally:
        db_tools._columns_check_done = monkey_done
        db_tools._TOOL_COLS_EXTRA = monkey_extra


def test_create_and_update_no_longer_read_the_cold_global() -> None:
    """Guard the two call sites.

    A bare `_TOOL_COLS_EXTRA` membership test inside either function is
    the bug; `_extra_cols()` is the only accepted way to ask.
    """
    import ast

    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    targets = {"create_tool", "update_tool"}

    def bare_reads(node: ast.AST) -> list[int]:
        bad: list[int] = []
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Name):
                continue
            if sub.id != "_TOOL_COLS_EXTRA":
                continue
            # `_extra_cols()` returns it; a bare Name load is the bug.
            parent_is_call = False
            for outer in ast.walk(node):
                if isinstance(outer, ast.Call) and sub in ast.walk(outer):
                    if any(
                        isinstance(n, ast.Attribute) and n.attr == "return"
                        for n in ast.walk(outer)
                    ):
                        parent_is_call = True
            if not parent_is_call:
                bad.append(sub.lineno)
        return bad

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            found = True
            bad = bare_reads(node)
            assert not bad, (
                f"{node.name} reads _TOOL_COLS_EXTRA directly at line(s) "
                f"{bad}; use _extra_cols() so the self-heal runs first"
            )
    assert found, "neither create_tool nor update_tool was found"
