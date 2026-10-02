"""Regression guard for the 2026-10-02 log-noise cleanup.

The TTS observability chain (TTS_RAW_SEGMENT / TTS_SANITIZED_SEGMENT /
TTS_FORMATTED_SEGMENT, formerly joined by TTS_INWORLD_BODY) dumped
transcript text at INFO on every segment of every production call, even
though TTS_DEBUG_LOG exists precisely to switch it off. The existing
tests only asserted the env flag PARSES correctly — nothing asserted that
anything actually respected it, which is why the leak survived.

These are source-level checks on purpose: they are the smallest thing that
fails if someone adds an ungated transcript log again, and they need no
TTS provider, no network, and no session.
"""
import ast
from pathlib import Path

import pytest

TURN_MANAGER = Path("STT_server/services/turn_manager.py")
INWORLD_TTS = Path("STT_server/adapters/inworld_tts.py")

CHAIN = ("TTS_RAW_SEGMENT", "TTS_SANITIZED_SEGMENT", "TTS_FORMATTED_SEGMENT")


def _log_calls(tree, needle):
    """Every log.<level>() in *tree* whose format string contains *needle*."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") in
                {"info", "debug", "warning", "error", "exception"}):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and needle in arg.value:
                out.append(node)
                break
    return out


def _parents(tree):
    return {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}


def _guarded_by_tts_debug_log(parents, call):
    """True when an enclosing `if TTS_DEBUG_LOG:` guards *call*.

    Walks ancestors, not direct children: the log call sits inside an
    ast.Expr statement, so it is a grandchild of the If at best.
    """
    node = call
    while (parent := parents.get(node)) is not None:
        node = parent
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if "TTS_DEBUG_LOG" in names:
                return True
    return False


def test_chain_logs_are_gated_by_tts_debug_log():
    """Every transcript-dumping log must sit behind TTS_DEBUG_LOG."""
    tree = ast.parse(TURN_MANAGER.read_text(encoding="utf-8"))
    parents = _parents(tree)
    for needle in CHAIN:
        calls = _log_calls(tree, needle)
        assert calls, f"{needle} log disappeared — did the cleanup delete it by mistake?"
        for call in calls:
            assert _guarded_by_tts_debug_log(parents, call), (
                f"{needle} at turn_manager.py:{call.lineno} is not behind "
                "TTS_DEBUG_LOG, so it dumps transcript text on every segment "
                "of every production call"
            )


def test_tts_inworld_body_log_is_gone():
    """TTS_INWORLD_BODY dumped the full request body per segment.

    Deliberately deleted rather than gated: the body is determined by
    (voice, model, speakingRate, deliveryMode), all fixed per agent, so it
    repeated a known value on every segment of every call.
    """
    assert _log_calls(
        ast.parse(INWORLD_TTS.read_text(encoding="utf-8")), "TTS_INWORLD_BODY"
    ) == [], (
        "TTS_INWORLD_BODY came back — it was the highest-volume log in the "
        "pipeline and ignored TTS_DEBUG_LOG entirely"
    )


def test_transcript_summary_logs_are_not_warnings():
    """Per-turn 'Usuario'/'Agente' summaries at WARNING poisoned alerting.

    Railway tags anything on stderr as an error, so every turn produced two
    false WARNING lines per call.
    """
    tree = ast.parse(Path("STT_server/services/turn_manager.py").read_text(encoding="utf-8"))
    for needle in ("Usuario (%s)", "Agente (%s)"):
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "warning"):
                continue
            assert not any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and needle in a.value
                for a in node.args
            ), f"{needle} logged at WARNING at turn_manager.py:{node.lineno}"


def test_httpx_logger_is_quieted():
    """httpx logs one INFO line per HTTP request; we log turn timings."""
    src = Path("STT_server/STT_Server.py").read_text(encoding="utf-8")
    assert '"httpx"' in src, "httpx is back at INFO — one line per LLM/STT request"