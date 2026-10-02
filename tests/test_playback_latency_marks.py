"""Checks for the 2026-10-02 latency/observability fixes.

Three regressions, three asserts. No pytest — these are the smallest
runnable checks that fail if the logic breaks.
"""
import ast
import sys
import time

sys.path.insert(0, ".")

from STT_server.domain.session import CallSession  # noqa: E402
from STT_server.services._instrumentation import StageTimer, Stages  # noqa: E402

FRAME_MS = 20.0
MARGIN = 3.0

# ponytail: importing STT_server.STT_Server pulls in the openai SDK, which
# is not installed in every checkout. Lift the pure watchdog predicate out
# of the source with ast instead — still the real code under test.
_stt_src = open("STT_server/STT_Server.py", encoding="utf-8").read()
_stt_tree = ast.parse(_stt_src)
_fn = next(
    n for n in _stt_tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_speaking_stuck_reason"
)
_ns: dict = {}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), "STT_Server.py", "exec"), _ns)
_speaking_stuck_reason = _ns["_speaking_stuck_reason"]

# ── 1. assistant_expected_end_at must accumulate across chunks ──────────
s = CallSession(session_key="c1")
started = time.perf_counter()
s.assistant_speaking = True
s.assistant_started_at = started
s.assistant_frames_sent = 0

# Simulate playback_service: two chunks of the SAME speaking stretch.
for chunk_frames in (5, 7):
    for _ in range(chunk_frames):
        s.assistant_frames_sent += 1
        s.assistant_expected_end_at = (
            started + s.assistant_frames_sent * (FRAME_MS / 1000.0) + MARGIN
        )

# 12 frames = 240 ms of audio. Deadline must reflect all 12, not just the
# last chunk's 7 (which would unstick the flag 100 ms too early).
expected = started + 0.240 + MARGIN
assert abs(s.assistant_expected_end_at - expected) < 1e-9, (
    f"expected end {s.assistant_expected_end_at} != {expected} — "
    "per-chunk counter leaked into the deadline"
)
# Mid-audio the watchdog must NOT fire.
assert _speaking_stuck_reason(s, started + 0.30) is None, "cleared assistant_speaking mid-playback"
# Past the deadline it must fire (lost mark ack).
assert _speaking_stuck_reason(s, expected + 0.1) == "past-expected-end"

# Fresh speaking stretch resets the counter.
s.assistant_speaking = False
s.assistant_frames_sent = 0
assert s.assistant_frames_sent == 0

# ── 2. mark ack must stamp the PER-TURN timer, not the session one ─────
tree = _stt_tree
stale = [
    n.lineno for n in ast.walk(tree)
    if isinstance(n, ast.Call)
    and getattr(n.func, "attr", "") == "getattr"
    and len(n.args) == 3
    and isinstance(n.args[1], ast.Constant)
    and n.args[1].value == "stage_timer"
]
assert not stale, (
    f"STT_Server.py reads session.stage_timer (the session-lifetime timer) "
    f"at lines {stale}; audio_ingest rebinds session._stage_timer per turn, "
    "so the mark ack must stamp _stage_timer"
)

# ── 3. FIRST_160_FRAME_SENT must be stamped once per GENERATION ───────
pb = open("STT_server/services/playback_service.py", encoding="utf-8").read()
assert "first_frame_marked = False" not in pb, (
    "playback_loop still has the call-wide one-shot first_frame_marked; "
    "every turn after generation 0 inherits turn 0's TTFB"
)
assert "first_frame_marked_gen = generation" in pb, "per-generation stamp not recorded"

# The stage timer itself must still be coherent.
st = StageTimer(call_id="c1", turn_id=0, generation=0)
st.mark(Stages.STT_FIRST_RESULT)
time.sleep(0.01)
st.mark(Stages.FIRST_160_FRAME_SENT)
d = st.summary()["deltas"]
assert d[Stages.STT_FIRST_RESULT] == 0.0 and d[Stages.FIRST_160_FRAME_SENT] >= 5, d

print("latency_fixes: OK")