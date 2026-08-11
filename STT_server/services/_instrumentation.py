"""Pipeline stage timing — monotonic-delta tracking for one call/turn/generation.

Used by other agents to measure latency across STT -> LLM -> TTS -> network stages
without dragging in Prometheus or any external time-series.
"""
import time


class Stages:
    """Canonical stage-name constants. Reuse rather than free-typing."""
    EOS = "eos"
    STT_FIRST_RESULT = "stt_first_result"
    LLM_FIRST_TOKEN = "llm_first_token"
    TTS_FIRST_BYTE = "tts_first_byte"
    FIRST_160_FRAME_SENT = "first_160_frame_sent"
    TWILIO_MARK_ACK = "twilio_mark_ack"
    END_TO_END_FIRST_AUDIO = "end_to_end_first_audio"


class StageTimer:
    """Records monotonic timestamps for arbitrary named stages in one generation."""

    def __init__(self, call_id: str, turn_id: int, generation: int):
        self.call_id = call_id
        self.turn_id = turn_id
        self.generation = generation
        self._stages: dict[str, float] = {}
        self._first_t: float | None = None

    def mark(self, stage: str) -> None:
        """Record now (monotonic) for *stage*. Re-marking overwrites the prior timestamp."""
        now = time.monotonic()
        if self._first_t is None:
            self._first_t = now
        self._stages[stage] = now

    def summary(self) -> dict:
        """Return {call_id, turn_id, generation, stages, deltas} where deltas are ms floats."""
        keys = list(self._stages.keys())
        deltas: dict[str, float] = {}
        prev = self._first_t
        for k in keys:
            cur = self._stages[k]
            if prev is not None:
                deltas[k] = round((cur - prev) * 1000.0, 3)
            prev = cur
        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "generation": self.generation,
            "stages": dict(self._stages),
            "deltas": deltas,
        }

    def to_log_line(self) -> str:
        """Single line suitable for log.info(...): ``call=... turn=... gen=... deltas: k1=Xms k2=Yms``."""
        s = self.summary()
        head = f"call={s['call_id']} turn={s['turn_id']} gen={s['generation']}"
        body = " ".join(f"{k}={v}ms" for k, v in s["deltas"].items())
        return f"{head} deltas: {body}" if body else head


if __name__ == "__main__":
    import time as _t

    st = StageTimer(call_id="c1", turn_id=0, generation=1)
    st.mark(Stages.STT_FIRST_RESULT)
    _t.sleep(0.05)
    st.mark(Stages.LLM_FIRST_TOKEN)
    _t.sleep(0.05)
    st.mark(Stages.TTS_FIRST_BYTE)

    s = st.summary()

    # First-stage delta is ~0 (since first_t == mark-1 timestamp).
    assert s["deltas"][Stages.STT_FIRST_RESULT] == 0.0, "first-stage delta should be 0ms"
    # Subsequent deltas reflect ~50ms sleeps.
    assert s["deltas"][Stages.LLM_FIRST_TOKEN] >= 45, "expected >=45ms after 50ms sleep"
    assert s["deltas"][Stages.TTS_FIRST_BYTE] >= 45
    # All deltas are positive ms floats.
    for v in s["deltas"].values():
        assert v >= 0

    print("summary:", s)
    print("log_line:", st.to_log_line())
    print("_instrumentation: OK")
