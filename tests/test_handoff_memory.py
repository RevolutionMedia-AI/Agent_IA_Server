"""Transfer-chain handoff memory.

When the AI hands a call to a <Dial> action our WebSocket dies and
cleanup_session() pops the session (and its history) out of `sessions`.
If the whole chain goes unanswered, Twilio re-opens a stream for the
SAME call_sid and the returning AI used to start with zero memory of
what the caller had already said.

These tests pin the stash/pop contract: round-trip, destructive pop,
no-op guards, TTL expiry and the max-size cap (a human who DOES answer
never resumes, so nothing ever pops that entry).
"""
from STT_server.services import session_runtime as sr


def _fresh():
    sr._handoff_memory.clear()


def test_round_trip_preserves_conversation():
    _fresh()
    history = [
        {"role": "user", "content": "Soy Ana, orden 4471"},
        {"role": "assistant", "content": "Te transfiero ahora mismo."},
        {"role": "tool", "content": "Tool 'Recepcion' result: transferring..."},
    ]
    sr.stash_handoff_history("CA123", history)
    got = sr.pop_handoff_history("CA123")
    assert got == history
    # destructive: a second pop must not resurrect it
    assert sr.pop_handoff_history("CA123") == []


def test_stash_copies_so_later_history_edits_do_not_leak():
    _fresh()
    history = [{"role": "user", "content": "hola"}]
    sr.stash_handoff_history("CA999", history)
    history.append({"role": "user", "content": " appended after the stash"})
    assert len(sr.pop_handoff_history("CA999")) == 1


def test_no_op_guards():
    _fresh()
    sr.stash_handoff_history(None, [{"role": "user", "content": "x"}])
    sr.stash_handoff_history("CA000", [])
    sr.stash_handoff_history("CA000", None)
    assert len(sr._handoff_memory) == 0
    assert sr.pop_handoff_history(None) == []
    assert sr.pop_handoff_history("CA-never-stashed") == []


def test_ttl_expires_unresumed_handoffs():
    """A human that answered never resumes, so the entry would sit
    there forever without the TTL."""
    _fresh()
    sr.stash_handoff_history("CA-old", [{"role": "user", "content": "x"}])
    # pretend it was stashed well past the TTL
    _ts, history = sr._handoff_memory["CA-old"]
    sr._handoff_memory["CA-old"] = (_ts - sr.HANDOFF_MEMORY_TTL_SEC - 1, history)
    assert sr.pop_handoff_history("CA-old") == []
    assert "CA-old" not in sr._handoff_memory


def test_max_size_cap_evicts_oldest():
    _fresh()
    for i in range(sr.HANDOFF_MEMORY_MAX + 5):
        sr.stash_handoff_history(f"CA{i}", [{"role": "user", "content": str(i)}])
    assert len(sr._handoff_memory) == sr.HANDOFF_MEMORY_MAX
    # oldest evicted, newest kept
    assert "CA0" not in sr._handoff_memory
    assert "CA1" not in sr._handoff_memory
    assert sr.pop_handoff_history(f"CA{sr.HANDOFF_MEMORY_MAX + 4}") != []


def test_resume_history_is_carried_plus_failure_note():
    """The shape the media-stream start handler builds on
    transfer_resume: the pre-transfer conversation, then the note."""
    _fresh()
    carried_before = [
        {"role": "user", "content": "Soy Ana, orden 4471"},
        {"role": "assistant", "content": "Te transfiero ahora mismo."},
        {"role": "tool", "content": "Tool 'Recepcion' result: transferring..."},
    ]
    sr.stash_handoff_history("CA777", carried_before)

    _carried = sr.pop_handoff_history("CA777")
    note = {"role": "system", "content": "nadie contestó la transferencia"}
    resumed = _carried + [note]

    assert len(resumed) == 4
    # the caller info the AI needs survives ...
    assert resumed[0]["content"] == "Soy Ana, orden 4471"
    # ... the AI already knows it tried ...
    assert "Te transfiero" in resumed[1]["content"]
    # ... and it is told not to loop into another transfer
    assert resumed[-1] is note


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("test_handoff_memory: all green")
