"""Audio-pipeline error taxonomy.

The audio pipeline mixes three different kinds of failure:

  - Network / IO / timeout — the upstream provider dropped a WebSocket,
    Twilio's media stream paused, httpx timed out. The component can
    reset (reconnect, drop the in-flight chunk) and keep going.

  - Schema / validation — the inbound base64 was oversized, the
    Twilio JSON envelope was malformed, the format header was wrong.
    The component logs the violation and skips this event. No
    reconnect needed.

  - Programming bug — NameError, AttributeError, KeyError, TypeError,
    ImportError, ValueError surfaced from inside a function the
    runtime relies on. **These must NOT be swallowed.** They signal
    broken code; the right response is a loud global exception
    handler that fires `play_error_and_hangup` so the operator sees
    the bug instead of a half-broken call that limps along.

The class hierarchy gives adapters one place to translate native
exceptions (httpx, websockets, ConnectionError, …) into a semantic
``RecoverableAudioError`` so ``media_stream`` can keep the call
alive on the first kind, surface the second as a logged skip, and
let the third fall through to the global exception handler that
hasn't moved.

Keep this taxonomy small. Adding a sub-class per adapter pushes
the decision of "is this recoverable?" to the throw site — where
the context that makes it answerable is — instead of to the catch
site, which has no way to tell.
"""
from __future__ import annotations


class AudioError(Exception):
    """Base for every audio-pipeline-specific exception.

    Native exceptions (httpx, websockets, ConnectionError, …) should
    NOT subclass this directly; instead, catch them at the boundary
    and re-raise as ``RecoverableAudioError`` or ``FatalAudioError``.
    That keeps the throw site honest about intent and gives
    ``media_stream`` one place to switch on.
    """


class RecoverableAudioError(AudioError):
    """Transient failure: the audio component can reset and the call
    keeps going. Maps to ``httpx.ConnectError``, ``httpx.Timeout``,
    ``websockets.ConnectionClosed``, plain ``ConnectionError`` /
    ``TimeoutError``, JSON-decode failures on inbound Twilio events,
    oversized / invalid base64 payloads, and adapter WS handshake
    failures.

    The catch site should reset the affected component's transient
    state (VAD buffer, mute buffer, playback queue head) and log the
    recovery so the operator can see it in the per-call summary.
    """


class FatalAudioError(AudioError):
    """Programming bug surfaced as an exception. Must NOT be silently
    caught or downgraded to ``RecoverableAudioError`` by an adapter.
    Maps to ``NameError``, ``AttributeError``, ``KeyError``,
    ``TypeError``, ``ValueError``, ``ImportError``, ``AssertionError``,
    ``SyntaxError`` — anything that means "the code as shipped is
    broken". Falls through to the global ``media_stream`` exception
    handler that fires ``play_error_and_hangup`` so the operator
    sees the bug.

    If you find yourself wanting to catch this in a new place,
    stop. The right answer is to fix the underlying bug, not to
    add another swallow.
    """


if __name__ == "__main__":
    # Smoke: instance-of checks + inheritance chain.
    assert issubclass(RecoverableAudioError, AudioError)
    assert issubclass(FatalAudioError, AudioError)
    assert isinstance(RecoverableAudioError("x"), AudioError)
    assert isinstance(FatalAudioError("x"), AudioError)
    # Recoverable is NOT Fatal and vice versa — they are siblings, not
    # interchangeable. An adapter that catches one must explicitly
    # catch the other; no surprise cross-catch.
    assert not isinstance(RecoverableAudioError("x"), FatalAudioError)
    assert not isinstance(FatalAudioError("x"), RecoverableAudioError)
    print("audio_errors: OK")