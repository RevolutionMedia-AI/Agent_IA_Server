"""Event-based wait signals for the voice pipeline.

Replaces the old ``STREAM_SID_WAIT_MAX_MS / STREAM_SID_WAIT_POLL_MS`` polling
loop with asyncio.Event semantics. Producers ``set`` the event when the
condition is met (e.g. ``stream_sid`` arrives); consumers ``await`` it.

Also exposes the single ``mark_stage`` helper that other modules call to
stamp a StageTimer without each callsite needing to know how the timer
is created or what the log line should look like.
"""
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")


def _ensure_timer(session: "CallSession") -> None:
    """Lazy-init the StageTimer exactly once per session.

    ponytail: session_runtime.register_session is the canonical init point,
    but adapters that import mark_stage() may run before the start-event
    handler reaches register_session() (e.g. a debug endpoint that runs
    TTS synchronously). Falling back here keeps the timer usable in every
    code path without surfacing an AttributeError.
    """
    if session.stage_timer is None:
        # ponytail: deferred import avoids a cycle: domain.session.py
        # is the bottom of the dep graph, _instrumentation.py is one
        # rung up, and adapters import both. Importing
        # _instrumentation at module load would force it to be loaded
        # before domain.session can resolve its reference.
        from STT_server.services._instrumentation import StageTimer
        session.stage_timer = StageTimer(
            call_id=session.session_key,
            turn_id=0,
            generation=session.active_generation,
        )


def mark_stage(session: "CallSession", stage_name: str) -> None:
    """Stamp *stage_name* on the session's StageTimer and log the delta line.

    No-op when the session has no timer yet AND we can't create one (closed,
    torn down, etc. — never crash the live path on a metrics write).
    """
    if session.closed:
        return
    try:
        _ensure_timer(session)
        timer = session.stage_timer
        if timer is None:
            return
        timer.mark(stage_name)
        log.info(timer.to_log_line())
    except Exception as exc:
        # ponytail: instrumentation must never break the call. A bad
        # logging handler, a corrupt timer, anything — log and move on.
        log.warning("[stage_timer] mark_stage(%s) failed: %s", stage_name, exc)


def set_stream_ready(session: "CallSession") -> None:
    """Signal that the ``start`` event has been received and stream_sid is set.

    Consumed by ``playback_service.playback_loop`` (next phase) and any
    other code that previously polled ``session.stream_sid`` on a 50ms
    timer. The event is idempotent: calling it twice is a no-op.
    """
    event = getattr(session, "stream_ready", None)
    if event is None:
        # ponytail: the dataclass field might not exist on a session
        # constructed with __init__ kwargs that pre-date this PR
        # (e.g. tests). Fall back to a fresh Event so the awaiter
        # still wakes up.
        event = asyncio.Event()
        try:
            session.stream_ready = event  # type: ignore[attr-defined]
        except Exception:
            pass
    event.set()


async def wait_stream_ready(session: "CallSession", timeout: float | None = None) -> bool:
    """Await ``stream_ready`` up to *timeout* seconds. Returns True if set.
    If *timeout* is None, falls back to ``STREAM_SID_WAIT_TIMEOUT_MS / 1000``,
    or 5 seconds if that env var is 0 (which matches the pre-Phase-2 legacy
    behavior the welcome greeting path expects)."""
    if timeout is None:
        from STT_server.config import STREAM_SID_WAIT_TIMEOUT_MS
        ms = STREAM_SID_WAIT_TIMEOUT_MS
        # ponytail: hotfix — STREAM_SID_WAIT_TIMEOUT_MS default was 0 which
        # made the wait effectively instant. With timeout=0 the wait
        # returns immediately even if the start event hasn't fired,
        # so the welcome greeting is enqueued before stream_sid exists
        # and gets dropped. The legacy behavior was a 5s polling wait;
        # we keep that as the fallback default so the welcome greeting
        # always reaches Twilio.
        timeout = (ms / 1000.0) if ms > 0 else 5.0
    event = getattr(session, "stream_ready", None)
    if event is None:
        return False
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
