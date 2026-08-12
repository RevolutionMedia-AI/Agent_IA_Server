import asyncio
import contextlib
import json
import logging
import os
import time

from fastapi import WebSocket

from STT_server.config import IDLE_SILENCE_TIMEOUT_SEC, MAX_CALL_DURATION_SEC
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop
from STT_server.services.usage_store import has_user_stored_key, record_call


log = logging.getLogger("stt_server")

sessions: dict[str, CallSession] = {}

# ponytail: idle/duration monitor polling cadence. Landed here (instead of
# config.py) so the constant is colocated with the only two functions that
# read it. Kept tiny — the goal is "responsive shutdown on tear-down",
# not a config knob for operators.
IDLE_MONITOR_POLL_SEC = 5.0
MAX_DURATION_MONITOR_POLL_SEC = 10.0

# ponytail: agents.json lookup for the per-call usage record. We need
# the agent's stt_provider/llm_provider (the session only carries
# tts_provider); loading the whole file here is fine — it's small and
# the lookup happens once per call end.
_AGENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "agents.json",
)
_TOOLS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "agent_tools.json",
)


def _load_agent_providers(agent_id: str | None) -> tuple[str | None, str | None]:
    if not agent_id:
        return (None, None)
    try:
        with open(_AGENTS_FILE, "r", encoding="utf-8") as f:
            agents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return (None, None)
    if not isinstance(agents, list):
        return (None, None)
    for a in agents:
        if isinstance(a, dict) and a.get("id") == agent_id:
            return (a.get("stt_provider"), a.get("llm_provider"))
    return (None, None)


def _load_agent_tools(agent_id: str | None, user_id: str | None = None) -> list[dict]:
    """Load tools for an agent from agent_tools.json.

    ponytail: includes "shared" tools too — tools stored with
    agent_id="__shared__" and a matching user_id are loaded as
    if they belonged to the agent. Per-user scope: a user can only
    attach their own shared tools to their agents.
    """
    if not agent_id:
        return []
    try:
        with open(_TOOLS_FILE, "r", encoding="utf-8") as f:
            all_tools = json.load(f) or []
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return []
    out = [t for t in all_tools if isinstance(t, dict) and t.get("agent_id") == agent_id]
    if user_id:
        out.extend(
            t for t in all_tools
            if isinstance(t, dict)
            and t.get("agent_id") == "__shared__"
            and t.get("user_id") == user_id
        )
    return out


def track_task(session: CallSession, task: asyncio.Task) -> asyncio.Task:
    session.tasks.add(task)
    task.add_done_callback(session.tasks.discard)
    return task


async def register_session(session: CallSession) -> None:
    sessions[session.session_key] = session
    # ponytail: lazy-init the per-call StageTimer here, exactly once
    # per session. STT_Server.py's start-event handler is the only
    # caller, so this runs before any adapter can emit its first
    # stage. wait_signals.mark_stage() also lazy-inits as a safety
    # net for code paths that bypass register_session (debug
    # endpoints, tests).
    if session.stage_timer is None:
        from STT_server.services._instrumentation import StageTimer
        session.stage_timer = StageTimer(
            call_id=session.session_key,
            turn_id=0,
            generation=session.active_generation,
        )
        # ponytail: alias the underscore-prefixed name so audio_ingest.py
        # (which already attaches to session._stage_timer) reuses the
        # same timer instead of constructing a parallel one. Same
        # instance, same stage timestamps — the dashboard sees one
        # timeline per call.
        session._stage_timer = session.stage_timer  # type: ignore[attr-defined]
    # ponytail: P2 round-3 — attach the per-call metrics container so
    # all adapters can `session.metrics.incr(...)`/`.observe_ms(...)`
    # without the attach_metrics idempotency dance. Idempotent.
    from STT_server.services.audio_metrics import attach_metrics
    attach_metrics(session, call_id=session.session_key)
    # ponytail: H2 from the call-flow audit. Mirror the in-memory
    # registration to Postgres so a server crash doesn't orphan
    # the CallSession forever (memory leak across restarts). Best
    # effort: the in-memory registration already happened and the
    # call works without the DB write. A subsequent redeploy will
    # see the row via list_open_sessions() and can mark it closed.
    #
    # db_call_sessions.register_session is a sync function (psycopg2
    # is sync) returning a dict on success / None on skip. We
    # previously `await`-ed it, which raised
    # `object dict can't be used in 'await' expression` and broke
    # every call. Just call it.
    try:
        from STT_server import db_call_sessions
        db_call_sessions.register_session(
            session.session_key,
            tenant_id=session.tenant_id,
            call_sid=session.call_sid,
            preferred_language=session.preferred_language,
            tts_provider=session.tts_provider,
            custom_prompt=session.custom_prompt,
            started_at=session.started_at,
        )
    except Exception as exc:  # noqa: BLE001 — DB write must never block a call
        log.warning("[runtime] DB register_session failed for %s: %s",
                    session.session_key, exc)


async def cleanup_session(session: CallSession, ws: WebSocket) -> None:
    if session.closed:
        return

    session.closed = True

    # ponytail: P2 round-3 — emit the consolidated end-of-call summary
    # BEFORE the usage record and queue teardown so all adapter-recorded
    # counters are still attached to the session. Idempotent + best-effort:
    # nothing here can crash cleanup.
    try:
        from STT_server.services.call_summary import emit_call_summary
        await emit_call_summary(log, session)
    except Exception:
        log.exception("emit_call_summary failed")

    # ponytail: write the per-call usage record BEFORE we lose the
    # session data. The aggregation in /api/usage reads from this
    # ledger, so dropping the write means the call is invisible to
    # billing.
    try:
        _record_usage_for(session)
    except Exception as exc:  # noqa: BLE001 — never crash cleanup on billing
        log.warning("[usage] record failed for %s: %s", session.session_key, exc)

    with contextlib.suppress(Exception):
        await enqueue_with_drop(session.stt_audio_queue, None, "stt_audio_queue")

    with contextlib.suppress(Exception):
        await enqueue_with_drop(session.realtime_audio_queue, None, "realtime_audio_queue")

    session.generation_changed.set()

    # AUDIO-005: free every per-call audio buffer so the dataclass holds
    # zero audio bytes after cleanup. speech_frames is already a bounded
    # deque (maxlen=SPEECH_FRAMES_MAX) but we still .clear() to release
    # the bytes eagerly. vad_buffer is an unbounded bytearray in the happy
    # path it drains to ~zero, but a misbehaving caller could leave a
    # few KB behind; clear it too. pre_speech_frames is bounded by
    # PRE_SPEECH_FRAMES but the slot-holding bytes can be large.
    session.speech_frames.clear()
    session.vad_buffer.clear()
    session.pre_speech_frames.clear()
    session.stt_mute_buffer.clear()
    # ponytail: AUDIO-005 — reset the once-per-session cap warning
    # flag. cleanup_session may run after a subsequent call reuses
    # the same session_key (rare but possible in tests); reset
    # defensively.
    session._speech_frames_cap_warned = False

    for task in list(session.tasks):
        task.cancel()

    sessions.pop(session.session_key, None)

    try:
        await asyncio.gather(*session.tasks, return_exceptions=True)
    except Exception:
        pass

    # ponytail: H2 from the call-flow audit. Mark the call closed
    # in Postgres so list_open_sessions() at startup can recover
    # any sessions that didn't close cleanly (server crash, deploy).
    # Best effort — if the DB write fails, the in-memory pop already
    # ran and the call is fully torn down. The row will be recovered
    # by a future startup sweep.
    #
    # db_call_sessions.close_session is sync (psycopg2 is sync);
    # don't await it.
    try:
        from STT_server import db_call_sessions
        db_call_sessions.close_session(session.session_key)
    except Exception as exc:  # noqa: BLE001 — DB write must never block cleanup
        log.warning("[runtime] DB close_session failed for %s: %s",
                    session.session_key, exc)

    try:
        await ws.close()
    except Exception:
        pass


def _record_usage_for(session: CallSession) -> None:
    """Best-effort: write one usage row per call. Skipped silently
    when there's no user_id (anonymous test calls) or no start ts.
    """
    user_id = session.user_id
    if not user_id:
        return
    if session.started_at is None:
        return
    stt_provider, llm_provider = _load_agent_providers(session.agent_id)
    providers = {
        "stt": stt_provider,
        "llm": llm_provider,
        "tts": session.tts_provider,
    }
    used_platform = any(
        p and not has_user_stored_key(user_id, p)
        for p in providers.values()
    )
    record_call(
        user_id=user_id,
        agent_id=session.agent_id,
        tenant_id=session.tenant_id,
        call_sid=session.call_sid,
        started_at=session.started_at,
        ended_at=time.time(),
        providers=providers,
        used_platform_keys=used_platform,
    )


async def monitor_idle_silence(session: CallSession, ws: WebSocket) -> None:
    """Close the call if both parties are silent for IDLE_SILENCE_TIMEOUT_SEC."""
    if IDLE_SILENCE_TIMEOUT_SEC <= 0:
        return
    try:
        while not session.closed:
            if session.assistant_speaking:
                await asyncio.sleep(IDLE_MONITOR_POLL_SEC)
                continue
            remaining = IDLE_SILENCE_TIMEOUT_SEC - (time.monotonic() - session.last_activity_at)
            if remaining <= 0:
                log.info(
                    "Idle silence timeout (%.0fs) en %s, cerrando llamada",
                    IDLE_SILENCE_TIMEOUT_SEC,
                    session.session_key,
                )
                try:
                    await ws.close()
                except Exception:
                    pass
                break
            await asyncio.sleep(min(remaining, IDLE_MONITOR_POLL_SEC))
    except asyncio.CancelledError:
        return
    except Exception:
        log.exception("Error en monitor_idle_silence")


async def monitor_max_call_duration(session: CallSession, ws: WebSocket) -> None:
    """Hard-timeout: close the call if it exceeds MAX_CALL_DURATION_SEC.

    This prevents phantom calls that consume infinite STT/LLM units when
    the idle-silence monitor cannot fire (e.g. one party is always speaking,
    or the audio stream stays active without meaningful conversation).
    """
    if MAX_CALL_DURATION_SEC <= 0:
        return
    try:
        while not session.closed:
            await asyncio.sleep(MAX_DURATION_MONITOR_POLL_SEC)
            if session.closed:
                return
            if session.started_at is None:
                continue
            elapsed = time.time() - session.started_at
            if elapsed >= MAX_CALL_DURATION_SEC:
                log.warning(
                    "Max call duration (%.0fs) exceeded in %s, force-closing. "
                    "started_at=%.1f elapsed=%.1f",
                    MAX_CALL_DURATION_SEC,
                    session.session_key,
                    session.started_at,
                    elapsed,
                )
                try:
                    await ws.close()
                except Exception:
                    pass
                break
    except asyncio.CancelledError:
        return
    except Exception:
        log.exception("Error en monitor_max_call_duration")