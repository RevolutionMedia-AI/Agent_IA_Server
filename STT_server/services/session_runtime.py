import asyncio
import contextlib
import json
import logging
import os
import time

from fastapi import WebSocket

from STT_server.config import IDLE_SILENCE_TIMEOUT_SEC
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop
from STT_server.services.usage_store import has_user_stored_key, record_call


log = logging.getLogger("stt_server")

sessions: dict[str, CallSession] = {}

# ponytail: agents.json lookup for the per-call usage record. We need
# the agent's stt_provider/llm_provider (the session only carries
# tts_provider); loading the whole file here is fine — it's small and
# the lookup happens once per call end.
_AGENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "agents.json",
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


def track_task(session: CallSession, task: asyncio.Task) -> asyncio.Task:
    session.tasks.add(task)
    task.add_done_callback(session.tasks.discard)
    return task


async def register_session(session: CallSession) -> None:
    sessions[session.session_key] = session
    # ponytail: H2 from the call-flow audit. Mirror the in-memory
    # registration to Postgres so a server crash doesn't orphan
    # the CallSession forever (memory leak across restarts). Best
    # effort: the in-memory registration already happened and the
    # call works without the DB write. A subsequent redeploy will
    # see the row via list_open_sessions() and can mark it closed.
    try:
        from STT_server import db_call_sessions
        await db_call_sessions.register_session(
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

    if session.speech_frames:
        session.speech_frames.clear()

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
    try:
        from STT_server import db_call_sessions
        await db_call_sessions.close_session(session.session_key)
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
                await asyncio.sleep(5)
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
            await asyncio.sleep(min(remaining, 5))
    except asyncio.CancelledError:
        return
    except Exception:
        log.exception("Error en monitor_idle_silence")