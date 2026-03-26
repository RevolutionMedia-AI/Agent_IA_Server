import asyncio
import contextlib
import logging
import time

from fastapi import WebSocket

from STT_server.config import IDLE_SILENCE_TIMEOUT_SEC
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop


log = logging.getLogger("stt_server")

sessions: dict[str, CallSession] = {}


def track_task(session: CallSession, task: asyncio.Task) -> asyncio.Task:
    session.tasks.add(task)
    task.add_done_callback(session.tasks.discard)
    return task


def register_session(session: CallSession) -> None:
    sessions[session.session_key] = session


async def cleanup_session(session: CallSession, ws: WebSocket) -> None:
    if session.closed:
        return

    session.closed = True

    with contextlib.suppress(Exception):
        await enqueue_with_drop(session.stt_audio_queue, None, "stt_audio_queue")

    if session.speech_frames:
        session.speech_frames.clear()

    for task in list(session.tasks):
        task.cancel()

    sessions.pop(session.session_key, None)

    try:
        await asyncio.gather(*session.tasks, return_exceptions=True)
    except Exception:
        pass

    try:
        await ws.close()
    except Exception:
        pass


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