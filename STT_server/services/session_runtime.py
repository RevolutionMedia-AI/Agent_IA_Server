import asyncio
import contextlib

from fastapi import WebSocket

from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop


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