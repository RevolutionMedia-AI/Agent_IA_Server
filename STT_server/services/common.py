import asyncio
import logging

from fastapi import HTTPException

from STT_server.config import ENABLE_DEBUG_ENDPOINTS


log = logging.getLogger("stt_server")


def enqueue_nowait_with_drop(queue: asyncio.Queue, item, queue_name: str) -> bool:
    while True:
        try:
            queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                log.warning("No se pudo drenar la cola %s", queue_name)
                return False


async def enqueue_with_drop(queue: asyncio.Queue, item, queue_name: str) -> bool:
    return enqueue_nowait_with_drop(queue, item, queue_name)


def drain_queue_nowait(queue: asyncio.Queue) -> int:
    drained = 0
    while True:
        try:
            queue.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            return drained


def require_debug_endpoints() -> None:
    if not ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")