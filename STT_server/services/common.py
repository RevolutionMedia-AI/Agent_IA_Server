import asyncio
import logging

from fastapi import HTTPException

from STT_server.config import ENABLE_DEBUG_ENDPOINTS


log = logging.getLogger("stt_server")

# ponytail: AUDIO-001 — instrumentation only, drop policy unchanged. Counts per-queue drops + high-water marks so the operator can correlate with sequence gaps and STT failures before sizing policy.
_QUEUE_DROP_TOTAL: dict[str, int] = {}
_QUEUE_HIGH_WATER: dict[str, int] = {}
_QUEUE_CONTROL_DROP_TOTAL: dict[str, int] = {}


def _drop_prefer_audio(queue: asyncio.Queue, queue_name: str) -> bool:
    """Drain `queue`, drop one item preferring type=='audio' (else the
    oldest item), re-enqueue the rest. Returns True if anything dropped.

    AUDIO-001: the previous drop-oldest policy could evict a `clear` or
    `segment_end` control item, desyncing the playback loop from Twilio
    (Twilio keeps playing audio while the loop thinks it stopped).
    Audio frames are fungible; control items are not. Drop audio first;
    fall back to dropping the oldest item only when no audio items
    remain in the queue. Counter `_QUEUE_CONTROL_DROP_TOTAL` flags the
    degraded path so operators can size the queue before it happens.
    """
    items: list = []
    try:
        while True:
            items.append(queue.get_nowait())
    except asyncio.QueueEmpty:
        pass
    if not items:
        return False
    audio_idx = next(
        (i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type") == "audio"),
        -1,
    )
    if audio_idx >= 0:
        drop_idx = audio_idx
    else:
        drop_idx = 0
        _QUEUE_CONTROL_DROP_TOTAL[queue_name] = _QUEUE_CONTROL_DROP_TOTAL.get(queue_name, 0) + 1
    kept = items[:drop_idx] + items[drop_idx + 1:]
    for it in kept:
        try:
            queue.put_nowait(it)
        except asyncio.QueueFull:  # ponytail: queue was bounded; re-enqueue fits after drain
            break
    return True


def enqueue_nowait_with_drop(queue: asyncio.Queue, item, queue_name: str) -> bool:
    while True:
        try:
            queue.put_nowait(item)
            size = queue.qsize()
            hw = _QUEUE_HIGH_WATER.get(queue_name, 0)
            if size > hw:
                _QUEUE_HIGH_WATER[queue_name] = size
            return True
        except asyncio.QueueFull:
            _QUEUE_DROP_TOTAL[queue_name] = _QUEUE_DROP_TOTAL.get(queue_name, 0) + 1
            if not _drop_prefer_audio(queue, queue_name):
                log.warning("No se pudo drenar la cola %s", queue_name)
                return False


async def enqueue_with_drop(queue: asyncio.Queue, item, queue_name: str) -> bool:
    return enqueue_nowait_with_drop(queue, item, queue_name)


def queue_overflow_stats() -> dict:
    return {
        "drops": dict(_QUEUE_DROP_TOTAL),
        "control_drops": dict(_QUEUE_CONTROL_DROP_TOTAL),
        "high_water": dict(_QUEUE_HIGH_WATER),
    }


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