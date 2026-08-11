"""20 ms / 160-byte μ-law framing for the voice pipeline.

Single owner of frame buffering: producers push raw bytes via ``feed``; downstream
callers (sync producers or async consumers via ``aiter_frames``) get completed
``frame_size``-byte slices. ``flush`` handles the trailing partial frame at EOF.
"""
import asyncio
from typing import AsyncIterator


class AudioFrameProcessor:
    def __init__(self, *, emit_silence_tail: bool = True, frame_size: int = 160):
        self._buf = bytearray()
        self._frame = frame_size
        self._emit_silence = emit_silence_tail
        self._closed = False
        self._stats: dict[str, int] = {
            "bytes_in": 0,
            "frames_out": 0,
            "padded_tail_frames": 0,
            "dropped_tail_bytes": 0,
        }
        self._queue: asyncio.Queue | None = None

    # ── sync API ─────────────────────────────────────────────────────────
    def feed(self, data: bytes) -> list[bytes]:
        """Append *data*; return every completed ``frame_size``-byte frame."""
        self._buf.extend(data)
        self._stats["bytes_in"] += len(data)
        out: list[bytes] = []
        while len(self._buf) >= self._frame:
            frame = bytes(self._buf[: self._frame])
            out.append(frame)
            self._stats["frames_out"] += 1
            del self._buf[: self._frame]
        self._publish(out)
        return out

    def flush(self) -> list[bytes]:
        """Handle trailing partial frame. Returns 0..1 frame; signals EOF to async consumer."""
        if self._closed:
            return []
        self._closed = True
        rem = len(self._buf)
        out: list[bytes] = []
        if rem == 0:
            self._publish_eof()
            return out
        if self._emit_silence and rem < self._frame:
            frame = bytes(self._buf) + b"\xff" * (self._frame - rem)  # μ-law silence
            self._stats["padded_tail_frames"] += 1
            self._stats["frames_out"] += 1
            out.append(frame)
        else:
            self._stats["dropped_tail_bytes"] += rem
        self._buf.clear()
        self._publish(out)
        self._publish_eof()
        return out

    def stats(self) -> dict:
        return dict(self._stats)

    # ── async API ────────────────────────────────────────────────────────
    async def aiter_frames(self) -> AsyncIterator[bytes]:
        """Yield completed frames as they appear; returns on EOF.

        Must be driven from the same event loop that flushes the producer.
        Sync calls to ``feed``/``flush`` also enqueue frames (no-op before the
        loop starts — they're still returned to the sync caller).
        """
        q = self._get_or_make_queue()
        while True:
            item = await q.get()
            if item is None:
                return
            yield item

    # ── internals ────────────────────────────────────────────────────────
    def _get_or_make_queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    def _publish(self, frames: list[bytes]) -> None:
        if self._queue is None:
            return
        for f in frames:
            self._queue.put_nowait(f)

    def _publish_eof(self) -> None:
        if self._queue is not None:
            self._queue.put_nowait(None)


# ── self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1 second @ 8 kHz = 8000 μ-law bytes = 50 frames @ 160 bytes.
    stream = bytes((0x00 if i % 2 == 0 else 0xFF) for i in range(8007))
    proc = AudioFrameProcessor(emit_silence_tail=False)
    for i in range(0, 8000, 200):
        proc.feed(stream[i : i + 200])
    tail = proc.flush()
    s = proc.stats()
    assert s["frames_out"] == 50, f"expected 50 frames at 8kHz/20ms, got {s['frames_out']}"
    assert tail == [], "no partial bytes left → flush should produce nothing"
    assert s["dropped_tail_bytes"] == 0
    assert s["padded_tail_frames"] == 0

    # Partial trailing frame with padding: 50 frames + 7 leftover bytes.
    proc = AudioFrameProcessor(emit_silence_tail=True)
    proc.feed(stream[:8007])
    final = proc.flush()
    s = proc.stats()
    assert final and len(final[0]) == 160, "padded tail frame should be exactly 160 bytes"
    assert final[0][-7:] == b"\xff" * 7, "silence pad should be 0xFF"
    assert s["padded_tail_frames"] == 1
    assert s["frames_out"] == 51

    # emit_silence_tail=False drops the remainder instead (999 bytes = 6*160 + 39 leftover).
    proc = AudioFrameProcessor(emit_silence_tail=False)
    proc.feed(stream[:999])
    leftover = proc.flush()
    s = proc.stats()
    assert leftover == []
    assert s["dropped_tail_bytes"] == 39
    assert s["frames_out"] == 6

    print("audio_frame_processor: OK", s)
