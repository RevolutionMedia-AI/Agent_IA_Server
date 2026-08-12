"""Unit tests for STT_server.services.audio_frame_processor.

Covers the ``AudioFrameProcessor`` 20 ms / 160-byte μ-law framing contract:
  - aligned feed yields exactly N frames with no tail loss,
  - partial trailing frame is either padded with μ-law silence (0xFF) or dropped,
  - chunked feed across multiple ``feed()`` calls matches a single big feed,
  - ``stats()`` returns a defensive copy,
  - async ``aiter_frames`` consumes until EOF after ``flush()``.
"""
from __future__ import annotations

import pytest

from STT_server.services.audio_frame_processor import AudioFrameProcessor


# ── aligned feed ───────────────────────────────────────────────────────


def test_aligned_feed_one_second_emits_50_frames_no_tail() -> None:
    """1 s @ 8 kHz = 8000 μ-law bytes = 50 frames @ 160 bytes."""
    proc = AudioFrameProcessor()
    frames = proc.feed(b"\x00" * 8000)
    assert len(frames) == 50
    assert all(len(f) == 160 for f in frames)
    tail = proc.flush()
    assert tail == []
    s = proc.stats()
    assert s["frames_out"] == 50
    assert s["bytes_in"] == 8000
    assert s["dropped_tail_bytes"] == 0
    assert s["padded_tail_frames"] == 0


# ── partial trailing frame: padded vs dropped ──────────────────────────


def test_partial_tail_with_padding_emits_extra_silence_padded_frame() -> None:
    """8007 bytes → 50 frames + 1 padded frame ending in 7×0xFF."""
    proc = AudioFrameProcessor(emit_silence_tail=True)
    frames = proc.feed(b"\x00" * 8007)
    assert len(frames) == 50
    tail = proc.flush()
    assert len(tail) == 1
    assert len(tail[0]) == 160
    assert tail[0][-7:] == b"\xff" * 7
    s = proc.stats()
    assert s["frames_out"] == 51
    assert s["padded_tail_frames"] == 1
    assert s["dropped_tail_bytes"] == 0


def test_partial_tail_dropped_when_emit_silence_tail_false() -> None:
    """8007 bytes with emit_silence_tail=False → 50 frames + 7 bytes dropped."""
    proc = AudioFrameProcessor(emit_silence_tail=False)
    frames = proc.feed(b"\x00" * 8007)
    assert len(frames) == 50
    tail = proc.flush()
    assert tail == []
    s = proc.stats()
    assert s["frames_out"] == 50
    assert s["dropped_tail_bytes"] == 7
    assert s["padded_tail_frames"] == 0


# ── chunked feed ───────────────────────────────────────────────────────


def test_chunked_feed_matches_single_feed() -> None:
    """Splitting the same 8000 bytes across N chunks must produce identical frames."""
    big = AudioFrameProcessor()
    big_frames = big.feed(b"\x00" * 8000)
    assert len(big_frames) == 50

    chunked = AudioFrameProcessor()
    chunked_frames: list[bytes] = []
    chunk = 1600  # 5 chunks of 1600 = 8000
    for i in range(0, 8000, chunk):
        chunked_frames.extend(chunked.feed(b"\x00" * chunk))
    assert chunked_frames == big_frames


# ── stats defensive copy ───────────────────────────────────────────────


def test_stats_returns_independent_copy() -> None:
    """Mutating the returned dict must NOT affect the processor's internal state."""
    proc = AudioFrameProcessor()
    proc.feed(b"\x00" * 1600)
    snap = proc.stats()
    snap["frames_out"] = 99999
    snap["bytes_in"] = -1
    fresh = proc.stats()
    assert fresh["frames_out"] == 10
    assert fresh["bytes_in"] == 1600


# ── async EOF after flush ──────────────────────────────────────────────


async def test_aiter_frames_yields_until_eof_after_flush() -> None:
    proc = AudioFrameProcessor()
    proc.feed(b"\x00" * 320)  # 2 frames
    proc.flush()

    collected: list[bytes] = []
    async for frame in proc.aiter_frames():
        collected.append(frame)
    assert len(collected) == 2
    assert all(len(f) == 160 for f in collected)


async def test_aiter_frames_terminates_without_data() -> None:
    """No feed, no flush → aiter_frames still terminates on the EOF sentinel."""
    proc = AudioFrameProcessor()
    proc.flush()
    seen: list[bytes] = []
    async for f in proc.aiter_frames():
        seen.append(f)
    assert seen == []
