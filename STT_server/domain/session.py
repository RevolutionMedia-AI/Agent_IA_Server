from collections import deque
from dataclasses import dataclass, field
import asyncio
import time

from STT_server.config import (
    PLAYBACK_QUEUE_MAXSIZE,
    PRE_SPEECH_FRAMES,
    REALTIME_AUDIO_QUEUE_MAXSIZE,
    STT_AUDIO_QUEUE_MAXSIZE,
    STT_MUTE_BUFFER_CHUNKS,
    TRANSCRIPT_QUEUE_MAXSIZE,
)


@dataclass
class CallSession:
    session_key: str
    call_sid: str | None = None
    stream_sid: str | None = None
    preferred_language: str | None = None
    vad_buffer: bytearray = field(default_factory=bytearray)
    pre_speech_frames: deque[bytes] = field(default_factory=lambda: deque(maxlen=PRE_SPEECH_FRAMES))
    speech_frames: list[bytes] = field(default_factory=list)
    speech_frame_count: int = 0
    voice_streak: int = 0
    silence_frames: int = 0
    active_generation: int = 0
    response_active: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    utterance_queue: asyncio.Queue[tuple[int, bytes]] = field(default_factory=asyncio.Queue)
    playback_queue: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE))
    stt_audio_queue: asyncio.Queue[bytes | None] = field(default_factory=lambda: asyncio.Queue(maxsize=STT_AUDIO_QUEUE_MAXSIZE))
    stt_mute_buffer: deque[bytes] = field(default_factory=lambda: deque(maxlen=STT_MUTE_BUFFER_CHUNKS))
    transcript_queue: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=TRANSCRIPT_QUEUE_MAXSIZE))
    tasks: set[asyncio.Task] = field(default_factory=set)
    pending_marks: set[str] = field(default_factory=set)
    mark_counter: int = 0
    assistant_speaking: bool = False
    assistant_started_at: float | None = None
    current_transcript: str = ""
    reply_source_text: str = ""
    reply_task: asyncio.Task | None = None
    partial_reply_task: asyncio.Task | None = None
    prefetched_reply_source_text: str = ""
    prefetched_reply_text: str = ""
    prefetched_reply_task: asyncio.Task | None = None

    deferred_final_text: str = ""
    deferred_final_language: str | None = None
    deferred_final_flush_task: asyncio.Task | None = None
    collected_data: dict[str, str] = field(default_factory=dict)
    realtime_audio_queue: asyncio.Queue[bytes | None] = field(default_factory=lambda: asyncio.Queue(maxsize=REALTIME_AUDIO_QUEUE_MAXSIZE))
    realtime_text_queue: asyncio.Queue | None = None
    generation_changed: asyncio.Event = field(default_factory=asyncio.Event)
    stt_failure_announced: bool = False
    last_processed_user_text: str = ""
    closed: bool = False
    last_activity_at: float = field(default_factory=time.monotonic)



