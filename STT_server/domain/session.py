from collections import deque
from dataclasses import dataclass, field
import asyncio
import time

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    DEFAULT_TTS_PROVIDER,
    PLAYBACK_QUEUE_MAXSIZE,
    PRE_SPEECH_FRAMES,
    REALTIME_AUDIO_QUEUE_MAXSIZE,
    STT_AUDIO_QUEUE_MAXSIZE,
    STT_MUTE_BUFFER_CHUNKS,
    TRANSCRIPT_QUEUE_MAXSIZE,
    UTTERANCE_QUEUE_MAXSIZE,
)

# Valid TTS providers and languages
VALID_TTS_PROVIDERS = {"elevenlabs", "rime", "openai", "deepgram", "inworld"}
VALID_LANGUAGES = {"en", "es"}


@dataclass
class CallSession:
    session_key: str
    call_sid: str | None = None
    stream_sid: str | None = None
    # ponytail: usage tracking. started_at is wall-clock seconds since
    # epoch, set when Twilio sends the `start` event. agent_id comes
    # from the <Parameter name="agent_id" /> Twilio passes through
    # customParameters. Both feed the per-call record written at
    # cleanup_session time.
    started_at: float | None = None
    agent_id: str | None = None
    preferred_language: str = field(default_factory=lambda: DEFAULT_CALL_LANGUAGE)
    # Per-session custom system prompt (overrides default if set)
    custom_prompt: str | None = None
    # ponytail: the agent's welcome message, set on the session at
    # call start if the linked phone number has an agent with one.
    # played by play_initial_greeting() so the caller hears the
    # agent speak first.
    welcome_message: str | None = None
    # Per-session TTS provider: "elevenlabs" or "rime"
    tts_provider: str = field(default_factory=lambda: DEFAULT_TTS_PROVIDER)
    # ponytail: tts_model = concrete model id sent to the TTS provider.
    # Set from the agent config at call start so the dispatcher picks
    # the right model (e.g. "inworld-tts-2", "aura-asteria-en").
    tts_model: str | None = None
    # ponytail: voice_id for TTS. ElevenLabs / Inworld take a voice id,
    # Deepgram / OpenAI take a voice name. The agent's voice field is
    # also kept on the session as `tts_voice` for UI display.
    voice_id: str | None = None
    tts_voice: str | None = None
    # ponytail: which LLM provider the agent picked. Mirrors the FE's
    # llm_provider dropdown — read by openai_llm._client_for_session so
    # the dispatch goes to OpenAI, MiniMax (OpenAI-compat with custom
    # base_url), Anthropic or Gemini. The 'start' event handler in
    # STT_Server.py ALWAYS sets this from (a) the agent config or
    # (b) the per-user credential auto-detect; no env-var default.
    # If both are empty, the call adapter surfaces a clear error.
    llm_provider: str = ""
    # Concrete model id to send to that provider (gpt-4o-mini,
    # claude-3-5-sonnet-20241022, gemini-1.5-pro, minimax, ...).
    # Comes from the agent config at call start.
    llm_model: str | None = None
    # ponytail: per-agent runtime knobs (006_agent_runtime_params.sql).
    # None = use the adapter's default (0.2 / MAX_RESPONSE_TOKENS / 1.0).
    # The LLM adapter reads these on every chat-completions call; the
    # TTS adapter reads tts_speed where the provider supports it
    # (OpenAI / ElevenLabs / Inworld; Deepgram & Rime ignore).
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    tts_speed: float | None = None
    # ponytail: which STT provider the agent picked. Mirrors the FE's
    # stt_provider dropdown. Set by the 'start' event handler in
    # STT_Server.py from (a) the agent config or (b) the per-user
    # credential auto-detect. The field is declared here so the
    # dispatch below doesn't raise AttributeError on an unconfigured
    # session; the dispatch logs a clear error if it's still empty.
    stt_provider: str = ""
    stt_model: str | None = None
    # Tenant ID this session belongs to (set when call comes from a configured tenant)
    tenant_id: str | None = None
    # Owning user — resolved from the tenant (or set by an admin tool call).
    # Adapters read this to pick per-user provider credentials. None means
    # "no per-user config, use system env-var defaults."
    user_id: str | None = None
    vad_buffer: bytearray = field(default_factory=bytearray)
    pre_speech_frames: deque[bytes] = field(default_factory=lambda: deque(maxlen=PRE_SPEECH_FRAMES))
    speech_frames: list[bytes] = field(default_factory=list)
    speech_frame_count: int = 0
    voice_streak: int = 0
    silence_frames: int = 0
    active_generation: int = 0
    response_active: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    utterance_queue: asyncio.Queue[tuple[int, bytes]] = field(default_factory=lambda: asyncio.Queue(maxsize=UTTERANCE_QUEUE_MAXSIZE))
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



