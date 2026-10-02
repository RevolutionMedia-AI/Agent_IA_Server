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
    SPEECH_FRAMES_MAX,
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
    # ponytail: master handoff switch (024). Denormalized from the
    # agent row at call start. False = Full-AI: transfer tools are
    # filtered from tools[] and invoking one returns a tool error
    # instead of dialing. Default True (legacy rows predate it).
    transfer_enabled: bool = True
    # ponytail: loop-guard state for this call. `call_plan` is the sealed
    # CallPlan (services/transfer_cascade.py) that arrived on the
    # <Stream> when a transfer chain handed the call back to the AI. It
    # carries the running rounds/attempts so the tool executor can refuse
    # a further human handoff as a REAL guard rather than trusting a
    # sentence in the system prompt. `handoff_disabled` is sticky: once
    # set, no further handoff happens for the rest of the call.
    # None/absent = no prior chain on this call, counters start at zero.
    call_plan: object | None = None
    handoff_disabled: bool = False
    # Per-session custom system prompt (overrides default if set)
    custom_prompt: str | None = None
    # ponytail: the agent's welcome message, set on the session at
    # call start if the linked phone number has an agent with one.
    # played by play_initial_greeting() so the caller hears the
    # agent speak first.
    welcome_message: str | None = None
    # ponytail: 2026-09-23 — wall-clock time of the initial greeting
    # mark_ack (when Twilio confirms the caller has heard it). The idle
    # silence monitor uses this to skip the first deadline after the
    # greeting so the user isn't flagged "silent" while they're still
    # listening. Without this the monitor fires "Are you still there?"
    # ~5s after the TTS playback ends, which is normal pause-for-thought
    # time. None until set by the mark_ack handler.
    initial_greeting_ended_at: float | None = None
    # ponytail: 2026-09-23 — symmetric for idle prompts. Same problem
    # class: long first_timeout/sub_timeout + a slow LLM reply can
    # make the silence baseline stale. The monitor seeds the next
    # interval's baseline from the last mark_ack for any idle prompt.
    last_idle_prompt_ended_at: float | None = None
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
    # ponytail: idle / silence detection (008_agent_idle_settings.sql).
    # All None = fall back to the global IDLE_SILENCE_TIMEOUT_SEC (legacy
    # single-timeout-then-close behaviour). When idle_enabled=True the
    # monitor in session_runtime.monitor_idle_silence plays the prompt
    # messages at the configured intervals and closes the websocket after
    # idle_max_attempts + idle_disconnect_timeout_sec of silence.
    idle_enabled: bool | None = None
    idle_first_timeout_sec: int | None = None
    idle_first_message: str | None = None
    idle_subsequent_timeout_sec: int | None = None
    idle_final_message: str | None = None
    idle_disconnect_timeout_sec: int | None = None
    idle_max_attempts: int | None = None
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
    # ponytail: Twilio subaccount creds that own this call. Denormalized
    # at WS start from the phone_numbers row matched by (user_id, agent_id)
    # via db_phone_numbers.find_for_agent. Used by the call_transfer tool
    # executor — Twilio's calls(call_sid).update API call has to authenticate
    # against the subaccount that owns the call, and that's whichever sub
    # the operator wired to the agent's assigned number. None on legacy
    # sessions (no phone_number row matched) → call_transfer is silently
    # disabled for that call.
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    # ponytail: per-slot credential source toggle (009_agent_use_own_key.sql).
    # False (default) lets the resolver fall back to platform env vars
    # (Railway OPENAI_API_KEY etc.) when no per-user key is stored.
    # True forces the resolver to ignore platform env and require a
    # per-user credential. Denormalized at WS start so the TTS/STT/LLM
    # adapters can pick the right resolver mode without a second DB
    # round-trip per call.
    stt_use_own_key: bool = False
    llm_use_own_key: bool = False
    tts_use_own_key: bool = False
    vad_buffer: bytearray = field(default_factory=bytearray)
    pre_speech_frames: deque[bytes] = field(default_factory=lambda: deque(maxlen=PRE_SPEECH_FRAMES))
    # AUDIO-005: deque(maxlen=...) bounds the per-utterance PCM buffer at
    # SPEECH_FRAMES_MAX frames (~60 s default). Replaces the unbounded
    # list — a sustained-signal caller (or a stuck VAD) could otherwise
    # grow memory until END_SILENCE_FRAMES fired. drop-oldest is the
    # right policy here: the leading frames are stale by definition once
    # the utterance exceeds the cap.
    speech_frames: deque[bytes] = field(default_factory=lambda: deque(maxlen=SPEECH_FRAMES_MAX))
    speech_frame_count: int = 0
    voice_streak: int = 0
    silence_frames: int = 0
    # ponytail: adaptive noise floor for THIS line. Measured 2026-10-01:
    # webrtcvad flags lowpassed noise as speech from ~2% full scale upward
    # at every aggressiveness mode (0..3), so WEBRTC_VAD_MODE is a no-op
    # for line hiss and MIN_VOICE_RMS is the only real gate. On a noisy
    # PSTN line that fixed gate is either too low (hiss opens phantom
    # turns) or too high (quiet speech gets dropped). Seeded at 0 and
    # treated as "not yet estimated" so the first frames still use
    # MIN_VOICE_RMS.
    noise_floor_rms: int = 0
    # ponytail: 2026-10-01 — bumped once per end-of-speech in
    # audio_ingest. The OpenAI transcription session (adapters/
    # openai_stt_transcription.py) has no server_vad, so it has to send
    # `input_audio_buffer.commit` itself; watching this counter lets it
    # reuse THIS VAD (and the adaptive floor above) instead of running a
    # second energy detector that would disagree on exactly the noisy
    # lines that motivated the floor.
    stt_turn_end_seq: int = 0
    # ponytail: 2026-10-01 — monotonic timestamp of that same
    # end-of-speech bump. The transcription adapter subtracts it from the
    # arrival of the first partial / the final so the recorded
    # stt_partial_ms / stt_final_ms are REAL end-of-speech-to-transcript
    # measurements rather than the static estimates in the UI. These are
    # what will eventually replace the estimated table.
    stt_turn_end_at: float | None = None
    # ponytail: 2026-10-01 — monotonic timestamp of the VOICE START that
    # opened the turn. Partials arrive while the caller is still talking,
    # so their latency is measured from here. Finals are measured from
    # stt_turn_end_at. Measuring a partial from the END stamp produced
    # 12-18 s figures that were the time since the previous turn ended,
    # not a latency at all.
    stt_turn_start_at: float | None = None
    # 027_agent_stt_latency_mode.sql. The agent's OpenAI latency/accuracy
    # dial. None for models that have no dial (gpt-transcribe) and for
    # legacy rows, where the adapter substitutes the platform default.
    stt_latency_mode: str | None = None
    active_generation: int = 0
    # ponytail: every generation <= this is invalid; producers abort when
    # their gen <= cancelled_through. Set by interrupt_current_turn after
    # bumping active_generation. Replaces per-adapter cancel handshakes —
    # producers pull active_generation at each emit and compare.
    # Initial value MUST be -1 so the ``generation <= cancelled_through``
    # check in playback_loop is false for the FIRST generation (gen=0)
    # before any barge-in has happened. Setting this to 0 caused the
    # in-band greeting to be silently skipped on every call, producing
    # 30+ s of dead silence (the 2026-08-17 production incident).
    cancelled_through: int = -1
    # ponytail: monotonic ts of last barge-in, for diagnostics only.
    # Producers must NOT branch on this; it's instrumentation.
    barge_in_at: float | None = None
    response_active: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    utterance_queue: asyncio.Queue[tuple[int, bytes]] = field(default_factory=lambda: asyncio.Queue(maxsize=UTTERANCE_QUEUE_MAXSIZE))
    playback_queue: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE))
    stt_audio_queue: asyncio.Queue[bytes | None] = field(default_factory=lambda: asyncio.Queue(maxsize=STT_AUDIO_QUEUE_MAXSIZE))
    stt_mute_buffer: deque[bytes] = field(default_factory=lambda: deque(maxlen=STT_MUTE_BUFFER_CHUNKS))
    # ponytail: 2026-09-03 — cursor into stt_mute_buffer marking the
    # offset at which the most recent assistant turn started. Anything
    # captured BEFORE the cursor was either genuine user audio or audio
    # of the previous (now-finished) assistant turn; both are safe to
    # re-inject as fresh STT input. Anything captured AFTER the cursor
    # belongs to the live assistant turn and must NOT be replayed into
    # STT (it would echo the agent's own voice back into the
    # transcription pipeline).
    stt_mute_buffer_cursor: int = 0
    transcript_queue: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=TRANSCRIPT_QUEUE_MAXSIZE))
    tasks: set[asyncio.Task] = field(default_factory=set)
    # ponytail: name → monotonic-ts when mark was sent, so the consumer in
    # STT_Server.py's mark handler can compute RTT.
    pending_marks: dict[str, float] = field(default_factory=dict)
    mark_counter: int = 0
    # ponytail: AUDIO echo gate — count of segments of the ACTIVE
    # generation whose Mark events Twilio has not yet acked.
    # ``pending_marks`` (above) is a per-mark dictionary used for RTT
    # computation; ``pending_playback_marks`` is the per-generation
    # counter used to decide when the assistant has truly finished
    # playing — only when this reaches zero AND ``active_generation``
    # matches, ``assistant_speaking`` drops. Earlier code checked
    # ``not pending_marks`` after every pop, which fired on every
    # segment of a multi-segment reply (one at a time) and let VAD
    # grab the audio echo / click / TTS-tail that lands between
    # segments. See the 2026-08-14 audio review: each seg-9/seg-10/
    # seg-11 ack of gen=7 logged "last mark cleared" and dropped
    # ``assistant_speaking`` three times in a row.
    pending_playback_marks: int = 0
    assistant_speaking: bool = False
    assistant_started_at: float | None = None
    # ponytail: 2026-10-02 — monotonic timestamp by which the audio sent
    # so far should have finished playing (started_at + frames * 20 ms +
    # margin). The watchdog compares against this instead of a flat
    # timeout, so a lost mark ack on a 2 s reply unsticks in ~5 s while a
    # lost mark on a 9 s greeting unsticks in ~12 s. Without it the flag
    # stayed True for a fixed 30 s no matter how short the audio was.
    assistant_expected_end_at: float | None = None
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
    # ponytail: set by the 'start' event handler when Twilio sends streamSid.
    # Replaces the polling loop in playback_service — wait_signals.wait_stream_ready
    # blocks on this event instead of spinning on a 50ms timer.
    stream_ready: asyncio.Event = field(default_factory=asyncio.Event)
    # ponytail: per-call pipeline stage timer. Lazily set by
    # session_runtime.register_session; mark_stage() in wait_signals.py
    # will lazy-init if mark arrives before register_session. Defined
    # here as Optional + forward reference to dodge the circular import
    # between domain.session and services._instrumentation.
    stage_timer: object | None = None
    stt_failure_announced: bool = False
    last_processed_user_text: str = ""
    closed: bool = False
    last_activity_at: float = field(default_factory=time.monotonic)



