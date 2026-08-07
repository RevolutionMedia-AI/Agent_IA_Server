import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "entornoLocal.env")


# ponytail: this file splits into two camps.
#
#   (A) Provider API keys: REMOVED. The user's rule was "Env fallback
#       de las API Keys: Elimínalos por completo." So
#       OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, etc.
#       are gone — every adapter must read from per-user
#       tools_integrations or per-agent agent row only. resolve_provider()
#       drops the env-fallback loop entirely.
#
#   (B) Operational tunables: KEPT. Things like sample rate, retry
#       count, queue size, webrtc VAD mode, fail-prompts, partial-
#       transcript debounce, billing per-minute — these are deploy
#       configuration, not user config. An operator edits them once
#       per environment (Railway dashboard) and they tune behaviour
#       across every call. Removing them would force a code change
#       for every adjustment.
#
# The grep that decides whether something belongs to (A) or (B):
#   - if the value is a credential used by an adapter to authenticate
#     to a provider → (A) per-user only
#   - if the value is a number/string that tunes how an adapter
#     behaves → (B) env tunable

PORT = int(os.environ.get("PORT", 8080))
PUBLIC_URL = os.getenv("PUBLIC_URL")

TWILIO_SR = 8000
TWILIO_CHANNELS = 1
FRAME_DURATION_MS = 20
TWILIO_OUTBOUND_CHUNK_BYTES = 160
TWILIO_OUTBOUND_PACING_MS = float(os.getenv("TWILIO_OUTBOUND_PACING_MS", "20"))

# webrtc VAD mode: 0..3 (0 less aggressive, 3 most aggressive).
WEBRTC_VAD_MODE = int(os.getenv("WEBRTC_VAD_MODE", "1"))

# Default call language — per-call override comes from agent row
# (session.preferred_language). Default 'es' for the platform's
# primary locale.
DEFAULT_CALL_LANGUAGE = os.getenv("DEFAULT_CALL_LANGUAGE", "es").strip().lower()

# If false, buffer the full assistant reply and run TTS once per reply.
REALTIME_TTS_STREAMING = os.getenv("REALTIME_TTS_STREAMING", "false").strip().lower() in {"1", "true", "yes", "on"}

LOG_TWILIO_PLAYBACK = os.getenv("LOG_TWILIO_PLAYBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
SAVE_TWILIO_FRAMES = os.getenv("SAVE_TWILIO_FRAMES", "false").strip().lower() in {"1", "true", "yes", "on"}
FILLER_TTS_ENABLED = os.getenv("FILLER_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").strip().lower() in {"1", "true", "yes", "on"}
LOG_TRANSCRIPT_CONTENT = os.getenv("LOG_TRANSCRIPT_CONTENT", "false").strip().lower() in {"1", "true", "yes", "on"}

STREAM_SID_WAIT_MAX_MS = int(os.getenv("STREAM_SID_WAIT_MAX_MS", "5000"))
STREAM_SID_WAIT_POLL_MS = int(os.getenv("STREAM_SID_WAIT_POLL_MS", "50"))

STT_AUDIO_QUEUE_MAXSIZE = int(os.getenv("STT_AUDIO_QUEUE_MAXSIZE", "300"))
REALTIME_AUDIO_QUEUE_MAXSIZE = int(os.getenv("REALTIME_AUDIO_QUEUE_MAXSIZE", "300"))
STT_MUTE_BUFFER_CHUNKS = int(os.getenv("STT_MUTE_BUFFER_CHUNKS", "25"))
TRANSCRIPT_QUEUE_MAXSIZE = int(os.getenv("TRANSCRIPT_QUEUE_MAXSIZE", "32"))
PLAYBACK_QUEUE_MAXSIZE = int(os.getenv("PLAYBACK_QUEUE_MAXSIZE", "1024"))
TEXT_SEGMENT_QUEUE_MAXSIZE = int(os.getenv("TEXT_SEGMENT_QUEUE_MAXSIZE", "16"))
UTTERANCE_QUEUE_MAXSIZE = int(os.getenv("UTTERANCE_QUEUE_MAXSIZE", "256"))

# Streaming segmentation (LLM token-stream → TTS chunk boundaries).
# ponytail: previous defaults (200) plus the old min_punct (5/15) cut a
# typical 194-char reply into 3 segments → 3 sequential Inworld TTS
# round-trips → ~2.1s of pure overhead per turn. Bumping these to 400
# (and the min_punct in pop_streaming_segments to 30/60) lets most
# customer-service replies ship as one TTS call. Inworld handles
# 400-char inputs fine. Override with the env var if needed.
STREAMING_SEGMENT_MAX_CHARS = int(os.getenv("STREAMING_SEGMENT_MAX_CHARS", "400"))
STREAMING_FIRST_SEGMENT_CHARS = int(os.getenv("STREAMING_FIRST_SEGMENT_CHARS", "400"))
PARTIAL_TRANSCRIPT_START_CHARS = int(os.getenv("PARTIAL_TRANSCRIPT_START_CHARS", "20"))
PARTIAL_TRANSCRIPT_DEBOUNCE_MS = int(os.getenv("PARTIAL_TRANSCRIPT_DEBOUNCE_MS", "200"))
FINAL_RESTART_DELTA_CHARS = int(os.getenv("FINAL_RESTART_DELTA_CHARS", "12"))
PARTIAL_PREFETCH_MAX_DELTA_CHARS = int(os.getenv("PARTIAL_PREFETCH_MAX_DELTA_CHARS", "40"))
FINAL_TRANSCRIPT_GRACE_MS = int(os.getenv("FINAL_TRANSCRIPT_GRACE_MS", "800"))
DIGIT_DICTATION_GRACE_MS = int(os.getenv("DIGIT_DICTATION_GRACE_MS", "2000"))
SHORT_FINAL_MAX_WORDS = int(os.getenv("SHORT_FINAL_MAX_WORDS", "3"))

# STT reconnect policy.
STT_RECONNECT_MAX_ATTEMPTS = int(os.getenv("STT_RECONNECT_MAX_ATTEMPTS", "3"))
STT_RECONNECT_BASE_DELAY_MS = int(os.getenv("STT_RECONNECT_BASE_DELAY_MS", "250"))
STT_RECONNECT_MAX_DELAY_MS = int(os.getenv("STT_RECONNECT_MAX_DELAY_MS", "2000"))

# Fallback prompts the assistant speaks when STT reconnects after
# a sustained outage. Per-call user_id-specific prompts live on the
# agent; these are the platform-default fallbacks.
STT_FAILURE_PROMPT_EN = os.getenv("STT_FAILURE_PROMPT_EN", "I'm having trouble hearing you right now.").strip()
STT_FAILURE_PROMPT_ES = os.getenv("STT_FAILURE_PROMPT_ES", "Estoy teniendo problemas para escucharte en este momento.").strip()

IDLE_SILENCE_TIMEOUT_SEC = float(os.getenv("IDLE_SILENCE_TIMEOUT_SEC", "45"))

# Max call duration guardrail: hard-timeout to prevent phantom calls
# consuming infinite STT/LLM units. Defaults to 30 minutes.
MAX_CALL_DURATION_SEC = float(os.getenv("MAX_CALL_DURATION_SEC", "1800"))

# ── Deepgram STT tunables (operational, NOT credentials) ────────────
# These tune the WebSocket URL that Deepgram sees. Per-user override
# of model is in tools_integrations.credentials.model; these env
# vars set the platform defaults.
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_STT_PUNCTUATE = os.getenv("DEEPGRAM_STT_PUNCTUATE", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_SMART_FORMAT = os.getenv("DEEPGRAM_STT_SMART_FORMAT", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_DETECT_LANGUAGE = os.getenv("DEEPGRAM_STT_DETECT_LANGUAGE", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_LANGUAGE_HINT = os.getenv("DEEPGRAM_STT_LANGUAGE_HINT", "es").strip().lower() or None
DEEPGRAM_STT_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_STT_ENDPOINTING_MS", "500"))
DEEPGRAM_STT_NUMERALS = os.getenv("DEEPGRAM_STT_NUMERALS", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_KEYWORDS: list[str] = [
    kw.strip()
    for kw in os.getenv(
        "DEEPGRAM_STT_KEYWORDS",
        "zero:2,one:2,two:2,three:2,four:2,five:2,six:2,seven:2,eight:2,nine:2,oh:1,order number:2",
    ).split(",")
    if kw.strip()
]

# Deepgram TTS tunables.
DEEPGRAM_TTS_ENCODING = os.getenv("DEEPGRAM_TTS_ENCODING", "mulaw").strip().lower()
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "8000"))

# ElevenLabs TTS tunables.
ELEVENLABS_TTS_MODEL_ID = os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_flash_v2_5")
ELEVENLABS_TTS_VOICE_ID = os.getenv("ELEVENLABS_TTS_VOICE_ID", "r8iaJkwUpytwsK5jNHRG")

# Rime TTS tunables.
RIME_TTS_MODEL_ID = os.getenv("RIME_TTS_MODEL_ID", "mist-v2")
RIME_TTS_SAMPLE_RATE = int(os.getenv("RIME_TTS_SAMPLE_RATE", "8000"))

# Default TTS provider — operational default the runtime uses when
# session.tts_provider is unset (debug calls, test paths).
DEFAULT_TTS_PROVIDER = os.getenv("DEFAULT_TTS_PROVIDER", "elevenlabs").strip().lower()

# VAD / barge-in / pre-speech buffer tunables (audio_ingest).
MIN_UTTERANCE_MS = int(os.getenv("MIN_UTTERANCE_MS", "180"))
MIN_SPEECH_FRAMES = int(os.getenv("MIN_SPEECH_FRAMES", "5"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "14"))
SPEECH_START_FRAMES = int(os.getenv("SPEECH_START_FRAMES", "1"))
MIN_BARGE_IN_FRAMES = int(os.getenv("MIN_BARGE_IN_FRAMES", "12"))
PRE_SPEECH_FRAMES = int(os.getenv("PRE_SPEECH_FRAMES", "5"))
TRIM_TRAILING_SILENCE_FRAMES = int(os.getenv("TRIM_TRAILING_SILENCE_FRAMES", "6"))
MIN_VOICE_RMS = int(os.getenv("MIN_VOICE_RMS", "260"))
BARGE_IN_MIN_RMS = int(os.getenv("BARGE_IN_MIN_RMS", "900"))
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "true").strip().lower() in {"1", "true", "yes", "on"}
ASSISTANT_ECHO_IGNORE_MS = float(os.getenv("ASSISTANT_ECHO_IGNORE_MS", "3000"))

# LLM context window and response length — runtime tunables.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "150"))

# OpenAI Realtime temperature (API minimum 0.6).
# ponytail: DEAD CODE — no adapter reads this. The Realtime GA migration
# removed session-level temperature; we never send it. Kept here so the
# env var still parses (some ops dashboards watch it), but don't expect
# changing it to do anything. Add per-call override support when GA
# exposes response.create events.
OPENAI_REALTIME_TEMPERATURE = float(os.getenv("OPENAI_REALTIME_TEMPERATURE", "0.7"))

# Filler text the LLM emits during pause (operational content, not
# per-user config). Empty by default — the LLM fills its own pauses.
FILLER_TEXT_EN = os.getenv("FILLER_TEXT_EN", "").strip()
FILLER_TEXT_ES = os.getenv("FILLER_TEXT_ES", "").strip()
# Per-stage timeout tunables. STT_TIMEOUT_SEC=0 disables the cap
# (legacy behavior). LLM and TTS defaults are reasonable for voice
# agents; operators tune per-deploy if they hit timeouts.
STT_TIMEOUT_SEC = float(os.getenv("STT_TIMEOUT_SEC", "0"))
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "5.0"))
TTS_TTFB_TIMEOUT_SEC = float(os.getenv("TTS_TTFB_TIMEOUT_SEC", "15.0"))
TTS_IDLE_TIMEOUT_SEC = float(os.getenv("TTS_IDLE_TIMEOUT_SEC", "3.0"))
TTS_TIMEOUT_SEC = float(os.getenv("TTS_TIMEOUT_SEC", "45.0"))
TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "1"))
TTS_RETRY_BACKOFF_MS = int(os.getenv("TTS_RETRY_BACKOFF_MS", "250"))
FILLER_DELAY_MS = int(os.getenv("FILLER_DELAY_MS", "1200"))

# ponytail: per-minute pricing in USD. Two tiers:
# - own_key: the user is bringing their own provider credential
# - platform_key: the user is using our provisioned credential
PRICE_OWN_KEY_PER_MIN = float(os.getenv("PRICE_OWN_KEY_PER_MIN", "0.07"))
PRICE_PLATFORM_KEY_PER_MIN = float(os.getenv("PRICE_PLATFORM_KEY_PER_MIN", "0.14"))
