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
# ponytail: AUDIO-006 — per-event base64 payload cap. Twilio sends
# ~160-800 byte PCMU payloads; 8192 decoded bytes leaves generous
# headroom (2 frames at 8 kHz / 20 ms is 320 bytes; 8192 is 25x).
# Bounds memory if an upstream proxy or test harness sends a huge
# event. Compare against b64 length (4/3 inflation), so we never
# even allocate the decoded buffer.
MAX_MEDIA_PAYLOAD_BYTES = int(os.getenv("MAX_MEDIA_PAYLOAD_BYTES", "8192"))

# webrtc VAD mode: 0..3 (0 less aggressive, 3 most aggressive).
# ponytail: P0 follow-up — bumped default 1 -> 2 per operator feedback.
# Mode 1 fires on noise bursts (clicks, line hiss) reaching BARGE_IN_MIN_RMS
# sustained for MIN_BARGE_IN_FRAMES; mode 2 rejects more non-speech while
# still triggering on real human speech (which has distinctive formant
# transitions that VAD mode 2 catches reliably).
WEBRTC_VAD_MODE = int(os.getenv("WEBRTC_VAD_MODE", "2"))

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

# ponytail: P0 — was 5000ms polling; now event-based via wait_signals.py
STREAM_SID_WAIT_TIMEOUT_MS = int(os.getenv("STREAM_SID_WAIT_TIMEOUT_MS", "5000"))

STT_AUDIO_QUEUE_MAXSIZE = int(os.getenv("STT_AUDIO_QUEUE_MAXSIZE", "300"))
REALTIME_AUDIO_QUEUE_MAXSIZE = int(os.getenv("REALTIME_AUDIO_QUEUE_MAXSIZE", "300"))
STT_MUTE_BUFFER_CHUNKS = int(os.getenv("STT_MUTE_BUFFER_CHUNKS", "25"))
TRANSCRIPT_QUEUE_MAXSIZE = int(os.getenv("TRANSCRIPT_QUEUE_MAXSIZE", "32"))
PLAYBACK_QUEUE_MAXSIZE = int(os.getenv("PLAYBACK_QUEUE_MAXSIZE", "1024"))
# ponytail: P1 — bumped from 16 to 64. The previous maxsize silently
# dropped the LLM reply opener when load exceeded ~16 segments
# enqueued faster than the TTS consumer drained them.
TEXT_SEGMENT_QUEUE_MAXSIZE = int(os.getenv("TEXT_SEGMENT_QUEUE_MAXSIZE", "64"))
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
# ponytail: 2026-08-14 audio review — A/B-test the segmentation hypothesis
# (multiple TTS calls per reply produce audible discontinuities that
# AMR re-encoding turns into metallic noise). When this flag is true,
# pop_streaming_segments() and split_tts_segments() both return the
# whole reply as a single segment, regardless of length. Off by
# default; flip on via env var for the operator's A/B test.
# Expected behaviour: 1 reply = 1 TTS request = 1 playback with no
# inter-segment concatenation discontinuity. If the artifacts
# disappear in this mode, segmentation is the cause; if they
# remain, the cause is downstream of TTS (pacing, mux, codec).
TTS_SINGLE_SEGMENT_PER_REPLY = os.getenv("TTS_SINGLE_SEGMENT_PER_REPLY", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
# ponytail: opt-in audio capture for the 2026-08-14 A/B test. When
# this env var points at an existing directory, the runtime writes
# two files per Twilio callSid:
#   <dir>/A_inworld_<callSid>.mulaw  — exact bytes the TTS adapter
#                                       produced (post-resample, post-
#                                       mu-law encode), in order.
#   <dir>/B_twilio_<callSid>.mulaw   — exact 160-byte frames sent
#                                       to Twilio, in order.
# Diffing A and B against each other and against the AMR recording
# (capture C, from Twilio/carrier) tells us exactly which stage
# introduces the noise. Default empty = disabled; capture is
# best-effort (a write failure logs once and disables for the
# call, never crashes the live path).
TTS_AUDIO_CAPTURE_DIR = os.getenv("TTS_AUDIO_CAPTURE_DIR", "").strip()
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

# ponytail: 2026-08-14 — initial greeting the agent speaks as soon
# as the call connects. Sent DIRECTLY to TTS (no STT, no LLM
# intermediate), so the caller hears the agent greet them without
# waiting for them to speak first. Per-agent override lives on
# agent.welcome_message (higher priority); the env vars below
# are the platform fallback when the agent has no welcome_message
# configured. Default-on so a fresh deployment without any per-agent
# config still greets the caller instead of dead-air silence.
INITIAL_GREETING_ENABLED = os.getenv("INITIAL_GREETING_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
INITIAL_GREETING_TEXT_EN = os.getenv(
    "INITIAL_GREETING_TEXT_EN",
    "Thank you for calling. My name is Athenas, your virtual assistant. How can I help you today?",
).strip()
INITIAL_GREETING_TEXT_ES = os.getenv(
    "INITIAL_GREETING_TEXT_ES",
    "Hola, gracias por llamar. Mi nombre es Athenas, soy su asistente virtual. ¿En qué puedo ayudarle?",
).strip()
# ponytail: 2026-08-14 production log review — the operator
# reported "no se escucha nada hasta después de 28 segundos"
# after the call connects. Root cause was a 236-char welcome_message
# in Spanish that took ~18 s to play at speakingRate=1.15; plus
# the user's reaction time + barge-in detection + LLM TTFB added
# another ~10 s before the first non-greeting agent message. The
# 28 s was real and avoidable without changing the welcome text:
# cap the greeting to a length that plays in < ~10 s, regardless
# of what the agent row has stored. Agents with shorter greetings
# are unaffected; agents with longer greetings get truncated at
# the nearest sentence boundary (or the configured fallback) so
# the caller hears the agent within a few seconds of connecting.
# Default 200 chars (~10 s of speech at 1.15x). Set via env var.
INITIAL_GREETING_MAX_CHARS = int(os.getenv("INITIAL_GREETING_MAX_CHARS", "200"))

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
# ponytail: 2026-08-14 production log review (CA5f48c4...) — VAD was
# firing on 80 ms of voice + 280 ms of silence and forwarding 3-char
# transcripts ("len=3") to the LLM, which then responded with
# "Disculpe, esa parte no la escuché bien. ¿Me la podría repetir?"
# every time the user started speaking briefly. The LLM correctly
# identified 3 chars as un-understandable; the bug was that the VAD
# treated such short utterances as a turn at all. End the turn on
# more sustained silence; require more sustained voice before
# starting one.
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "25"))
# ponytail: 2026-08-14 production log review — bump 4 → 6 (~120 ms).
# 4 frames (~80 ms) was still triggerable by a single click or
# post-playback tail. 6 frames requires ~120 ms of sustained
# voice-positive + RMS-above-threshold before INICIO DE VOZ fires.
# Still well under a human phoneme (~100 ms is the shortest stop
# consonant); latency cost is negligible (~40 ms).
SPEECH_START_FRAMES = int(os.getenv("SPEECH_START_FRAMES", "6"))
# ponytail: 2026-08-14 — minimum voice frames for an utterance to be
# forwarded to the LLM. Below this threshold the VAD reset + drop
# the buffer as noise (a click, a consonant, a single frame of
# echo). Without this filter every echo / click reached the LLM and
# the LLM correctly responded "I didn't understand" because the
# input was uninterpretable. At 20 ms/frame, 25 frames = 500 ms of
# sustained voice — well below a real-word duration but well above
# anything a click / echo burst can sustain.
MIN_UTTERANCE_VOICE_FRAMES = int(os.getenv("MIN_UTTERANCE_VOICE_FRAMES", "25"))
# ponytail: P0 follow-up — bumped default 12 -> 16 (~320ms of consecutive
# VAD-positive frames required). Real human speech sustains voice
# activity for 320ms easily; click/noise bursts typically don't.
MIN_BARGE_IN_FRAMES = int(os.getenv("MIN_BARGE_IN_FRAMES", "16"))
PRE_SPEECH_FRAMES = int(os.getenv("PRE_SPEECH_FRAMES", "5"))
# AUDIO-005: hard cap on the per-utterance PCM buffer (audio_ingest).
# 20 ms @ 8 kHz PCM16 = 320 bytes/frame. 3000 frames = 60 s ≈ 960 KB.
# Bounded via deque(maxlen=...) so a sustained-signal caller cannot grow
# memory unboundedly while waiting for END_SILENCE_FRAMES. Drop-oldest
# is fine here: the leading frames are stale by definition once the
# utterance exceeds the cap.
SPEECH_FRAMES_MAX = int(os.getenv("SPEECH_FRAMES_MAX", "3000"))
# AUDIO-005: hard cap on the partial-decoded vad_buffer. Stays small in
# the happy path (under 1 s of audio, ~16 KB). This guard rejects
# pathological inputs (bursts of oversized media payloads) before the
# bytearray grows past a sane ceiling. Trim from the left when full.
VAD_BUFFER_MAX_BYTES = int(os.getenv("VAD_BUFFER_MAX_BYTES", "65536"))  # 64 KB ≈ 2 s
# ponytail: P0 follow-up — bumped 260 -> 800. The RMS threshold is the
# PRIMARY filter for non-speech noise. Telephony line noise + clicks
# typically peak well below 800; real human voice easily clears 1000+.
MIN_VOICE_RMS = int(os.getenv("MIN_VOICE_RMS", "800"))
# ponytail: P0 follow-up — bumped 900 -> 2500. The pre-barge-in RMS
# check uses pre_speech_frames[-MIN_BARGE_IN_FRAMES:] averaged, so the
# absolute RMS must be loud AND sustained for barge-in to fire.
BARGE_IN_MIN_RMS = int(os.getenv("BARGE_IN_MIN_RMS", "2500"))
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "true").strip().lower() in {"1", "true", "yes", "on"}
# ponytail: P0 follow-up — bumped 3000 -> 5000ms. The echo of the AI's
# own voice reaching the user's phone mic can sustain VAD-trigger
# levels past the previous 3s window. 5s covers any reasonable
# greeting reply (the welcome is ~12s of speech; we accept barge-in
# risk on the tail of the reply rather than mid-greeting).
ASSISTANT_ECHO_IGNORE_MS = float(os.getenv("ASSISTANT_ECHO_IGNORE_MS", "5000"))

# LLM context window and response length — runtime tunables.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
# ponytail: lowered 150 -> 100 per operator feedback (the LLM was
# hitting the cap with verbose enumeration — "le recomiendo el plan
# de veintitrés con veinte balboas, incluye cinco gigabytes,
# doscientos cincuenta minutos, ..." — 100 tokens forces tighter
# replies (≈400 chars), shorter TTS playback, less perceived
# repetition. Per-agent override via session.llm_max_tokens still wins.
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "100"))

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
