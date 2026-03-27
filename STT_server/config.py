import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "entornoLocal.env")


PORT = int(os.environ.get("PORT", 8080))
PUBLIC_URL = os.getenv("PUBLIC_URL")



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
OPENAI_REALTIME_TEMPERATURE = float(os.getenv("OPENAI_REALTIME_TEMPERATURE", "0.7"))  # API minimum is 0.6
USE_OPENAI_REALTIME = os.getenv("USE_OPENAI_REALTIME", "true").strip().lower() in {"1", "true", "yes", "on"}


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_STT_PUNCTUATE = os.getenv("DEEPGRAM_STT_PUNCTUATE", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_SMART_FORMAT = os.getenv("DEEPGRAM_STT_SMART_FORMAT", "true").strip().lower() in {"1", "true", "yes", "on"}
# Language detection disabled — full English mode.
# To re-enable, change the default back to "true".
DEEPGRAM_STT_DETECT_LANGUAGE = os.getenv("DEEPGRAM_STT_DETECT_LANGUAGE", "false").strip().lower() in {"1", "true", "yes", "on"}
# Force English language hint for Deepgram STT.
DEEPGRAM_STT_LANGUAGE_HINT = os.getenv("DEEPGRAM_STT_LANGUAGE_HINT", "en").strip().lower() or None
DEEPGRAM_STT_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_STT_ENDPOINTING_MS", "500"))
DEEPGRAM_UTTERANCE_END_MS = int(os.getenv("DEEPGRAM_UTTERANCE_END_MS", "1000"))
DEEPGRAM_STT_NUMERALS = os.getenv("DEEPGRAM_STT_NUMERALS", "true").strip().lower() in {"1", "true", "yes", "on"}
DEEPGRAM_STT_KEYWORDS: list[str] = [
    kw.strip()
    for kw in os.getenv(
        "DEEPGRAM_STT_KEYWORDS",
        "zero:2,one:2,two:2,three:2,four:2,five:2,six:2,seven:2,eight:2,nine:2,oh:1,order number:2"
    ).split(",")
    if kw.strip()
]


RIME_API_KEY = os.getenv("RIME_API_KEY")
RIME_TTS_MODEL_ID = os.getenv("RIME_TTS_MODEL_ID", "arcana")
RIME_TTS_SPEAKER_EN = os.getenv("RIME_TTS_SPEAKER_EN", "lyra")
RIME_TTS_SPEAKER_ES = os.getenv("RIME_TTS_SPEAKER_ES", "celestino")
RIME_TTS_SAMPLE_RATE = int(os.getenv("RIME_TTS_SAMPLE_RATE", "8000"))


TWILIO_SR = 8000
TWILIO_CHANNELS = 1
FRAME_DURATION_MS = 30
TWILIO_OUTBOUND_CHUNK_BYTES = 160
TWILIO_OUTBOUND_PACING_MS = float(os.getenv("TWILIO_OUTBOUND_PACING_MS", "20"))


MIN_UTTERANCE_MS = int(os.getenv("MIN_UTTERANCE_MS", "180"))
MIN_SPEECH_FRAMES = int(os.getenv("MIN_SPEECH_FRAMES", "5"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "18"))
SPEECH_START_FRAMES = int(os.getenv("SPEECH_START_FRAMES", "2"))
MIN_BARGE_IN_FRAMES = int(os.getenv("MIN_BARGE_IN_FRAMES", "12"))
PRE_SPEECH_FRAMES = int(os.getenv("PRE_SPEECH_FRAMES", "5"))
TRIM_TRAILING_SILENCE_FRAMES = int(os.getenv("TRIM_TRAILING_SILENCE_FRAMES", "6"))
MIN_VOICE_RMS = int(os.getenv("MIN_VOICE_RMS", "260"))
BARGE_IN_MIN_RMS = int(os.getenv("BARGE_IN_MIN_RMS", "900"))
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "true").strip().lower() in {"1", "true", "yes", "on"}
ASSISTANT_ECHO_IGNORE_MS = float(os.getenv("ASSISTANT_ECHO_IGNORE_MS", "1200"))


STT_TIMEOUT_SEC = float(os.getenv("STT_TIMEOUT_SEC", "0"))
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "5.0"))
TTS_TIMEOUT_SEC = float(os.getenv("TTS_TIMEOUT_SEC", "10.0"))
TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "1"))
TTS_RETRY_BACKOFF_MS = int(os.getenv("TTS_RETRY_BACKOFF_MS", "250"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "150"))
DEFAULT_CALL_LANGUAGE = os.getenv("DEFAULT_CALL_LANGUAGE", "en").strip().lower()


LOG_TWILIO_PLAYBACK = os.getenv("LOG_TWILIO_PLAYBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
FILLER_TTS_ENABLED = os.getenv("FILLER_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").strip().lower() in {"1", "true", "yes", "on"}


STT_AUDIO_QUEUE_MAXSIZE = int(os.getenv("STT_AUDIO_QUEUE_MAXSIZE", "300"))
REALTIME_AUDIO_QUEUE_MAXSIZE = int(os.getenv("REALTIME_AUDIO_QUEUE_MAXSIZE", "300"))
# How many Twilio audio chunks (20ms each) to keep during mute.
# 25 chunks = 500ms of audio to replay when assistant stops speaking.
STT_MUTE_BUFFER_CHUNKS = int(os.getenv("STT_MUTE_BUFFER_CHUNKS", "25"))
TRANSCRIPT_QUEUE_MAXSIZE = int(os.getenv("TRANSCRIPT_QUEUE_MAXSIZE", "32"))
PLAYBACK_QUEUE_MAXSIZE = int(os.getenv("PLAYBACK_QUEUE_MAXSIZE", "256"))
TEXT_SEGMENT_QUEUE_MAXSIZE = int(os.getenv("TEXT_SEGMENT_QUEUE_MAXSIZE", "16"))


STREAMING_SEGMENT_MAX_CHARS = int(os.getenv("STREAMING_SEGMENT_MAX_CHARS", "120"))
STREAMING_FIRST_SEGMENT_CHARS = int(os.getenv("STREAMING_FIRST_SEGMENT_CHARS", "30"))
FILLER_TEXT_EN = os.getenv("FILLER_TEXT_EN", "").strip()
FILLER_TEXT_ES = os.getenv("FILLER_TEXT_ES", "").strip()
FILLER_DELAY_MS = int(os.getenv("FILLER_DELAY_MS", "1200"))
PARTIAL_TRANSCRIPT_START_CHARS = int(os.getenv("PARTIAL_TRANSCRIPT_START_CHARS", "20"))
PARTIAL_TRANSCRIPT_DEBOUNCE_MS = int(os.getenv("PARTIAL_TRANSCRIPT_DEBOUNCE_MS", "200"))
FINAL_RESTART_DELTA_CHARS = int(os.getenv("FINAL_RESTART_DELTA_CHARS", "12"))
PARTIAL_PREFETCH_MAX_DELTA_CHARS = int(os.getenv("PARTIAL_PREFETCH_MAX_DELTA_CHARS", "40"))
FINAL_TRANSCRIPT_GRACE_MS = int(os.getenv("FINAL_TRANSCRIPT_GRACE_MS", "800"))
DIGIT_DICTATION_GRACE_MS = int(os.getenv("DIGIT_DICTATION_GRACE_MS", "2000"))
SHORT_FINAL_MAX_WORDS = int(os.getenv("SHORT_FINAL_MAX_WORDS", "3"))
STT_RECONNECT_MAX_ATTEMPTS = int(os.getenv("STT_RECONNECT_MAX_ATTEMPTS", "3"))
STT_RECONNECT_BASE_DELAY_MS = int(os.getenv("STT_RECONNECT_BASE_DELAY_MS", "250"))
STT_RECONNECT_MAX_DELAY_MS = int(os.getenv("STT_RECONNECT_MAX_DELAY_MS", "2000"))
STT_FAILURE_PROMPT_EN = os.getenv("STT_FAILURE_PROMPT_EN", "I'm having trouble hearing you right now.").strip()
STT_FAILURE_PROMPT_ES = os.getenv("STT_FAILURE_PROMPT_ES", "Estoy teniendo problemas para escucharte en este momento.").strip()
INITIAL_GREETING_ENABLED = os.getenv("INITIAL_GREETING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
INITIAL_GREETING_TEXT = os.getenv(
	"INITIAL_GREETING_TEXT",
	"Thank you for calling Cialix Customer Support. My name is Tessa. How can I help you today?",
).strip()
IDLE_SILENCE_TIMEOUT_SEC = float(os.getenv("IDLE_SILENCE_TIMEOUT_SEC", "45"))

