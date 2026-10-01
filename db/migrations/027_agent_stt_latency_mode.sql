-- ============================================================================
-- 027 · agent_stt_latency_mode.sql
-- ----------------------------------------------------------------------------
-- Per-agent STT latency/accuracy dial for OpenAI's transcription models.
--
-- This product is a CASCADE voice agent, not speech-to-speech:
--   Twilio audio -> STT -> text -> independent LLM -> independent TTS
-- OpenAI is used for transcription only.
--
-- gpt-live-transcribe and gpt-realtime-whisper expose a `delay` dial
-- (minimal / low / medium / high / xhigh) that trades transcript latency
-- against accuracy. gpt-transcribe is committed-turn and has no dial, so
-- its row stores NULL.
--
-- NULL means "use the platform default for that model" (low for the two
-- streaming models, none for gpt-transcribe). That is also exactly what a
-- pre-migration row looks like, so this migration is additive: no agent
-- needs a data backfill and nothing is rewritten.
--
-- This column is deliberately NOT `NOT NULL DEFAULT 'low'` — a default
-- would write a value into gpt-transcribe rows, which cannot accept it.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS stt_latency_mode TEXT;
