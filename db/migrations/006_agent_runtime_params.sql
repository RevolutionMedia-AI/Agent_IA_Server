-- ============================================================================
-- 006 · agent_runtime_params.sql
-- ----------------------------------------------------------------------------
-- Add operator-tunable runtime knobs to the agents row.
--
-- What we add (all nullable — NULL means "use the platform default"):
--
--   llm_temperature REAL
--     Per-agent sampling temperature for the chat-completions LLM
--     (OpenAI / Anthropic / Gemini / MiniMax-compat). 0.0 = deterministic,
--     2.0 = max chaos. NULL falls back to the env-default the adapter
--     has been using all along (currently 0.2 hardcoded in
--     openai_llm.py). OpenAI Realtime is unaffected today (sends no
--     temperature post-GA migration).
--
--   llm_max_tokens INTEGER
--     Per-agent max tokens for the LLM response. NULL falls back to
--     config.MAX_RESPONSE_TOKENS (default 150). Higher = longer agent
--     replies, more tokens billed.
--
--   tts_speed REAL
--     Per-agent TTS playback speed. 1.0 = normal, 0.5 = half speed,
--     2.0 = double. NULL = no override (provider default).
--     Supported by: OpenAI TTS (always accepted), ElevenLabs (via
--     voice_settings.speed), Inworld (via audioConfig.speakingRate).
--     NOT supported by: Deepgram Aura, Rime. Adapters for the latter
--     silently ignore the field — there's no API knob to turn.
--
-- All three are bounded by CHECK constraints so a typo can't crash
-- the call path with a 4xx from the provider.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS llm_temperature REAL
    CHECK (llm_temperature IS NULL OR (llm_temperature >= 0.0 AND llm_temperature <= 2.0)),
  ADD COLUMN IF NOT EXISTS llm_max_tokens  INTEGER
    CHECK (llm_max_tokens  IS NULL OR (llm_max_tokens  > 0   AND llm_max_tokens  <= 4096)),
  ADD COLUMN IF NOT EXISTS tts_speed      REAL
    CHECK (tts_speed      IS NULL OR (tts_speed      >= 0.5 AND tts_speed      <= 2.0));
