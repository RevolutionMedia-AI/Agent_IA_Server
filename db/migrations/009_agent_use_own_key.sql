-- ============================================================================
-- 009 · agent_use_own_key.sql
-- ----------------------------------------------------------------------------
-- Per-agent toggle: "use my own API key for this service slot" vs "use
-- the platform key". NULL/false (default) means the agent falls back to
-- platform env vars (OPENAI_API_KEY / DEEPGRAM_API_KEY / etc. on
-- Railway) when no per-user key is configured. true means the operator
-- wants the agent to use ONLY the per-user / per-agent credential — the
-- resolver ignores platform env vars for that slot.
--
-- One toggle per service (stt / llm / tts) because each slot has its own
-- credential source — a user might have an OpenAI key but a platform
-- Twilio number, or vice versa.
--
-- Backwards compat: existing rows default to false (use platform fallback).
-- A user who already stored keys in tools_integrations keeps using them
-- when the toggle is false — the FE shows the toggle as OFF but the BE
-- resolver still consults tools_integrations first. The toggle only
-- enforces "platform-only" when explicitly true; we never silently drop
-- a stored credential.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS stt_use_own_key BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS llm_use_own_key BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS tts_use_own_key BOOLEAN NOT NULL DEFAULT FALSE;