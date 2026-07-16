-- ============================================================================
-- 007 · agent_pricing_overrides.sql
-- ----------------------------------------------------------------------------
-- Per-agent per-model pricing overrides. Used when the operator picks a
-- tier / plan that the public catalog doesn't cover (e.g. Inworld
-- Enterprise, custom per-account discounts on Anthropic, negotiated
-- Deepgram rates, etc.). The FE writes to this table when the operator
-- selects "Enterprise" or "Custom" on the per-service tier dropdown
-- and enters a number; the runtime cost summary merges these with the
-- public MODEL_PRICING catalog at resolve time.
--
-- Schema mirrors the catalog shape:
--   agent_id, service, provider, model_id -- composite PK
--   unit                                  -- 'minute'|'hour'|'1k_chars'|'1m_chars'|'1m_tokens'
--   price        NUMERIC (nullable)       -- single-price (STT/TTS)
--   input_price  NUMERIC (nullable)       -- LLM input  per 1M tokens
--   output_price NUMERIC (nullable)       -- LLM output per 1M tokens
--   source       TEXT                      -- 'enterprise'|'manual'|...
--   updated_at   TIMESTAMPTZ
--
-- Resolution order (getAgentPrice in db_pricing_overrides.py):
--   1. agent_pricing_overrides for (agent_id, service, provider, model_id) — wins
--   2. catalog fallback (LLM_TOKENS_PER_MIN, TTS_CHARS_PER_MIN normalized)
--   3. null if neither has the row
-- ============================================================================

CREATE TABLE agent_pricing_overrides (
  agent_id     TEXT        NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  service      TEXT        NOT NULL CHECK (service IN ('stt', 'tts', 'llm')),
  provider     TEXT        NOT NULL,
  model_id     TEXT        NOT NULL,
  unit         TEXT        NOT NULL CHECK (unit IN ('minute', 'hour', '1k_chars', '1m_chars', '1m_tokens')),
  price        NUMERIC,
  input_price  NUMERIC,
  output_price NUMERIC,
  source       TEXT        NOT NULL DEFAULT 'manual',
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (agent_id, service, provider, model_id),
  CHECK (
    price IS NOT NULL OR input_price IS NOT NULL OR output_price IS NOT NULL
  )
);

CREATE INDEX idx_agent_pricing_overrides_agent ON agent_pricing_overrides (agent_id);
