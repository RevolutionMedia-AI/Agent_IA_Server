-- ============================================================================
-- RevolutionMedia-AI · Migration 004
-- ----------------------------------------------------------------------------
-- Add the call_usage table. Replaces the legacy calls.json ledger that
-- services/usage_store.py used to write to. One row per completed call,
-- appended at cleanup_session() time.
--
-- The 001_schema.sql was already updated to include this table for new
-- deploys; this migration is for existing deploys that ran 001 before
-- the table existed. Idempotent (CREATE TABLE IF NOT EXISTS + IF NOT
-- EXISTS on the indexes).
-- ============================================================================

CREATE TABLE IF NOT EXISTS call_usage (
  id                  BIGSERIAL   PRIMARY KEY,
  user_id             TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  agent_id            TEXT,
  tenant_id           TEXT        REFERENCES tenants(tenant_id) ON DELETE SET NULL,
  call_sid            TEXT,
  started_at          TIMESTAMPTZ NOT NULL,
  ended_at            TIMESTAMPTZ NOT NULL,
  duration_seconds    REAL        NOT NULL,
  stt_provider        TEXT,
  llm_provider        TEXT,
  tts_provider        TEXT,
  used_platform_keys  BOOLEAN     NOT NULL DEFAULT FALSE,
  cost_usd            REAL        NOT NULL DEFAULT 0.0,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_usage_user_started  ON call_usage (user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_usage_agent         ON call_usage (agent_id);
CREATE INDEX IF NOT EXISTS idx_call_usage_tenant        ON call_usage (tenant_id);
