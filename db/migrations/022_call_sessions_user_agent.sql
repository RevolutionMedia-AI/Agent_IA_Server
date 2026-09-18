-- ============================================================================
-- 022 · call_sessions_user_agent.sql
-- ----------------------------------------------------------------------------
-- The runtime (services/session_runtime.py) inserts user_id + agent_id on
-- register_session so the dashboard live-calls roster can scope counts
-- per owner, but the base schema (001) never had those columns — every
-- /voice hit on Postgres logged:
--   column "user_id" of relation "call_sessions" does not exist
-- and the session row was lost (call invisible to the dashboard).
-- Nullable TEXT, no backfill: legacy rows simply read as unscoped.
-- ============================================================================

ALTER TABLE call_sessions
  ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE call_sessions
  ADD COLUMN IF NOT EXISTS agent_id TEXT;
CREATE INDEX IF NOT EXISTS idx_call_sessions_user_open
  ON call_sessions (user_id, closed, started_at DESC);
