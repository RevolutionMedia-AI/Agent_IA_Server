-- ============================================================================
-- 024 · transfer_enabled.sql
-- ----------------------------------------------------------------------------
-- Master on/off for human handoff per agent. TRUE (default) = today's
-- behaviour: assigned transfer tools reach the LLM (tools[] +
-- prompt sections) and the ordered chain can run. FALSE = Full-AI
-- service: transfer tools are filtered from the call, their prompt
-- sections are stripped by the reconciler, and invoking one returns
-- a tool error instead of dialing. No other behaviour changes.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS transfer_enabled BOOL NOT NULL DEFAULT TRUE;
