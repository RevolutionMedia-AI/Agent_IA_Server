-- ============================================================================
-- 023 · transfer_chain.sql
-- ----------------------------------------------------------------------------
-- Ordered, user-configured call-transfer chains ("AI answers first, then
-- phone 1, phone 2, ... then back to the AI").
--
-- Two halves:
--   1. agent_tools.ring_timeout_sec — per-transfer-tool ring budget
--      (seconds Twilio rings the destination before the fallback
--      fires). 5..60 enforced in Python; DEFAULT 20 matches the
--      pre-AI cascade step default.
--   2. agents.transfer_chain — ordered JSONB list of call_transfer
--      tool ids for one agent. Maintained by assign/unassign (append /
--      remove) and editable in the agent modal (up/down). [] = today's
--      behaviour: the single invoked tool dials, then back to the AI.
-- ============================================================================

ALTER TABLE agent_tools
  ADD COLUMN IF NOT EXISTS ring_timeout_sec INT NOT NULL DEFAULT 20;
ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS transfer_chain JSONB NOT NULL DEFAULT '[]'::jsonb;
