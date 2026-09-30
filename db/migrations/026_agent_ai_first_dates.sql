-- ============================================================================
-- 026 · agent_ai_first_dates.sql
-- ----------------------------------------------------------------------------
-- Per-agent list of specific calendar dates on which the AI answers FIRST.
--
-- The operator's ask: not a business-hours grid (Mon-Fri 7-17), just a set
-- of named days — company holidays, special closures. On those days the
-- pre-AI cascade must not ring anybody: the AI picks up on the first ring,
-- and the humans become the post-AI chain instead.
--
-- Stored as JSONB (a sorted list of "YYYY-MM-DD" strings) rather than a
-- child table, for the same reason transfer_cascade/transfer_chain are
-- JSONB: the value is a small ordered blob owned entirely by the agent
-- row, never queried across rows, never joined. One column means the
-- create/update/select/return machinery in db_agents.py carries it with
-- no new query and delete cascades for free.
--
-- Empty list = every day uses the configured handoff order, i.e. this
-- migration changes nothing until an operator adds a date.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS ai_first_dates JSONB NOT NULL DEFAULT '[]'::jsonb;
