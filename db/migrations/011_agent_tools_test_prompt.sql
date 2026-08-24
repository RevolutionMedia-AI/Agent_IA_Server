-- ============================================================================
-- 011 · agent_tools_test_prompt.sql
-- ----------------------------------------------------------------------------
-- Adds the optional `test_prompt` column. When set, the Test button
-- asks the user's configured LLM to generate realistic data matching
-- the tool's parameters schema before POSTing to the n8n webhook.
-- The operator curates one test_prompt per integration (e.g. "Mexican
-- dentist appointment, name='María López', today 3pm, 60min") so the
-- data the workflow actually exercises is plausible — instead of the
-- default placeholder behavior which n8n rejects.
-- ============================================================================

ALTER TABLE agent_tools
  ADD COLUMN IF NOT EXISTS test_prompt TEXT;