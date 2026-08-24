-- ============================================================================
-- 013 · agent_tools_test_data_model.sql
-- ----------------------------------------------------------------------------
-- The Test data generator (services/test_data_generator.py) needs a
-- configurable LLM model per provider so the operator can pick
-- gpt-4o-mini / gpt-4o / gpt-4-turbo from the Integrations Connect
-- modal. The model is stored alongside the user's encrypted
-- credentials in agent_tools (the same table that holds API keys
-- after migration 010 — name persists from the legacy
-- tools_integrations.json file) so it lives with the rest of the
-- provider's per-user config.
-- ============================================================================

ALTER TABLE agent_tools
  ADD COLUMN IF NOT EXISTS test_data_model TEXT NOT NULL DEFAULT 'gpt-4o-mini';
