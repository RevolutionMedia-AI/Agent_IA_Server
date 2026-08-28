-- ============================================================================
-- 016 · agent_tools_integration.sql
-- ----------------------------------------------------------------------------
-- Adds the columns that let an agent_tool reference an integration
-- instead of (or in addition to) carrying its own webhook_url.
--
--   integration_id  TEXT  NULL  →  logical FK to integrations.id.
--                                  NOT a real FK because the JSON-file
--                                  fallback can't enforce referential
--                                  integrity; the route layer
--                                  (db_integrations + AgentTool
--                                  validation) is the gatekeeper.
--   action          TEXT  NULL  →  provider-specific action the LLM
--                                  triggers (find_customer,
--                                  get_tickets, ...). Server-injected
--                                  into the n8n body; the LLM never
--                                  controls this value.
--
-- Tools created before this migration keep working unchanged:
-- integration_id=NULL falls back to tool.webhook_url, action=NULL
-- means "no provider dispatch" (legacy n8n workflows that switch on
-- tool_name only).
--
-- The kind='call_transfer' constraint (no integration_id) is enforced
-- in AgentTool.validate(), not as a CHECK on the table, because
-- legacy rows may predate both columns.
-- ============================================================================

ALTER TABLE agent_tools
  ADD COLUMN IF NOT EXISTS integration_id TEXT,
  ADD COLUMN IF NOT EXISTS action         TEXT;

-- ponytail: index on integration_id so the dependent-tools COUNT in
-- db_integrations.delete_integration (RESTRICT) is cheap. Without the
-- index the COUNT would sequential-scan agent_tools on every delete.
CREATE INDEX IF NOT EXISTS idx_agent_tools_integration_id
  ON agent_tools (integration_id)
  WHERE integration_id IS NOT NULL;