-- ============================================================================
-- 015 · integrations.sql
-- ----------------------------------------------------------------------------
-- Adds the `integrations` table. One row per third-party connection the user
-- wants the agent to talk to (Salesforce, Zendesk, Dynamics 365, Genesys
-- Cloud CX, NICE CXone, Generic Webhook, ...). Tools reference an
-- integration by `agent_tools.integration_id` (added in 016) — one
-- integration feeds many tools, so configuring Zendesk happens once even
-- when find_customer / get_tickets / create_ticket / add_comment /
-- update_ticket each get their own tool row.
--
-- Credentials live encrypted (Fernet via encrypt_credentials) in
-- credentials_encrypted; configuration is plain JSONB (provider-specific
-- settings like Zendesk's subdomain — those are not secrets). The
-- agent_id column mirrors the tool-side convention so a Shared
-- Integration (agent_id='__shared__') is reachable by every agent the
-- owner owns, and a per-agent integration is scoped to that one agent.
--
-- The `connection_status` column is the last result of running the
-- provider's test_fn (or the preflight at create time). It's a hint for
-- the FE; the source of truth is last_test_message + the live ping
-- itself.
-- ============================================================================

CREATE TABLE IF NOT EXISTS integrations (
  id                    TEXT        PRIMARY KEY,
  user_id               TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  agent_id              TEXT        NOT NULL DEFAULT '__shared__',  -- '__shared__' | 'agent-<uuid8>'
  provider              TEXT        NOT NULL,                       -- zendesk|salesforce|dynamics365|genesys_cloud|nice_cxone|generic_webhook
  name                  TEXT        NOT NULL,
  configuration         JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- provider-specific, NO secrets
  credentials_encrypted BYTEA,                                       -- Fernet ciphertext; NULL = not yet configured
  credentials_cipher    TEXT        NOT NULL DEFAULT 'fernet-v1',
  connection_status     TEXT        NOT NULL DEFAULT 'unknown',     -- unknown|connected|failed
  last_tested_at        TIMESTAMPTZ,
  last_test_message     TEXT,                                       -- last test result message (sanitized, no secrets)
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: list integrations for a user (optionally filtered by agent_id).
CREATE INDEX IF NOT EXISTS idx_integrations_user_agent
  ON integrations (user_id, agent_id);

-- Used by the tools executor to check ownership + RESTRICT-on-delete counts.
CREATE INDEX IF NOT EXISTS idx_integrations_provider
  ON integrations (user_id, provider);