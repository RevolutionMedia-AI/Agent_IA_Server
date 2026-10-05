-- ============================================================================
-- 029 · integration_n8n_token.sql
-- ----------------------------------------------------------------------------
-- Per-integration n8n credentials, so an integration is not welded to the
-- platform env vars and tenants do not share one identity.
--
-- Before this, n8n authenticated to us with a single global
-- INTEGRATIONS_N8N_TOKEN and the credentials endpoint used the UNSCOPED
-- integration lookup (no user_id in the caller context). Consequence: any
-- n8n holding that token could read the decrypted access token of ANY
-- integration belonging to ANY user, by guessing the integration id.
--
-- This adds the lookup half only. The token itself is stored encrypted
-- inside credentials_encrypted (Fernet) like every other credential; a
-- hash prefix is kept in clear so the row can be FOUND without decrypting
-- every integration on every n8n request. The plaintext token is shown to
-- the operator exactly once, at creation/rotation.
--
-- NULL = "this integration still uses the platform token", which is the
-- default and the backwards-compatible path. So this migration is purely
-- additive: no existing integration changes behaviour, and no credential is
-- rewritten.
-- ============================================================================

ALTER TABLE integrations
  ADD COLUMN IF NOT EXISTS n8n_token_prefix TEXT;

-- Partial index: the column is NULL for every integration that has not
-- opted into its own token, so indexing only the non-NULL rows keeps the
-- index small and makes the lookup a range scan on an equality match.
CREATE INDEX IF NOT EXISTS idx_integrations_n8n_token_prefix
  ON integrations (n8n_token_prefix)
  WHERE n8n_token_prefix IS NOT NULL;