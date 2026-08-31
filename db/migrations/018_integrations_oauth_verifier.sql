-- ============================================================================
-- 018 · integrations_oauth_verifier.sql
-- ----------------------------------------------------------------------------
-- Adds the encrypted PKCE code_verifier column to integrations.
-- Migration 017 added the state hash + expires_at; this one adds the
-- matching PKCE fields. The verifier itself is encrypted at rest
-- (same Fernet scheme as credentials_encrypted) because it is a bearer
-- secret — anyone with the row + the Fernet key can exchange an
-- authorization code for tokens.
--
-- Lifecycle (mirrors oauth_state_hash):
--   start_oauth_flow    ->  write encrypted verifier
--   complete_oauth_flow ->  clear it (NULL)
--   clear_oauth_state   ->  clear it (NULL)  (failure path)
--
-- The state hash is HASHED (we only need to compare). The verifier is
-- ENCRYPTED (we need to send the original back to Salesforce in the
-- token-exchange call). Two different protections, two different
-- columns.
-- ============================================================================

ALTER TABLE integrations
  ADD COLUMN IF NOT EXISTS oauth_code_verifier_encrypted BYTEA;
