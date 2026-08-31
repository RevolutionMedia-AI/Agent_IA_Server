-- ============================================================================
-- 017 · integrations_oauth.sql
-- ----------------------------------------------------------------------------
-- Adds the columns the OAuth (Authorization Code) flow needs. Salesforce is
-- the first provider on this path; the catalog entries for `auth_type="oauth"`
-- drive which integrations get this treatment. Future providers (Dynamics,
-- Google, HubSpot) reuse the same machinery — only `oauth_providers.py`
-- changes per provider.
--
-- Why these columns exist:
--   oauth_state_hash         SHA-256 of the `state` token we sent to the
--                            provider. We never store the raw state — a leak
--                            of the DB row is no longer an open redirect.
--                            The callback hashes the incoming state and
--                            compares to this value with constant-time
--                            compare.
--   oauth_state_expires_at   TTL for the state lookup. Defaults to 10 minutes
--                            after `/oauth/start`; the callback rejects a
--                            state whose expires_at is in the past. Single-
--                            use is enforced by clearing the column inside
--                            the same transaction that writes the tokens, so
--                            a replay never succeeds.
--   oauth_scope              Scopes the provider actually granted back via
--                            the token response. Can differ from the
--                            requested scopes (the user may have approved
--                            a subset). Surfaced in the integration detail
--                            so the operator can see what they're using.
--
-- Five-state connection_status:
--   unknown      row just created, no test ever run
--   pending      OAuth flow in progress (row created, no tokens yet)
--   connected    tokens valid, last test passed
--   failed       test failed OR refresh token revoked/expired
--   disconnected operator-initiated disconnect; tokens wiped
--
--   Existing rows (pre-017) keep their stored status string. The CHECK
--   constraint below is permissive — the FE / BE write paths are the
--   only producers and they emit one of these five literals, so any
--   legacy value stays valid until next write.
-- ============================================================================

ALTER TABLE integrations
  ADD COLUMN IF NOT EXISTS oauth_state_hash       TEXT,
  ADD COLUMN IF NOT EXISTS oauth_state_expires_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS oauth_scope            TEXT;

-- ponytail: relax an existing implicit constraint. The original 015
-- schema didn't enforce status values (CHECK constraints were left to
-- the application layer) but we tighten the constraint here so a
-- typo in a future route doesn't silently write a sixth value. The
-- `IF NOT EXISTS` form isn't supported for CHECK constraints, so we
-- drop and re-add; on a fresh DB the DROP is a no-op because the
-- constraint never existed.
ALTER TABLE integrations DROP CONSTRAINT IF EXISTS integrations_connection_status_check;
ALTER TABLE integrations ADD CONSTRAINT integrations_connection_status_check
  CHECK (connection_status IN ('unknown','pending','connected','failed','disconnected'));

-- ponytail: index on oauth_state_hash. The callback does a single
-- lookup by hash per request; with the index the lookup is a single
-- row fetch even on busy tables. Partial index — only rows currently
-- mid-flow have a hash set, so the index stays tiny.
CREATE INDEX IF NOT EXISTS idx_integrations_oauth_state_hash
  ON integrations (oauth_state_hash)
  WHERE oauth_state_hash IS NOT NULL;