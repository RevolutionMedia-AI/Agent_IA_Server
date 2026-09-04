-- ============================================================================
-- 019 · twilio_credentials.sql
-- ----------------------------------------------------------------------------
-- Multi-credential Twilio support. Settings → API used to store a single
-- Twilio row in `agent_tools` (id='twilio', agent_id='__shared__') under the
-- `credentials` JSONB column, with `phone_number` as one of the field names.
-- One organisation with multiple Twilio sub-accounts had no way to register
-- more than one pair of SID+Token without the second PUT overwriting the first.
--
-- This migration introduces a dedicated `twilio_credentials` table: one row
-- per (user, name) pair, each holding an encrypted Account SID + Auth Token.
-- Phone numbers later reference these rows via `phone_numbers.twilio_credential_id`
-- (added in this same migration), and the Settings modal exposes a CRUD list
-- instead of a single row.
--
-- ponytail: backfill from existing rows. Pre-019 phone numbers carry
-- twilio_account_sid + twilio_auth_token directly. On first run after this
-- migration, we create one twilio_credentials row per (user, sid) and point
-- the phone number at it. Same phone numbers already on Postgres are
-- migrated; the existing twilio_account_sid / twilio_auth_token columns stay
-- (NOT NULL-safe) so legacy code paths keep reading them until the resolver
-- is rewired. The FE drops them after this PR lands; the columns can be
-- dropped in a follow-up migration once no callers read them.
-- ============================================================================

CREATE TABLE IF NOT EXISTS twilio_credentials (
  id                            TEXT        PRIMARY KEY,            -- "twcred-{uuid12}"
  user_id                       TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name                          TEXT        NOT NULL,               -- operator-supplied label, e.g. "Sales subaccount"
  account_sid_encrypted         TEXT        NOT NULL,               -- Fernet ciphertext of the Account SID
  auth_token_encrypted         TEXT        NOT NULL,               -- Fernet ciphertext of the Auth Token
  account_sid_last4             TEXT        NOT NULL,               -- last 4 chars of the SID (plaintext) so the FE can render a card without revealing the secret
  status                        TEXT        NOT NULL DEFAULT 'unknown'
                              CHECK (status IN ('connected', 'invalid', 'connection_error', 'unknown')),
  last_tested_at                TIMESTAMPTZ,
  last_test_message             TEXT,
  created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                    TIMESTAMPTZ
);

-- ponytail: a user can't have two credentials with the same display name.
-- Same logic as agent_tools' (user_id, name) uniqueness — operators can
-- pick any free-form label, but the column has to be unique within the
-- tenant so the dropdown in ModalConnectNumber isn't ambiguous.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_twilio_credentials_user_name
  ON twilio_credentials (user_id, lower(name));

-- ponytail: the resolver's hot path is "list this user's credentials
-- ordered by created_at DESC". Composite index on (user_id, created_at)
-- serves the read without touching the table.
CREATE INDEX IF NOT EXISTS idx_twilio_credentials_user_created
  ON twilio_credentials (user_id, created_at DESC);


-- ── Phone number FK ──────────────────────────────────────────────────────────
-- ponytail: phone_numbers gains a FK to twilio_credentials. The existing
-- twilio_account_sid / twilio_auth_token columns stay populated until every
-- caller has been rewired — that's a separate PR. New numbers coming in
-- through the updated FE will set twilio_credential_id instead.

ALTER TABLE phone_numbers
  ADD COLUMN IF NOT EXISTS twilio_credential_id TEXT
                              REFERENCES twilio_credentials(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_phone_numbers_twilio_credential
  ON phone_numbers (twilio_credential_id)
  WHERE twilio_credential_id IS NOT NULL;
