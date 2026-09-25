-- ============================================================================
-- 020 · twilio_credentials_backfill.sql
-- ----------------------------------------------------------------------------
-- One-shot backfill that turns every existing phone_numbers row with inline
-- twilio_account_sid + twilio_auth_token into a twilio_credentials row and
-- points the phone number at it. Runs after 019_twilio_credentials.sql.
--
-- ponytail: idempotent. Safe to re-run — the SELECT targets rows whose
-- twilio_credential_id IS NULL, so a second pass is a no-op. Run it once
-- per environment.
--
-- HOW TO RUN:
--   1. Take a database snapshot (Railway → Postgres → Backups, or `pg_dump`).
--   2. `psql $DATABASE_URL -f db/migrations/020_twilio_credentials_backfill.sql`
--   3. Verify with the SELECT at the bottom.
--   4. Re-deploy the application.
--
-- The BE now resolves Twilio auth by joining phone_numbers.twilio_credential_id
-- onto twilio_credentials. Rows that still have NULL twilio_credential_id fall
-- back to the legacy inline columns (kept for this transitional step) — see
-- the resolver changes in STT_server/services/credentials_resolver.py. Drop
-- the inline columns in a follow-up migration once every row is migrated.
-- ============================================================================

BEGIN;

-- ponytail: encrypt the inline credentials into a per-(user, sid) twilio_credentials
-- row. We use the BE's Fernet key (the same one used by agent_tools.credentials)
-- so the new column decrypts with the existing STT_server.security.credentials
-- module. The Fernet encrypt call happens server-side via the BE; this SQL
-- only creates the placeholder rows.
--
-- Since the operator must keep the BE's Fernet key handy (it's
-- CREDENTIAL_ENCRYPTION_KEY in Railway), the recommended flow is:
--   a) Run 019 + this migration.
--   b) Boot the BE. A small startup hook (see STT_server/db_twilio_credentials.py
--      backfill_from_json) re-encrypts any plaintext rows from the legacy
--      data/twilio_credentials.json file (if present).
--   c) Use the /twilio-credentials endpoints to re-save each credential. The
--      BE re-encrypts on save and the resolver joins phone_numbers on the new
--      column from then on.
--
-- The SQL below only materialises the schema-level FKs and indexes; the
-- actual row-by-row re-encryption is handled by the BE. We pre-create one
-- twilio_credentials row per (user, sid) here, marked status='unknown',
-- so the FK can be set without breaking existing phone numbers.
--
-- ponyltail: do NOT populate account_sid_encrypted / auth_token_encrypted
-- here — those columns require Fernet ciphertext and writing raw plaintext
-- would lock out the operator from re-saving. Instead we run the BE's
-- backfill helper (db_twilio_credentials.backfill_from_phone_numbers) after
-- this script. See the operator-facing note at the bottom.
INSERT INTO twilio_credentials (id, user_id, name, account_sid_encrypted, auth_token_encrypted, account_sid_last4, status)
SELECT
  'twcred-' || substr(md5(random()::text || p.user_id || p.twilio_account_sid), 1, 12),
  p.user_id,
  'Subaccount ' || substr(p.twilio_account_sid, 1, 6) || '…' || substr(p.twilio_account_sid, length(p.twilio_account_sid) - 3),
  '__PENDING_BACKFILL__',  -- placeholder; the BE rewrites this on the next save
  '__PENDING_BACKFILL__',
  substr(p.twilio_account_sid, length(p.twilio_account_sid) - 3),
  'unknown'
FROM phone_numbers p
WHERE p.twilio_account_sid IS NOT NULL
  AND p.twilio_account_sid <> ''
  AND p.twilio_credential_id IS NULL
ON CONFLICT (user_id, lower(name)) DO NOTHING;  -- ponyltail: avoid dup rows if rerun


-- ponytail: now point every phone_number with an inline SID at its matching
-- twilio_credentials row. We pick the most-recently-created credential for
-- the (user, sid) pair so two numbers from the same sub-account share one
-- row instead of getting duplicates.
UPDATE phone_numbers p
SET twilio_credential_id = (
  SELECT c.id
  FROM twilio_credentials c
  WHERE c.user_id = p.user_id
    AND c.account_sid_last4 = substr(p.twilio_account_sid, length(p.twilio_account_sid) - 3)
    AND c.status = 'unknown'
  ORDER BY c.created_at DESC
  LIMIT 1
)
WHERE p.twilio_account_sid IS NOT NULL
  AND p.twilio_account_sid <> ''
  AND p.twilio_credential_id IS NULL;

COMMIT;


-- ── Sanity queries (run separately, do NOT bundle in the migration) ────────

-- How many phone numbers were linked to a credential?
-- Expected after a clean run: every row with a non-empty twilio_account_sid
-- shows twilio_credential_id IS NOT NULL.
-- Example:
--   SELECT count(*) AS total,
--          count(twilio_credential_id) AS linked,
--          count(*) FILTER (WHERE twilio_credential_id IS NULL
--                            AND twilio_account_sid IS NOT NULL) AS still_unlinked
--   FROM phone_numbers;

-- How many credentials were created by the backfill (status='unknown'
-- means "the BE hasn't re-encrypted it yet")?
-- Example:
--   SELECT status, count(*) FROM twilio_credentials GROUP BY status;


-- ── Operator follow-up ──────────────────────────────────────────────────────
-- The __PENDING_BACKFILL__ sentinel rows need their secrets re-encrypted by
-- the BE before they can be used to place calls. Two options:
--
--   1. UI path (recommended): open Settings → API → Twilio sub-accounts in
--      the browser, click Edit on each row, re-paste the SID + Token, Save.
--      The BE encrypts on save and flips status to 'connected' after a
--      successful Twilio auth check.
--
--   2. One-shot script (if you have many rows and the plaintext SID + Token
--      pairs from the legacy data/phone_numbers.json): point the BE at the
--      same DATABASE_URL and let db_twilio_credentials.backfill_from_json()
--      run on startup. See the helper at the bottom of
--      STT_server/db_twilio_credentials.py.
--
-- After either path, drop the placeholder sentinel with:
--
--   DELETE FROM twilio_credentials
--   WHERE account_sid_encrypted = '__PENDING_BACKFILL__';
--
-- In a follow-up migration (021) we will:
--   - DROP COLUMN phone_numbers.twilio_account_sid;
--   - DROP COLUMN phone_numbers.twilio_auth_token;
-- to remove the legacy columns once every row has a real credential FK.
