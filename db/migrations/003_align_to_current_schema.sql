-- ============================================================================
-- RevolutionMedia-AI · Migration 003
-- ----------------------------------------------------------------------------
-- Align the pre-cleanup schema to what the current code reads/writes.
--
-- The new 001_schema.sql uses CREATE TABLE (not IF NOT EXISTS). On
-- existing deployments the tables already exist, so the startup
-- script logs "relation already exists" and walks past — but it
-- never adds the new columns. The old 004_extend_business_tables.sql
-- that did the ALTERs was deleted in the cleanup, so existing DBs
-- are now missing voice_id, welcome_message, the per-agent provider
-- overrides, the per-number Twilio/SIP/WhatsApp creds, etc.
--
-- Run this once after 002. All ALTERs are idempotent (IF NOT EXISTS)
-- so re-running is a no-op. Safe to apply on a DB that already has
-- the new schema.
-- ============================================================================


-- ── agents ────────────────────────────────────────────────────────────
ALTER TABLE agents ADD COLUMN IF NOT EXISTS voice_id          TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS welcome_message   TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS stt_provider      TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS stt_model         TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tts_provider      TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tts_model         TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS llm_provider      TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS llm_model         TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS updated_at        TIMESTAMPTZ;


-- ── phone_numbers ─────────────────────────────────────────────────────
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS twilio_account_sid        TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS twilio_auth_token         TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS sip_host                  TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS sip_username              TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS sip_password              TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS whatsapp_phone_number_id  TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS whatsapp_access_token     TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS campaign                  TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS updated_at                 TIMESTAMPTZ;


-- ── tools_integrations ────────────────────────────────────────────────
-- Old PK was single-column (id). New PK is composite (user_id, id).
-- Drop the old PK first if it's still single-column, then add the
-- composite. The DO block is defensive: if the PK is already
-- composite, the DROP CONSTRAINT would fail, so we only run it when
-- the old shape is detected.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'tools_integrations_pkey'
      AND pg_get_constraintdef(oid) LIKE 'PRIMARY KEY (id)%'
  ) THEN
    ALTER TABLE tools_integrations DROP CONSTRAINT tools_integrations_pkey;
    ALTER TABLE tools_integrations ADD PRIMARY KEY (user_id, id);
  END IF;
END $$;
ALTER TABLE tools_integrations ADD COLUMN IF NOT EXISTS display_name  TEXT;
ALTER TABLE tools_integrations ADD COLUMN IF NOT EXISTS category      TEXT;
ALTER TABLE tools_integrations ADD COLUMN IF NOT EXISTS updated_at    TIMESTAMPTZ;


-- ── settings ──────────────────────────────────────────────────────────
ALTER TABLE settings ADD COLUMN IF NOT EXISTS updated_at    TIMESTAMPTZ;


-- ── tenants ───────────────────────────────────────────────────────────
-- ponytail: drop the 4 legacy provider key columns per Opción 2
-- (provider creds live on tools_integrations, not on tenants). The
-- new code in db_tenants.py never reads or writes these — keeping
-- them in the schema would be dead data. If you have rows in here
-- with non-null values, they're already orphaned (no code path
-- surfaces them to the user), so dropping is safe.
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS user_id          TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE tenants DROP COLUMN IF EXISTS openai_api_key;
ALTER TABLE tenants DROP COLUMN IF EXISTS elevenlabs_api_key;
ALTER TABLE tenants DROP COLUMN IF EXISTS elevenlabs_voice_id;
ALTER TABLE tenants DROP COLUMN IF EXISTS deepgram_api_key;


-- ── call_sessions ─────────────────────────────────────────────────────
ALTER TABLE call_sessions ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ;


-- ── Indexes (parallel to what 001_schema.sql creates for fresh DBs) ────
CREATE INDEX IF NOT EXISTS idx_agents_user_status       ON agents (user_id, status);
CREATE INDEX IF NOT EXISTS idx_phone_numbers_campaign    ON phone_numbers (campaign) WHERE campaign IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_phone_numbers_digits      ON phone_numbers ((regexp_replace(number, '\D', '', 'g')));
CREATE INDEX IF NOT EXISTS idx_tools_user_category       ON tools_integrations (user_id, category);
CREATE INDEX IF NOT EXISTS idx_tenants_user              ON tenants (user_id);
