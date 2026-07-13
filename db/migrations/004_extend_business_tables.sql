-- ============================================================================
-- RevolutionMedia-AI · Migration 004
-- ----------------------------------------------------------------------------
-- Extend the business tables with the columns the FE/BE actually write.
-- The original 001_schema.sql only carried the UI-facing columns; the
-- New Agent flow writes stt/tts/llm provider+model and welcome_message
-- which have been landing in STT_server/data/*.json. This migration
-- brings the DB columns in line so the Postgres-backed routes can take
-- over without losing the data the FE sends.
--
-- All ALTERs are idempotent (IF NOT EXISTS) so re-running on an
-- already-migrated DB is a no-op.
-- ============================================================================

-- agents: per-service provider/model + welcome_message + voice (rename safe)
ALTER TABLE agents ADD COLUMN IF NOT EXISTS welcome_message  TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS stt_provider      TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS stt_model         TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tts_provider      TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tts_model         TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS llm_provider      TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS llm_model         TEXT;
-- voice is the display name (e.g. "Aria", "Dennis"). Some FE copies use
-- `voice_id` for the underlying provider voice id; rename was deferred
-- to avoid touching 001. Keep them side-by-side for now.
ALTER TABLE agents ADD COLUMN IF NOT EXISTS voice_id          TEXT;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS updated_at        TIMESTAMPTZ;

-- phone_numbers: Twilio + SIP + WhatsApp creds + label + agent FK
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS twilio_account_sid         TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS twilio_auth_token          TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS sip_host                   TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS sip_username               TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS sip_password               TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS whatsapp_phone_number_id   TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS whatsapp_access_token       TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS updated_at                  TIMESTAMPTZ;

-- tools_integrations: keep id as TEXT so existing service ids ("openai",
-- "deepgram", "elevenlabs", ...) work as PKs. id stays the provider id;
-- user_id becomes part of the composite key.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'tools_integrations_pkey'
  ) THEN
    -- add a composite PK on (user_id, id) so one user can have their own
    -- row per provider without colliding on id.
    ALTER TABLE tools_integrations
      DROP CONSTRAINT IF EXISTS tools_integrations_pkey;
    ALTER TABLE tools_integrations
      ADD PRIMARY KEY (user_id, id);
  END IF;
END $$;
ALTER TABLE tools_integrations ADD COLUMN IF NOT EXISTS display_name  TEXT;
ALTER TABLE tools_integrations ADD COLUMN IF NOT EXISTS category      TEXT;
ALTER TABLE tools_integrations ADD COLUMN IF NOT EXISTS updated_at    TIMESTAMPTZ;

-- settings: name + company + timezone + notifications already exist (001).
-- Add the JSON shape the FE sends (notifications object).
ALTER TABLE settings ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ;
