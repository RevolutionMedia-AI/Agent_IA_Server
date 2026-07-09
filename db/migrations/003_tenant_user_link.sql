-- ============================================================================
-- Add user_id to tenants (links a Twilio sub-account to an admin user).
-- Required for the voice path to resolve which user's per-user OpenAI /
-- Twilio / Deepgram / ElevenLabs keys to use for a given inbound call.
-- ============================================================================

ALTER TABLE tenants
  ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_tenants_user ON tenants (user_id);
