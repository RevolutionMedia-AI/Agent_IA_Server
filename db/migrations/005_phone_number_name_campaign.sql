-- ============================================================================
-- RevolutionMedia-AI · Migration 005
-- ----------------------------------------------------------------------------
-- Add name + campaign columns to phone_numbers. Both are optional; the FE
-- sends them when the user fills the new fields in the Connect Number
-- modal. Idempotent — IF NOT EXISTS so re-running is safe.
-- ============================================================================

ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS name    TEXT;
ALTER TABLE phone_numbers ADD COLUMN IF NOT EXISTS campaign TEXT;
CREATE INDEX IF NOT EXISTS idx_phone_numbers_campaign ON phone_numbers (campaign);
