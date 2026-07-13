-- ============================================================================
-- RevolutionMedia-AI · Migration 008
-- ----------------------------------------------------------------------------
-- Drop the `name` column from phone_numbers. The number's display name
-- is now derived automatically from the assigned agent (phone_numbers
-- .label falls back to the linked agent's name). A free-text name on
-- the number record was redundant and easy to drift out of sync.
-- Idempotent — IF EXISTS so re-running is safe.
-- ============================================================================

ALTER TABLE phone_numbers DROP COLUMN IF EXISTS name;
