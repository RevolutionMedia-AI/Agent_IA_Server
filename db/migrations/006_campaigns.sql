-- ============================================================================
-- RevolutionMedia-AI · Migration 006
-- ----------------------------------------------------------------------------
-- Global campaigns catalog. Both the agent modal and the phone-number
-- modal write the campaign the user types (when it's not already in
-- the curated list). The /campaigns endpoint then serves the union of
-- the curated list + everything ever typed, so the user sees their own
-- choices as suggestions on the next modal open.
-- ============================================================================

CREATE TABLE IF NOT EXISTS campaigns (
  name       TEXT        PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_campaigns_name ON campaigns (lower(name));
