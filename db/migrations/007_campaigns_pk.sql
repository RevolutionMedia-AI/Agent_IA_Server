-- ============================================================================
-- RevolutionMedia-AI · Migration 007
-- ----------------------------------------------------------------------------
-- Migration 006 declared `name TEXT PRIMARY KEY` for `campaigns`. On a
-- project that already had an earlier campaigns table without the PK
-- (or that hit the CREATE TABLE IF NOT EXISTS path before migration
-- 006 landed) the constraint is missing and ON CONFLICT (name) fails
-- at runtime with `there is no unique or exclusion constraint matching
-- the ON CONFLICT specification`. This migration defensively adds the
-- primary key so the upsert path works on every environment.
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'campaigns'::regclass
      AND contype = 'p'
  ) THEN
    -- Backfill any NULL name rows so the PK constraint can be added
    -- safely. Empty-string names are treated as 'unnamed-campaign'.
    UPDATE campaigns SET name = 'unnamed-campaign' WHERE name IS NULL;
    ALTER TABLE campaigns ADD PRIMARY KEY (name);
  END IF;
END $$;

-- Also defensively add an index for case-insensitive search if it's
-- not there (the original migration declared it but CREATE TABLE IF
-- NOT EXISTS on a pre-existing table skipped it).
CREATE INDEX IF NOT EXISTS idx_campaigns_name ON campaigns (lower(name));
