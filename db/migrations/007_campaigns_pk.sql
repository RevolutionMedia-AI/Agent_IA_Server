-- ============================================================================
-- RevolutionMedia-AI · Migration 007
-- ----------------------------------------------------------------------------
-- Migration 006 declared `name TEXT PRIMARY KEY` for `campaigns`. On a
-- project that already had an earlier campaigns table without the PK
-- (or that hit the CREATE TABLE IF NOT EXISTS path before migration
-- 006 landed) the constraint is missing and ON CONFLICT (name) fails
-- at runtime with `there is no unique or exclusion constraint matching
-- the ON CONFLICT specification`.
--
-- This migration adds a UNIQUE INDEX (not a PK constraint - the
-- existing table might already have a PK; ALTER TABLE ... ADD
-- CONSTRAINT PRIMARY KEY would fail if so). ON CONFLICT works the
-- same way whether the constraint is a PK or a UNIQUE index, so the
-- upsert path is satisfied either way.
-- ============================================================================

-- Backfill any NULL names so we can build a unique index on the column.
UPDATE campaigns SET name = 'unnamed-campaign-' || gen_random_uuid()::text
  WHERE name IS NULL;

-- Build a unique index if one doesn't exist. ON CONFLICT accepts both
-- PRIMARY KEY and UNIQUE constraints as its conflict target, so this
-- unblocks upsert_campaign() regardless of which kind of constraint
-- (if any) was previously created.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public'
      AND tablename = 'campaigns'
      AND indexname = 'campaigns_name_unique_idx'
  ) THEN
    EXECUTE 'CREATE UNIQUE INDEX campaigns_name_unique_idx ON campaigns (name)';
  END IF;
END $$;

-- Drop the old case-insensitive search index if it exists - that one
-- was for LIKE '%foo%' lookups and we don't actually use it.
DROP INDEX IF EXISTS idx_campaigns_name;
