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
-- ponytail: this migration must be a SINGLE statement. The startup
-- script runs `cur.execute(sql)` which doesn't accept multiple
-- statements separated by `;` (the prior version used a DO block +
-- CREATE INDEX + DROP INDEX which silently all-but-the-first got
-- dropped on the wire). CREATE UNIQUE INDEX IF NOT EXISTS works in
-- Postgres 9.5+ and ON CONFLICT accepts a unique index as its
-- conflict target.
-- ============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS campaigns_name_unique_idx ON campaigns (name);
