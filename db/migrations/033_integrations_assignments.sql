-- ============================================================================
-- 033 A– integrations_assignments.sql
-- ----------------------------------------------------------------------------
-- Adds the `assignments` column to `integrations`.
--
-- The column holds the ids of the agents a SHARED integration is
-- reachable from. It was created at runtime by the Python backend's
-- self-heal (`db_integrations._ensure_integrations_table`), so a
-- database created by the migrations alone did not have it: the
-- column only appeared when the Python backend booted and noticed it
-- was missing.
--
-- That is a schema change hidden in application code, and it breaks
-- the moment the traffic moves to the Go backend: the Go service does
-- not run the Python's self-heal, so the column would simply be
-- missing there.
--
-- The migration makes it explicit and idempotent. The self-heal stays
-- in the Python backend for the deploys whose migration history is
-- already broken, but a fresh database gets the column from here.
-- ============================================================================

ALTER TABLE integrations
  ADD COLUMN IF NOT EXISTS assignments JSONB NOT NULL DEFAULT '[]'::jsonb;
