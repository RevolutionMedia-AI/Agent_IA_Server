-- 032_settings_test_data_model.sql
--
-- The settings table gets the model the "Test" buttons use.
--
-- ponytail: this column was NEVER migrated: it only existed because
-- db_settings.get_settings() self-heals it at runtime with an ALTER
-- the first time it reads the row. A database created from the
-- migrations alone (CI, a fresh deploy, the Go service's own runner)
-- was missing the column, and the settings endpoint answered 500 with
-- "column test_data_model does not exist".
--
-- The self-heal stays: it is what saved the existing deployments.
-- This migration is what makes a fresh database correct from the
-- start.
ALTER TABLE settings
    ADD COLUMN IF NOT EXISTS test_data_model TEXT DEFAULT 'gpt-4o-mini';
