ALTER TABLE settings
  ADD COLUMN IF NOT EXISTS test_data_model TEXT NOT NULL DEFAULT 'gpt-4o-mini';
