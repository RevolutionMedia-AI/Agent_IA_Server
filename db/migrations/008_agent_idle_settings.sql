-- ============================================================================
-- 008 · agent_idle_settings.sql
-- ----------------------------------------------------------------------------
-- Per-agent idle / silence detection (the "are you still there?" flow).
--
-- All 7 columns are nullable so legacy rows keep working — a NULL means
-- "fall back to the platform global IDLE_SILENCE_TIMEOUT_SEC / no prompts",
-- which is exactly what existing agents had before this migration ran.
--
-- Flow when idle_enabled = TRUE:
--   t = idle_first_timeout_sec   of silence → speak idle_first_message
--   t = idle_subsequent_timeout_sec of silence → speak idle_final_message
--   … repeat idle_subsequent_timeout_sec up to idle_max_attempts prompts
--   t = idle_disconnect_timeout_sec of silence → close the websocket
--
-- With the defaults below (5 / "Are you still there?" / 5 / final / 5 / 2),
-- this matches the spec's worked example:
--   5 s → "Are you still there?"
--   5 s → "I'm not hearing a response, so I'm going to disconnect the call."
--   5 s → hang up.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS idle_enabled                BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS idle_first_timeout_sec      INTEGER
    CHECK (idle_first_timeout_sec      IS NULL OR idle_first_timeout_sec      >  0),
  ADD COLUMN IF NOT EXISTS idle_first_message          TEXT
    CHECK (idle_first_message          IS NULL OR length(idle_first_message)          <= 1000),
  ADD COLUMN IF NOT EXISTS idle_subsequent_timeout_sec INTEGER
    CHECK (idle_subsequent_timeout_sec IS NULL OR idle_subsequent_timeout_sec >  0),
  ADD COLUMN IF NOT EXISTS idle_final_message          TEXT
    CHECK (idle_final_message          IS NULL OR length(idle_final_message)          <= 1000),
  ADD COLUMN IF NOT EXISTS idle_disconnect_timeout_sec INTEGER
    CHECK (idle_disconnect_timeout_sec IS NULL OR idle_disconnect_timeout_sec >  0),
  ADD COLUMN IF NOT EXISTS idle_max_attempts           INTEGER
    CHECK (idle_max_attempts           IS NULL OR (idle_max_attempts >= 1 AND idle_max_attempts <= 10));