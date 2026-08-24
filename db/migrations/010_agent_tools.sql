-- ============================================================================
-- 010 · agent_tools.sql
-- ----------------------------------------------------------------------------
-- agent_tools.json was an ephemeral file under STT_server/data/ — every
-- Railway deploy / container restart wiped it, which meant operators
-- came back from a deploy to find all their tools gone. The fix is
-- the same pattern we used for db_agents in 001..009: a real Postgres
-- table that survives the deploy and the JSON file becomes a one-time
-- backfill source on first boot.
--
-- Schema mirrors the JSON shape returned by _load_tools() today, plus the
-- observability fields record_tool_result writes (last_*_at, last_*_result,
-- last_*_error). The `assignments` column is a JSONB array of agent_ids
-- (only meaningful for shared rows where agent_id = '__shared__').
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_tools (
  id                TEXT        PRIMARY KEY,
  user_id           TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  agent_id          TEXT        NOT NULL,            -- real agent id, or '__shared__'
  name              TEXT        NOT NULL,
  description       TEXT,
  webhook_url       TEXT        NOT NULL DEFAULT '',
  filler_phrase     TEXT        NOT NULL DEFAULT 'Let me check the system...',
  parameters        JSONB       NOT NULL DEFAULT '{"type": "object", "properties": {}, "required": []}'::jsonb,
  kind              TEXT        NOT NULL DEFAULT 'webhook',  -- 'webhook' | 'call_transfer'
  destination       TEXT,                                 -- E.164 for call_transfer
  assignments       JSONB       NOT NULL DEFAULT '[]'::jsonb,
  function_name     TEXT        NOT NULL DEFAULT '',
  -- observability columns written by record_tool_result
  last_tested_at            TIMESTAMPTZ,
  last_test_result          TEXT,                  -- 'ok' | 'fail'
  last_test_error           TEXT,
  last_test_error_at        TIMESTAMPTZ,
  last_invoked_at           TIMESTAMPTZ,
  last_invocation_status    TEXT,                  -- 'ok' | 'fail'
  last_invocation_error     TEXT,
  last_invocation_error_at  TIMESTAMPTZ,
  invocation_count         INTEGER     NOT NULL DEFAULT 0,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ
);

-- ponytail: query paths we already use. user_id is the hot path
-- (every CRUD operation filters by user_id first for ownership).
CREATE INDEX IF NOT EXISTS idx_agent_tools_user_id ON agent_tools (user_id);
CREATE INDEX IF NOT EXISTS idx_agent_tools_user_agent ON agent_tools (user_id, agent_id);