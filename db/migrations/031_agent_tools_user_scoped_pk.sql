-- ============================================================================
-- 031 · agent_tools_user_scoped_pk.sql
-- ----------------------------------------------------------------------------
-- The per-user provider credentials are stored in agent_tools with
-- id = service_id ("openai", "elevenlabs", ...) and agent_id = '__shared__'.
-- The idea is that each user has their own row for each provider, so the
-- table's primary key has to be (user_id, id).
--
-- It was NOT. Migration 010 created the table with:
--
--     id TEXT PRIMARY KEY
--
-- and migration 014 believed it was already "(id, user_id)":
--
--     -- ponytail: the existing pk (id, user_id) already lets multiple
--     -- services for the same user coexist. No additional index needed
--
-- That comment was wrong: the primary key was (id) alone. The
-- consequence is that a provider id can only exist ONCE in the whole
-- table, for the whole platform.
--
-- What that breaks, end to end (reproduced against a fresh database):
--
--     PUT /settings/api-keys/openai   as user A   -> 200 OK
--     PUT /settings/api-keys/openai   as user B   -> 500
--
--     UniqueViolation: duplicate key value violates unique constraint
--     "agent_tools_pkey"
--     DETAIL:  Key (id)=(openai) already exists.
--
-- The route looks the row up scoped to the caller
-- (db_get_tool(service_id, user_id)), finds nothing for the second user
-- because the row belongs to the first one, and then INSERTs a row with
-- the same id -- which the primary key forbids. So:
--
--   * only the FIRST user of the platform can save a credential for a
--     given provider, and
--   * every other user gets a 500 when they try.
--
-- This migration makes the key what the code and the comments always
-- assumed: (user_id, id).
--
-- It is safe to apply on a live database:
--
--   * the current key (id) is STRICTER than the new one, so the new
--     key cannot be violated by the existing rows: there is at most
--     one row per id today, and the new key allows one row per
--     (user_id, id). No data is lost and no row needs rewriting.
--   * every query against agent_tools is already scoped by user_id
--     (list_tools, get_tool, update_tool, delete_tool, the assignment
--     helpers). The one exception is the tool dispatcher's
--     "SELECT user_id FROM agent_tools WHERE id = %s LIMIT 1", which
--     finds the owner of a TOOL. Tool ids are generated as UUID-style
--     strings, so they do not collide across users; the rows that DO
--     share an id are the provider credential rows, which are not
--     dispatchable.
--
-- The index that the old key provided on id alone is kept as a
-- non-unique index, so the dispatcher's lookup does not lose its
-- index.
-- ============================================================================

ALTER TABLE agent_tools DROP CONSTRAINT IF EXISTS agent_tools_pkey;

ALTER TABLE agent_tools
  ADD CONSTRAINT agent_tools_pkey PRIMARY KEY (user_id, id);

-- The old key's index on id alone is gone with it; the dispatcher's
-- "WHERE id = %s" lookup still wants one.
CREATE INDEX IF NOT EXISTS idx_agent_tools_id ON agent_tools (id);
