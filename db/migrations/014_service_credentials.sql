-- ============================================================================
-- 014 · service_credentials.sql
-- ----------------------------------------------------------------------------
-- The per-user provider credentials (OpenAI api_key, Twilio SID+token+number,
-- ElevenLabs voice_id, etc.) used to live in a JSON file under
-- STT_server/data/tools_integrations.json — wiped by every Railway redeploy.
-- Migration 010 moved n8n agent_tools to Postgres; 014 finishes the same job
-- for service credentials by giving agent_tools a `credentials` column and a
-- `(user_id, id)` uniqueness convention.
--
-- Storage convention: a per-user service row is an agent_tools row with
--   id = service_id (e.g. "openai", "deepgram", "twilio")
--   agent_id = '__shared__' (the row isn't tied to any specific agent)
--   function_name = service_id (the function_name column is reused as the
--     canonical service id; AgentTool computes it from `name` if empty so
--     on a fresh INSERT we just set it to the service id)
--   credentials = JSONB encrypted dict (encrypt_credentials() writes Fernet
--     ciphertext plus a tiny wrapper)
-- All other tool-shaped columns (webhook_url, filler_phrase, parameters,
-- kind, destination, assignments) stay at their tool-shape defaults — the
-- resolver ignores them.
--
-- The credentials_resolver._read_per_user() helper was already looking for
-- `r.get("credentials")` on these rows but the column didn't exist; this
-- migration makes the read path whole again.
-- ============================================================================

ALTER TABLE agent_tools
  ADD COLUMN IF NOT EXISTS credentials JSONB;

-- ponytail: the existing pk (id, user_id) already lets multiple services
-- for the same user coexist. No additional index needed — the lookup is
-- `WHERE id = %s AND user_id = %s` which the composite PK covers.
