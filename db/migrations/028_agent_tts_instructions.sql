-- ============================================================================
-- 028 · agent_tts_instructions.sql
-- ----------------------------------------------------------------------------
-- Per-agent voice instructions for TTS providers that accept free-text
-- steering. Today only OpenAI's gpt-4o-mini-tts does: the Speech API
-- takes an `instructions` field that controls accent, tone, emotional
-- range, intonation and pace ("Speak in Mexican Spanish, warm and
-- conversational"). The legacy tts-1 / tts-1-hd models reject it, so
-- the adapter drops the field for them.
--
-- NULL (not '') means "no instructions" — TTS must keep working when an
-- operator never configures this, and NULL is also exactly what a
-- pre-migration row looks like. So the migration is purely additive: no
-- backfill, no rewrite, no existing agent changes behaviour.
--
-- Bounded to 600 chars so a pasted paragraph cannot bloat the row or the
-- outbound request; the adapter trims to the same limit.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS tts_instructions TEXT
    CHECK (tts_instructions IS NULL OR char_length(tts_instructions) <= 600);