-- ============================================================================
-- 025 · agent_transfer_unavailable_message.sql
-- ----------------------------------------------------------------------------
-- Per-agent line for the "nobody picked up" moment of a handoff chain.
--
-- When a transfer chain (023) or a pre-AI cascade (021) runs out of
-- destinations, the call returns to the AI. The AI is LIVE at that point:
-- it is already connected to the realtime model, and what the caller hears
-- is the model's own reply, driven by the system note seeded in
-- STT_Server's transfer_resume branch. Setting session.welcome_message
-- there does nothing on that path — the greeting is only spoken on a fresh
-- call, by play_initial_greeting.
--
-- So the only way to control the sentence is to put it in the note. This
-- column is that sentence, in the operator's own words and language.
--
-- NULL = keep the built-in behaviour, which picks Spanish or English from
-- agents.language. That is what every existing row gets, so this migration
-- changes nothing until an operator sets a value.
--
-- Capped at 1000 chars like the other per-agent message columns; a TTS
-- utterance longer than that is a configuration mistake, not a message.
-- ============================================================================

ALTER TABLE agents
  ADD COLUMN IF NOT EXISTS transfer_unavailable_message TEXT
    CHECK (transfer_unavailable_message IS NULL
           OR length(transfer_unavailable_message) <= 1000);
