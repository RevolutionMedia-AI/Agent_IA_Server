-- ============================================================================
-- RevolutionMedia-AI / Migration 015
-- ----------------------------------------------------------------------------
-- Add the audio_capture table for the 2026-08-28 forensic A/B test.
--
-- One row per capture write (one Inworld chunk for stage='inworld', one
-- 160-byte Twilio frame for stage='twilio'). The raw mu-law bytes go in
-- the BYTEA column so the operator can SELECT them out and diff against
-- the AMR recording from Twilio.
--
-- SHA-256 is computed in Python before INSERT, so the forensic
-- predicate "did the pipeline touch a single byte?" is one SQL
-- statement: SELECT sha256, byte_count FROM audio_capture
-- WHERE call_sid=... AND generation=N AND stage IN ('inworld','twilio')
-- GROUP BY ... — if A.sha256 == B.sha256 the bytes are byte-identical.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS + IF NOT EXISTS on indexes.
-- ============================================================================

CREATE TABLE IF NOT EXISTS audio_capture (
  id              BIGSERIAL    PRIMARY KEY,
  call_sid        TEXT         NOT NULL,
  generation      INTEGER      NOT NULL,
  stage           TEXT         NOT NULL CHECK (stage IN ('inworld', 'twilio')),
  seg             INTEGER,
  byte_count      INTEGER      NOT NULL,
  sha256          TEXT         NOT NULL,
  payload         BYTEA        NOT NULL,
  ts              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_capture_call_gen_stage
  ON audio_capture (call_sid, generation, stage);

CREATE INDEX IF NOT EXISTS idx_audio_capture_ts
  ON audio_capture (ts DESC);
