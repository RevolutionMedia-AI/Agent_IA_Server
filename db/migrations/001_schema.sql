-- ============================================================================
-- RevolutionMedia-AI · Postgres schema (clean)
-- ----------------------------------------------------------------------------
-- Single migration replacing 001..008 (and 009). The DB is empty at the
-- time of writing, so this is a clean slate.
--
-- Design rules
--   * No dead tables, no dead columns, no dead migrations. The 005+008
--     add-then-drop of phone_numbers.name, the 006 `SELECT 1;` placeholder,
--     and the 007 DROP INDEX of an index that never existed are gone.
--   * Tenants = Twilio sub-account + per-call config ONLY. Provider
--     credentials live in tools_integrations (the canonical catalog from
--     PROVIDER_CATALOG) and are resolved per-user at call time. The
--     resolver chain never reads from tenants.
--   * PKs are right the first time. No DO $$ patches to undo a mistake
--     the original CREATE TABLE made.
--   * Soft-FK (ON DELETE SET NULL) only for agent->phone_numbers, where
--     the number record must survive the agent being deleted. Everything
--     else cascades on user delete.
--   * All secrets stored as TEXT. Fernet column-level encryption is a
--     later migration (already in code at security/credentials.py); we
--     don't smuggle that complexity into the base schema.
--   * TIMESTAMPTZ everywhere. created_at defaults to NOW(); updated_at
--     defaults to NULL and the app sets it on every UPDATE.
--   * TEXT ids preserve the existing "user-{uuid12}" / "agent-{uuid8}" /
--     "num-{uuid8}" / "tenant-{uuid12}" contract the FE and the JSON
--     fallback files still use. Migrating to native UUID is a separate
--     decision and not done here.
--   * Functional + partial indexes for the queries the code actually
--     runs (see comments on each index).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- reserved for column-level encryption


-- ============================================================================
-- 1 · users
-- ----------------------------------------------------------------------------
-- Replaces STT_server/data/users.json. The single admin bootstrap row is
-- inserted by 002_seed_admin.sql. password is SHA-256 hex; migrate to
-- bcrypt (with a real salt) before exposing this server publicly.
-- ============================================================================
CREATE TABLE users (
  id          TEXT        PRIMARY KEY,                          -- "user-{uuid12}"
  name        TEXT        NOT NULL,
  email       TEXT        NOT NULL UNIQUE,
  password    TEXT        NOT NULL,                              -- SHA-256 hex (64 chars)
  role        TEXT        NOT NULL DEFAULT 'admin'
                          CHECK (role IN ('admin', 'user')),
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at  TIMESTAMPTZ
);
-- UNIQUE on email already builds the lookup index. No extra idx_users_email.


-- ============================================================================
-- 2 · auth_sessions
-- ----------------------------------------------------------------------------
-- Replaces STT_server/data/sessions.json. PK is the Bearer token
-- (secrets.token_urlsafe(32)). auth_sessions is also kept hot in an
-- in-memory cache by routes/api.py, but Postgres is the source of truth.
-- ============================================================================
CREATE TABLE auth_sessions (
  token        TEXT        PRIMARY KEY,                          -- the bearer token
  user_id      TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  email        TEXT        NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at   TIMESTAMPTZ NOT NULL                              -- 7 days from creation
);
CREATE INDEX idx_auth_sessions_user    ON auth_sessions (user_id);
CREATE INDEX idx_auth_sessions_expires ON auth_sessions (expires_at);  -- periodic cleanup


-- ============================================================================
-- 3 · settings
-- ----------------------------------------------------------------------------
-- One row per user. Replaces STT_server/data/settings/<user_id>.json.
-- notifications is JSONB with shape:
--   { "calls": bool, "qa": bool, "weekly": bool, "marketing": bool }
-- ============================================================================
CREATE TABLE settings (
  user_id        TEXT        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  name           TEXT,
  company        TEXT,
  timezone       TEXT        NOT NULL DEFAULT 'America/Mexico_City',
  notifications  JSONB       NOT NULL DEFAULT
                              '{"calls": true, "qa": true, "weekly": false, "marketing": false}'::jsonb,
  updated_at     TIMESTAMPTZ
);


-- ============================================================================
-- 4 · agents
-- ----------------------------------------------------------------------------
-- A voice agent owned by a user. voice is the display name ("Aria"),
-- voice_id is the underlying provider voice id. The per-service
-- provider+model columns override the user-level provider credentials
-- resolved by credentials_resolver at call time.
--
-- calls is TEXT (formatted "8,432") because the FE renders that string
-- verbatim; perf is a 0..100 score.
-- ============================================================================
CREATE TABLE agents (
  id                TEXT        PRIMARY KEY,                    -- "agent-{uuid8}"
  user_id           TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name              TEXT        NOT NULL,
  description       TEXT,
  tone              TEXT,
  prompt            TEXT,
  welcome_message   TEXT,
  voice             TEXT,                                        -- display name
  voice_id          TEXT,                                        -- provider voice id
  language          TEXT        NOT NULL DEFAULT 'English'
                                  CHECK (language IN ('English', 'Spanish', 'Bilingual')),
  status            TEXT        NOT NULL DEFAULT 'Active'
                                  CHECK (status IN ('Active', 'Training', 'Paused')),
  campaign          TEXT,                                        -- free-form tag, see db_campaigns

  -- per-agent provider overrides
  stt_provider      TEXT,
  stt_model         TEXT,
  tts_provider      TEXT,
  tts_model         TEXT,
  llm_provider      TEXT,
  llm_model         TEXT,

  -- denormalized counters
  calls             TEXT        NOT NULL DEFAULT '0',           -- pre-formatted "8,432"
  perf              SMALLINT    NOT NULL DEFAULT 0
                                  CHECK (perf BETWEEN 0 AND 100),

  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ
);
CREATE INDEX idx_agents_user        ON agents (user_id);
CREATE INDEX idx_agents_user_status ON agents (user_id, status);  -- most common filter
CREATE INDEX idx_agents_campaign    ON agents (campaign) WHERE campaign IS NOT NULL;


-- ============================================================================
-- 5 · phone_numbers
-- ----------------------------------------------------------------------------
-- A phone number (or SIP trunk, or WhatsApp sender) owned by a user,
-- optionally assigned to an agent. twilio_account_sid / twilio_auth_token
-- are the per-number sub-account override; when null, the call falls
-- through to the tenant's Twilio creds (see tenants table).
--
-- find_by_number() does a suffix match on digits (so "+521551234567"
-- matches stored "21551234567"); the functional index on the
-- digit-only form keeps that O(log n) instead of a seq scan.
-- ============================================================================
CREATE TABLE phone_numbers (
  id                        TEXT        PRIMARY KEY,             -- "num-{uuid8}"
  user_id                   TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider                  TEXT        NOT NULL DEFAULT 'twilio'
                                       CHECK (provider IN ('twilio', 'sip', 'whatsapp')),
  country                   TEXT        NOT NULL DEFAULT '+1',  -- e.g. "+52"
  number                    TEXT        NOT NULL,                -- E.164
  display                   TEXT,                                -- "+52 55 12 34 56 78"
  label                     TEXT,                                -- optional override (FE label)
  agent                     TEXT        REFERENCES agents(id) ON DELETE SET NULL,
  campaign                  TEXT,                                -- free-form tag
  calls                     TEXT        NOT NULL DEFAULT '0',
  status                    TEXT        NOT NULL DEFAULT 'Active'
                                       CHECK (status IN ('Active', 'Inactive')),

  -- per-number Twilio sub-account override
  twilio_account_sid        TEXT,
  twilio_auth_token         TEXT,
  -- SIP trunk
  sip_host                  TEXT,
  sip_username              TEXT,
  sip_password              TEXT,
  -- WhatsApp business
  whatsapp_phone_number_id  TEXT,
  whatsapp_access_token     TEXT,

  created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                TIMESTAMPTZ,

  -- one row per (user, provider, number)
  UNIQUE (user_id, provider, number)
);
CREATE INDEX idx_phone_numbers_user     ON phone_numbers (user_id);
CREATE INDEX idx_phone_numbers_agent    ON phone_numbers (agent);
CREATE INDEX idx_phone_numbers_campaign ON phone_numbers (campaign) WHERE campaign IS NOT NULL;
CREATE INDEX idx_phone_numbers_digits   ON phone_numbers ((regexp_replace(number, '\D', '', 'g')));


-- ============================================================================
-- 6 · tools_integrations
-- ----------------------------------------------------------------------------
-- The SINGLE home for per-user provider credentials (the canonical
-- catalog lives in STT_server/services/credentials_resolver.py:PROVIDER_CATALOG).
-- One row per (user, provider_id). The composite PK lets every user
-- carry their own row for the same provider without colliding.
--
-- credentials is JSONB and is the target of Fernet column-level
-- encryption (see security/credentials.py). Storing the encrypted
-- blob as a JSONB string keeps the column shape stable when the
-- encryption key rotates.
-- ============================================================================
CREATE TABLE tools_integrations (
  user_id        TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  id             TEXT        NOT NULL,                          -- provider id (PROVIDER_CATALOG)
  display_name   TEXT        NOT NULL,
  category       TEXT        NOT NULL
                              CHECK (category IN ('llm', 'stt', 'tts', 'telephony')),
  connected      BOOLEAN     NOT NULL DEFAULT FALSE,
  credentials    JSONB       NOT NULL DEFAULT '{}'::jsonb,
  connected_at   TIMESTAMPTZ,
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  PRIMARY KEY (user_id, id)
);
CREATE INDEX idx_tools_user_category ON tools_integrations (user_id, category);


-- ============================================================================
-- 7 · tenants
-- ----------------------------------------------------------------------------
-- A tenant = a Twilio sub-account with its own call configuration.
-- Provider API keys are NEVER stored here — they live on the owning
-- user's tools_integrations row and are resolved by credentials_resolver.
--
-- In-memory TenantStore (domain/tenant.py) is the current implementation;
-- this table is the DB-backed target. A partial UNIQUE on twilio_phone_number
-- enforces "one tenant per phone number" across the whole platform
-- without blocking tenants that haven't configured a number yet.
-- ============================================================================
CREATE TABLE tenants (
  tenant_id            TEXT        PRIMARY KEY,                  -- "tenant-{uuid12}"
  user_id              TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name                 TEXT,
  twilio_account_sid   TEXT,
  twilio_auth_token    TEXT,
  twilio_phone_number  TEXT,
  custom_prompt        TEXT,
  tts_provider         TEXT        NOT NULL DEFAULT 'elevenlabs'
                                  CHECK (tts_provider IN ('elevenlabs', 'deepgram', 'rime', 'inworld')),
  preferred_language   TEXT        NOT NULL DEFAULT 'es'
                                  CHECK (preferred_language IN ('en', 'es')),
  webhook_configured   BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ
);
CREATE UNIQUE INDEX uniq_tenants_phone ON tenants (twilio_phone_number)
  WHERE twilio_phone_number IS NOT NULL;
CREATE INDEX idx_tenants_user ON tenants (user_id);


-- ============================================================================
-- 8 · call_sessions
-- ----------------------------------------------------------------------------
-- Live call state. Created on /voice POST, marked closed=true on hangup.
-- Today the in-memory session_runtime (services/session_runtime.py) is the
-- source of truth; this table is the DB-backed target. Soft-FK to tenant
-- so a tenant deletion doesn't drop call history.
--
-- Kept in the base schema (not deferred to a later migration) because
-- /sessions and /dashboard already reference closed/started_at fields
-- in routes/api.py — the column shape is stable.
-- ============================================================================
CREATE TABLE call_sessions (
  session_key         TEXT        PRIMARY KEY,                  -- generated per call
  tenant_id           TEXT        REFERENCES tenants(tenant_id) ON DELETE SET NULL,
  call_sid            TEXT,                                     -- Twilio CallSid
  preferred_language  TEXT,
  tts_provider        TEXT,
  custom_prompt       TEXT,
  assistant_speaking  BOOLEAN     NOT NULL DEFAULT FALSE,
  closed              BOOLEAN     NOT NULL DEFAULT FALSE,
  started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at            TIMESTAMPTZ
);
CREATE INDEX idx_call_sessions_tenant       ON call_sessions (tenant_id);
CREATE INDEX idx_call_sessions_closed_open  ON call_sessions (closed, started_at DESC);
