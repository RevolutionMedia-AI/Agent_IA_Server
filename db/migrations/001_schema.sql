-- ============================================================================
-- RevolutionMedia-AI · Postgres schema
-- ----------------------------------------------------------------------------
-- Reemplaza los JSON files de STT_server/data/. Cuando termines de correr
-- 001_schema.sql, el backend (api.py) puede migrarse de _load/_save a
-- queries SQLAlchemy/asyncpg — los contratos de los routes no cambian.
--
-- Orden de ejecución (recomendado) en DBeaver:
--   1. db/migrations/001_schema.sql
--   2. db/migrations/002_seed_admin.sql
--
-- Probado con Postgres 14+. Compatible con 13+.
-- ============================================================================


-- ============================================================================
-- Sección 0 · Extensiones
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- para gen_random_uuid() si decides migrar de TEXT ids a UUID


-- ============================================================================
-- Sección 1 · users
-- ----------------------------------------------------------------------------
-- Reemplaza STT_server/data/users.json.
-- Cada user puede tener muchos agents / phone_numbers / tools_integrations
-- / settings / auth_sessions. El id es TEXT (no UUID) para conservar los
-- ids existentes del JSON ("user-{uuid12}"). Cuando migres a UUID
-- type, deja TEXT pero los nuevos ids pueden ser uuid() reales.
-- ============================================================================
CREATE TABLE users (
  id          TEXT        PRIMARY KEY,                       -- "user-{uuid12}"
  name        TEXT        NOT NULL,
  email       TEXT        NOT NULL UNIQUE,
  password    TEXT        NOT NULL,                           -- SHA-256 hex digest (64 chars)
  role        TEXT        NOT NULL DEFAULT 'admin',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at  TIMESTAMPTZ
);
CREATE INDEX idx_users_role  ON users (role);
CREATE INDEX idx_users_email ON users (email);  -- UNIQUE ya crea uno, este es redundante pero explícito


-- ============================================================================
-- Sección 2 · auth_sessions
-- ----------------------------------------------------------------------------
-- Reemplaza STT_server/data/sessions.json. El token Bearer es la PK.
-- api.py hace require_auth() leyendo esta tabla + un cache en memoria.
-- ============================================================================
CREATE TABLE auth_sessions (
  token       TEXT        PRIMARY KEY,                       -- el bearer token (secrets.token_urlsafe(32))
  user_id     TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  email       TEXT        NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at  TIMESTAMPTZ NOT NULL                          -- 7 días desde creación
);
CREATE INDEX idx_auth_sessions_user    ON auth_sessions (user_id);
CREATE INDEX idx_auth_sessions_expires ON auth_sessions (expires_at);  -- útil para limpieza periódica


-- ============================================================================
-- Sección 3 · agents
-- ----------------------------------------------------------------------------
-- Reemplaza STT_server/data/agents.json. Un agent pertenece a un único
-- user. voice es un id libre (e.g. "rime-warm-f"), language es
-- "English" | "Spanish" | "Bilingual", status valida 3 valores.
-- calls se guarda como TEXT (formateado con coma, e.g. "8,432") para
-- matchear el shape que devuelve el backend hoy.
-- ============================================================================
CREATE TABLE agents (
  id          TEXT        PRIMARY KEY,                       -- "agent-{uuid8}"
  user_id     TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name        TEXT        NOT NULL,
  voice       TEXT,
  language    TEXT        NOT NULL DEFAULT 'English',
  campaign    TEXT,
  status      TEXT        NOT NULL DEFAULT 'Active'
                          CHECK (status IN ('Active', 'Training', 'Paused')),
  description TEXT,
  tone        TEXT,
  prompt      TEXT,
  calls       TEXT        NOT NULL DEFAULT '0',              -- pre-formateado, ver api.py
  perf        INTEGER     NOT NULL DEFAULT 0,                -- 0–100
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_agents_user        ON agents (user_id);
CREATE INDEX idx_agents_user_status ON agents (user_id, status);  -- filtro + listado


-- ============================================================================
-- Sección 4 · phone_numbers
-- ----------------------------------------------------------------------------
-- Reemplaza STT_server/data/phone_numbers.json. agent es soft-FK al
-- id de agents: si borras un agent, el campo queda NULL (los números
-- sobreviven). display y label son derivados del country+number para UI.
-- ============================================================================
CREATE TABLE phone_numbers (
  id          TEXT        PRIMARY KEY,                       -- "num-{uuid8}"
  user_id     TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider    TEXT        NOT NULL DEFAULT 'twilio',
  country     TEXT        NOT NULL DEFAULT '+1',
  number      TEXT        NOT NULL,
  display     TEXT,                                          -- "52 55 1234 5678"
  label       TEXT,
  agent       TEXT        REFERENCES agents(id) ON DELETE SET NULL,  -- soft FK
  calls       TEXT        NOT NULL DEFAULT '0',
  status      TEXT        NOT NULL DEFAULT 'Active',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_phone_numbers_user  ON phone_numbers (user_id);
CREATE INDEX idx_phone_numbers_agent ON phone_numbers (agent);


-- ============================================================================
-- Sección 5 · tools_integrations
-- ----------------------------------------------------------------------------
-- Reemplaza STT_server/data/tools_integrations.json. credentials es JSONB
-- (puede contener tokens de Twilio, OpenAI, etc — cifrar a nivel de
-- columna si vas a producción real).
-- ============================================================================
CREATE TABLE tools_integrations (
  id            TEXT        PRIMARY KEY,                     -- "twilio" | "openai-realtime" | "webhooks" | "cli-deploy"
  user_id       TEXT        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  connected     BOOLEAN     NOT NULL DEFAULT FALSE,
  credentials   JSONB       NOT NULL DEFAULT '{}'::jsonb,
  connected_at  TIMESTAMPTZ
);
CREATE INDEX idx_tools_user ON tools_integrations (user_id);


-- ============================================================================
-- Sección 6 · settings
-- ----------------------------------------------------------------------------
-- Reemplaza STT_server/data/settings/<user_id>.json. Una fila por user.
-- notifications es JSONB con la forma:
--   { "calls": true, "qa": true, "weekly": false, "marketing": false }
-- ============================================================================
CREATE TABLE settings (
  user_id       TEXT        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  name          TEXT,
  company       TEXT,
  timezone      TEXT        NOT NULL DEFAULT 'America/Mexico_City',
  notifications JSONB       NOT NULL DEFAULT '{"calls": true, "qa": true, "weekly": false, "marketing": false}'::jsonb
);


-- ============================================================================
-- Sección 7 · tenants
-- ----------------------------------------------------------------------------
-- Reemplaza el in-memory store de STT_server/domain/tenant.py:tenant_store.
-- Aquí vive la config de Twilio por sub-cuenta. Los secrets (auth_token,
-- api_keys) se guardan en texto plano — cifrar la columna si producción
-- real lo requiere.
-- ============================================================================
CREATE TABLE tenants (
  tenant_id            TEXT        PRIMARY KEY,             -- "tenant-{uuid12}"
  name                  TEXT,
  twilio_account_sid    TEXT,
  twilio_auth_token     TEXT,
  twilio_phone_number   TEXT,
  custom_prompt         TEXT,
  tts_provider          TEXT,
  preferred_language    TEXT,
  openai_api_key        TEXT,
  elevenlabs_api_key    TEXT,
  elevenlabs_voice_id   TEXT,
  deepgram_api_key      TEXT,
  webhook_configured    BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ
);


-- ============================================================================
-- Sección 8 · call_sessions
-- ----------------------------------------------------------------------------
-- Reemplaza el in-memory STT_server/services/session_runtime.py:sessions.
-- Una fila por llamada activa. Cuando closed=true, el row se puede
-- archivar/borrar sin pérdida. tenant_id es soft-FK.
-- ============================================================================
CREATE TABLE call_sessions (
  session_key        TEXT        PRIMARY KEY,
  tenant_id          TEXT        REFERENCES tenants(tenant_id) ON DELETE SET NULL,
  call_sid           TEXT,
  preferred_language TEXT,
  tts_provider       TEXT,
  custom_prompt      TEXT,
  assistant_speaking BOOLEAN     NOT NULL DEFAULT FALSE,
  closed             BOOLEAN     NOT NULL DEFAULT FALSE,
  started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at           TIMESTAMPTZ
);
CREATE INDEX idx_call_sessions_tenant ON call_sessions (tenant_id);
CREATE INDEX idx_call_sessions_closed ON call_sessions (closed);


-- ============================================================================
-- Listo. Próximo paso: db/migrations/002_seed_admin.sql
-- ============================================================================
