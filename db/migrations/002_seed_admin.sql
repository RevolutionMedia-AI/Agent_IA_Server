-- ============================================================================
-- RevolutionMedia-AI · Seed admin
-- ----------------------------------------------------------------------------
-- Inserts the initial admin user. The password hash corresponds to
-- "Adminrevolutionmedia@109" (SHA-256 hex). To rotate the password:
--
--   python -c "import hashlib; print(hashlib.sha256(b'NEW').hexdigest())"
--
-- then replace the literal below, or update it via the FE after the
-- first login (PUT /settings/password).
-- ============================================================================

INSERT INTO users (id, name, email, password, role)
VALUES (
  'user-admin-001',
  'Revolution Media Admin',
  'admin@revolutionmedia.ai',
  '55fb08cb36cf4be5c4c9d1dc620479c13c8abd62ae9569b4f2b67ff138d3a15a',
  'admin'
)
ON CONFLICT (id) DO UPDATE SET
  name        = EXCLUDED.name,
  email       = EXCLUDED.email,
  password    = EXCLUDED.password,
  role        = EXCLUDED.role,
  updated_at  = NOW();
