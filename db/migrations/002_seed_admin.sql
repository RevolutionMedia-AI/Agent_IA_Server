-- ============================================================================
-- RevolutionMedia-AI · Seed admin
-- ----------------------------------------------------------------------------
-- Inserta el usuario admin inicial. El hash corresponde a la password
-- "Adminrevolutionmedia@109" (SHA-256 hex).
--
-- Si querés cambiar la password, generá el hash nuevo:
--   python -c "import hashlib; print(hashlib.sha256(b'TuPassword').hexdigest())"
-- y reemplazá el valor de "password" en el INSERT.
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
  name = EXCLUDED.name,
  email = EXCLUDED.email,
  password = EXCLUDED.password,
  role = EXCLUDED.role,
  updated_at = NOW();
