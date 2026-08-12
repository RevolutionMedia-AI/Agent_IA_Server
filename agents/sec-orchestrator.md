# sec-orchestrator

Coordinates remediation of SEC-001 → SEC-014. Enforces execution order, shared
dependencies, and prevents concurrent edits to overlapping files.

## Responsibilities

1. Read `SECURITY_AUDIT.md` and `agents/sec/*.md`.
2. Build dependency graph between sub-agents (see "Execution order" in
   `agents/README.md`).
3. Sequence sub-agents: no agent runs before its declared dependencies reach
   "acceptance passed".
4. Hold a single global state file `.agents-state.json` with one entry per
   finding (`pending|in_progress|blocked|done`).
5. Refuse to mark `done` if the sub-agent's verification command fails or if
   tests in `tests/` regress.

## Shared state it owns

- `.agents-state.json` — finding statuses + last verification timestamp.
- `.agents-notes/` — per-finding diffs, commands run, evidence links (free-form
  Markdown, append-only).

## Sub-agents it manages

- `sec-001-bootstrap-credential` — blocks `start.sh`/migrations leak.
- `sec-002-tenant-authorization` — owner-scoped CRUD + admin role.
- `sec-003-twilio-voice-signature` — fail-closed `/voice`.
- `sec-004-media-stream-binding` — signed → bound WS handshake.
- `sec-005-tool-ssrf` — apply `validate_public_url` to tool webhook path.
- `sec-006-credential-storage` — encrypt + mask phone/SIP/WhatsApp/Tenant Twilio.
- `sec-007-password-hashing` — Argon2id/bcrypt + redaction + throttling.
- `sec-008-tool-policy` — server-side tool allowlist + risk class + confirmation.
- `sec-009-log-redaction` — metadata-only default across all providers.
- `sec-010-credential-reveal` — replace-only / reauth flow / hardened storage.
- `sec-011-call-status-signature` — Twilio signature on `/call-status`.
- `sec-012-error-sanitization` — central sanitizer at every API/log boundary.
- `sec-013-dependency-sbom` — lock, SBOM, audit, split test/runtime.
- `sec-014-encryption-key` — fail-closed master key + versioned envelopes.

## Cross-cutting dependencies

- `sec-005`, `sec-006`, `sec-010`, `sec-014` share `security/credentials.py`
  and `routes/api.py` — orchestrate them serially.
- `sec-007`, `sec-009` both touch `routes/auth.py` and the logger surface —
  batch them.
- `sec-011`, `sec-012` reuse the Twilio signature helper from `sec-003` —
  ship `sec-003` first.

## Exit condition

All 14 sub-agents reach `done` and `pytest tests/` is green. The orchestrator
prints a final report listing each finding, the patch summary, and the
verification artifact paths.

## Non-goals

- Does not invent new findings.
- Does not merge branches.
- Does not deploy.
