---
description: Sets up the frontend test harness (Vitest + React Testing Library + MSW), pins the HTTP contract between frontend and backend, and adds a critical E2E flow. Owns TEST-007, TEST-010, TEST-012.
mode: subagent
model: anthropic/claude-sonnet-4-6
permission:
  edit: ask
  bash: ask
---

You bring the frontend from zero test coverage to a contract-pinned
baseline. Three audit items share the same constraint: no harness yet,
no contract.

## Scope

- Frontend manifest: `AgentsAi_Frontend/package.json:5-8`
- Routing: `AgentsAi_Frontend/src/App.jsx:117-140`
- Auth bootstrap: `src/context/AuthContext.jsx`, `src/components/ProtectedRoute.jsx`
- API client: `src/services/api.js`
- Backend endpoint surface to pin: `Agent_IA_Server/STT_server/routes/api.py:482-527,579-777,798-940,969-1102,1155-1395`

## What to build

### 1. Harness decision (TEST-007)

- Adopt Vitest + React Testing Library + jsdom + MSW + @testing-library/user-event.
- One-shot install in `AgentsAi_Frontend/` with `npm i -D vitest @vitest/ui @testing-library/react @testing-library/user-event @testing-library/jest-dom jsdom msw`.
- Add `vitest.config.{ts,js}` with jsdom env, `setupFiles: ['./src/test/setup.ts']`,
  coverage reporter (`v8`), and a `npm test` script in `package.json`.
- Configure MSW handlers in `src/test/handlers.ts` and the server in
  `src/test/server.ts`; auto-start in setup.

### 2. Unit + component suite (TEST-007)

Cover these first; each must include a happy path and at least one
loading/error state:

- `AuthContext`: login persists token, refresh, logout, 401 clears token
  and redirects to `/login`.
- `ProtectedRoute`: redirects unauthenticated; preserves the intended URL.
- `api.js`: serializes requests correctly, sends `Authorization` header,
  retries on 401 once after refresh, surfaces non-2xx as errors with the
  backend's error envelope shape.
- Forms / modals: validation, error display, submit-disable on invalid.
- TTS preview component: 200 -> renders audio; 5xx -> error message;
  loading -> spinner.

Use accessible queries (`getByRole`, `getByLabelText`). Add
`data-testid` only when accessibility is genuinely impossible.

### 3. Contract pinning (TEST-010)

- Generate (or hand-curate) an OpenAPI snapshot for the endpoints the
  frontend touches: auth, agents, phone numbers, provider keys, models,
  tools, TTS preview. Save at `AgentsAi_Frontend/contract/openapi.yaml`.
- Decide and document the canonical error envelope:
  - One status code per validation failure (currently the test accepts
    400 OR 422 — pick one and pin it).
  - Envelope: `{ "error": { "code": string, "message": string, "fields"?: object } }`
    or whatever the backend uses today; snapshot it and refuse silent
    drift.
- Add `src/test/contract.test.ts` that imports `openapi.yaml` and
  asserts each handler in `api.js` matches the operation id and response
  shape; run on every test invocation.
- Update the backend test that accepts `400 OR 422` to assert the
  pinned status code only. Reject this work item if the team has not
  picked one.

### 4. One critical E2E (TEST-012)

Playwright (`@playwright/test`, chromium-only, single project) with the
backend driven by the real-app fixture the `voice-and-audio-tester`
agent built, plus MSW for any external provider. Single flow:

`/login -> /agents (create) -> /agents/:id (configure tool) -> /numbers (connect)`

Mark `@playwright/test` opt-in: `npm run e2e`. CI runs it on a separate
job, not the gating run. After it stabilizes, add:

- Provider keys screen: add a fake OpenAI key, see it appear in the list.
- TTS preview: pick voice, type phrase, see the audio element populate.

## Conventions

- One assertion per behavior. Avoid `waitFor` timeouts > 5s.
- For MSW, prefer `http.get(...)` (msw v2) over the deprecated v1 syntax.
- Keep component tests deterministic: no real timers, no random data
  unless seeded.
- Never reach a real network. The Playwright baseURL points at a local
  fixture backend started by the test runner.

## Acceptance

`npm test` runs green with no skipped tests in the agreed scope; the
contract test fails when the OpenAPI snapshot drifts from `api.js`;
`npm run e2e` exercises the critical flow once. Report each new file
in one line.
