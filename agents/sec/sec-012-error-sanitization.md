# sec-012-error-sanitization

**Severity:** MEDIUM — Sensitive logging / Input validation gap.

> Status: baseline fix landed in commit `382c102` (sanitized upstream exception
> messages). Sub-agent maintains coverage for the remaining branches.

## Scope (files)

- `STT_server/routes/api.py` — health endpoint around 227-259, TTS preview
  around 1389-1391, other raw-`str(exc)` paths.
- `STT_server/services/credentials_resolver.py` — model listing traceback.

## Approach (SAFE_CHANGE)

1. Define `STT_server/utils/safe_errors.py` with one `sanitize_exception(exc) -> dict`
   helper: returns `{code, message, hint}` and never echoes the raw exception
   string.
2. Replace every direct `str(exc)` return and every `traceback.print_exc`
   with the helper.
3. Health endpoint returns only `{status, components: {name: "ok"|"degraded"}}`.
4. Add tests with synthetic exceptions that contain fake secrets and verify
   that they never reach the response body or the log.

## Sub-agents

- `sec-012a-safe-errors-helper` — module under `utils/`.
- `sec-012b-call-site-sweep` — find/replace every raw exception emit.
- `sec-012c-redaction-tests` — `tests/test_safe_errors.py` with synthetic
  secrets in `str(exc)`.

## Dependencies

- None.

## Verification

```python
def test_sanitize_does_not_leak_secret_in_message():
    e = RuntimeError("connection to https://user:sk-LIVE@api.openai.com failed")
    out = sanitize_exception(e)
    assert "sk-LIVE" not in out["message"]
    assert "user:" not in out["message"]

def test_health_endpoint_does_not_return_exceptions():
    monkeypatch.setattr(db, "ping", lambda: (_ for _ in ()).throw(RuntimeError("boom with secret X")))
    r = client.get("/health")
    assert r.status_code == 503
    assert "secret" not in r.text
    assert "boom" not in r.text
```

## Acceptance

- No API response body or log line carries the raw exception string.
- Health endpoint exposes only structured pass/fail.
- Redaction tests pass against synthetic secrets in the exception text.
