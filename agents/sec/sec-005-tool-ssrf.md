# sec-005-tool-ssrf

**Severity:** HIGH — Input validation gap (SSRF sink in tool webhook path).

## Scope (files)

- `STT_server/domain/tool.py` — URL validator around 117-130.
- `STT_server/services/tool_executor.py` — execution path 83-120.
- `STT_server/routes/api.py` — tool CRUD around 653-777.

## Approach (NEEDS_TESTS_FIRST)

1. Apply the existing `validate_public_url` from `utils/safe_http.py:88-142` at
   create / update / immediate pre-request.
2. Default scheme to `https`; reject http unless explicitly allowed per tool.
3. Disable redirects by default; if kept, validate each hop (DNS resolve +
   `validate_public_url`) before following.
4. Optional per-tool hostname allowlist, surfaced in the tool config model.
5. Audit stored URLs against the new policy before enforcement to avoid
   breaking existing tools; provide a `migrate_tools.py` script.

## Sub-agents

- `sec-005a-tool-url-validator` — wraps `validate_public_url` with tool
  defaults (https, no redirects).
- `sec-005b-redirect-hop-validator` — custom `RedirectTransport` that re-checks
  each hop.
- `sec-005c-tool-migration` — script to dry-run the new validator against
  existing rows; logs which would be rejected.

## Dependencies

- `utils/safe_http.py` already provides `validate_public_url` — reuse.

## Verification

```python
# tests/test_tool_ssrf.py
@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/admin",
    "http://10.0.0.1",
    "http://169.254.169.254/latest/meta-data/",  # AWS metadata
    "http://[::1]/",
    "http://metadata.google.internal/",
    "http://user:pass@evil.example.com/",
    "ftp://example.com/",
])
def test_tool_create_rejects_internal(url):
    r = client.post("/tools", json={"name": "x", "url": url, "method": "POST"})
    assert r.status_code in (400, 422)

def test_tool_create_accepts_public_https():
    r = client.post("/tools", json={"name": "x", "url": "https://example.com/hook", "method": "POST"})
    assert r.status_code == 200

def test_executor_follows_redirect_only_if_public():
    # mock redirect to 127.0.0.1; expect refusal
    ...
```

## Acceptance

- All loopback, RFC1918, link-local, metadata, and userinfo URLs are rejected
  at create/update and at request time.
- Existing `validate_public_url` is the single source of truth — tool path
  uses no other validator.
- Existing tools audited; migration script logs any that would fail and blocks
  enforcement until resolved.
