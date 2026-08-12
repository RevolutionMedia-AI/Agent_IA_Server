# sec-009-log-redaction

**Severity:** MEDIUM — Sensitive logging.

> Status: baseline partially addressed in commit `f325245` (Realtime path
> default metadata-only). Outstanding branches still emit content.

## Scope (files)

- `STT_server/adapters/openai_realtime.py` — verify all emit sites.
- `STT_server/services/turn_manager.py` — 555-567, 692, 754-764, 816-824.
- `STT_server/services/tool_executor.py` — 107-112, 132-136.
- `STT_server/services/turn_manager.py` — 374-395.

## Approach (SAFE_CHANGE)

1. Define a single `safe_log(event, **fields)` helper that whitelists fields
   and rejects anything that smells like content (transcript, reply, tool
   args/result, password hash, secret-bearing URL).
2. Replace every remaining `logger.info/warning(...content...)` with
   `safe_log(...)` carrying metadata only.
3. Promote `LOG_TRANSCRIPT_CONTENT` to the diagnostic mode it should be:
   time-bounded, debug-only, off by default, never on in production.
4. Centralize redaction; document the policy in `docs/logging.md`.

## Sub-agents

- `sec-009a-safe-log-helper` — module under `STT_server/utils/`.
- `sec-009b-call-site-sweep` — find/replace every unsafe log emit.
- `sec-009c-retention-policy-doc` — `docs/logging.md` covering PII / retention.

## Dependencies

- None. Pure logging refactor.

## Verification

```python
def test_safe_log_drops_transcript_field():
    rec = safe_log("turn.final", transcript="hello", turn_id="t1")
    assert rec["turn_id"] == "t1"
    assert "transcript" not in rec

def test_tool_url_with_userinfo_is_redacted():
    rec = safe_log("tool.call", url="https://user:pw@x.com/")
    assert "user:pw" not in rec["url"]
```

Static check:

```bash
# No bare transcript/reply/args/result content logging
! grep -RnE 'logger\.(info|warning|error).*(transcript|reply|tool_args|tool_result)' \
    STT_server/ | grep -v '_test.py'
```

## Acceptance

- Single `safe_log` helper used at every call site.
- No raw transcripts / tool args / results in production logs.
- `docs/logging.md` documents redaction + retention policy.
