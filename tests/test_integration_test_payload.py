"""Test Connection must rehearse the real request, not ping with `{}`.

Three behaviours pinned here:

1. A generic_webhook with a bound tool POSTs the tool's real arguments,
   not an empty object. A script reading e.postData.contents.name never
   saw a field before this.
2. A 401/403 is a FAILURE. Reporting it as "webhook reachable" marked the
   integration connected and then failed on the first live call, which is
   how a Google Apps Script deployed as "Only myself" hid for days.
3. 404/405 still count as reachable — wrong verb is not a broken link.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT_server.services.agent_prompt_tools import (  # noqa: E402
    build_integration_section,
    integration_actions_for_prompt,
)
from STT_server.services.integrations_tester import (  # noqa: E402
    _test_webhook_reachable,
    _wants_request_body,
    run_integration_test,
)

CONFIG = {"webhook_url": "https://script.google.com/a/macros/x/exec", "webhook_method": "POST"}


class _FakeHTTPError(urllib.error.HTTPError):
    """HTTPError with a fixed code, raisable from the patched urlopen."""

    def __init__(self, code: int):
        super().__init__("https://x", code, "err", {}, None)


def _patched(monkeypatch, code=None, ok_code=None):
    """Replace urlopen. `code` raises an HTTPError; `ok_code` returns 200."""
    sent: list[dict] = []

    def fake_urlopen(req, timeout=None):
        sent.append({
            "method": req.get_method(),
            "data": req.data,
            "full_url": req.full_url,
        })
        if code is not None:
            raise _FakeHTTPError(code)
        return _FakeResp(ok_code or 200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


class _FakeResp:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── 1. the body is the tool's real arguments ────────────────────────────────

def test_posts_the_real_body_not_an_empty_object(monkeypatch) -> None:
    sent = _patched(monkeypatch, ok_code=200)
    body = {"tool_name": "Submit candidate", "arguments": {"candidate_name": "John Doe"}}
    valid, msg = _test_webhook_reachable(CONFIG, {}, body)
    assert valid is True
    assert sent, "no request was made"
    assert sent[0]["data"] == json.dumps(body).encode("utf-8")
    # The regression: an empty object proved nothing about the script.
    assert sent[0]["data"] != b"{}"


def test_falls_back_to_empty_object_when_no_body(monkeypatch) -> None:
    sent = _patched(monkeypatch, ok_code=200)
    valid, _msg = _test_webhook_reachable(CONFIG, {}, None)
    assert valid is True
    assert sent[0]["data"] == b"{}"


def test_dispatcher_forwards_the_body_only_where_it_is_declared(monkeypatch) -> None:
    sent = _patched(monkeypatch, ok_code=200)
    body = {"tool_name": "t", "arguments": {"a": 1}}
    ok, _ = run_integration_test(
        "_test_webhook_reachable", CONFIG, {}, request_body=body,
    )
    assert ok is True
    assert sent[0]["data"] == json.dumps(body).encode("utf-8")
    # A two-argument provider must not be handed a third argument.
    assert _wants_request_body(_test_webhook_reachable) is True
    assert _wants_request_body(lambda c, cr: (True, "x")) is False


# ── 2. auth failures are failures, not "reachable" ──────────────────────────

def test_401_is_a_failure_with_an_actionable_message(monkeypatch) -> None:
    _patched(monkeypatch, code=401)
    valid, msg = _test_webhook_reachable(CONFIG, {}, {"a": 1})
    assert valid is False, "a 401 must not be reported as connected"
    assert "401" in msg
    # The message has to tell the operator what to DO, not just that it broke.
    assert "Anyone" in msg


def test_403_is_also_a_failure(monkeypatch) -> None:
    _patched(monkeypatch, code=403)
    valid, msg = _test_webhook_reachable(CONFIG, {}, None)
    assert valid is False
    assert "403" in msg


# ── 3. wrong verb still counts as reachable ─────────────────────────────────

def test_404_still_counts_as_reachable(monkeypatch) -> None:
    _patched(monkeypatch, code=404)
    valid, msg = _test_webhook_reachable(CONFIG, {}, None)
    assert valid is True
    assert "reachable" in msg


def test_405_still_counts_as_reachable(monkeypatch) -> None:
    _patched(monkeypatch, code=405)
    valid, _msg = _test_webhook_reachable(CONFIG, {}, None)
    assert valid is True


def test_5xx_is_not_swallowed(monkeypatch) -> None:
    # 5xx is a server-side fault, not "wrong verb". The loop exhausts and
    # reports unreachable rather than pretending the host is fine.
    _patched(monkeypatch, code=503)
    valid, msg = _test_webhook_reachable(CONFIG, {}, None)
    assert valid is False
    assert msg == "unreachable"


def test_missing_url_is_a_failure(monkeypatch) -> None:
    valid, msg = _test_webhook_reachable({"webhook_method": "POST"}, {}, None)
    assert valid is False
    assert "webhook_url" in msg


# ── 4. a webhook integration's prompt block carries the tool's JSON ────────

def test_webhook_integration_prompt_renders_the_tool_schema() -> None:
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
    )

    spec = get_integration_provider_spec("generic_webhook")
    # The catalog is empty BY DESIGN for this provider.
    assert len(spec.actions) == 0

    tools = [{
        "id": "tool-1",
        "action": "submit_candidate",
        "name": "Submit candidate",
        "description": "Registers a candidate in the sheet.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string"},
                "resume_link": {"type": "string"},
            },
            "required": ["candidate_name"],
        },
    }]
    actions = integration_actions_for_prompt(
        {"id": "int-1", "user_id": "u1"}, spec, bound_tools=tools,
    )
    assert [a["id"] for a in actions] == ["submit_candidate"]

    block = build_integration_section("int-1", "Job Portal", actions)
    # The regression: the block said "no actions configured yet" while a
    # tool with a full schema was bound to the same integration.
    assert "no actions configured" not in block
    assert "Submit candidate" in block
    assert "candidate_name" in block
    assert "```json" in block
    assert "Required: candidate_name" in block


def test_official_provider_still_uses_the_catalog() -> None:
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
    )

    spec = get_integration_provider_spec("google_calendar")
    if not spec or not spec.actions:
        return  # catalog not loaded in this environment
    actions = integration_actions_for_prompt(
        {"id": "i", "user_id": "u"}, spec, bound_tools=[],
    )
    # Catalog wins; the tool fallback must not hijack a provider that
    # already declares its own verbs.
    assert [a["id"] for a in actions] == [a.id for a in spec.actions]


def test_no_tools_and_no_catalog_actions_is_still_empty() -> None:
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
    )

    spec = get_integration_provider_spec("generic_webhook")
    actions = integration_actions_for_prompt(
        {"id": "int-1", "user_id": "u1"}, spec, bound_tools=[],
    )
    assert actions == []
    block = build_integration_section("int-1", "Empty", actions)
    assert "no actions configured" in block


def test_blank_tools_are_skipped_not_rendered_as_action() -> None:
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
    )

    spec = get_integration_provider_spec("generic_webhook")
    actions = integration_actions_for_prompt(
        {"id": "i", "user_id": "u"}, spec,
        bound_tools=[{"id": "", "action": "", "name": "  ", "parameters": {}}],
    )
    assert actions == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name} ...", end=" ", flush=True)
            # pytest supplies the fixture; the smoke runner does not, so
            # skip the ones that need a patched urlopen.
            if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                print("(needs pytest)")
                continue
            fn()
            print("ok")
    print("\nrun with: python -m pytest tests/test_integration_test_payload.py -q")
