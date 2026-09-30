"""An integration-bound tool must survive a restart and be callable.

Production incident this pins: `nullify_stale_tool_integration_pointers`
ran `UPDATE agent_tools SET integration_id = NULL WHERE integration_id IS
NOT NULL` on every boot, and both dispatchers read the URL off the TOOL
row instead of the integration. A generic_webhook keeps its URL in
integrations.configuration.webhook_url, so the combination meant a bound
tool could never fire: the call raised "missing webhook_url" and the
agent told the caller it could not save their data. The Test Connection
button stayed green because it reads the URL off the integration row.

Two invariants, one per bug:
  1. the pointer is load-bearing and must not be wiped
  2. the dispatchers resolve the endpoint through the integration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeResp:
    """httpx-shaped response. httpx uses `status_code`, not `status`."""

    def __init__(self, status_code: int = 200, body: bytes = b'{"ok":true}'):
        self.status_code = status_code
        self._body = body
        self.headers: dict = {}
        self.content = body

    def read(self, limit: int = -1) -> bytes:
        return self._body if limit < 0 else self._body[:limit]

    def json(self):
        import json as _json
        return _json.loads(self._body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── 1. the pointer survives ───────────────────────────────────────────────

def test_the_wipe_backfill_is_gone() -> None:
    """The destructive UPDATE must not be reachable again.

    Not "returns 0" — the function must not exist. A no-op version would
    be one refactor away from wiping the column again.
    """
    import STT_server.db_integrations as db_int

    assert not hasattr(db_int, "nullify_stale_tool_integration_pointers"), (
        "the backfill that nulls agent_tools.integration_id is back; "
        "it silently kills every integration-bound tool on the next boot"
    )


def test_nothing_in_the_lifespan_wipes_the_pointer() -> None:
    """Guard the call site, not just the definition.

    The boot log read "[backfill] nullified stale integration_id on 1
    row(s)" — a single wiped row silently killed the automation.
    Comments naming the function are fine (and wanted: the removal
    needs explaining); an import or a call is not.
    """
    server_src = (
        Path(__file__).resolve().parents[1] / "STT_server" / "STT_Server.py"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in server_src.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "nullify_stale_tool_integration_pointers" not in code, (
        "lifespan still invokes the pointer-wiping backfill"
    )


# ── 2. the dispatcher resolves through the integration ───────────────────

def _allow_https(monkeypatch) -> None:
    """Neutralise the SSRF guard for a fake host.

    The guard resolves the hostname and rejects private/unresolvable
    ones. That is correct behaviour we do not want to weaken in the
    test — so stub the resolution, not the check.
    """
    from STT_server.services import tool_executor as te

    monkeypatch.setattr(te, "_resolve_host_ips", lambda host: ["93.184.216.34"])


@pytest.mark.asyncio
async def test_bound_tool_resolves_its_url_from_the_integration(monkeypatch) -> None:
    """The regression, end to end.

    The tool row has NO webhook_url (that is where a generic_webhook
    keeps it), the integration does. The call must reach the webhook
    instead of bailing on "missing webhook_url".
    """
    import httpx

    from STT_server.services import tool_executor as te

    sent: list[dict] = {}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, **kw):
            sent["method"] = method
            sent["url"] = url
            sent["json"] = kw.get("json")
            return _FakeResp(200)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(te, "get_tool_executor", lambda: te.ToolExecutor())
    _allow_https(monkeypatch)

    import STT_server.db_integrations as db_int
    integration = {
        "id": "int-1",
        "user_id": "u1",
        "provider": "generic_webhook",
        "configuration": {
            "webhook_url": "https://script.google.com/macros/x/exec",
            "webhook_method": "POST",
        },
        "credentials_encrypted": None,
    }
    monkeypatch.setattr(db_int, "get_integration", lambda iid, uid: integration)

    tool = {
        "id": "tool-1",
        "name": "guardar datos",
        "function_name": "guardar_datos",
        "kind": "webhook",
        # The whole point: no webhook_url on the row.
        "webhook_url": "",
        "integration_id": "int-1",
        "action": "insertar",
    }

    result = await te.execute_tool_call(
        tool=tool, user_id="u1", llm_arguments={"full_name": "Juan"},
    )

    body = sent["json"]
    # The regression: the endpoint came from the integration, and the
    # envelope matches what a live call sends.
    assert sent["url"] == "https://script.google.com/macros/x/exec"
    assert body["tool_name"] == "guardar_datos"
    assert body["arguments"] == {"full_name": "Juan"}
    assert body["action"] == "insertar"
    assert body["integration_id"] == "int-1"
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_dangling_integration_fails_loudly_not_silently(monkeypatch) -> None:
    """A deleted integration with a bound tool must be a clear error.

    With the 409 gate restored, this is the belt-and-braces path: the
    operator gets "missing or revoked" naming the integration instead
    of a generic webhook failure with a candidate waiting.
    """
    from STT_server.services import tool_executor as te
    import STT_server.db_integrations as db_int

    monkeypatch.setattr(db_int, "get_integration", lambda iid, uid: None)

    with pytest.raises(te.ToolExecutionError) as exc:
        await te.execute_tool_call(
            tool={
                "id": "tool-1", "name": "t", "function_name": "t",
                "kind": "webhook", "webhook_url": "", "integration_id": "gone",
            },
            user_id="u1",
            llm_arguments={},
        )
    assert "gone" in str(exc.value)


@pytest.mark.asyncio
async def test_legacy_tool_without_integration_still_works(monkeypatch) -> None:
    """The pre-integration path must not regress.

    A tool carrying its own webhook_url and no integration_id is the
    shape every pre-refactor row has.
    """
    import httpx

    from STT_server.services import tool_executor as te

    sent: dict = {}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, **kw):
            sent["url"] = url
            return _FakeResp(200)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(te, "get_tool_executor", lambda: te.ToolExecutor())
    _allow_https(monkeypatch)

    result = await te.execute_tool_call(
        tool={
            "id": "tool-2", "name": "legacy", "function_name": "legacy",
            "kind": "webhook",
            "webhook_url": "https://n8n.example/webhook/legacy",
            "integration_id": None, "action": None,
        },
        user_id="u1",
        llm_arguments={"a": 1},
    )
    assert sent["url"] == "https://n8n.example/webhook/legacy"
    assert result == {"ok": True}


def test_both_dispatchers_call_the_integration_aware_executor() -> None:
    """Guard the call sites.

    Both used to read `tool_def["webhook_url"]` and bail when empty.
    Regression guard for exactly that string, in both files.
    """
    root = Path(__file__).resolve().parents[1] / "STT_server"
    for rel in ("adapters/openai_realtime.py", "services/turn_manager.py"):
        src = (root / rel).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "execute_tool_call" in code, f"{rel} no longer uses the executor"
        assert "has no webhook_url" not in code, (
            f"{rel} still bails on an empty tool-row webhook_url; "
            "an integration-bound tool has none by design"
        )
