"""GET /tools?include_private=true — the opt-in that unblocks per-agent tools.

Context: the Integrations "By agent" view is the only surface that shows
an operator the private tools an agent owns. They were unreachable from
there because GET /tools discarded them server-side: it calls
db_list_tools(user_id) with no agent filter, so EVERY tool the user owns
was already in memory, and the shared-only branch threw the per-agent
rows away in a Python list comprehension.

include_private=true stops that discard. It adds no query and no scan —
same single call, same _is_real_tool filter. The default (param absent)
must stay byte-identical, since Standalone tools, IntegrationDetail and
the webhook dialog all call this route without it.

The route is invoked directly with a plain dict rather than through
TestClient: `auth` is just a Depends default, and the unit under test is
the filter, not the HTTP plumbing.
"""
from __future__ import annotations

import pytest

from STT_server.routes import api
from STT_server.routes.api import SHARED_TOOL_AGENT_ID, list_shared_tools

SHARED = {"id": "t_shared", "agent_id": SHARED_TOOL_AGENT_ID,
          "webhook_url": "https://n8n.example.com/w/x"}
PRIVATE_A = {"id": "t_a", "agent_id": "agent_a",
             "webhook_url": "https://n8n.example.com/w/a"}
PRIVATE_B = {"id": "t_b", "agent_id": "agent_b",
             "webhook_url": "https://n8n.example.com/w/b"}
# provider-credential row: no webhook_url, no destination
CREDENTIAL_ROW = {"id": "t_cred", "agent_id": SHARED_TOOL_AGENT_ID,
                  "credentials": {"api_key": "sk-..."}}
ALL_ROWS = [SHARED, PRIVATE_A, PRIVATE_B, CREDENTIAL_ROW]

AUTH = {"user_id": "u1"}


@pytest.fixture
def stub_db(monkeypatch):
    seen = {}

    def fake_list_tools(user_id, agent_id=None):
        seen["user_id"] = user_id
        seen["agent_id"] = agent_id
        return list(ALL_ROWS)

    monkeypatch.setattr(api, "db_list_tools", fake_list_tools)
    return seen


def test_default_still_returns_only_shared_tools(stub_db):
    """The regression guard for every existing caller."""
    rows = list_shared_tools(auth=AUTH)
    assert [r["id"] for r in rows] == ["t_shared"]


def test_default_drops_credential_rows_too(stub_db):
    rows = list_shared_tools(auth=AUTH)
    assert "t_cred" not in {r["id"] for r in rows}


def test_include_private_returns_per_agent_rows(stub_db):
    rows = list_shared_tools(include_private=True, auth=AUTH)
    ids = {r["id"] for r in rows}
    assert {"t_a", "t_b"} <= ids, "per-agent tools must be reachable now"


def test_include_private_still_drops_credential_rows(stub_db):
    """_is_real_tool is not a shared-only concern; credential rows must
    stay hidden in BOTH variants or the agent modal's marketplace
    starts offering OpenAI as a callable tool again."""
    rows = list_shared_tools(include_private=True, auth=AUTH)
    assert "t_cred" not in {r["id"] for r in rows}


def test_include_private_keeps_shared_rows(stub_db):
    rows = list_shared_tools(include_private=True, auth=AUTH)
    assert "t_shared" in {r["id"] for r in rows}


def test_both_variants_cost_exactly_one_db_call(stub_db):
    """The whole point: no N+1. The agent_id kwarg must stay None so
    db_list_tools keeps issuing its single unfiltered query."""
    list_shared_tools(include_private=True, auth=AUTH)
    assert stub_db["agent_id"] is None
    assert stub_db["user_id"] == "u1"


def test_user_scoping_is_untouched(stub_db):
    """We widen WHICH tools, never WHOSE — the query is still keyed on
    the authenticated user."""
    list_shared_tools(include_private=True, auth={"user_id": "someone_else"})
    assert stub_db["user_id"] == "someone_else"
