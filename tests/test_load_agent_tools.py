"""Unit tests for the shared/private tool filtering in
STT_server.services.session_runtime._load_agent_tools.

Contract (010_agent_tools.sql + db_tools.list_tools):

  * a private tool (agent_id = the agent) ships only to that agent;
  * a shared tool (agent_id = "__shared__") ships only to the agents
    listed in its `assignments` array — the operator assigns it
    explicitly from the agent modal;
  * tools owned by another user are never visible.

The "every agent of the same user" rule of the legacy JSON store is
GONE: a shared row without an explicit assignment reaches nobody.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from STT_server.services.session_runtime import _load_agent_tools


SAMPLE_TOOLS = [
    # Alice's private tool — only her agent can invoke it
    {"id": "t-priv-a", "agent_id": "agent-a", "user_id": "alice",
     "name": "private_a", "description": "x", "webhook_url": "https://x"},
    # Alice's shared tool — assigned to agent-a AND agent-other-a
    {"id": "t-shared-a", "agent_id": "__shared__", "user_id": "alice",
     "assignments": ["agent-a", "agent-other-a"],
     "name": "shared_a", "description": "x", "webhook_url": "https://x"},
    # Alice's shared tool with NO assignment — reachable by nobody
    {"id": "t-shared-unassigned-a", "agent_id": "__shared__", "user_id": "alice",
     "assignments": [],
     "name": "shared_unassigned_a", "description": "x", "webhook_url": "https://x"},
    # Bob's shared tool — must NEVER show for alice
    {"id": "t-shared-b", "agent_id": "__shared__", "user_id": "bob",
     "assignments": ["agent-b"],
     "name": "shared_b", "description": "x", "webhook_url": "https://x"},
    # Bob's private tool on a different agent
    {"id": "t-priv-b", "agent_id": "agent-b", "user_id": "bob",
     "name": "private_b", "description": "x", "webhook_url": "https://x"},
]


@pytest.fixture
def seed_tools(data_dir: Path) -> None:
    """Write the sample toolset to the tmp file the store reads from.

    The `data_dir` fixture already redirects every JSON store
    (STT_server/db_tools.py included) to the isolated dir, so writing
    agent_tools.json there is enough — no monkeypatching of module
    constants needed.
    """
    (data_dir / "agent_tools.json").write_text(json.dumps(SAMPLE_TOOLS), encoding="utf-8")


def test_private_tool_only_for_owning_agent(seed_tools: None) -> None:
    tools = _load_agent_tools("agent-a", "alice")
    ids = {t["id"] for t in tools}
    assert "t-priv-a" in ids, "alice's private tool must load for her agent"
    assert "t-shared-a" in ids, "alice's shared tool must load for her assigned agent"
    assert "t-priv-b" not in ids, "bob's private tool must NOT load for alice"
    assert "t-shared-b" not in ids, "bob's shared tool must NOT load for alice"


def test_shared_tool_visible_to_every_assigned_agent(seed_tools: None) -> None:
    # Alice owns two agents; BOTH are assigned the shared tool.
    for agent in ("agent-a", "agent-other-a"):
        tools = _load_agent_tools(agent, "alice")
        ids = {t["id"] for t in tools}
        assert "t-shared-a" in ids, f"shared tool must load for assigned {agent}"
        assert "t-shared-b" not in ids, f"other user's shared tool must NOT load for {agent}"


def test_unassigned_shared_tool_reaches_nobody(seed_tools: None) -> None:
    # A shared row with an empty `assignments` array is not offered to
    # any agent — the operator has to assign it explicitly.
    for agent in ("agent-a", "agent-other-a"):
        ids = {t["id"] for t in _load_agent_tools(agent, "alice")}
        assert "t-shared-unassigned-a" not in ids, (
            f"unassigned shared tool must NOT load for {agent}"
        )


def test_other_users_shared_tool_is_hidden(seed_tools: None) -> None:
    # Even Bob's own private agent must not see alice's shared tool.
    tools = _load_agent_tools("agent-b", "bob")
    ids = {t["id"] for t in tools}
    assert "t-priv-b" in ids
    assert "t-shared-b" in ids
    assert "t-shared-a" not in ids, "alice's shared tool must NOT leak to bob"
    assert "t-priv-a" not in ids, "alice's private tool must NOT leak to bob"


def test_missing_agent_id_returns_empty() -> None:
    assert _load_agent_tools(None, "alice") == []
    assert _load_agent_tools("", "alice") == []


def test_missing_file_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("STT_server.db_tools._AGENT_TOOLS_FILE",
                        tmp_path / "does-not-exist.json", raising=False)
    assert _load_agent_tools("agent-a", "alice") == []
