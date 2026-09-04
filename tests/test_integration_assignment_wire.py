"""Regression tests for the integration assignment wire shape.

Three production incidents live here:
  1. private integrations (agent_id != '__shared__') must expose an
     `assigned_agents` list that includes their owner. The previous
     wire shape only carried `assignments`, so the FE rendered
     "No agents assigned. Pick from below." even though the integration
     already belonged to the agent.
  2. shared integrations must expose `assigned_agents` mirroring the
     JSONB `assignments` array (BE normalizes this so the FE never has
     to).
  3. credentials + OAuth-internal fields are still stripped.
"""
from __future__ import annotations

import importlib

import pytest


def _import_api():
    api = importlib.import_module("STT_server.routes.api")
    return importlib.reload(api)


def _integration_row(**overrides):
    base = {
        "id": "integ-1",
        "user_id": "user-1",
        "agent_id": "agent-A",
        "name": "ACME CRM",
        "provider": "salesforce",
        "configuration": {},
        "credentials_encrypted": b"\x00encrypted_bytes",
        "credentials_cipher": "fernet-v1",
        "connection_status": "connected",
        "oauth_state_hash": "secret",
        "oauth_state_expires_at": None,
        "assignments": [],
        "created_at": "2026-09-01T00:00:00Z",
        "updated_at": "2026-09-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_strip_wire_exposes_assigned_agents_for_private_integration():
    """Private integration (agent_id='agent-A') should be marked as
    assigned to its owner on the wire even though `assignments` is
    empty (private rows don't use the JSONB array)."""
    api = _import_api()
    out = api._strip_integration_for_wire(_integration_row())
    assert out["agent_id"] == "agent-A"
    assert out["assigned_agents"] == ["agent-A"], (
        "Private integrations expose their owner through "
        "`assigned_agents` so the FE doesn't render an empty Assign list."
    )
    assert "credentials_encrypted" not in out
    assert "credentials_cipher" not in out
    assert "oauth_state_hash" not in out
    assert "oauth_state_expires_at" not in out


def test_strip_wire_exposes_assigned_agents_for_shared_integration():
    api = _import_api()
    out = api._strip_integration_for_wire(_integration_row(
        agent_id="__shared__",
        assignments=["agent-X", "agent-Y"],
    ))
    assert out["assigned_agents"] == ["agent-X", "agent-Y"]
    assert out["agent_id"] == "__shared__"


def test_strip_wire_preserves_existing_assigned_agents_field():
    """If the caller already stamped `assigned_agents` (e.g. via a
    convenience lookup), don't clobber it."""
    api = _import_api()
    row = _integration_row()
    row["assigned_agents"] = ["agent-Preserved"]
    out = api._strip_integration_for_wire(row)
    assert out["assigned_agents"] == ["agent-Preserved"]
