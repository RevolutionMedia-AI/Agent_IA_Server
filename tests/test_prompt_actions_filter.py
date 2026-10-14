"""The prompt block lists only the actions the operator turned into tools.

The bug this pins: integration_actions_for_prompt took the WHOLE catalog
for an official provider. Assigning a Dynamics 365 integration to an agent
therefore wrote all 47 action blocks into the System Prompt — ~8.8k
tokens — while turn_manager builds the OpenAI tools[] payload from
session.agent_tools and never reads the catalog. The model could invoke 3
tools and was told about 47, so 44 blocks were pure noise and an
invitation to emit a tool call for a function name that isn't in tools[].

The rules:
  1. Tools bound -> intersect the catalog with their `action` values.
  2. No tools bound -> full catalog. Connect-then-create-tools is a
     two-step flow and an empty block on step 1 reads as a bug.
  3. generic_webhook has an empty catalog by design -> the tool rows
     themselves are the action list (unchanged behaviour).
  4. A bound tool whose action is not in the catalog must still render,
     via the tool-shaped fallback, or the model is told a tool it can
     actually call does not exist.
"""
from __future__ import annotations

import pytest

from STT_server.services.agent_prompt_tools import (
    build_integration_section,
    integration_actions_for_prompt,
)
from STT_server.services.integrations_catalog import (
    INTEGRATION_PROVIDERS as PROVIDERS,
)

DYNAMICS = next(p for p in PROVIDERS if p.id == "dynamics365")
GENERIC = next(p for p in PROVIDERS if p.id == "generic_webhook")


def tool(action: str, name: str = "", **kw):
    return {
        "id": f"t-{action}",
        "action": action,
        "name": name or action.replace("_", " ").title(),
        "description": kw.get("description", "desc"),
        "parameters": kw.get("parameters", {"type": "object", "properties": {}}),
        "integration_id": "int-1",
    }


# ── the intersection ────────────────────────────────────────────────────

def test_three_tools_render_three_actions():
    acts = integration_actions_for_prompt(
        {}, DYNAMICS,
        bound_tools=[tool("find_customer"), tool("get_cases"), tool("create_case")],
    )
    assert {a["id"] for a in acts} == {"find_customer", "get_cases", "create_case"}


def test_order_follows_the_catalog_not_the_tool_list():
    """Stable, meaningful order beats insertion order: re-ordering the
    tools must not churn the stored System Prompt."""
    forward = integration_actions_for_prompt(
        {}, DYNAMICS, bound_tools=[tool("find_customer"), tool("create_case")],
    )
    backward = integration_actions_for_prompt(
        {}, DYNAMICS, bound_tools=[tool("create_case"), tool("find_customer")],
    )
    assert [a["id"] for a in forward] == [a["id"] for a in backward]
    assert [a["id"] for a in forward] == ["find_customer", "create_case"]


def test_no_tools_renders_full_catalog():
    """Step 1 of the flow: connect, before any tool exists."""
    acts = integration_actions_for_prompt({}, DYNAMICS, bound_tools=[])
    assert len(acts) == 47


def test_catalog_copy_wins_over_tool_row():
    """The tool row's description must not override the catalog's."""
    acts = integration_actions_for_prompt(
        {}, DYNAMICS,
        bound_tools=[tool("find_customer", description="texto del operador")],
    )
    assert len(acts) == 1
    assert acts[0]["description"] == "Search contacts and accounts."


def test_bound_tool_missing_from_catalog_still_renders():
    """A renamed action must not vanish while it is still in tools[]."""
    acts = integration_actions_for_prompt(
        {}, DYNAMICS,
        bound_tools=[tool("accion_renombrada", name="Renombrada")],
    )
    assert len(acts) == 1
    assert acts[0]["id"] == "accion_renombrada"
    assert acts[0]["when_to_use_es"] == ""


def test_mixed_known_and_unknown_keeps_both():
    acts = integration_actions_for_prompt(
        {}, DYNAMICS,
        bound_tools=[tool("get_cases"), tool("fantasma")],
    )
    ids = sorted(a["id"] for a in acts)
    assert ids == ["fantasma", "get_cases"]


def test_blank_action_does_not_widen_the_filter():
    """A tool with no usable `action` must not count as a filter key, or a
    blank would widen the block back to all 47. The tool itself still
    renders under its own name — it is bound, so the model can call it."""
    acts = integration_actions_for_prompt(
        {}, DYNAMICS,
        bound_tools=[tool("get_cases"), {"action": "  ", "name": "X"}],
    )
    ids = [a["id"] for a in acts]
    assert len(acts) == 2, f"the blank must not widen the catalog, got {len(acts)}"
    assert "get_cases" in ids and "X" in ids


# ── generic_webhook is unchanged ────────────────────────────────────────

def test_generic_webhook_uses_tool_rows():
    acts = integration_actions_for_prompt(
        {}, GENERIC,
        bound_tools=[tool("mi_accion", name="Mi Accion")],
    )
    assert len(acts) == 1
    assert acts[0]["id"] == "mi_accion"


def test_generic_webhook_empty_stays_empty():
    assert integration_actions_for_prompt({}, GENERIC, bound_tools=[]) == []


# ── the thing that motivated it ─────────────────────────────────────────

def test_three_tools_do_not_shrink_to_the_generic_fallback():
    """Guards the trap: a narrow intersection must not trip the
    generic-webhook branch, which renders the TOOL ROW's schema instead
    of the catalog's. get_cases is in the catalog, so its catalog schema
    has to be the one that lands in the block."""
    catalog_schema = next(
        a.parameters_schema
        for a in DYNAMICS.actions if a.id == "get_cases"
    )
    acts = integration_actions_for_prompt(
        {}, DYNAMICS,
        bound_tools=[tool("get_cases", parameters={"type": "object", "properties": {"DEL FALLBACK": {}}})],
    )
    assert acts[0]["parameters_schema"] == catalog_schema
    assert "DEL FALLBACK" not in acts[0]["parameters_schema"].get("properties", {})


def test_prompt_block_shrinks_roughly_proportionally():
    """3 tools must cost about 3 action blocks, not 47."""
    full = build_integration_section(
        "int-1", "Microsoft Dynamics 365",
        integration_actions_for_prompt({}, DYNAMICS, bound_tools=[]),
    )
    narrow = build_integration_section(
        "int-1", "Microsoft Dynamics 365",
        integration_actions_for_prompt(
            {}, DYNAMICS, bound_tools=[tool("get_cases")],
        ),
    )
    assert len(narrow) < len(full) / 10, (
        f"expected a big shrink, got {len(full)} -> {len(narrow)}"
    )
    assert len(narrow) < 2_000, f"one action should stay small, got {len(narrow)}"