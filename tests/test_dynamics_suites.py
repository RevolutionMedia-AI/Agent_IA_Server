"""Dynamics 365 suite lineup (STT_server/services/integrations_catalog.py).

Dynamics 365 is sold as separate products over ONE tenant, ONE OAuth grant
and ONE Dataverse dataset. Before this change the catalog advertised all 47
actions flat, so picking "create case" meant scrolling past 20 CRM actions
and 10 field-ops ones.

These tests pin the mapping so it cannot rot silently:

  * every action the provider ships carries a suite
  * every suite has at least one action (no empty card in the picker)
  * the lookup has no key for an action that no longer exists -- a renamed
    action must FAIL here rather than quietly drop to suite=""
  * shares_data_with only names real suites

The suite is TAXONOMY. The runtime gate stays where it was: `capability`
plus the explicit _require_field_service() tenant probe. Suite selection
must never be mistaken for an availability check -- Field Service actions
still raise CAPABILITY_NOT_AVAILABLE when the module is not installed on
the tenant, which is asserted separately in the dynamics365 suite.
"""
from __future__ import annotations

import pytest

from STT_server.services.integrations_catalog import (
    DYNAMICS_SUITES,
    _DYNAMICS_SUITE_BY_ACTION,
    get_integration_provider_spec,
    list_integration_providers,
)

EXPECTED_COUNTS = {
    "crm": 21,
    "customer_service": 6,
    "field_service": 16,
    "customer_insights": 4,
}


@pytest.fixture(scope="module")
def spec():
    return get_integration_provider_spec("dynamics365")


@pytest.fixture(scope="module")
def actions(spec):
    return list(spec.actions)


def test_every_action_has_a_suite(actions):
    missing = [a.id for a in actions if not a.suite]
    assert missing == [], f"actions with no suite: {missing}"


def test_suite_counts_match_the_lineup(actions):
    from collections import Counter

    counts = Counter(a.suite for a in actions)
    assert dict(counts) == EXPECTED_COUNTS


def test_no_action_is_orphaned_from_every_suite(actions):
    """21+6+16+4 must reconstruct the whole catalog. A typo'd suite id
    would silently shrink the total."""
    assert sum(EXPECTED_COUNTS.values()) == len(actions) == 47


def test_every_suite_has_at_least_one_action(actions):
    used = {a.suite for a in actions}
    empty = [s.id for s in DYNAMICS_SUITES if s.id not in used]
    assert empty == [], f"picker would show a card with no actions: {empty}"


def test_lookup_has_no_dangling_keys(actions):
    real = {a.id for a in actions}
    dangling = [k for k in _DYNAMICS_SUITE_BY_ACTION if k not in real]
    assert dangling == [], f"mapping names actions that no longer exist: {dangling}"


def test_lookup_covers_every_action(actions):
    real = {a.id for a in actions}
    uncovered = sorted(real - set(_DYNAMICS_SUITE_BY_ACTION))
    assert uncovered == [], f"actions missing from the suite mapping: {uncovered}"


def test_shares_data_with_names_real_suites():
    valid = {s.id for s in DYNAMICS_SUITES}
    for s in DYNAMICS_SUITES:
        bad = [x for x in s.shares_data_with if x not in valid]
        assert bad == [], f"suite {s.id} shares_data_with unknown suite {bad}"
        assert s.id not in s.shares_data_with, f"{s.id} lists itself"


def test_crm_is_the_hub(actions):
    """CRM is the product the others attach records to, so it is the one
    every other suite declares. If that ever stops being true the FE
    "shares the same data" copy is lying."""
    crm = next(s for s in DYNAMICS_SUITES if s.id == "crm")
    others = [s for s in DYNAMICS_SUITES if s.id != "crm"]
    # CRM points at every other suite...
    for o in others:
        assert o.id in crm.shares_data_with, f"{o.id} missing from crm.shares_data_with"
    # ...and every other suite points back at CRM. A one-way edge would
    # make the picker show the note on only one of the two cards.
    for o in others:
        assert "crm" in o.shares_data_with, f"{o.id} does not declare crm"


def test_suite_is_taxonomy_not_the_capability_gate(actions):
    """Field Service actions are suite=field_service but capability=core:
    suite is a UI grouping, capability is the wire-level classification.
    Conflating them would make the FE hide Field Service for the wrong
    reason."""
    fs = [a for a in actions if a.suite == "field_service"]
    assert fs, "expected Field Service actions"
    assert {a.capability for a in fs} == {"core"}
    ci = [a for a in actions if a.suite == "customer_insights"]
    assert {a.capability for a in ci} == {"customer_insights"}


def test_provider_declares_the_suites(spec):
    assert [s.id for s in spec.suites] == [s.id for s in DYNAMICS_SUITES]


def test_single_product_providers_have_no_suites():
    """The FE branches on spec.suites.length, so a provider with none must
    serialize an empty list rather than omitting the key."""
    multi = {"dynamics365"}
    for p in list_integration_providers():
        if p.id in multi:
            assert p.suites, f"{p.id} should declare suites"
        else:
            assert p.suites == (), f"{p.id} should have no suites"


def test_no_other_provider_inherits_a_dynamics_suite():
    """Regression. The suite lookup was first applied globally inside _a(),
    which stamped Dynamics suite ids onto other providers' actions because
    action ids are NOT namespaced across the catalog: Salesforce and
    Zendesk both ship find_customer / create_case / log_call. Zendesk came
    back with suite='crm' and Salesforce with 7 of them. The lookup is now
    scoped to _dynamics365_actions()."""
    leaked = {
        p.id: [a.id for a in p.actions if getattr(a, "suite", "")]
        for p in list_integration_providers()
        if p.id != "dynamics365"
    }
    dirty = {k: v for k, v in leaked.items() if v}
    assert dirty == {}, f"suite leaked onto other providers: {dirty}"


def test_action_id_collisions_exist_so_the_scoped_lookup_stays_necessary():
    """Guard for the guard. If no other provider ever reuses a Dynamics
    action id again, the scoping could be relaxed -- but until then the
    narrow lookup is load-bearing, not superstition."""
    dyn_ids = {a.id for a in get_integration_provider_spec("dynamics365").actions}
    collisions = {}
    for p in list_integration_providers():
        if p.id == "dynamics365":
            continue
        shared = sorted(dyn_ids & {a.id for a in p.actions})
        if shared:
            collisions[p.id] = shared
    assert collisions, "expected at least one cross-provider id collision"
    assert "find_customer" in collisions.get("zendesk", [])


def test_erp_is_absent_not_empty():
    """Finance & Operations is a separate product with its own APIs (sales
    orders, invoices, vendors) and is deliberately NOT shipped yet. It must
    not appear as a selectable card with zero actions."""
    ids = {s.id for s in DYNAMICS_SUITES}
    assert "erp" not in ids
    assert not any("finance" in s.id or s.id == "fo" for s in DYNAMICS_SUITES)