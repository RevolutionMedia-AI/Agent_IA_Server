"""Unit tests for STT_server.domain.tool.validate_json_schema.

Covers the three rules the tool creator must obey:
  - the value is an object (dict),
  - if `type == "object"`, the `properties` value is a dict of dicts,
  - each property only uses the allowed field set
    (type, description, enum, properties, items, required).
"""
from __future__ import annotations

import pytest

from STT_server.domain.tool import validate_json_schema


# ── happy path ───────────────────────────────────────────────────────


def test_valid_object_schema_passes() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "user name"},
            "age": {"type": "integer"},
            "status": {"type": "enum", "enum": ["active", "inactive"]},
        },
        "required": ["name"],
    }
    ok, err = validate_json_schema(schema)
    assert ok is True
    assert err is None


def test_empty_object_schema_passes() -> None:
    """An object with no properties is still a valid (empty) schema."""
    ok, err = validate_json_schema({"type": "object", "properties": {}})
    assert ok is True
    assert err is None


# ── rejection cases ──────────────────────────────────────────────────


def test_non_dict_rejected() -> None:
    ok, err = validate_json_schema("not a dict")
    assert ok is False
    assert "object" in err.lower()


def test_list_rejected() -> None:
    ok, err = validate_json_schema([{"type": "object"}])
    assert ok is False
    assert "object" in err.lower()


def test_non_object_properties_rejected() -> None:
    """type=object but properties is a list, not a dict."""
    schema = {"type": "object", "properties": ["a", "b"]}
    ok, err = validate_json_schema(schema)
    assert ok is False
    assert "properties" in err.lower()


def test_property_must_be_object() -> None:
    schema = {"type": "object", "properties": {"x": "string"}}
    ok, err = validate_json_schema(schema)
    assert ok is False
    assert "object" in err.lower()


def test_unknown_field_in_property_rejected() -> None:
    """OpenAI's spec only accepts a small field set per property.
    Anything else must be rejected so a typo doesn't silently break
    downstream tool execution."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minimum": 1},  # 'minimum' is unknown here
        },
    }
    ok, err = validate_json_schema(schema)
    assert ok is False
    assert "unknown" in err.lower() or "minimum" in err.lower()


# ── regression coverage: the practical "bad enum" shape ─────────────


def test_bad_enum_shape_still_validates_object_shape() -> None:
    """An enum is just a field on a property — `validate_json_schema`
    only enforces object-shape, not enum value types. A non-list enum
    is allowed at this layer (the FE rejects it separately)."""
    ok, err = validate_json_schema({
        "type": "object",
        "properties": {"color": {"type": "enum", "enum": "red,blue"}},
    })
    assert ok is True
    assert err is None