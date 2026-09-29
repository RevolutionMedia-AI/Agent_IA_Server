"""The agents.language validator must agree with the DDL's CHECK.

This exists because it did not agree, and the only environment that could
have told us was production:

    psycopg2.errors.CheckViolation: new row for relation "agents"
    violates check constraint "agents_language_check"

A previous revision made every write emit the canonical codes 'en' / 'es',
because tenants.preferred_language and call_sessions.preferred_language are
code enums. agents.language is not — 001_schema.sql declares it as a display
enum:

    language TEXT NOT NULL DEFAULT 'English'
              CHECK (language IN ('English', 'Spanish', 'Bilingual')),

So a perfectly valid-looking PUT /agents/{id} 500s the moment it reaches
Postgres. The local suite cannot catch that class of bug: db_agents falls
back to a JSON file whenever DATABASE_URL is unset, and the JSON path has no
constraints at all.

So parse the DDL and assert that every value the validator can possibly emit
is inside the CHECK. The two files cannot drift again without this failing.
"""
import re
from pathlib import Path

import pytest

from STT_server.routes.api import (
    _AGENT_LANGUAGE_ALIASES,
    _canonical_agent_language,
    AgentCreate,
    AgentUpdate,
)

SCHEMA = Path(__file__).resolve().parents[1] / "db" / "migrations" / "001_schema.sql"


def _allowed_values_from_ddl() -> set:
    """Pull the literal set out of `CHECK (language IN (...))`."""
    text = SCHEMA.read_text(encoding="utf-8", errors="replace")
    # Only the agents table's column, not tenants' preferred_language.
    agents_block = text.split("CREATE TABLE agents")[1].split("CREATE TABLE")[0]
    m = re.search(
        r"language\s+TEXT\s+NOT NULL\s+DEFAULT\s+'([^']+)'\s*"
        r"CHECK\s*\(\s*language\s+IN\s*\(([^)]*)\)",
        agents_block,
    )
    assert m, "agents.language CHECK not found in 001_schema.sql"
    return {v.strip().strip("'") for v in m.group(2).split(",") if v.strip()}


ALLOWED = _allowed_values_from_ddl()


def test_the_ddl_check_was_actually_parsed():
    """If the regex ever stops matching, every other test here would pass
    vacuously against an empty set."""
    assert ALLOWED == {"English", "Spanish", "Bilingual"}


def test_every_alias_maps_into_the_check():
    """The exact assertion that was missing."""
    for alias in _AGENT_LANGUAGE_ALIASES:
        assert _canonical_agent_language(alias) in ALLOWED, alias


def test_every_writable_path_emits_an_allowed_value():
    """Both schemas, across the whole alias surface, plus the default."""
    for alias in list(_AGENT_LANGUAGE_ALIASES) + [None, "", "  "]:
        assert AgentCreate(name="a", language=alias).language in ALLOWED | {None}
        assert AgentUpdate(language=alias).language in ALLOWED | {None}


def test_codes_are_accepted_on_input_and_stored_as_words():
    """The FE sends 'en' / 'es'; the column stores the word."""
    assert AgentCreate(name="a", language="es").language == "Spanish"
    assert AgentUpdate(language="en").language == "English"


def test_the_default_is_an_allowed_value():
    assert AgentCreate(name="a").language in ALLOWED


def test_codes_are_not_what_the_column_wants():
    """The regression, pinned. The codes are right everywhere else in the
    schema and wrong here, so assert the code itself is never stored."""
    assert "en" not in ALLOWED and "es" not in ALLOWED
    # a code in, a word out — never the code
    assert _canonical_agent_language("es") == "Spanish"
    assert _canonical_agent_language("en") == "English"
    assert _canonical_agent_language("es") != "es"
    assert _canonical_agent_language("en") != "en"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
