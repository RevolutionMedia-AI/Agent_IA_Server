"""Pre-AI transfer cascade: pure-function contract tests.

No app imports, no JSON-file writes, no fixtures — this file is
order-independent by construction (it never touches the shared
STT_server/data/*.json backends that made the legacy endpoint
tests flaky).
"""
from STT_server.services.transfer_cascade import (
    parse_cascade,
    validate_cascade,
    dial_twiml,
    connect_stream_twiml,
    cascade_action_url,
)


def test_validate_accepts_good_steps_with_defaults():
    steps, err = validate_cascade([
        {"destination": "+15550001111", "timeout_sec": 25},
        {"destination": "+15550002222"},
    ])
    assert err is None
    assert steps == [
        {"destination": "+15550001111", "timeout_sec": 25},
        {"destination": "+15550002222", "timeout_sec": 20},
    ]


def test_validate_rejects_bad_destination():
    steps, err = validate_cascade([{"destination": "12345"}])
    assert steps is None
    assert "E.164" in err


def test_validate_rejects_bad_timeout_and_too_many_steps():
    _, err = validate_cascade([{"destination": "+15550001111", "timeout_sec": 999}])
    assert "between" in err
    _, err = validate_cascade([{"destination": "+15550001111"}] * 6)
    assert "at most 5" in err


def test_parse_is_lenient_for_voice_path():
    # /voice must never 500 on a bad row — worst case straight to AI.
    assert parse_cascade(None) == []
    assert parse_cascade("not-json") == []
    assert parse_cascade([{"destination": "bad"}]) == []
    assert parse_cascade([{"destination": "+15550001111", "timeout_sec": 999}]) == [
        {"destination": "+15550001111", "timeout_sec": 60}
    ]


def test_dial_twiml_carries_timeout_and_action():
    xml = dial_twiml("+15550001111", 25, "https://x.test/voice/cascade?step=1")
    assert '<Dial timeout="25"' in xml
    assert "+15550001111" in xml
    assert "action=" in xml


def test_cascade_action_url_is_stateless():
    url = cascade_action_url("https://x.test/", "agent-1", 2, tenant_id="t-9")
    assert url.startswith("https://x.test/voice/cascade?")
    assert "agent_id=agent-1" in url and "step=2" in url and "tenant_id=t-9" in url


def test_connect_twiml_matches_legacy_shape():
    xml = connect_stream_twiml("wss://x.test", '<Parameter name="agent_id" value="a1" />')
    assert "<Stream" in xml and "media-stream" in xml and "agent_id" in xml
