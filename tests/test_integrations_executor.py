"""Server-side integration action executor.

The /internal/integrations/{id}/execute endpoint delegates to
services/integrations_executor. Today only Google Calendar ships
executor actions (``check_availability`` + ``create_appointment``).
The dispatcher is shaped so adding a provider is a one-line
registry entry; the tests pin the contract today.
"""
from __future__ import annotations

import importlib
import json
import sys
import urllib.error
from unittest import mock

import pytest


def _import_executor():
    if "STT_server.services.integrations_executor" in sys.modules:
        del sys.modules["STT_server.services.integrations_executor"]
    return importlib.import_module("STT_server.services.integrations_executor")


_GCAL_ROW = {
    "id": "int-google",
    "provider": "google_calendar",
    "configuration": {
        "calendar_id": "cal-team@example.com",
        "timezone": "America/Tijuana",
    },
}
_GCAL_CREDS = {
    "access_token": "TOKEN_X",
    "refresh_token": "REFRESH_X",
    "expires_at": "2099-01-01T00:00:00Z",
}


def _json_response(payload: dict, status: int = 200):
    """urllib mock that returns a usable body for read()/readline()."""
    body = json.dumps(payload).encode("utf-8")

    class _Resp:
        def __init__(self):
            self._body = body

        def read(self, _n: int = -1) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Resp()


def test_check_availability_happy_path():
    exec_mod = _import_executor()

    fake_busy = []
    fake_fb_response = {"calendars": {"cal-team@example.com": {"busy": fake_busy}}}

    with mock.patch.object(
        exec_mod,
        "_google_http",
        return_value=fake_fb_response,
    ) as fake_http:
        ok, data, err = exec_mod._google_check_availability(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={"datetime": "2026-09-08T14:00:00", "duration_minutes": 30},
        )

    assert ok is True
    assert err is None
    assert data["available"] is True
    assert data["conflicts"] == 0
    assert data["busy"] == []
    # ponytail: freebusy.query against the integration's calendar_id,
    # not "primary", so the spec stays aligned with the same source
    # of truth that the FE Configures.
    sent = fake_http.call_args
    assert sent.args[0] == "POST"
    assert sent.args[1] == exec_mod._GOOGLE_API_BASE + "/calendar/v3/freeBusy"
    assert sent.args[2] == "TOKEN_X"
    sent_body = sent.kwargs["body"]
    assert sent_body["items"] == [{"id": "cal-team@example.com"}]
    assert sent_body["timeMin"].startswith("2026-09-08T14:00:00")
    assert sent_body["timeMax"].startswith("2026-09-08T14:30:00")
    assert sent_body["timeZone"] == "America/Tijuana"


def test_check_availability_reports_conflicts():
    exec_mod = _import_executor()
    fake_busy = [
        {"start": "2026-09-08T14:00:00Z", "end": "2026-09-08T14:30:00Z"},
    ]
    fake_fb_response = {
        "calendars": {"cal-team@example.com": {"busy": fake_busy}},
    }
    with mock.patch.object(exec_mod, "_google_http", return_value=fake_fb_response):
        ok, data, err = exec_mod._google_check_availability(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={"datetime": "2026-09-08T14:00:00", "duration_minutes": 30},
        )
    assert ok is True
    assert err is None
    assert data["available"] is False
    assert data["conflicts"] == 1
    assert data["busy"][0]["start"].startswith("2026-09-08T14:00:00")


def test_check_availability_refuses_missing_calendar_id():
    exec_mod = _import_executor()
    ok, data, err = exec_mod._google_check_availability(
        integration_row={"configuration": {}},
        credentials=_GCAL_CREDS,
        arguments={"datetime": "2026-09-08T14:00:00", "duration_minutes": 30},
    )
    assert ok is False
    assert data is None
    assert "calendar_id" in err


def test_check_availability_refuses_bad_datetime():
    exec_mod = _import_executor()
    ok, data, err = exec_mod._google_check_availability(
        integration_row=_GCAL_ROW,
        credentials=_GCAL_CREDS,
        arguments={"datetime": "not-an-iso", "duration_minutes": 30},
    )
    assert ok is False
    assert "not-a" in err or "ISO" in err


def test_create_appointment_happy_path():
    exec_mod = _import_executor()
    fake_event = {
        "id": "evt-1",
        "htmlLink": "https://calendar.google.com/event?eid=evt-1",
        "conferenceData": {
            "entryPoints": [
                {
                    "entryPointType": "video",
                    "uri": "https://meet.google.com/abc-defg-hij",
                }
            ]
        },
        "start": {"dateTime": "2026-09-08T14:00:00Z"},
        "end": {"dateTime": "2026-09-08T14:30:00Z"},
        "summary": "Test (test@example.com)",
        "status": "confirmed",
    }

    with mock.patch.object(
        exec_mod, "_google_http",
        side_effect=[
            # 1) freebusy.query probe
            {"calendars": {"cal-team@example.com": {"busy": []}}},
            # 2) events.insert response
            fake_event,
        ],
    ) as fake_http:
        ok, data, err = exec_mod._google_create_appointment(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={
                "name": "Test",
                "email": "test@example.com",
                "datetime": "2026-09-08T14:00:00",
                "duration_minutes": 30,
                "notes": "hello",
            },
        )

    assert fake_http.call_count == 2
    assert ok is True
    assert err is None
    assert data["id"] == "evt-1"
    assert data["meet_link"] == "https://meet.google.com/abc-defg-hij"
    assert data["htmlLink"].startswith("https://calendar.google.com")
    # ponytail: events.insert asks Google to attach a Meet link.
    # ``sendUpdates=all`` triggers invitations to the attendee.
    assert fake_http.call_args_list[1].args[0] == "POST"
    assert "/calendars/cal-team%40example.com/events" in fake_http.call_args_list[1].args[1]
    assert fake_http.call_args_list[1].kwargs["body"]["conferenceData"]["createRequest"]["conferenceSolutionKey"]["type"] == "hangoutsMeet"
    body = fake_http.call_args_list[1].kwargs["body"]
    # ponytail: sendUpdates=all on the URL query triggers Google to
    # email the attendee; the body itself never carries it.
    assert "cal-team%40example.com/events" in fake_http.call_args_list[1].args[1]
    assert "sendUpdates=all" in fake_http.call_args_list[1].args[1]
    assert body["attendees"] == [{"email": "test@example.com", "displayName": "Test"}]
    assert body["conferenceDataVersion"] == 1


def test_create_appointment_rejects_slots_already_taken():
    exec_mod = _import_executor()
    busy = [{"start": "2026-09-08T14:00:00Z", "end": "2026-09-08T14:30:00Z"}]
    with mock.patch.object(
        exec_mod, "_google_http",
        return_value={"calendars": {"cal-team@example.com": {"busy": busy}}},
    ) as fake_http:
        ok, data, err = exec_mod._google_create_appointment(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={
                "name": "Test",
                "email": "test@example.com",
                "datetime": "2026-09-08T14:00:00",
            },
        )

    # ponytail: freebusy refused → no insert attempt. createAppointment
    # surfaces the conflict so the agent can suggest a different time.
    assert ok is False
    assert err == "slot_taken"
    assert data["available"] is False
    assert data["conflicts"] == 1
    assert fake_http.call_count == 1, "must NOT have called events.insert"


def test_create_appointment_validates_email():
    exec_mod = _import_executor()
    with mock.patch.object(
        exec_mod, "_google_http",
        return_value={"calendars": {"cal-team@example.com": {"busy": []}}},
    ):
        # Missing TLD after the @ — our permissive regex catches it.
        ok, data, err = exec_mod._google_create_appointment(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={
                "name": "Test",
                "email": "kevin@revolutionmedia",
                "datetime": "2026-09-08T14:00:00",
            },
        )
    assert ok is False
    assert "not a valid" in err


def test_dispatcher_rejects_unsupported_action():
    exec_mod = _import_executor()
    ok, data, err = exec_mod.execute_action(
        provider="google_calendar",
        action="nuke_planet",
        integration_row=_GCAL_ROW,
        credentials=_GCAL_CREDS,
        arguments={},
    )
    assert ok is False
    assert "unsupported" in err


def test_supported_actions_lists_google_calendar_only():
    exec_mod = _import_executor()
    actions = exec_mod.supported_actions("google_calendar")
    assert set(actions) == {"check_availability", "create_appointment"}


def test_google_http_propagates_http_errors():
    exec_mod = _import_executor()
    # Build a real HTTPError pointing at a body so the message includes
    # the truncated payload.
    err = urllib.error.HTTPError(
        "https://x/y", 403, "Forbidden", {}, None,
    )

    def _raise(*a, **kw):
        raise err

    err.read = lambda *_a, **_kw: b'{"error":"forbidden"}'
    with mock.patch(
        "urllib.request.urlopen", side_effect=_raise,
    ):
        with pytest.raises(RuntimeError) as info:
            exec_mod._google_http("GET", "https://x/y", "TOKEN")
        # ponytail: the helper includes the response body so the
        # n8n workflow gets a deterministic error message.
        assert "HTTP 403" in str(info.value)
        assert "forbidden" in str(info.value)
