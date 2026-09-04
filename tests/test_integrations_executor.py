"""Server-side integration action executor.

The /internal/integrations/{id}/execute endpoint delegates to
services/integrations_executor. Today only Google Calendar ships
executor actions (``check_availability`` + ``create_appointment``).
The dispatcher is shaped so adding a provider is a one-line
registry entry; the tests pin the contract today.
"""
from __future__ import annotations

import importlib
import sys
import urllib.error
from unittest import mock
from zoneinfo import ZoneInfo

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


# ── Timezone helper unit tests ───────────────────────────────────────────


def test_parse_local_naive_uses_integration_timezone():
    """A naive ``2026-09-08T14:00:00`` against America/Tijuana should
    produce an aware datetime at 14:00 -07:00 — NOT 14:00 UTC.
    """
    exec_mod = _import_executor()
    dt = exec_mod._parse_local("2026-09-08T14:00:00", "America/Tijuana")
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == -7 * 3600
    # ponytail: the wall-clock moment in the integration's timezone
    # is what we care about, NOT the absolute UTC time. 14:00
    # Tijuana == 21:00 UTC. The previous bug treated naive strings
    # as UTC and produced 14:00 UTC == 07:00 -07:00.
    assert dt.astimezone(ZoneInfo("UTC")).hour == 21
    assert dt.astimezone(ZoneInfo("America/Tijuana")).hour == 14


def test_parse_local_offset_preserves_caller_value():
    """A string with an explicit offset must NOT be re-interpreted.
    14:00 -07:00 == 21:00 UTC always."""
    exec_mod = _import_executor()
    dt = exec_mod._parse_local("2026-09-08T14:00:00-07:00", "America/Tijuana")
    assert dt.utcoffset().total_seconds() == -7 * 3600
    assert dt.astimezone(ZoneInfo("UTC")).hour == 21


def test_format_google_datetime_emits_offset_not_z():
    exec_mod = _import_executor()
    dt = exec_mod._parse_local("2026-09-08T14:00:00", "America/Tijuana")
    formatted = exec_mod._format_google_datetime(dt)
    # The 2026-09-04 incident shipped ...Z which Google combined
    # with ``timeZone: America/Tijuana`` to land the event 7 hours
    # off. The new formatter emits ``...-07:00`` so the wall-clock
    # moment is unambiguous.
    assert formatted == "2026-09-08T14:00:00-07:00"
    assert not formatted.endswith("Z")


# ── Action dispatch ────────────────────────────────────────────────────


def test_check_availability_happy_path():
    exec_mod = _import_executor()
    fake_fb_response = {"calendars": {"cal-team@example.com": {"busy": []}}}
    with mock.patch.object(
        exec_mod, "_google_http", return_value=fake_fb_response,
    ) as fake_http:
        ok, data, err = exec_mod._google_check_availability(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={"datetime": "2026-09-08T14:00:00", "duration_minutes": 30},
        )
    assert ok is True
    assert err is None
    assert data == {"available": True, "conflicts": 0, "busy": []}
    sent = fake_http.call_args
    assert sent.args[0] == "POST"
    assert sent.args[1] == exec_mod._GOOGLE_API_BASE + "/calendar/v3/freeBusy"
    sent_body = sent.kwargs["body"]
    assert sent_body["items"] == [{"id": "cal-team@example.com"}]
    # ponytail: 2026-09-04 fix. The body must carry the offset, not
    # a bare "Z" (UTC). Without the offset, Google reinterprets
    # the moment as UTC and the integration's timeZone field eats
    # 7 hours on Tijuana.
    assert sent_body["timeMin"] == "2026-09-08T14:00:00-07:00"
    assert sent_body["timeMax"] == "2026-09-08T14:30:00-07:00"
    assert sent_body["timeZone"] == "America/Tijuana"


def test_check_availability_reports_conflicts():
    exec_mod = _import_executor()
    fake_busy = [
        {"start": "2026-09-08T14:00:00-07:00", "end": "2026-09-08T14:30:00-07:00"},
    ]
    with mock.patch.object(
        exec_mod, "_google_http",
        return_value={"calendars": {"cal-team@example.com": {"busy": fake_busy}}},
    ):
        ok, data, err = exec_mod._google_check_availability(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={"datetime": "2026-09-08T14:00:00", "duration_minutes": 30},
        )
    assert ok is True
    assert data["available"] is False
    assert data["conflicts"] == 1
    assert data["busy"][0]["start"] == "2026-09-08T14:00:00-07:00"


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
    assert err
    # ponytail: the user-facing message must hint at ISO 8601 so the
    # operator can fix their input on the next call.
    assert "ISO" in err or "not-an-iso" in err


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
        "start": {"dateTime": "2026-09-08T14:00:00-07:00"},
        "end": {"dateTime": "2026-09-08T14:30:00-07:00"},
        "summary": "Test (test@example.com)",
        "status": "confirmed",
    }
    with mock.patch.object(
        exec_mod, "_google_http",
        side_effect=[
            {"calendars": {"cal-team@example.com": {"busy": []}}},
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
    assert data["start"] == "2026-09-08T14:00:00-07:00"
    assert data["end"] == "2026-09-08T14:30:00-07:00"
    assert data["meet_pending"] is False

    # ponytail: freebusy + events.insert URLs and bodies. events.insert
    # MUST carry conferenceDataVersion=1 so Google actually creates
    # the Meet (the 2026-09-04 incident was missing this flag and
    # meet_link came back null).
    freebusy = fake_http.call_args_list[0]
    insert = fake_http.call_args_list[1]
    assert freebusy.kwargs["body"]["timeMin"] == "2026-09-08T14:00:00-07:00"
    assert insert.args[1].endswith("?conferenceDataVersion=1&sendUpdates=all")
    assert insert.kwargs["body"]["conferenceDataVersion"] == 1
    body = insert.kwargs["body"]
    assert body["conferenceData"]["createRequest"]["conferenceSolutionKey"]["type"] == "hangoutsMeet"
    # ponytail: requestId is a uuid4 hex, not a timestamp — the
    # previous implementation could collide on two calls in the
    # same second and Google returned CONFERENCE_REQUEST_ALREADY_EXISTS.
    request_id = body["conferenceData"]["createRequest"]["requestId"]
    assert request_id.startswith("revolutionmedia-")
    assert len(request_id.split("-", 1)[1]) == 32  # uuid4 hex
    assert body["attendees"] == [{"email": "test@example.com", "displayName": "Test"}]


def test_create_appointment_falls_back_to_hangoutLink():
    """The 2026-09-04 incident: meet_link was null because Google
    hadn't populated entryPoints yet. We now also read
    ``hangoutLink`` (the canonical field) as a fallback."""
    exec_mod = _import_executor()
    fake_event = {
        "id": "evt-2",
        "htmlLink": "https://calendar.google.com/event?eid=evt-2",
        "hangoutLink": "https://meet.google.com/direct-hangout-link",
        "start": {"dateTime": "2026-09-08T14:00:00-07:00"},
        "end": {"dateTime": "2026-09-08T14:30:00-07:00"},
        "summary": "Test (test@example.com)",
        "status": "confirmed",
    }
    with mock.patch.object(
        exec_mod, "_google_http",
        side_effect=[
            {"calendars": {"cal-team@example.com": {"busy": []}}},
            fake_event,
        ],
    ):
        ok, data, _ = exec_mod._google_create_appointment(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={
                "name": "Test",
                "email": "test@example.com",
                "datetime": "2026-09-08T14:00:00",
            },
        )
    assert ok is True
    assert data["meet_link"] == "https://meet.google.com/direct-hangout-link"


def test_create_appointment_marks_meet_pending_when_status_pending():
    """When Google returns ``conferenceData.createRequest.status.statusCode
    == "pending"`` the Meet URL is being generated async. The
    response surfaces ``meet_pending: true`` so the FE can retry."""
    exec_mod = _import_executor()
    fake_event = {
        "id": "evt-3",
        "htmlLink": "https://calendar.google.com/event?eid=evt-3",
        "conferenceData": {
            "createRequest": {
                "status": {"statusCode": "pending"},
            }
        },
        "start": {"dateTime": "2026-09-08T14:00:00-07:00"},
        "end": {"dateTime": "2026-09-08T14:30:00-07:00"},
        "summary": "Test (test@example.com)",
        "status": "confirmed",
    }
    with mock.patch.object(
        exec_mod, "_google_http",
        side_effect=[
            {"calendars": {"cal-team@example.com": {"busy": []}}},
            fake_event,
        ],
    ):
        ok, data, _ = exec_mod._google_create_appointment(
            integration_row=_GCAL_ROW,
            credentials=_GCAL_CREDS,
            arguments={
                "name": "Test",
                "email": "test@example.com",
                "datetime": "2026-09-08T14:00:00",
            },
        )
    assert ok is True
    assert data["meet_pending"] is True
    assert data["meet_link"] is None


def test_create_appointment_rejects_slots_already_taken():
    exec_mod = _import_executor()
    busy = [{"start": "2026-09-08T14:00:00-07:00", "end": "2026-09-08T14:30:00-07:00"}]
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
        ok, _, err = exec_mod._google_create_appointment(
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
    err = urllib.error.HTTPError("https://x/y", 403, "Forbidden", {}, None)

    def _raise(*a, **kw):
        raise err

    err.read = lambda *_a, **_kw: b'{"error":"forbidden"}'
    with mock.patch("urllib.request.urlopen", side_effect=_raise):
        with pytest.raises(RuntimeError) as info:
            exec_mod._google_http("GET", "https://x/y", "TOKEN")
        assert "HTTP 403" in str(info.value)
        assert "forbidden" in str(info.value)
