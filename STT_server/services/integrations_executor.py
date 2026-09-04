"""Server-side action executors for OAuth integrations.

The /internal/integrations/{id}/execute endpoint calls into this
module. The dispatch table maps (provider, action) to a callable that
takes the integration row + its already-decrypted credentials + the
agent-supplied arguments and returns a JSON-safe response body.

Today only Google Calendar ships actions (check_availability +
create_appointment). The dispatcher is shaped so adding a new
provider is a one-line registry entry — see ``_REGISTRY``.

Both Google actions use stdlib ``urllib`` (no SDK dependency): the
BE already lives behind an LTS Python and pinning ``google-api-python-
client`` would balloon the install footprint. Google endpoints
considered here:
  * GET  https://www.googleapis.com/calendar/v3/freeBusy
  * POST https://www.googleapis.com/calendar/v3/calendars/{id}/events
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Tuple

log = logging.getLogger("stt_server.integrations_executor")


# ponytail: signatures must stay uniform so the dispatcher never has
# to special-case arity. Every executor receives the bare minimum it
# needs and returns a dict that's already JSON-safe (no bytes, no
# datetime objects).
ExecutorFn = Callable[[dict, dict, dict], Tuple[bool, dict | None, str | None]]
#     (integration_row, credentials_plain, arguments) →
#     (success, data, error_message)


# ── Google Calendar ──────────────────────────────────────────────────────

_GOOGLE_API_BASE = "https://www.googleapis.com"


def _google_iso_to_z(iso_or_dt: str) -> str:
    """Normalize a Google `dateTime` field. We accept either an ISO
    string (the LLM sometimes hands us a naive one) and coerce to the
    `YYYY-MM-DDTHH:MM:SSZ` shape that freeBusy + events expect. We
    trust operator's timezone configuration on the integration row
    rather than inferring it."""
    if not iso_or_dt:
        return ""
    raw = iso_or_dt.strip()
    if raw.endswith("Z"):
        return raw
    # ponytail: keep it simple — append the trailing Z so the BE
    # doesn't have to walk pytz just to book one appointment. The
    # integration row's `configuration.timezone` is authoritative;
    # when the LLM sends a tz-naive string we treat it as UTC
    # (timezone is set on the calendar event itself below).
    if "+" in raw[10:] or raw.count("-") > 2:
        return raw
    return raw + "Z"


def _google_http(
    method: str,
    url: str,
    access_token: str,
    body: dict | None = None,
    timeout: float = 8.0,
) -> dict:
    """One-call Google API helper.

    Returns the parsed JSON body on a 2xx response. Raises
    ``RuntimeError`` on a 4xx/5xx or a network error so the route
    layer can map it to a structured ``{success: false, error}``.
    """
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Accept", "application/json")
    payload = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    try:
        with urllib.request.urlopen(req, data=payload, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"google {method} {url} → HTTP {exc.code}: {body}"
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(f"google {method} {url} → network: {exc.reason}")
    if not raw:
        return {}
    return json.loads(raw)


def _google_check_availability(
    integration_row: dict,
    credentials: dict,
    arguments: dict,
) -> Tuple[bool, dict | None, str | None]:
    """Implements Google Calendar ``freebusy.query``.

    The FE/agent passes ``{"datetime": "ISO", "duration_minutes": int}``.
    ``calendar_id`` + ``timezone`` come from the integration row's
    configuration; ``access_token`` from the decrypted credentials.
    """
    calendar_id = (integration_row.get("configuration") or {}).get("calendar_id")
    if not calendar_id:
        return False, None, "calendar_id is not configured on this integration"

    raw_dt = arguments.get("datetime")
    duration = int(arguments.get("duration_minutes") or 30)
    if not raw_dt:
        return False, None, "datetime is required"

    # ponytail: feed Google a [start, end) window. We anchor at the
    # requested moment and expand by the call duration. If the
    # caller passes 30 min and 14:00, we probe 14:00 → 14:30.
    start_iso = _google_iso_to_z(raw_dt)
    from datetime import datetime, timedelta, timezone as _tz
    try:
        # ponytail: Google returns plain `dateTime` (no timezone) when
        # we hit the API with an offset-naive string. We re-derive an
        # offset-aware datetime so ``+ timedelta`` doesn't crash.
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except ValueError:
        return False, None, f"datetime '{raw_dt}' is not a valid ISO 8601 string"

    end_iso = (dt + timedelta(minutes=duration)).astimezone(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg_tz = (integration_row.get("configuration") or {}).get("timezone", "UTC")
    body = {
        "timeMin": start_iso,
        "timeMax": end_iso,
        "items": [{"id": calendar_id}],
        "timeZone": cfg_tz,
    }
    url = f"{_GOOGLE_API_BASE}/calendar/v3/freeBusy"
    try:
        resp = _google_http("POST", url, credentials["access_token"], body=body)
    except RuntimeError as exc:
        return False, None, str(exc)

    busy = (resp.get("calendars") or {}).get(calendar_id, {}).get("busy") or []
    # ponytail: `busy` is a list of {start, end} windows. Surface the
    # exact start so the agent can say "I'm booked at 14:00" rather
    # than a generic "you have conflicts".
    conflicts = [
        {
            "start": window.get("start"),
            "end": window.get("end"),
        }
        for window in busy
    ]
    return True, {
        "available": not busy,
        "conflicts": len(conflicts),
        "busy": conflicts,
    }, None


def _google_create_appointment(
    integration_row: dict,
    credentials: dict,
    arguments: dict,
) -> Tuple[bool, dict | None, str | None]:
    """Implements Google Calendar ``events.insert``.

    The FE/agent passes ``{name, email, datetime, duration_minutes,
    notes}``. We always re-run the freebusy probe inside this
    executor so a double-booking returns ``{success: false,
    reason: "slot_taken"}`` instead of an event with overlapping
    attendees. Google Meet (conferenceData) is requested when the
    operator's account supports it — Google Workspace responds with
    a hangout link; consumer accounts no-op (the field stays null).
    """
    cfg = integration_row.get("configuration") or {}
    calendar_id = cfg.get("calendar_id")
    if not calendar_id:
        return False, None, "calendar_id is not configured on this integration"

    name = (arguments.get("name") or "").strip()
    email = (arguments.get("email") or "").strip()
    raw_dt = arguments.get("datetime")
    duration = int(arguments.get("duration_minutes") or 30)
    notes = (arguments.get("notes") or "").strip()
    timezone = cfg.get("timezone", "UTC")

    if not name or not email or not raw_dt:
        return False, None, "name, email and datetime are required"

    start_iso = _google_iso_to_z(raw_dt)
    from datetime import datetime, timedelta, timezone as _tz
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except ValueError:
        return False, None, f"datetime '{raw_dt}' is not a valid ISO 8601 string"
    end_iso = (dt + timedelta(minutes=duration)).astimezone(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ponytail: freebusy guard. We refuse the insert when the slot is
    # already taken — the FE can then suggest a nearby time. Without
    # this guard two concurrent calls would silently double-book.
    fb_body = {
        "timeMin": start_iso,
        "timeMax": end_iso,
        "items": [{"id": calendar_id}],
        "timeZone": timezone,
    }
    try:
        fb = _google_http(
            "POST",
            f"{_GOOGLE_API_BASE}/calendar/v3/freeBusy",
            credentials["access_token"],
            body=fb_body,
        )
    except RuntimeError as exc:
        return False, None, f"freebusy check failed: {exc}"
    busy = (fb.get("calendars") or {}).get(calendar_id, {}).get("busy") or []
    if busy:
        return False, {
            "available": False,
            "conflicts": len(busy),
            "busy": [
                {"start": w.get("start"), "end": w.get("end")}
                for w in busy
            ],
        }, "slot_taken"

    # ponytail: email validation is permissive on purpose. RFC 5322
    # is way too strict for the kinds of addresses agents will see in
    # a phone call. We just require ``something@something.something``.
    if "@" not in email or "." not in email.split("@", 1)[1]:
        return False, None, f"email '{email}' is not a valid address"

    summary = f"{name} ({email})"
    description_parts = []
    if email:
        description_parts.append(f"Attendee: {email}")
    if notes:
        description_parts.append(notes)
    description = "\n\n".join(description_parts) if description_parts else ""

    event_body: Dict[str, Any] = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": timezone},
        "end": {"dateTime": end_iso, "timeZone": timezone},
        "attendees": [{"email": email, "displayName": name}],
        # ponytail: ask Google to attach a Meet link. Works for
        # Workspace + education accounts; consumer accounts leave
        # conferenceData empty and we surface the htmlLink instead.
        "conferenceData": {
            "createRequest": {
                "requestId": f"revolutionmedia-{int(dt.timestamp())}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
        "conferenceDataVersion": 1,
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 60},
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    cal_path = urllib.parse.quote(calendar_id, safe="")
    url = f"{_GOOGLE_API_BASE}/calendar/v3/calendars/{cal_path}/events?sendUpdates=all"
    try:
        event = _google_http("POST", url, credentials["access_token"], body=event_body)
    except RuntimeError as exc:
        return False, None, str(exc)

    hangout = (
        (event.get("conferenceData") or {}).get("entryPoints") or [{}]
    )
    meet_link = next(
        (ep.get("uri") for ep in hangout if ep.get("entryPointType") == "video"),
        None,
    )
    return True, {
        "id": event.get("id"),
        "htmlLink": event.get("htmlLink"),
        "meet_link": meet_link,
        "start": (event.get("start") or {}).get("dateTime"),
        "end": (event.get("end") or {}).get("dateTime"),
        "summary": event.get("summary"),
        "status": event.get("status"),
        "calendar_id": calendar_id,
    }, None


# ── Dispatcher ────────────────────────────────────────────────────────────

_REGISTRY: Dict[Tuple[str, str], ExecutorFn] = {
    ("google_calendar", "check_availability"): _google_check_availability,
    ("google_calendar", "create_appointment"): _google_create_appointment,
}


def supported_actions(provider: str) -> list[str]:
    """Public helper for the FE: enumerate the action ids the
    dispatcher knows for this provider. Used by the integrations page
    so the operator can pre-test the actions n8n will call."""
    return sorted(
        action for (prov, action), _ in _REGISTRY.items() if prov == provider
    )


def execute_action(
    provider: str,
    action: str,
    integration_row: dict,
    credentials: dict,
    arguments: dict,
) -> Tuple[bool, dict | None, str | None]:
    """Single entry point. Resolves the executor from ``_REGISTRY``,
    runs it, returns ``(success, data, error_message)``.

    Unknown (provider, action) tuples land here too — caller wires
    them to a 422 / structured error. Keeping this dispatcher in the
    one place means new providers ship without a route change.
    """
    fn = _REGISTRY.get((provider, action))
    if fn is None:
        return False, None, f"unsupported action '{action}' for provider '{provider}'"
    return fn(integration_row, credentials, arguments or {})
