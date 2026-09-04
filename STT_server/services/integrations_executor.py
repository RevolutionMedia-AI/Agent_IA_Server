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
import uuid
from datetime import datetime, timedelta, timezone as _tz
from typing import Any, Callable, Dict, Tuple
from zoneinfo import ZoneInfo

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


# ponytail: shared timezone math for the Google executors. The
# 2026-09-08 incident: the LLM (and the operator's manual input)
# sends ``"2026-09-08T14:00:00"`` with no offset — that's
# "2 PM in whatever timezone the integration is configured for",
# not "2 PM UTC". Google combines the bare dateTime + the
# ``timeZone`` field, and we were passing ``14:00Z`` (UTC), so
# Google converted 14:00 UTC → 07:00 -07:00 in America/Tijuana.
# Fix: parse with ``ZoneInfo(timezone)`` when the string is naive,
# otherwise honor whatever offset the caller already encoded. Format
# the resulting aware datetime as RFC 3339 with the integration's
# offset so the wire shape Google sees is unambiguous.
def _parse_local(iso_or_dt: str, tz_name: str) -> datetime:
    """Parse an ISO 8601 string honoring the integration's timezone.

    * tz-naive string ("2026-09-08T14:00:00") → attach the
      integration's tz, never assume UTC.
    * tz-aware string with offset ("2026-09-08T14:00:00-07:00")
      → leave the offset alone (caller already specified a wall-clock
      moment in some concrete zone).
    * trailing "Z" → UTC, then converted to the integration's tz
      so the resulting datetime matches the operator's calendar.
    """
    raw = (iso_or_dt or "").strip()
    if not raw:
        raise ValueError("datetime is empty")
    try:
        zone = ZoneInfo(tz_name) if tz_name else ZoneInfo("UTC")
    except Exception:
        # ponytail: the integration row's timezone is operator
        # config; a typo must not crash the call. Fall back to UTC
        # and let Google apply its own conversion. The operator will
        # see the off-by-N-hours in the result and fix the row.
        zone = ZoneInfo("UTC")
    # datetime.fromisoformat handles "Z" in Python 3.11+. Older 3.x
    # needs the explicit "+00:00" rewrite. We always do the rewrite
    # so the same code runs on 3.10 + 3.13.
    normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"datetime '{raw}' is not a valid ISO 8601 string") from exc
    if dt.tzinfo is None:
        # ponytail: CRITICAL — naive strings are interpreted in the
        # integration's local timezone, not UTC. This is the bug
        # the operator reported on 2026-09-04: a 2 PM Tijuana
        # booking was being created at 7 AM because the BE assumed
        # UTC and let Google's tz conversion eat 7 hours.
        return dt.replace(tzinfo=zone)
    return dt.astimezone(zone)


def _format_google_datetime(dt: datetime) -> str:
    """RFC 3339 with offset, e.g. ``2026-09-08T14:00:00-07:00``.

    Google accepts both ``...-0700`` and ``...-07:00``; we emit the
    colon-separated form (RFC 3339 strict) so the wire shape matches
    the integration's wall-clock and the operator's mental model.
    No accidental UTC normalization — the offset already carries the
    intent.
    """
    raw = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    if len(raw) >= 5 and raw[-5] in "+-" and raw[-3] != ":":
        # Insert the colon between hours and minutes of the offset.
        return raw[:-2] + ":" + raw[-2:]
    return raw


# Legacy helper kept for callers that want a plain z-naive UTC
# marker. The Google executors below no longer use it; the calendar
# timezone is authoritative now.
def _google_iso_to_z(iso_or_dt: str) -> str:  # noqa: D401
    """DEPRECATED: do not use for Google ``dateTime`` fields. Naive
    strings were previously appended with ``Z`` (treating them as
    UTC), which produced 7-hour-off bookings on Tijuana operators.
    Use ``_parse_local`` + ``_format_google_datetime`` instead."""
    if not iso_or_dt:
        return ""
    raw = iso_or_dt.strip()
    if raw.endswith("Z"):
        return raw
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
    cfg = integration_row.get("configuration") or {}
    calendar_id = cfg.get("calendar_id")
    if not calendar_id:
        return False, None, "calendar_id is not configured on this integration"
    timezone = cfg.get("timezone") or "UTC"

    raw_dt = arguments.get("datetime")
    duration = int(arguments.get("duration_minutes") or 30)
    if not raw_dt:
        return False, None, "datetime is required"

    # ponytail: feed Google a [start, end) window in the integration's
    # timezone, NOT in UTC. The 2026-09-04 incident proved the naive
    # approach loses 7 hours on Tijuana. We parse the caller string
    # honoring the integration's timezone, then emit RFC 3339 with
    # the correct offset so Google doesn't apply its own conversion.
    try:
        start_dt = _parse_local(raw_dt, timezone)
    except ValueError as exc:
        return False, None, str(exc)
    end_dt = start_dt + timedelta(minutes=duration)
    body = {
        "timeMin": _format_google_datetime(start_dt),
        "timeMax": _format_google_datetime(end_dt),
        "items": [{"id": calendar_id}],
        "timeZone": timezone,
    }
    url = f"{_GOOGLE_API_BASE}/calendar/v3/freeBusy"
    try:
        resp = _google_http("POST", url, credentials["access_token"], body=body)
    except RuntimeError as exc:
        return False, None, str(exc)

    busy = (resp.get("calendars") or {}).get(calendar_id, {}).get("busy") or []
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
    timezone = cfg.get("timezone") or "UTC"

    name = (arguments.get("name") or "").strip()
    email = (arguments.get("email") or "").strip()
    raw_dt = arguments.get("datetime")
    duration = int(arguments.get("duration_minutes") or 30)
    notes = (arguments.get("notes") or "").strip()

    if not name or not email or not raw_dt:
        return False, None, "name, email and datetime are required"

    # ponytail: parse the caller's datetime honoring the integration's
    # timezone (not UTC). The 2026-09-04 incident shipped
    # 14:00Z + America/Tijuana and Google converted 14:00 UTC → 7:00
    # -07:00. With the fix we send RFC 3339 + offset, so the offset
    # already carries the operator's wall-clock intent.
    try:
        start_dt = _parse_local(raw_dt, timezone)
    except ValueError as exc:
        return False, None, str(exc)
    end_dt = start_dt + timedelta(minutes=duration)

    # ponytail: freebusy guard. We refuse the insert when the slot is
    # already taken — the FE can then suggest a nearby time. Without
    # this guard two concurrent calls would silently double-book.
    fb_body = {
        "timeMin": _format_google_datetime(start_dt),
        "timeMax": _format_google_datetime(end_dt),
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

    # ponytail: meet request. The 2026-09-04 incident had ``meet_link: null``
    # because the request was missing ``conferenceDataVersion=1`` and
    # used a non-unique requestId. We now always include the version
    # flag (so Google actually creates the conference) and a unique
    # requestId (so retries don't dedupe against the prior attempt).
    event_body: Dict[str, Any] = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": _format_google_datetime(start_dt),
            "timeZone": timezone,
        },
        "end": {
            "dateTime": _format_google_datetime(end_dt),
            "timeZone": timezone,
        },
        "attendees": [{"email": email, "displayName": name}],
        "conferenceData": {
            "createRequest": {
                # ponytail: requestId must be globally unique per
                # Google; collisions throw ``CONFERENCE_REQUEST_ALREADY_EXISTS``
                # on the second create. We mint it from a uuid4 hex
                # so two calls booked in the same second don't clash.
                "requestId": f"revolutionmedia-{uuid.uuid4().hex}",
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
    url = (
        f"{_GOOGLE_API_BASE}/calendar/v3/calendars/{cal_path}"
        f"/events?conferenceDataVersion=1&sendUpdates=all"
    )
    try:
        event = _google_http("POST", url, credentials["access_token"], body=event_body)
    except RuntimeError as exc:
        return False, None, str(exc)

    # ponytail: 2026-09-04 incident — Google sometimes returns the
    # conferenceData block with status=``pending`` (the Meet URL is
    # being generated async). We surface that to the operator via a
    # log + a `meet_pending: true` flag so the FE can decide to
    # retry. Hangout link is the canonical field; entryPoints is the
    # fallback when Google returns the structured conferenceData
    # without ``hangoutLink`` (Workspace + newer API responses).
    meet_link = event.get("hangoutLink")
    meet_status = None
    conference = event.get("conferenceData") or {}
    if not meet_link:
        for entry in conference.get("entryPoints") or []:
            if entry.get("entryPointType") == "video":
                meet_link = entry.get("uri")
                break
    create_req = conference.get("createRequest") or {}
    if isinstance(create_req, dict):
        meet_status = (create_req.get("status") or {}).get("statusCode")
    if not meet_link:
        log.info(
            "[google.create] conferenceData pending integration_id=%s status=%s raw_keys=%s",
            integration_row.get("id"),
            meet_status,
            sorted(conference.keys()) if conference else [],
        )

    return True, {
        "id": event.get("id"),
        "htmlLink": event.get("htmlLink"),
        "meet_link": meet_link,
        "meet_pending": meet_status == "pending",
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
