"""Live test functions for each integration provider.

The IntegrationsProviderSpec.test_fn points at one of these (dotted
path). The runner calls them with `(configuration, credentials)` —
both already validated + cleaned by integrations_catalog — and
expects `(valid: bool, message: str)`.

V1 ships:
  * _test_zendesk         — real, hits /api/v2/users/me.json
  * _test_webhook_reachable — real, HEAD/GET on the generic_webhook URL
  * everything else       — stubs that return (False, "Test not yet
    implemented for {provider}")

The reason most are stubs: the user wants the FE to render the
"Configure" form so the operator can stash their credentials now, but
not validate against providers we haven't yet exercised. Adding a real
test_fn is a one-line change once someone has tested the OAuth flow
for that provider.
"""
from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("stt_server.services.integrations_tester")

# How much of a 2xx body to read for the success/false verdict. Enough
# for a JSON error object, small enough that a chatty endpoint cannot
# make the test button pull megabytes.
_BODY_INSPECT_BYTES = 8192


# ponytail: same credential sanitization used by credentials_resolver.
# If the test function raises (timeout, bad JSON, auth wall) we don't
# want the raw stack trace bubbling back to the FE — a short friendly
# message is enough for the operator to know whether to retry.
def _sanitize_error(msg: str, limit: int = 300) -> str:
    s = (msg or "").strip().replace("\n", " ").replace("\r", " ")
    return s[:limit]


def _stub(provider_id: str) -> tuple[bool, str]:
    return False, f"Test not yet implemented for {provider_id}"


def _wants_request_body(fn) -> bool:
    """True when `fn` declared a `request_body` parameter.

    ponytail: only _test_webhook_reachable has one. Rather than adding a
    third parameter to all eight providers (seven of which would ignore
    it and one of which is a stub), the dispatcher checks the signature.
    A test that does not exist in a coverage report is not a test, and a
    try/except TypeError here would swallow a genuine TypeError raised
    inside the function.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "request_body" in params:
        return True
    # A **kwargs catch-all can take it too.
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _test_zendesk(configuration: dict, credentials: dict) -> tuple[bool, str]:
    """Hit Zendesk's /api/v2/users/me.json — auth-protected, free."""
    subdomain = (configuration.get("subdomain") or "").strip()
    email = (credentials.get("email") or "").strip()
    api_token = (credentials.get("api_token") or "").strip()
    if not subdomain or not email or not api_token:
        return False, "missing subdomain, email, or api_token"
    # Zendesk's HTTP Basic is: "{email}/token:{api_token}"
    import base64
    user_pass = f"{email}/token:{api_token}".encode("utf-8")
    auth = base64.b64encode(user_pass).decode("ascii")
    url = f"https://{subdomain}.zendesk.com/api/v2/users/me.json"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = 200 <= resp.status < 300
            return ok, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))


def _verdict_from_body(method: str, status: int, raw: bytes) -> tuple[bool, str] | None:
    """Second opinion on a 2xx: does the endpoint say the WORK succeeded?

    HTTP 200 only proves the request was delivered and the handler ran.
    A Google Apps Script `doPost` that finds its required fields missing
    returns `{"success": false, "missing_fields": [...]}` and STILL
    answers 200, because ContentService always does. Reporting that as
    connected is the worst kind of false green: the operator sees a pass
    and no row was ever written.

    Conventions accepted, so this stays generic and does not assume n8n:
      {"success": false} / {"ok": false}  → failure, message forwarded
      {"success": true}  / {"ok": true}   → confirmed success
      anything else, or a non-JSON body   → no opinion, the status code
                                            is the only contract available

    `missing_fields` is quoted when present because "which fields?" is
    the one thing the operator needs to act, and it is the whole answer
    for a schema/argument mismatch.
    """
    if not raw:
        return None
    import json as _json
    try:
        parsed = _json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, TypeError):
        # Plain text, an HTML page, an empty 204 body. No business-level
        # verdict available — do not invent one.
        return None
    if not isinstance(parsed, dict):
        return None

    failed = parsed.get("success") is False or parsed.get("ok") is False
    if not failed:
        return None

    detail = str(
        parsed.get("message") or parsed.get("error") or ""
    ).strip()
    if isinstance(parsed.get("error"), dict):
        detail = str(parsed["error"].get("message") or parsed["error"]).strip()
    missing = parsed.get("missing_fields") or parsed.get("missing") or []
    if isinstance(missing, (list, tuple)) and missing:
        detail = (detail + " — missing: " + ", ".join(str(m) for m in missing)).strip()
    elif not detail:
        detail = "the endpoint reported success: false"
    return False, (
        f"{method} {status} but the endpoint reported failure — {detail}"
    )


def _test_webhook_reachable(
    configuration: dict,
    credentials: dict,
    request_body: dict | None = None,
) -> tuple[bool, str]:
    """Test the generic_webhook URL. Times out at 10s.

    ponytail: n8n/Make webhooks are often POST-only and return 404 on HEAD/GET.
    Respects configuration.webhook_method (GET/POST/PUT/PATCH/DELETE); defaults
    to POST. Any HTTP response (including 404/405) proves DNS/TLS works.
    Only network errors/timeouts are failures.

    `request_body` is what the caller wants on the wire. When the
    integration has a tool bound, the route passes the real
    {tool_name, arguments} envelope generated from that tool's parameter
    schema, so the test exercises the same shape a live call sends. An
    empty `{}` proved nothing: a script that reads
    e.postData.contents.name never saw a field.
    """
    url = (configuration.get("webhook_url") or "").strip()
    if not url:
        return False, "missing webhook_url"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme}' not allowed"
    # Respect configured method; fallback to POST for legacy rows.
    configured = (configuration.get("webhook_method") or "POST").strip().upper()
    if configured not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
        configured = "POST"
    import json as _json
    body_bytes = _json.dumps(request_body or {}).encode("utf-8")
    # Probe order: configured method first, then fallbacks for POST-only webhooks.
    probe_order = [configured]
    for m in ("HEAD", "GET", "POST"):
        if m not in probe_order:
            probe_order.append(m)
    for method in probe_order:
        try:
            kwargs = {"method": method}
            if method in ("POST", "PUT", "PATCH"):
                kwargs["data"] = body_bytes
                kwargs["headers"] = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, **kwargs)
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = 200 <= resp.status < 400
                if not ok:
                    return True, f"{method} {resp.status} — webhook reachable"
                # 2xx is not the last word. A handler that rejected the
                # payload at the business level still answers 200.
                if method in ("POST", "PUT", "PATCH"):
                    raw = resp.read(_BODY_INSPECT_BYTES)
                    verdict = _verdict_from_body(method, resp.status, raw)
                    if verdict is not None:
                        return verdict
                return True, f"{method} {resp.status}"
        except urllib.error.HTTPError as exc:
            # A 404/405 means "wrong verb, host is fine" — reachable. A
            # 401/403 means the endpoint refused us outright, which is
            # exactly how an Apps Script deployed as "Only myself" answers
            # an anonymous POST. Reporting that as "connected" marked the
            # integration green and then failed on the first live call, so
            # it is a failure with a message that names the cause.
            if exc.code in (401, 403):
                return False, (
                    f"HTTP {exc.code} — the endpoint rejected the request. "
                    "If this is a Google Apps Script, redeploy it with "
                    "access set to 'Anyone' (Deploy → Manage deployments → "
                    "Edit → Who has access). The URL changes on redeploy."
                )
            if 400 <= exc.code < 500:
                return True, f"HTTP {exc.code} — webhook reachable ({configured} expected)"
            continue
        except Exception as exc:
            return False, _sanitize_error(str(exc))
    return False, "unreachable"


def _test_google_calendar(configuration: dict, credentials: dict) -> tuple[bool, str]:
    """Live test against Google's APIs using the OAuth credentials the
    operator stored on the integration row.

    Three signals we want to see:
      1. The access_token authenticates against Google's userinfo
         endpoint. Without a valid token the operator's credentials
         are stale or the OAuth scope is wrong.
      2. configuration.calendar_id is set. The n8n workflow picks
         the host calendar from this field — empty value means every
         event creation will fail.
      3. configuration.timezone is set (IANA tz, e.g. America/Tijuana).
         Google API requires timezone-aware datetimes; without it
         every event is created in UTC and the operator's calendar
         shows the wrong wall-clock time.

    Returns (False, ...) if any check fails so the FE surfaces a
    clear "Reconnect" or "Set calendar_id + timezone" CTA.
    """
    access_token = (credentials.get("access_token") or "").strip()
    if not access_token:
        return False, "Connect Google Calendar first to test the connection"
    calendar_id = (configuration.get("calendar_id") or "").strip()
    if not calendar_id:
        return False, "Set the target calendar_id before running this test"
    timezone = (configuration.get("timezone") or "").strip()
    if not timezone:
        return False, "Set the timezone (e.g. America/Tijuana) before running this test"
    # Hit Google's userinfo endpoint with the stored access token.
    # 200 + email payload proves the token authenticates AND that the
    # user can read its own profile. We don't surface the email to the
    # operator; we just need the round-trip to succeed.
    try:
        req = urllib.request.Request(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if not (200 <= resp.status < 300):
                return False, "No se pudo conectar a Google — inténtalo de nuevo"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, "Sesión expirada con Google — reconecta la integración"
        return False, f"No se pudo conectar a Google (HTTP {exc.code})"
    except Exception as exc:
        return False, _sanitize_error(str(exc))
    return True, (
        f"Conectado correctamente a Google Calendar ({calendar_id}, {timezone})"
    )


# ponytail: Salesforce OAuth test. Hits the instance's REST API with the
# stored access_token. Lightweight, no side-effects, and the endpoint
# exists on every Salesforce org (including sandboxes).
def _test_salesforce(configuration: dict, credentials: dict) -> tuple[bool, str]:
    instance_url = (configuration.get("instance_url") or "").strip().rstrip("/")
    access_token = (credentials.get("access_token") or "").strip()
    if not instance_url or not access_token:
        return False, "Falta reconectar Salesforce para poder probar la conexión"
    for path in ("/services/oauth2/userinfo", "/services/data/v59.0/limits"):
        url = f"{instance_url}{path}"
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = 200 <= resp.status < 300
                # Mensaje natural para usuario final — sin tecnicismos HTTP
                return (True, "Conectado correctamente a Salesforce") if ok else (False, "No se pudo conectar a Salesforce")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, "Sesión expirada con Salesforce — reconecta la integración"
            if exc.code == 404 and path == "/services/oauth2/userinfo":
                continue
            return False, "No se pudo conectar a Salesforce — inténtalo de nuevo"
        except Exception as exc:
            return False, _sanitize_error(str(exc))
    return False, "No se pudo verificar la conexión con Salesforce"


# ponytail: every official provider gets an explicit stub so adding a
# real test later is a one-line change at the matching INTEGRATION_PROVIDERS
# entry, not a code-search hunt.


def _test_dynamics365(configuration: dict, credentials: dict) -> tuple[bool, str]:
    if not configuration.get("environment_url"):
        return False, "Select a Dynamics 365 environment first"
    if not credentials.get("access_token"):
        return False, "Reconnect Microsoft Dynamics 365"
    try:
        from STT_server.services.dynamics365 import _json_request, normalize_environment_url
        environment_url = normalize_environment_url(configuration["environment_url"])
        payload, _ = _json_request(
            "GET",
            f"{environment_url}/api/data/v9.2/WhoAmI",
            credentials["access_token"],
        )
        return bool(payload.get("UserId")), "Connected to Microsoft Dynamics 365"
    except Exception as exc:
        return False, _sanitize_error(str(exc))


def _test_genesys_cloud(configuration: dict, credentials: dict) -> tuple[bool, str]:
    return _stub("genesys_cloud")


def _test_nice_cxone(configuration: dict, credentials: dict) -> tuple[bool, str]:
    return _stub("nice_cxone")


def run_integration_test(
    test_fn_path: str,
    configuration: dict,
    credentials: dict,
    request_body: dict | None = None,
) -> tuple[bool, str]:
    """Resolve `test_fn_path` (dotted, e.g. "_test_zendesk" — caller
    prepends the module) and invoke. Returns (False, "...") if the
    path doesn't resolve — never raises.

    `request_body` is forwarded ONLY to a test_fn that declares it. The
    signature check keeps the eight other providers on their original
    two-argument contract instead of every one of them growing a
    parameter they would ignore.
    """
    if not test_fn_path:
        return False, "Test not yet implemented for this provider"
    # ponytail: dotted path of the form "module.symbol". We get the
    # module from this file (the only place test functions live) so
    # the caller doesn't have to know it.
    fn_name = test_fn_path.rsplit(".", 1)[-1]
    fn = globals().get(fn_name)
    if fn is None or not callable(fn):
        log.warning("[integrations_tester] unknown test_fn=%s", test_fn_path)
        return False, f"Test not yet implemented ({test_fn_path})"
    try:
        if request_body is not None and _wants_request_body(fn):
            return fn(configuration, credentials, request_body)
        return fn(configuration, credentials)
    except Exception as exc:
        log.exception("[integrations_tester] %s raised", test_fn_path)
        return False, _sanitize_error(str(exc))
