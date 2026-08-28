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


# ponytail: same credential sanitization used by credentials_resolver.
# If the test function raises (timeout, bad JSON, auth wall) we don't
# want the raw stack trace bubbling back to the FE — a short friendly
# message is enough for the operator to know whether to retry.
def _sanitize_error(msg: str, limit: int = 300) -> str:
    s = (msg or "").strip().replace("\n", " ").replace("\r", " ")
    return s[:limit]


def _stub(provider_id: str) -> tuple[bool, str]:
    return False, f"Test not yet implemented for {provider_id}"


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


def _test_webhook_reachable(configuration: dict, credentials: dict) -> tuple[bool, str]:
    """HEAD/GET the generic_webhook URL. Times out at 10s.

    ponytail: we use GET with a HEAD fallback because some webhook
    providers (Slack, certain n8n setups) reject HEAD with 405. Any
    2xx/3xx response counts as reachable; 4xx/5xx does not.
    """
    url = (configuration.get("webhook_url") or "").strip()
    if not url:
        return False, "missing webhook_url"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme}' not allowed"
    # Try HEAD first; fall back to GET on 405 (some endpoints reject HEAD).
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = 200 <= resp.status < 400
            return ok, f"HEAD {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    ok = 200 <= resp.status < 400
                    return ok, f"GET {resp.status}"
            except urllib.error.HTTPError as exc2:
                return False, f"HTTP {exc2.code}"
            except Exception as exc2:
                return False, _sanitize_error(str(exc2))
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, _sanitize_error(str(exc))


# ponytail: every official provider gets an explicit stub so adding a
# real test later is a one-line change at the matching INTEGRATION_PROVIDERS
# entry, not a code-search hunt.
def _test_salesforce(configuration: dict, credentials: dict) -> tuple[bool, str]:
    return _stub("salesforce")


def _test_dynamics365(configuration: dict, credentials: dict) -> tuple[bool, str]:
    return _stub("dynamics365")


def _test_genesys_cloud(configuration: dict, credentials: dict) -> tuple[bool, str]:
    return _stub("genesys_cloud")


def _test_nice_cxone(configuration: dict, credentials: dict) -> tuple[bool, str]:
    return _stub("nice_cxone")


def run_integration_test(test_fn_path: str, configuration: dict, credentials: dict) -> tuple[bool, str]:
    """Resolve `test_fn_path` (dotted, e.g. "_test_zendesk" — caller
    prepends the module) and invoke. Returns (False, "...") if the
    path doesn't resolve — never raises."""
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
        return fn(configuration, credentials)
    except Exception as exc:
        log.exception("[integrations_tester] %s raised", test_fn_path)
        return False, _sanitize_error(str(exc))