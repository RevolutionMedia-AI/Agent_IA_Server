"""Microsoft Dynamics 365 / Dataverse OAuth discovery and action executor."""
from __future__ import annotations

from dataclasses import replace
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


_DISCOVERY_URL = "https://globaldisco.crm.dynamics.com/api/discovery/v2.0/Instances"
_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_ENV_HOST_RE = re.compile(
    r"^[a-z0-9-]+(?:\.api)?\.crm\d*\."
    r"(?:dynamics\.com|microsoftdynamics\.us|appsplatform\.us|dynamics\.cn)$"
)


class DataverseError(RuntimeError):
    def __init__(self, message: str, status: int = 0, code: str = "dataverse_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def normalize_environment_url(value: str) -> str:
    raw = (value or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or not _ENV_HOST_RE.fullmatch(host)
    ):
        raise ValueError("Invalid Dynamics 365 environment URL")
    marker = "/api/data/"
    path = parsed.path or ""
    if marker in path.lower():
        path = path[: path.lower().index(marker)]
    return urllib.parse.urlunparse(("https", parsed.netloc, path.rstrip("/"), "", "", ""))


def _json_request(method: str, url: str, access_token: str, body: dict | None = None):
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Accept", "application/json")
    req.add_header("OData-MaxVersion", "4.0")
    req.add_header("OData-Version", "4.0")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    if method in ("POST", "PATCH"):
        req.add_header("Prefer", "return=representation")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8")
            return (json.loads(raw) if raw else {}), dict(response.headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:1000]
        try:
            payload_error = json.loads(raw).get("error") or {}
            code = payload_error.get("code") or "dataverse_http_error"
            message = payload_error.get("message") or raw
        except Exception:
            code, message = "dataverse_http_error", raw
        raise DataverseError(message or f"Dataverse HTTP {exc.code}", exc.code, code) from exc
    except urllib.error.URLError as exc:
        raise DataverseError(f"Dataverse network error: {exc.reason}") from exc


def discover_environments(access_token: str) -> list[dict]:
    query = urllib.parse.urlencode({
        "$select": "ApiUrl,FriendlyName,Id,EnvironmentId,TenantId,UniqueName,State"
    })
    payload, _ = _json_request("GET", f"{_DISCOVERY_URL}?{query}", access_token)
    environments = []
    for item in payload.get("value") or []:
        if item.get("State") not in (None, 0):
            continue
        try:
            api_url = normalize_environment_url(item.get("ApiUrl") or item.get("Url") or "")
        except ValueError:
            continue
        environments.append({
            "environment_url": api_url,
            "name": item.get("FriendlyName") or item.get("UniqueName") or api_url,
            "organization_id": item.get("Id"),
            "environment_id": item.get("EnvironmentId"),
            "tenant_id": item.get("TenantId"),
        })
    return environments


def prepare_oauth_connection(credentials: dict) -> dict:
    """Discover accessible Dataverse environments and hydrate OAuth metadata."""
    environments = discover_environments(credentials.get("access_token") or "")
    if not environments:
        raise DataverseError("No accessible Dynamics 365 environments were found")
    config = {
        "capabilities": ["sales", "customer_service"],
        "environments": environments,
        "tenant_id": environments[0].get("tenant_id"),
    }
    if len(environments) == 1:
        selected = environments[0]
        config.update({
            "environment_url": selected["environment_url"],
            "environment_name": selected["name"],
            "organization_id": selected.get("organization_id"),
            "environment_id": selected.get("environment_id"),
        })
        _replace_with_environment_token(
            credentials, selected["environment_url"], selected.get("tenant_id")
        )
    return config


def oauth_config_for_tenant(tenant_id: str | None):
    from STT_server.services.oauth_providers import get_oauth_config
    config = get_oauth_config("dynamics365")
    if tenant_id and _GUID_RE.fullmatch(tenant_id):
        return replace(
            config,
            token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        )
    return config


def _replace_with_environment_token(
    credentials: dict,
    environment_url: str,
    tenant_id: str | None = None,
) -> None:
    refresh_token = credentials.get("refresh_token")
    if not refresh_token:
        raise DataverseError("Microsoft did not return a refresh token; reconnect and grant offline_access")
    from STT_server.services.oauth_providers import (
        now_plus_seconds,
        refresh_access_token,
    )
    tokens = refresh_access_token(
        oauth_config_for_tenant(tenant_id),
        refresh_token,
        scopes=(f"{normalize_environment_url(environment_url)}/.default",),
    )
    credentials["access_token"] = tokens.access_token
    credentials["refresh_token"] = tokens.refresh_token or refresh_token
    if tokens.expires_in is not None:
        credentials["expires_at"] = now_plus_seconds(tokens.expires_in)
    if tokens.scope:
        credentials["scope"] = tokens.scope


class Dynamics365Client:
    def __init__(self, integration: dict, credentials: dict):
        self.integration = integration
        self.credentials = dict(credentials)
        self.environment_url = normalize_environment_url(
            (integration.get("configuration") or {}).get("environment_url") or ""
        )
        self.base_url = f"{self.environment_url}/api/data/v9.2/"

    def request(self, method: str, path: str, body: dict | None = None, query: dict | None = None):
        url = urllib.parse.urljoin(self.base_url, path.lstrip("/"))
        if query:
            url += "?" + urllib.parse.urlencode(query, safe="(),'$")
        try:
            return _json_request(method, url, self.credentials.get("access_token") or "", body)
        except DataverseError as exc:
            if exc.status != 401:
                raise
        self._refresh()
        try:
            return _json_request(method, url, self.credentials["access_token"], body)
        except DataverseError as exc:
            if exc.status == 401:
                self._mark_failed("Microsoft session expired; reconnect the integration")
            raise

    def _refresh(self) -> None:
        integration_id = self.integration["id"]
        user_id = self.integration["user_id"]
        from STT_server import db_integrations
        from STT_server.db import get_conn, is_postgres
        from STT_server.security.credentials import decrypt_credentials, encrypt_credentials
        from STT_server.services.oauth_providers import is_token_expiring

        def refresh_and_store(current: dict, cur=None):
            if not current.get("refresh_token"):
                raise DataverseError("Refresh token missing; reconnect Dynamics 365", 401, "reauth_required")
            _replace_with_environment_token(
                current,
                self.environment_url,
                (self.integration.get("configuration") or {}).get("tenant_id"),
            )
            db_integrations.update_integration_credentials(
                integration_id, user_id, encrypt_credentials(current), cur=cur
            )
            db_integrations.mark_integration_status(
                integration_id, user_id, "connected", last_test_message="Dynamics token refreshed", cur=cur
            )
            self.credentials = current

        if not is_postgres():
            refresh_and_store(dict(self.credentials))
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                db_integrations.acquire_advisory_xact_lock(cur, integration_id)
                fresh = db_integrations.get_integration_by_id(integration_id, cur=cur)
                current = decrypt_credentials(fresh.get("credentials_encrypted")) if fresh else dict(self.credentials)
                if current.get("access_token") != self.credentials.get("access_token") and not is_token_expiring(current.get("expires_at")):
                    self.credentials = current
                    return
                refresh_and_store(current, cur=cur)

    def _mark_failed(self, message: str) -> None:
        from STT_server.db_integrations import mark_integration_status
        mark_integration_status(
            self.integration["id"], self.integration["user_id"], "failed", last_test_message=message
        )


def _guid(arguments: dict, name: str) -> str:
    value = str(arguments.get(name) or "").strip()
    if not _GUID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a Dataverse GUID")
    return value


def _required(arguments: dict, name: str) -> str:
    value = str(arguments.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _odata(value: str) -> str:
    return value.replace("'", "''")


def _record(entity: str, row: dict) -> dict:
    clean = {k: v for k, v in row.items() if not k.startswith("@odata.")}
    key = {
        "contacts": "contactid", "accounts": "accountid", "leads": "leadid",
        "incidents": "incidentid", "opportunities": "opportunityid",
        "phonecalls": "activityid", "tasks": "activityid", "annotations": "annotationid",
        "msdyn_customerassets": "msdyn_customerassetid",
        "msdyn_workorders": "msdyn_workorderid",
        "bookableresourcebookings": "bookableresourcebookingid",
        "bookableresources": "bookableresourceid",
        "msdyn_agreements": "msdyn_agreementid",
        "bookingstatuses": "bookingstatusid",
        "msdyn_priorities": "msdyn_priorityid",
    }.get(entity)
    record_type = {
        "contacts": "contact", "accounts": "account", "leads": "lead",
        "incidents": "case", "opportunities": "opportunity",
        "phonecalls": "phonecall", "tasks": "task", "annotations": "note",
        "msdyn_customerassets": "asset", "msdyn_workorders": "workorder",
        "bookableresourcebookings": "booking", "bookableresources": "resource",
        "msdyn_agreements": "agreement",
        "bookingstatuses": "bookingstatus", "msdyn_priorities": "priority",
    }.get(entity, entity)
    return {"type": record_type, "id": clean.get(key), **clean}


def _rows(client: Dynamics365Client, entity: str, select: str, filter_: str = "", limit: int = 10):
    query = {"$select": select, "$top": max(1, min(int(limit), 50))}
    if filter_:
        query["$filter"] = filter_
    payload, _ = client.request("GET", entity, query=query)
    return [_record(entity, row) for row in payload.get("value") or []]


def _create(client: Dynamics365Client, entity: str, body: dict) -> dict:
    payload, headers = client.request("POST", entity, body=body)
    row = _record(entity, payload) if payload else {}
    entity_uri = headers.get("OData-EntityId") or headers.get("odata-entityid") or ""
    match = re.search(r"\(([0-9a-fA-F-]{36})\)", entity_uri)
    return {"created": True, "type": _record(entity, {}).get("type"), "id": row.get("id") or (match.group(1) if match else None), "record": row or None}


def _update(client: Dynamics365Client, entity: str, record_id: str, body: dict) -> dict:
    if not body:
        raise ValueError("At least one field to update is required")
    payload, _ = client.request("PATCH", f"{entity}({record_id})", body=body)
    return {"updated": True, "type": _record(entity, {}).get("type"), "id": record_id, "record": _record(entity, payload) if payload else None}


def _customer_bind(body: dict, arguments: dict) -> None:
    if not arguments.get("customer_id"):
        return
    customer_id = _guid(arguments, "customer_id")
    kind = arguments.get("customer_type") or "contact"
    if kind not in ("contact", "account"):
        raise ValueError("customer_type must be contact or account")
    body[f"customerid_{kind}@odata.bind"] = f"/{kind}s({customer_id})"


def _regarding_bind(body: dict, arguments: dict, prefix: str = "regardingobjectid") -> None:
    if not arguments.get("regarding_id"):
        return
    record_id = _guid(arguments, "regarding_id")
    kind = arguments.get("regarding_type")
    entities = {"contact": "contacts", "account": "accounts", "lead": "leads", "incident": "incidents", "opportunity": "opportunities", "workorder": "msdyn_workorders", "booking": "bookableresourcebookings", "asset": "msdyn_customerassets"}
    if kind not in entities:
        raise ValueError("regarding_type is required and invalid")
    body[f"{prefix}_{kind}@odata.bind"] = f"/{entities[kind]}({record_id})"


def _mapped(arguments: dict, mapping: dict[str, str]) -> dict:
    return {target: arguments[source] for source, target in mapping.items() if arguments.get(source) is not None}


# ── Field Service ─────────────────────────────────────────────────────────
# ponytail: Field Service is a CAPABILITY of provider "dynamics365", not a
# provider. Same OAuth, same Dynamics365Client, same executor contract.
# Entity metadata verified against Microsoft Learn (Field Service entity
# reference + "Create work orders using the Dataverse Web API" +
# "Search resource availability API"):
#   * msdyn_customerassets (msdyn_customerassetid, msdyn_name required;
#     msdyn_assettag = Asset Tag, msdyn_account->account,
#     msdyn_product->product). No serial_number / install_date in base schema.
#   * msdyn_workorders (msdyn_workorderid; msdyn_name is AUTO-numbered, never
#     written; msdyn_systemstatus required: Unscheduled 690970000,
#     Scheduled 690970001, In Progress 690970002, Completed 690970003,
#     Posted 690970004, Canceled 690970005; msdyn_priority is a LOOKUP).
#   * bookableresourcebookings (starttime/endtime/resource/bookingstatus
#     required; msdyn_workorder->msdyn_workorder).
#   * bookingstatuses + msdyn_fieldservicesettings (default scheduled/
#     canceled statuses); FS status Scheduled 690970000 / Canceled 690970005.
#   * msdyn_agreements (msdyn_serviceaccount required).
#   * msdyn_SearchResourceAvailability (Version 3 + IsWebApi + inline
#     msdyn_resourcerequirement + expando Settings).

FIELD_SERVICE_ACTIONS = (
    "find_asset", "get_asset", "create_asset", "update_asset",
    "create_work_order", "get_work_order", "get_work_orders",
    "update_work_order", "cancel_work_order", "complete_work_order",
    "get_available_resources", "create_booking", "update_booking",
    "cancel_booking", "find_service_agreement", "get_service_agreements",
)

_WO_SYSTEMSTATUS_COMPLETED = 690970003
_WO_SYSTEMSTATUS_CANCELED = 690970005
_WO_CLOSED_STATUSES = (690970003, 690970004, 690970005)  # Completed, Posted, Canceled
_BOOKING_FS_SCHEDULED = 690970000
_BOOKING_FS_CANCELED = 690970005
_CAPABILITY_ERROR = (
    "CAPABILITY_NOT_AVAILABLE: Field Service is not available "
    "for this Dynamics environment."
)

_ASSET_SELECT = (
    "msdyn_customerassetid,msdyn_name,msdyn_assettag,msdyn_account,"
    "msdyn_product,msdyn_parentasset,statecode,statuscode,createdon"
)
_WORKORDER_SELECT = (
    "msdyn_workorderid,msdyn_name,msdyn_systemstatus,msdyn_instructions,"
    "msdyn_serviceaccount,msdyn_billingaccount,msdyn_customerasset,"
    "msdyn_priority,msdyn_primaryincidenttype,msdyn_workordertype,"
    "msdyn_timefrompromised,msdyn_timetopromised,msdyn_datewindowstart,"
    "msdyn_datewindowend,statecode,statuscode,createdon"
)
_AGREEMENT_SELECT = (
    "msdyn_agreementid,msdyn_name,msdyn_description,msdyn_serviceaccount,"
    "msdyn_systemstatus,msdyn_startdate,msdyn_enddate,statecode,statuscode"
)


def _require_field_service(client: Dynamics365Client, integration: dict) -> None:
    """Gate every Field Service action on real availability.

    Source of truth is a lazy probe (GET msdyn_workorders?$top=1) cached
    in configuration.field_service_available. 404 on the probe segment
    means Field Service isn't installed -> CAPABILITY_NOT_AVAILABLE.
    Any other HTTP error (401/403/500...) is a transport/auth problem
    and propagates unchanged — never reported as missing capability.
    The cache is cleared whenever environment_url changes (PUT handler),
    so a re-probe happens against the new environment.
    """
    config = integration.get("configuration") or {}
    cached = config.get("field_service_available")
    if cached is True:
        return
    if cached is False:
        raise ValueError(_CAPABILITY_ERROR)
    try:
        client.request(
            "GET", "msdyn_workorders",
            query={"$select": "msdyn_workorderid", "$top": 1},
        )
    except DataverseError as exc:
        if exc.status != 404:
            raise
        _cache_field_service(integration, False)
        raise ValueError(_CAPABILITY_ERROR) from exc
    _cache_field_service(integration, True)


def _cache_field_service(integration: dict, available: bool) -> None:
    """Persist the probe result. Best-effort: a persistence failure must
    not fail the action itself — availability is already known."""
    try:
        integration_id = integration.get("id")
        user_id = integration.get("user_id")
        if not integration_id or not user_id:
            return
        from STT_server import db_integrations
        config = dict(integration.get("configuration") or {})
        config["field_service_available"] = available
        db_integrations.update_integration(
            integration_id, user_id, {"configuration": config}
        )
        integration["configuration"] = config
    except Exception:
        pass


def _resolve_priority_id(client: Dynamics365Client, arguments: dict):
    """msdyn_priority is a lookup: accept a GUID directly or resolve a
    priority name (e.g. "High") against msdyn_priorities. No invented codes."""
    if arguments.get("priority_id"):
        return _guid(arguments, "priority_id")
    name = str(arguments.get("priority") or "").strip()
    if not name:
        return None
    rows = _rows(
        client, "msdyn_priorities", "msdyn_priorityid,msdyn_name",
        f"contains(msdyn_name,'{_odata(name)}')", 5,
    )
    for row in rows:
        if str(row.get("msdyn_name") or "").strip().lower() == name.lower():
            return row["id"]
    if rows:
        return rows[0]["id"]
    raise ValueError(f"Priority '{name}' was not found in this environment")


def _default_booking_status_id(client: Dynamics365Client, kind: str) -> str:
    """Resolve the environment's default booking status GUID. Official
    mechanism first (msdyn_fieldservicesettings defaults), documented
    msdyn_fieldservicestatus fallback second. Never hardcoded GUIDs."""
    attr = (
        "msdyn_defaultscheduledbookingstatus"
        if kind == "scheduled" else "msdyn_defaultcanceledbookingstatus"
    )
    try:
        payload, _ = client.request(
            "GET", "msdyn_fieldservicesettings",
            query={"$select": f"{attr}", "$top": 1},
        )
        rows = payload.get("value") or []
        if rows and rows[0].get(f"_{attr}_value"):
            return rows[0][f"_{attr}_value"]
    except DataverseError as exc:
        if exc.status != 404:
            raise
    fs_status = _BOOKING_FS_SCHEDULED if kind == "scheduled" else _BOOKING_FS_CANCELED
    rows = _rows(
        client, "bookingstatuses", "bookingstatusid,name",
        f"msdyn_fieldservicestatus eq {fs_status}", 1,
    )
    if rows:
        return rows[0]["id"]
    raise ValueError(
        "No default booking status is configured in this environment"
    )


def _minutes_between(start: str, end: str):
    """Duration in whole minutes; None when unparseable (caller omits it)."""
    try:
        from datetime import datetime
        s = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        e = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return max(1, int((e - s).total_seconds() // 60))
    except (TypeError, ValueError):
        return None


def _note_on(client: Dynamics365Client, regarding_type: str, regarding_id: str, subject: str, text: str) -> None:
    """Timeline note written BEFORE the status PATCH it documents, so a
    PATCH failure never leaves a status change without its reason."""
    body = {"notetext": text, "subject": subject}
    _regarding_bind(
        body,
        {"regarding_id": regarding_id, "regarding_type": regarding_type},
        "objectid",
    )
    client.request("POST", "annotations", body=body)


def execute_dynamics_action(action: str, integration: dict, credentials: dict, arguments: dict):
    client = Dynamics365Client(integration, credentials)
    try:
        if action == "find_customer":
            q = _odata(_required(arguments, "query"))
            contacts = _rows(client, "contacts", "contactid,fullname,emailaddress1,telephone1", f"contains(fullname,'{q}') or contains(emailaddress1,'{q}') or contains(telephone1,'{q}')")
            accounts = _rows(client, "accounts", "accountid,name,emailaddress1,telephone1", f"contains(name,'{q}') or contains(emailaddress1,'{q}') or contains(telephone1,'{q}')")
            return True, {"records": contacts + accounts, "count": len(contacts) + len(accounts)}, None
        if action == "get_customer":
            record_id = _guid(arguments, "customer_id")
            if arguments.get("customer_type") not in (None, "contact", "account"):
                raise ValueError("customer_type must be contact or account")
            kinds = [arguments["customer_type"]] if arguments.get("customer_type") in ("contact", "account") else ["contact", "account"]
            for kind in kinds:
                entity, select = ("contacts", "contactid,fullname,firstname,lastname,emailaddress1,telephone1") if kind == "contact" else ("accounts", "accountid,name,emailaddress1,telephone1,websiteurl")
                try:
                    payload, _ = client.request("GET", f"{entity}({record_id})", query={"$select": select})
                    return True, {"record": _record(entity, payload)}, None
                except DataverseError as exc:
                    if exc.status != 404:
                        raise
            return False, None, "Customer not found"
        if action == "create_contact":
            _required(arguments, "last_name")
            body = _mapped(arguments, {"first_name": "firstname", "last_name": "lastname", "email": "emailaddress1", "phone": "telephone1"})
            if arguments.get("account_id"):
                body["parentcustomerid_account@odata.bind"] = f"/accounts({_guid(arguments, 'account_id')})"
            return True, _create(client, "contacts", body), None
        if action == "update_customer":
            record_id = _guid(arguments, "customer_id")
            kind = arguments.get("customer_type") or "contact"
            if kind not in ("contact", "account"):
                raise ValueError("customer_type must be contact or account")
            entity = "accounts" if kind == "account" else "contacts"
            mapping = {"name": "name", "email": "emailaddress1", "phone": "telephone1"} if entity == "accounts" else {"first_name": "firstname", "last_name": "lastname", "email": "emailaddress1", "phone": "telephone1"}
            return True, _update(client, entity, record_id, _mapped(arguments, mapping)), None
        if action == "find_account":
            q = _odata(_required(arguments, "query"))
            rows = _rows(client, "accounts", "accountid,name,emailaddress1,telephone1,websiteurl", f"contains(name,'{q}') or contains(emailaddress1,'{q}') or contains(telephone1,'{q}')")
            return True, {"records": rows, "count": len(rows)}, None
        if action in ("create_account", "update_account"):
            if action == "create_account":
                _required(arguments, "name")
            body = _mapped(arguments, {"name": "name", "email": "emailaddress1", "phone": "telephone1", "website": "websiteurl"})
            data = _create(client, "accounts", body) if action == "create_account" else _update(client, "accounts", _guid(arguments, "account_id"), body)
            return True, data, None
        if action == "find_lead":
            q = _odata(_required(arguments, "query"))
            rows = _rows(client, "leads", "leadid,fullname,subject,companyname,emailaddress1,telephone1,statecode,statuscode", f"contains(fullname,'{q}') or contains(companyname,'{q}') or contains(emailaddress1,'{q}')")
            return True, {"records": rows, "count": len(rows)}, None
        if action in ("create_lead", "update_lead"):
            if action == "create_lead":
                _required(arguments, "subject")
                _required(arguments, "last_name")
            body = _mapped(arguments, {"subject": "subject", "first_name": "firstname", "last_name": "lastname", "company": "companyname", "email": "emailaddress1", "phone": "telephone1"})
            data = _create(client, "leads", body) if action == "create_lead" else _update(client, "leads", _guid(arguments, "lead_id"), body)
            return True, data, None
        if action == "qualify_lead":
            record_id = _guid(arguments, "lead_id")
            body = {"CreateAccount": bool(arguments.get("create_account", False)), "CreateContact": bool(arguments.get("create_contact", True)), "CreateOpportunity": bool(arguments.get("create_opportunity", False)), "Status": int(arguments.get("status", 3)), "OpportunityCurrencyId": None, "ProcessInstanceId": None, "SourceCampaignId": None}
            payload, _ = client.request("POST", f"leads({record_id})/Microsoft.Dynamics.CRM.QualifyLead", body=body)
            return True, {"qualified": True, "lead_id": record_id, "result": payload or None}, None
        if action == "create_case":
            _required(arguments, "title")
            body = _mapped(arguments, {"title": "title", "description": "description"})
            _customer_bind(body, arguments)
            return True, _create(client, "incidents", body), None
        if action == "get_cases":
            filters = []
            if arguments.get("customer_id"):
                filters.append(f"_customerid_value eq {_guid(arguments, 'customer_id')}")
            if arguments.get("active_only", True):
                filters.append("statecode eq 0")
            rows = _rows(client, "incidents", "incidentid,ticketnumber,title,description,prioritycode,statecode,statuscode,createdon", " and ".join(filters), arguments.get("limit", 10))
            return True, {"records": rows, "count": len(rows)}, None
        if action == "get_case":
            record_id = _guid(arguments, "case_id")
            payload, _ = client.request("GET", f"incidents({record_id})", query={"$select": "incidentid,ticketnumber,title,description,prioritycode,statecode,statuscode,createdon"})
            return True, {"record": _record("incidents", payload)}, None
        if action == "update_case":
            return True, _update(client, "incidents", _guid(arguments, "case_id"), _mapped(arguments, {"title": "title", "description": "description", "priority_code": "prioritycode"})), None
        if action == "resolve_case":
            record_id = _guid(arguments, "case_id")
            body = {"IncidentResolution": {"@odata.type": "Microsoft.Dynamics.CRM.incidentresolution", "subject": arguments.get("resolution") or "Case resolved", "incidentid@odata.bind": f"/incidents({record_id})"}, "Status": int(arguments.get("status", 5))}
            payload, _ = client.request("POST", "CloseIncident", body=body)
            return True, {"resolved": True, "case_id": record_id, "result": payload or None}, None
        if action == "reopen_case":
            record_id = _guid(arguments, "case_id")
            return True, _update(client, "incidents", record_id, {"statecode": 0, "statuscode": 1}), None
        if action == "create_opportunity":
            _required(arguments, "name")
            body = _mapped(arguments, {"name": "name", "estimated_value": "estimatedvalue", "estimated_close_date": "estimatedclosedate"})
            _customer_bind(body, arguments)
            return True, _create(client, "opportunities", body), None
        if action == "get_opportunities":
            filters = []
            if arguments.get("customer_id"):
                filters.append(f"_customerid_value eq {_guid(arguments, 'customer_id')}")
            if arguments.get("active_only", True):
                filters.append("statecode eq 0")
            rows = _rows(client, "opportunities", "opportunityid,name,estimatedvalue,estimatedclosedate,statecode,statuscode,description", " and ".join(filters), arguments.get("limit", 10))
            return True, {"records": rows, "count": len(rows)}, None
        if action == "get_opportunity":
            record_id = _guid(arguments, "opportunity_id")
            payload, _ = client.request("GET", f"opportunities({record_id})", query={"$select": "opportunityid,name,estimatedvalue,estimatedclosedate,statecode,statuscode,description"})
            return True, {"record": _record("opportunities", payload)}, None
        if action == "update_opportunity":
            body = _mapped(arguments, {"name": "name", "estimated_value": "estimatedvalue", "estimated_close_date": "estimatedclosedate", "description": "description"})
            return True, _update(client, "opportunities", _guid(arguments, "opportunity_id"), body), None
        if action in ("close_opportunity_won", "close_opportunity_lost"):
            record_id = _guid(arguments, "opportunity_id")
            won = action.endswith("won")
            close = {"@odata.type": "Microsoft.Dynamics.CRM.opportunityclose", "subject": arguments.get("subject") or ("Opportunity won" if won else "Opportunity lost"), "opportunityid@odata.bind": f"/opportunities({record_id})"}
            if not won and arguments.get("description"):
                close["description"] = arguments["description"]
            payload, _ = client.request("POST", "WinOpportunity" if won else "LoseOpportunity", body={"OpportunityClose": close, "Status": int(arguments.get("status", 3 if won else 4))})
            return True, {"closed": True, "won": won, "opportunity_id": record_id, "result": payload or None}, None
        if action == "log_call":
            _required(arguments, "subject")
            if arguments.get("direction", "outgoing") not in ("incoming", "outgoing"):
                raise ValueError("direction must be incoming or outgoing")
            body = _mapped(arguments, {"subject": "subject", "description": "description", "phone": "phonenumber"})
            body["directioncode"] = arguments.get("direction", "outgoing") == "outgoing"
            _regarding_bind(body, arguments)
            return True, _create(client, "phonecalls", body), None
        if action == "add_note":
            body = {"notetext": _required(arguments, "text")}
            if arguments.get("subject"):
                body["subject"] = arguments["subject"]
            _regarding_bind(body, arguments, "objectid")
            return True, _create(client, "annotations", body), None
        if action == "create_task":
            _required(arguments, "subject")
            body = _mapped(arguments, {"subject": "subject", "description": "description", "due_at": "scheduledend"})
            _regarding_bind(body, arguments)
            return True, _create(client, "tasks", body), None
        if action == "complete_task":
            record_id = _guid(arguments, "task_id")
            return True, _update(client, "tasks", record_id, {"statecode": 1, "statuscode": 5}), None
        if action in FIELD_SERVICE_ACTIONS:
            _require_field_service(client, integration)
            return _execute_field_service_action(action, client, arguments)
        return False, None, f"Unsupported Dynamics 365 action '{action}'"
    except (ValueError, TypeError) as exc:
        return False, None, str(exc)
    except DataverseError as exc:
        return False, None, f"{exc.code}: {exc}"


def _execute_field_service_action(action: str, client: Dynamics365Client, arguments: dict):
    """Field Service actions. Only reached after _require_field_service."""
    if action == "find_asset":
        q = _odata(_required(arguments, "query"))
        filt = f"contains(msdyn_name,'{q}') or contains(msdyn_assettag,'{q}')"
        if arguments.get("account_id"):
            filt = f"({filt}) and _msdyn_account_value eq {_guid(arguments, 'account_id')}"
        rows = _rows(client, "msdyn_customerassets", _ASSET_SELECT, filt, arguments.get("limit", 10))
        return True, {"records": rows, "count": len(rows)}, None
    if action == "get_asset":
        record_id = _guid(arguments, "asset_id")
        payload, _ = client.request("GET", f"msdyn_customerassets({record_id})", query={"$select": _ASSET_SELECT})
        return True, {"record": _record("msdyn_customerassets", payload)}, None
    if action == "create_asset":
        # ponytail: msdyn_name is the only required attribute; there is
        # no serial_number / install_date in the base schema — callers
        # use asset_tag (msdyn_assettag).
        _required(arguments, "name")
        body = _mapped(arguments, {"name": "msdyn_name", "asset_tag": "msdyn_assettag"})
        if arguments.get("account_id"):
            body["msdyn_account@odata.bind"] = f"/accounts({_guid(arguments, 'account_id')})"
        if arguments.get("product_id"):
            body["msdyn_product@odata.bind"] = f"/products({_guid(arguments, 'product_id')})"
        if arguments.get("parent_asset_id"):
            body["msdyn_parentasset@odata.bind"] = f"/msdyn_customerassets({_guid(arguments, 'parent_asset_id')})"
        return True, _create(client, "msdyn_customerassets", body), None
    if action == "update_asset":
        record_id = _guid(arguments, "asset_id")
        body = _mapped(arguments, {"name": "msdyn_name", "asset_tag": "msdyn_assettag"})
        if arguments.get("account_id"):
            body["msdyn_account@odata.bind"] = f"/accounts({_guid(arguments, 'account_id')})"
        if arguments.get("product_id"):
            body["msdyn_product@odata.bind"] = f"/products({_guid(arguments, 'product_id')})"
        return True, _update(client, "msdyn_customerassets", record_id, body), None
    if action == "create_work_order":
        # ponytail: msdyn_name is auto-numbered by Field Service — never
        # accepted from the LLM. description lands on msdyn_instructions
        # (per the official create-work-order API doc + agreement mapping).
        body = _mapped(arguments, {"description": "msdyn_instructions"})
        if arguments.get("service_account_id"):
            body["msdyn_serviceaccount@odata.bind"] = f"/accounts({_guid(arguments, 'service_account_id')})"
        if arguments.get("billing_account_id"):
            body["msdyn_billingaccount@odata.bind"] = f"/accounts({_guid(arguments, 'billing_account_id')})"
        if arguments.get("asset_id"):
            body["msdyn_customerasset@odata.bind"] = f"/msdyn_customerassets({_guid(arguments, 'asset_id')})"
        if arguments.get("work_order_type_id"):
            body["msdyn_workordertype@odata.bind"] = f"/msdyn_workordertypes({_guid(arguments, 'work_order_type_id')})"
        if arguments.get("price_list_id"):
            body["msdyn_pricelist@odata.bind"] = f"/pricelevels({_guid(arguments, 'price_list_id')})"
        if arguments.get("incident_type_id"):
            body["msdyn_primaryincidenttype@odata.bind"] = f"/msdyn_incidenttypes({_guid(arguments, 'incident_type_id')})"
        priority_id = _resolve_priority_id(client, arguments)
        if priority_id:
            body["msdyn_priority@odata.bind"] = f"/msdyn_priorities({priority_id})"
        if arguments.get("scheduled_start"):
            body["msdyn_timefrompromised"] = arguments["scheduled_start"]
        if arguments.get("scheduled_end"):
            body["msdyn_timetopromised"] = arguments["scheduled_end"]
        body["msdyn_systemstatus"] = 690970000  # Unscheduled, per official doc
        body["msdyn_taxable"] = False
        return True, _create(client, "msdyn_workorders", body), None
    if action == "get_work_order":
        record_id = _guid(arguments, "work_order_id")
        payload, _ = client.request("GET", f"msdyn_workorders({record_id})", query={"$select": _WORKORDER_SELECT})
        return True, {"record": _record("msdyn_workorders", payload)}, None
    if action == "get_work_orders":
        filters = []
        if arguments.get("account_id"):
            filters.append(f"_msdyn_serviceaccount_value eq {_guid(arguments, 'account_id')}")
        if arguments.get("asset_id"):
            filters.append(f"_msdyn_customerasset_value eq {_guid(arguments, 'asset_id')}")
        if str(arguments.get("status", "open")).strip().lower() not in ("all",):
            if str(arguments.get("status", "open")).strip().lower() != "open":
                raise ValueError("status must be open or all")
            filters.append(" and ".join(f"msdyn_systemstatus ne {s}" for s in _WO_CLOSED_STATUSES))
        rows = _rows(client, "msdyn_workorders", _WORKORDER_SELECT, " and ".join(filters), arguments.get("limit", 10))
        return True, {"records": rows, "count": len(rows)}, None
    if action == "update_work_order":
        record_id = _guid(arguments, "work_order_id")
        body = _mapped(arguments, {"description": "msdyn_instructions"})
        if arguments.get("service_account_id"):
            body["msdyn_serviceaccount@odata.bind"] = f"/accounts({_guid(arguments, 'service_account_id')})"
        if arguments.get("asset_id"):
            body["msdyn_customerasset@odata.bind"] = f"/msdyn_customerassets({_guid(arguments, 'asset_id')})"
        priority_id = _resolve_priority_id(client, arguments)
        if priority_id:
            body["msdyn_priority@odata.bind"] = f"/msdyn_priorities({priority_id})"
        if arguments.get("scheduled_start"):
            body["msdyn_timefrompromised"] = arguments["scheduled_start"]
        if arguments.get("scheduled_end"):
            body["msdyn_timetopromised"] = arguments["scheduled_end"]
        return True, _update(client, "msdyn_workorders", record_id, body), None
    if action in ("cancel_work_order", "complete_work_order"):
        record_id = _guid(arguments, "work_order_id")
        notes = arguments.get("reason") if action == "cancel_work_order" else arguments.get("completion_notes")
        if notes:
            _note_on(client, "workorder", record_id, "Work order canceled" if action == "cancel_work_order" else "Work order completed", str(notes))
        status = _WO_SYSTEMSTATUS_CANCELED if action == "cancel_work_order" else _WO_SYSTEMSTATUS_COMPLETED
        return True, _update(client, "msdyn_workorders", record_id, {"msdyn_systemstatus": status}), None
    if action == "get_available_resources":
        start = _required(arguments, "start")
        end = _required(arguments, "end")
        try:
            duration = int(arguments.get("duration_minutes") or 0)
        except (TypeError, ValueError):
            raise ValueError("duration_minutes must be an integer") from None
        if not duration and arguments.get("work_order_id"):
            record_id = _guid(arguments, "work_order_id")
            payload, _ = client.request(
                "GET", f"msdyn_workorders({record_id})",
                query={"$select": "msdyn_primaryincidentestimatedduration"},
            )
            duration = int(payload.get("msdyn_primaryincidentestimatedduration") or 0)
        if not duration:
            duration = 60
        if duration < 1:
            raise ValueError("duration_minutes must be positive")
        body = {
            "Version": "3",
            "IsWebApi": True,
            "Requirement": {
                "msdyn_fromdate": start,
                "msdyn_todate": end,
                "msdyn_duration": duration,
                "msdyn_remainingduration": duration,
                "@odata.type": "Microsoft.Dynamics.CRM.msdyn_resourcerequirement",
            },
            "Settings": {"@odata.type": "Microsoft.Dynamics.CRM.expando"},
        }
        payload, _ = client.request("POST", "msdyn_SearchResourceAvailability", body=body)
        return True, {"resources": _normalize_availability(payload), "count": len(_normalize_availability(payload))}, None
    if action == "create_booking":
        record_id = _guid(arguments, "work_order_id")
        resource_id = _guid(arguments, "resource_id")
        start = _required(arguments, "start")
        end = _required(arguments, "end")
        body = {
            "starttime": start,
            "endtime": end,
            "resource@odata.bind": f"/bookableresources({resource_id})",
            "bookingstatus@odata.bind": f"/bookingstatuses({_default_booking_status_id(client, 'scheduled')})",
            "msdyn_workorder@odata.bind": f"/msdyn_workorders({record_id})",
        }
        minutes = _minutes_between(start, end)
        if minutes:
            body["duration"] = minutes
        return True, _create(client, "bookableresourcebookings", body), None
    if action == "update_booking":
        record_id = _guid(arguments, "booking_id")
        start = arguments.get("start")
        end = arguments.get("end")
        if not start and not end:
            raise ValueError("At least one of start or end is required")
        if bool(start) != bool(end):
            payload, _ = client.request(
                "GET", f"bookableresourcebookings({record_id})",
                query={"$select": "starttime,endtime"},
            )
            start = start or payload.get("starttime")
            end = end or payload.get("endtime")
        body = {"starttime": start, "endtime": end}
        minutes = _minutes_between(start, end)
        if minutes:
            body["duration"] = minutes
        return True, _update(client, "bookableresourcebookings", record_id, body), None
    if action == "cancel_booking":
        record_id = _guid(arguments, "booking_id")
        if arguments.get("reason"):
            _note_on(client, "booking", record_id, "Booking canceled", str(arguments["reason"]))
        return True, _update(
            client, "bookableresourcebookings", record_id,
            {"bookingstatus@odata.bind": f"/bookingstatuses({_default_booking_status_id(client, 'canceled')})"},
        ), None
    if action == "find_service_agreement":
        q = _odata(_required(arguments, "query"))
        filt = f"contains(msdyn_name,'{q}')"
        if arguments.get("account_id"):
            filt = f"({filt}) and _msdyn_serviceaccount_value eq {_guid(arguments, 'account_id')}"
        rows = _rows(client, "msdyn_agreements", _AGREEMENT_SELECT, filt, arguments.get("limit", 10))
        return True, {"records": rows, "count": len(rows)}, None
    if action == "get_service_agreements":
        filters = []
        if arguments.get("account_id"):
            filters.append(f"_msdyn_serviceaccount_value eq {_guid(arguments, 'account_id')}")
        if arguments.get("active_only", True):
            filters.append("statecode eq 0")
        rows = _rows(client, "msdyn_agreements", _AGREEMENT_SELECT, " and ".join(filters), arguments.get("limit", 10))
        return True, {"records": rows, "count": len(rows)}, None
    return False, None, f"Unsupported Dynamics 365 Field Service action '{action}'"


def _normalize_availability(payload: dict) -> list[dict]:
    """Best-effort normalization of the SearchResourceAvailability result.
    Slot shapes vary by URS version — known keys first, raw item kept under
    `details` so nothing the API returned is lost."""
    items = payload.get("Resources") or payload.get("resources") or payload.get("value") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slots = item.get("TimeSlots") or item.get("timeslots") or item.get("Slots") or item.get("slots") or []
        out.append({
            "resource_id": item.get("ResourceId") or item.get("resourceid") or item.get("_resource_value"),
            "resource_name": item.get("ResourceName") or item.get("name"),
            "slots": slots if isinstance(slots, list) else [],
            "details": {k: v for k, v in item.items() if not k.startswith("@odata.")},
        })
    return out


def _executor(action: str) -> Callable:
    return lambda integration, credentials, arguments: execute_dynamics_action(
        action, integration, credentials, arguments
    )


DYNAMICS365_EXECUTORS = {
    action: _executor(action)
    for action in (
        "find_customer", "get_customer", "create_contact", "update_customer",
        "find_account", "create_account", "update_account", "find_lead", "create_lead",
        "update_lead", "qualify_lead", "create_case", "get_cases", "get_case",
        "update_case", "resolve_case", "reopen_case", "create_opportunity",
        "get_opportunities", "get_opportunity", "update_opportunity",
        "close_opportunity_won", "close_opportunity_lost", "log_call", "add_note",
        "create_task", "complete_task",
    ) + FIELD_SERVICE_ACTIONS
}
