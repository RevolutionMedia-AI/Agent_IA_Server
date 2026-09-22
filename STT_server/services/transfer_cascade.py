"""Pre-AI transfer cascade ("hunt group" without STT/LLM/TTS).

When an agent has ``transfer_cascade`` configured, /voice answers with a
chain of Twilio <Dial> verbs instead of connecting straight to the AI
media stream. No audio pipeline runs during the cascade — Twilio rings
each destination, and only when nobody picks up does the call fall
through to <Connect><Stream> (the normal AI path).

Pure functions only (no imports from the app), so tests can exercise
the TwiML + validation without booting FastAPI.
"""
from __future__ import annotations

import re
import urllib.parse

# ponytail: same E.164 shape domain/tool.py enforces for call_transfer
# destinations. One regex for both features so the operator gets the
# same error everywhere.
E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")

DEFAULT_STEP_TIMEOUT_SEC = 20
MIN_STEP_TIMEOUT_SEC = 5
MAX_STEP_TIMEOUT_SEC = 60
MAX_CASCADE_STEPS = 5
MAX_CHAIN_TOOLS = 10


def parse_cascade(raw) -> list[dict]:
    """Normalize the stored cascade to a list of steps.

    Accepts None / list / JSON string (the JSON-file backend stores
    whatever the FE sent). Drops malformed steps instead of raising —
    /voice must never 500 on a bad row; worst case the call goes
    straight to the AI.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            import json
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    steps: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        dest = str(entry.get("destination") or "").strip()
        if not E164_PATTERN.match(dest):
            continue
        try:
            timeout = int(entry.get("timeout_sec") or DEFAULT_STEP_TIMEOUT_SEC)
        except (TypeError, ValueError):
            timeout = DEFAULT_STEP_TIMEOUT_SEC
        timeout = max(MIN_STEP_TIMEOUT_SEC, min(MAX_STEP_TIMEOUT_SEC, timeout))
        steps.append({"destination": dest, "timeout_sec": timeout})
    return steps[:MAX_CASCADE_STEPS]


def validate_cascade(raw) -> tuple[list[dict] | None, str | None]:
    """Strict version for the save path. Returns (steps, None) on
    success or (None, error_message) so the route can 400 with a
    useful message instead of silently dropping bad steps."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, "transfer_cascade must be a list"
    if len(raw) > MAX_CASCADE_STEPS:
        return None, f"transfer_cascade supports at most {MAX_CASCADE_STEPS} steps"
    steps: list[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None, f"transfer_cascade[{i}] must be an object"
        dest = str(entry.get("destination") or "").strip()
        if not E164_PATTERN.match(dest):
            return None, (
                f"transfer_cascade[{i}].destination must be E.164 "
                "(e.g. +15071234567)"
            )
        timeout = entry.get("timeout_sec", DEFAULT_STEP_TIMEOUT_SEC)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            return None, f"transfer_cascade[{i}].timeout_sec must be a number"
        if not MIN_STEP_TIMEOUT_SEC <= timeout <= MAX_STEP_TIMEOUT_SEC:
            return None, (
                f"transfer_cascade[{i}].timeout_sec must be between "
                f"{MIN_STEP_TIMEOUT_SEC} and {MAX_STEP_TIMEOUT_SEC} seconds"
            )
        steps.append({"destination": dest, "timeout_sec": timeout})
    return steps, None


def dial_twiml(
    destination: str,
    timeout_sec: int,
    action_url: str,
) -> str:
    """One <Dial> step. Twilio rings `destination` for `timeout_sec`
    seconds, then POSTs DialCallStatus to `action_url` (no-answer /
    busy / failed / cancel / completed).

    ponytail: the action URL rides in an XML attribute, so its query
    & separators are escaped to &amp;. Raw & is malformed TwiML and
    Twilio may drop the callback — which looks exactly like "the
    cascade dials but never falls through"."""
    action_esc = action_url.replace("&", "&amp;")
    return f"""<Response>
        <Dial timeout="{timeout_sec}" action="{action_esc}" method="POST">
            <Number>{destination}</Number>
        </Dial>
    </Response>"""


def connect_stream_twiml(
    ws_url: str,
    stream_params_str: str,
    play_section: str = "",
) -> str:
    """The normal AI answer (extracted from /voice so /voice and
    /voice/cascade render byte-identical TwiML)."""
    if play_section:
        return f"""
    <Response>
        {play_section}
        <Connect>
            <Stream url="{ws_url}/media-stream">{stream_params_str}</Stream>
        </Connect>
    </Response>
    """
    return f"""
    <Response>
        <Connect>
            <Stream url="{ws_url}/media-stream">{stream_params_str}</Stream>
        </Connect>
    </Response>
    """


def cascade_action_url(
    public_url: str,
    agent_id: str,
    step: int,
    tenant_id: str | None = None,
) -> str:
    """Action URL Twilio hits when a Dial step ends. Query params ride
    along so the callback is stateless (no BE-side call state to
    leak across deploys). Twilio signs the full URL including query,
    and /voice/cascade validates that signature the same way /voice
    does."""
    qs = {"agent_id": agent_id, "step": str(step)}
    if tenant_id:
        qs["tenant_id"] = tenant_id
    return f"{public_url.rstrip('/')}/voice/cascade?{urllib.parse.urlencode(qs)}"


def validate_transfer_chain(raw) -> tuple[list[str] | None, str | None]:
    """Strict version for the agent save path. A chain is an ordered
    list of call_transfer tool ids (user-ordered: phone 1, phone 2,
    ...). Returns (chain, None) or (None, error). Dedupe preserves
    first-seen order so a double-click in the UI can't ring twice."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, "transfer_chain must be a list of tool ids"
    if len(raw) > MAX_CHAIN_TOOLS:
        return None, f"transfer_chain supports at most {MAX_CHAIN_TOOLS} tools"
    chain: list[str] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, str) or not entry.strip():
            return None, f"transfer_chain[{i}] must be a tool id string"
        tid = entry.strip()
        if tid not in chain:
            chain.append(tid)
    return chain, None


def build_transfer_chain(
    invoked_tool_id: str,
    chain: list | None,
    tools_by_id: dict,
) -> list[dict]:
    """Order the Dial chain for one transfer invocation.

    Sequential routing (spec 1-2): the chain is the ordered list
    after the AI. For an explicit named request the LLM invokes the
    tool for that exact destination — route directly there instead of
    forcing the caller through earlier entries. That is a suffix of
    the configured chain starting at the invoked id. For the normal
    sequential case the invoked id is the next expected (first in
    chain), so the suffix is the whole chain. Tools missing a
    destination are dropped. Returns rows in ring order.
    """
    chain = [c for c in (chain or []) if isinstance(c, str)]
    if invoked_tool_id in chain:
        # explicit or sequential — start at its position, not from 0
        idx = chain.index(invoked_tool_id)
        ordered_ids = chain[idx:]
    else:
        # invoked not in configured chain (per-agent tool or legacy)
        # — dial it, then the configured chain as fallback
        ordered_ids = [invoked_tool_id] + [c for c in chain if c != invoked_tool_id]
    out: list[dict] = []
    for tid in ordered_ids[:MAX_CHAIN_TOOLS]:
        tool = tools_by_id.get(tid)
        if not isinstance(tool, dict):
            continue
        dest = str(tool.get("destination") or "").strip()
        if not E164_PATTERN.match(dest):
            continue
        try:
            timeout = int(tool.get("ring_timeout_sec") or DEFAULT_STEP_TIMEOUT_SEC)
        except (TypeError, ValueError):
            timeout = DEFAULT_STEP_TIMEOUT_SEC
        timeout = max(MIN_STEP_TIMEOUT_SEC, min(MAX_STEP_TIMEOUT_SEC, timeout))
        out.append({
            "id": tid,
            "name": tool.get("name") or tid,
            "destination": dest,
            "timeout_sec": timeout,
        })
    return out


def build_unified_routing(
    cascade_steps: list[dict],
    chain_ids: list[str],
    tools_by_id: dict,
) -> list[dict]:
    """Combine cascade (before AI) + AI + chain (after AI) into one
    ordered list for logging / diagnostics. Each entry has
    ``type`` ``human`` or ``ai``. Human entries carry
    ``destination``/``timeout_sec``. Used by tests and the /voice
    routing helpers to reason about a single sequence without creating
    a new DB column — the two stored lists remain the source of truth.
    """
    unified: list[dict] = []
    for s in cascade_steps or []:
        dest = str(s.get("destination") or "").strip()
        if not E164_PATTERN.match(dest):
            continue
        try:
            timeout = int(s.get("timeout_sec") or DEFAULT_STEP_TIMEOUT_SEC)
        except (TypeError, ValueError):
            timeout = DEFAULT_STEP_TIMEOUT_SEC
        timeout = max(MIN_STEP_TIMEOUT_SEC, min(MAX_STEP_TIMEOUT_SEC, timeout))
        unified.append({"type": "human", "destination": dest, "timeout_sec": timeout, "label": dest})
    unified.append({"type": "ai", "label": "AI agent"})
    for tid in chain_ids or []:
        tool = tools_by_id.get(tid) if isinstance(tools_by_id, dict) else None
        if not isinstance(tool, dict):
            continue
        dest = str(tool.get("destination") or "").strip()
        if not E164_PATTERN.match(dest):
            continue
        try:
            timeout = int(tool.get("ring_timeout_sec") or DEFAULT_STEP_TIMEOUT_SEC)
        except (TypeError, ValueError):
            timeout = DEFAULT_STEP_TIMEOUT_SEC
        timeout = max(MIN_STEP_TIMEOUT_SEC, min(MAX_STEP_TIMEOUT_SEC, timeout))
        unified.append({"type": "human", "destination": dest, "timeout_sec": timeout, "label": tool.get("name") or tid, "tool_id": tid})
    return unified


def transfer_fallback_url(
    public_url: str,
    agent_id: str,
    remaining_ids: list[str],
    tenant_id: str | None = None,
) -> str:
    """Action URL for a chain <Dial>. Same stateless contract as the
    cascade callback: remaining tool ids ride in the query so
    /voice/transfer-fallback can Dial the next one (or fall back to
    the AI stream when the chain is exhausted)."""
    qs = {"agent_id": agent_id, "remaining": ",".join(remaining_ids or [])}
    if tenant_id:
        qs["tenant_id"] = tenant_id
    return f"{public_url.rstrip('/')}/voice/transfer-fallback?{urllib.parse.urlencode(qs)}"


if __name__ == "__main__":  # smoke
    steps, err = validate_cascade([
        {"destination": "+15550001111", "timeout_sec": 25},
        {"destination": "not-a-number", "timeout_sec": 10},
    ])
    assert steps is None and "E.164" in err
    steps, err = validate_cascade([{"destination": "+15550001111"}])
    assert err is None and steps == [
        {"destination": "+15550001111", "timeout_sec": DEFAULT_STEP_TIMEOUT_SEC}
    ]
    # lenient parser drops the bad step instead of raising
    assert parse_cascade([{"destination": "bad"}, {"destination": "+15550002222", "timeout_sec": 99}]) == [
        {"destination": "+15550002222", "timeout_sec": MAX_STEP_TIMEOUT_SEC}
    ]
    chain, err = validate_transfer_chain(["t1", "t2", "t1", ""])
    assert chain is None and "tool id" in err
    chain, err = validate_transfer_chain(["t1", "t2", "t1"])
    assert err is None and chain == ["t1", "t2"]
    tools = {
        "t1": {"name": "Sales", "destination": "+15550001111", "ring_timeout_sec": 25},
        "t2": {"name": "NoDest", "destination": "", "ring_timeout_sec": 10},
        "t3": {"name": "Support", "destination": "+15550003333"},
    }
    # invoked not in chain → invoked + chain
    got = build_transfer_chain("t3", ["t1", "t2"], tools)
    assert [g["id"] for g in got] == ["t3", "t1"], got
    assert got[0]["timeout_sec"] == DEFAULT_STEP_TIMEOUT_SEC
    assert got[1]["timeout_sec"] == 25
    # invoked in chain → suffix from its position (explicit skip)
    tools2 = {
        "caf": {"name": "Cafeteria", "destination": "+15550001111", "ring_timeout_sec": 20},
        "rec": {"name": "Recruiting", "destination": "+15550002222", "ring_timeout_sec": 20},
    }
    got2 = build_transfer_chain("rec", ["caf", "rec"], tools2)
    assert [g["id"] for g in got2] == ["rec"], got2
    got3 = build_transfer_chain("caf", ["caf", "rec"], tools2)
    assert [g["id"] for g in got3] == ["caf", "rec"], got3
    print("transfer_cascade: OK")
