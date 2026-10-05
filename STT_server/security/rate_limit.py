"""Fixed-window rate limiter for the internal n8n endpoints.

Why it exists: /internal/integrations/{id}/* is authenticated by a bearer
token, and there was no throttle anywhere in the project. `compare_digest`
stops a timing leak on the token comparison, but it does nothing about
volume: an attacker could present guesses forever at line rate.

Scope: fixed window per client IP. A window is 60 s and the budget is
100 requests. Deliberately NOT applied to the public API — that would put
a shared quota in front of every user of one IP (a NAT'd office, a
carrier NAT behind Twilio) and throttle paying tenants.

Two caveats stated plainly rather than hidden:

  * The counter is per PROCESS. On Railway with more than one replica the
    effective budget is 100 x replicas. Still a brake on a brute force,
    which is the goal; it is not a global quota.
  * Memory is bounded by construction: entries are pruned on write, so
    the map holds at most (requests within the last window across
    distinct IPs) and nothing persists across restarts.

`Retry-After` is returned on 429 so a well-behaved client backs off
instead of hammering.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

# 100 requests / 60 s per client IP, per the operator's setting.
# Overridable because a busy tenant's n8n may legitimately need more,
# and raising it must not require a code deploy.
DEFAULT_INTERNAL_RATE_LIMIT = 100
DEFAULT_INTERNAL_RATE_WINDOW_SEC = 60.0


def _limit() -> int:
    raw = os.environ.get("INTERNAL_RATE_LIMIT", "").strip()
    if not raw:
        return DEFAULT_INTERNAL_RATE_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_INTERNAL_RATE_LIMIT


def _window() -> float:
    raw = os.environ.get("INTERNAL_RATE_WINDOW_SEC", "").strip()
    if not raw:
        return DEFAULT_INTERNAL_RATE_WINDOW_SEC
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_INTERNAL_RATE_WINDOW_SEC


_hits: dict[str, deque[float]] = {}
_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


def _prune(now: float, window: float) -> None:
    """Drop keys with no hits inside the window. Called under the lock."""
    stale = [k for k, dq in _hits.items() if not dq or dq[-1] <= now - window]
    for k in stale:
        _hits.pop(k, None)


def hit(client_key: str) -> tuple[bool, int, float]:
    """Record one request.

    Returns (allowed, remaining, retry_after_seconds). retry_after is 0
    when allowed.
    """
    limit = _limit()
    window = _window()
    now = _now()
    with _lock:
        _prune(now, window)
        dq = _hits.get(client_key)
        if dq is None:
            dq = deque()
            _hits[client_key] = dq
        dq.append(now)
        if len(dq) <= limit:
            return True, limit - len(dq), 0.0
        # Oldest hit in the window is when a slot frees up.
        oldest = dq[0]
        retry_after = max(0.0, (oldest + window) - now)
        return False, 0, retry_after


def reset() -> None:
    """Clear all counters. For tests."""
    with _lock:
        _hits.clear()


def snapshot() -> dict[str, int]:
    """Current window count per key. Diagnostics only, never a response."""
    now = _now()
    with _lock:
        _prune(now, _window())
        return {k: len(v) for k, v in _hits.items()}


def _client_key(request: Request) -> str:
    """Identify the caller.

    Prefers the leftmost X-Forwarded-For entry, because in production the
    app sits behind Railway's proxy and every request otherwise shares
    127.0.0.1 — which would make the whole platform one bucket.

    XFF is client-controlled, so this only raises the cost of an attack;
    it does not weaken authentication, which is the bearer token.
    """
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",")[0].strip()[:64]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_internal_rate_limit(request: Request) -> None:
    """FastAPI dependency. Raises 429 once the window budget is spent.

    `request` must be annotated as Request — without it FastAPI treats the
    parameter as a query field and every call 422s.
    """
    allowed, _remaining, retry_after = hit(_client_key(request))
    if allowed:
        return
    from starlette.responses import JSONResponse

    raise HTTPException(
        status_code=429,
        detail=(
            "Rate limit exceeded for the internal integrations API "
            f"({_limit()} requests / {int(_window())}s). Retry in "
            f"{retry_after:.0f}s."
        ),
        headers={
            "Retry-After": str(max(1, int(retry_after + 0.999))),
        },
    )


# ── Self-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    reset()
    limit = _limit()
    window = _window()

    # Under the limit.
    for i in range(limit):
        allowed, remaining, retry = hit("1.2.3.4")
        assert allowed, i
        assert remaining == limit - i - 1, (i, remaining)
        assert retry == 0.0

    # Over the limit: blocked with a sane Retry-After.
    allowed, remaining, retry = hit("1.2.3.4")
    assert allowed is False
    assert remaining == 0
    assert 0 < retry <= window, retry

    # A different client has its own budget.
    allowed, _r, _t = hit("5.6.7.8")
    assert allowed is True

    # Refusal does not grow without bound.
    for _ in range(500):
        hit("1.2.3.4")
    assert snapshot()["1.2.3.4"] <= limit + 501

    # Stale keys are pruned, so memory does not grow forever.
    reset()
    hit("9.9.9.9")
    assert "9.9.9.9" in snapshot()
    _hits["9.9.9.9"].clear()
    _prune(_now() - window - 1, window)
    assert snapshot() == {}

    print(f"internal_rate_limit: OK (limit={limit}, window={int(window)}s)")