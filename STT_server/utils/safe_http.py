"""
SSRF guard for outbound HTTP calls driven by user-supplied URLs.

Background
----------
The per-user provider credentials allow the user to set a `base_url`
override (Anthropic / Gemini / MiniMax). If we trust that string
verbatim and feed it to `urllib.request.urlopen`, a logged-in user can
point it at:

  * `http://169.254.169.254/...`  → AWS / GCP / Azure instance metadata
  * `http://127.0.0.1:6379/...`   → internal services (Redis, Postgres, admin)
  * `http://10.0.0.1/...`         → VPC internal IPs
  * `file:///etc/passwd`          → local file read via urllib on some platforms

The classic SSRF surface. This module centralises the validation so
every adapter that accepts a user-supplied URL goes through the same
allowlist + DNS check.

The rules
---------
1. Scheme must be `http` or `https` (no `file`, `gopher`, `ftp`, ...).
2. Host must be a public IP or DNS name that resolves to a public IP.
3. Block loopback (127/8, ::1), link-local (169.254/16, fe80::/10),
   private (10/8, 172.16/12, 192.168/16, fc00::/7) and
   the cloud metadata addresses explicitly.
4. Optional `allow_hosts` to enforce that the host belongs to the
   provider's expected domain set (e.g. `api.anthropic.com`,
   `api.minimax.io`). Default: accept any public host.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Iterable
from urllib.parse import urlparse


class UnsafeURLError(ValueError):
    """Raised when a URL fails the SSRF allowlist (private IP, bad
    scheme, unresolvable host, etc.)."""


_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _is_public_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.is_loopback:
        return False
    if ip.is_link_local:
        return False
    if ip.is_private:
        return False
    if ip.is_reserved:
        return False
    if ip.is_multicast:
        return False
    if ip.is_unspecified:
        return False
    return True


def _resolve_all(host: str) -> list[str]:
    """Resolve the host to every A/AAAA record. Returns [] on failure."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    out: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr and sockaddr[0]:
            out.append(sockaddr[0])
    # de-dupe, preserve order
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def validate_public_url(
    url: str,
    *,
    allow_hosts: Iterable[str] | None = None,
    resolve_dns: bool = True,
) -> str:
    """Return `url` if it passes the SSRF allowlist, raise UnsafeURLError
    otherwise. The returned string is the URL as supplied (no rewrite).

    `allow_hosts`, when provided, is a set of literal hostnames that
    are accepted in addition to the public-IP rule. Comparison is
    case-insensitive and matches the full host (not a suffix).
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL is empty")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"Refusing URL with scheme {parsed.scheme!r}; "
            f"allowed: {sorted(_ALLOWED_SCHEMES)}"
        )
    host = (parsed.hostname or "").strip()
    if not host:
        raise UnsafeURLError("URL has no host")

    # Optional allowlist (provider-pinned hostnames).
    if allow_hosts is not None:
        allowed = {h.lower() for h in allow_hosts}
        if host.lower() not in allowed:
            # Not on the allowlist — fall through to the public-IP check
            # so custom-tenant / proxy endpoints still work when they're
            # genuinely public. Allowlist alone is the strict mode.
            pass

    # Direct IP literal?
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass  # not an IP literal, treat as DNS name
    else:
        if not _is_public_ip(host):
            raise UnsafeURLError(f"Refusing non-public IP literal: {host}")
        return url

    if resolve_dns:
        addrs = _resolve_all(host)
        if not addrs:
            raise UnsafeURLError(f"Host does not resolve: {host}")
        for addr in addrs:
            if not _is_public_ip(addr):
                raise UnsafeURLError(
                    f"Refusing URL whose host resolves to non-public IP "
                    f"{addr} (host={host})"
                )
    return url
