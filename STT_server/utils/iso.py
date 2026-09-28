"""ISO-8601 serialization for row mappers.

One rule, seven call sites. Before this module every db_*.py row mapper
open-coded its own variant of:

    out[k] = out[k].isoformat() + "Z"

which is wrong for a tz-aware datetime. ``isoformat()`` already emits the
offset, so the result was::

    2026-09-28T12:34:56.789000+00:00Z

and ``Date.parse`` / ``new Date(...)`` return **NaN** for that string. The
browser cannot read it, so every timestamp the backend had correctly
stored rendered as "never" / "Never tested" in the admin UI — while the
same values were fine in the database. Postgres returns tz-aware
datetimes, so this hits production and not the JSON-file dev fallback
(where the value is already a clean ``...Z`` string written by
``_now_iso``).

Fix: append ``Z`` only when ``isoformat()`` left the offset off, i.e. for
a naive datetime. Aware datetimes keep their ``+00:00``, which the
browser parses correctly.

Do NOT reintroduce ``.isoformat() + "Z"`` in a new row mapper — import
``iso_utc`` from here instead.
"""
from __future__ import annotations

__all__ = ["iso_utc"]


def iso_utc(value):
    """Return ``value`` as an ISO string a browser can parse.

    Datetimes are normalized (the only case that was ever broken).
    Anything else — ``None``, an already-serialized string, a number —
    is returned untouched, which is what every call site did before:
    they guarded with ``hasattr(v, "isoformat")`` and left non-datetime
    values alone.
    """
    if not hasattr(value, "isoformat"):
        return value
    iso = value.isoformat()
    if iso.endswith("Z"):
        return iso
    # a tz-aware isoformat ends in "+00:00" / "-05:00"; the offset sign
    # sits 6 characters from the end. Only naive values need the Z.
    return iso if len(iso) > 10 and iso[-6:-5] in ("+", "-") else iso + "Z"
