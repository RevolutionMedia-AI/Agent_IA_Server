"""
Path-traversal guard for files derived from user-controlled strings.

Background
----------
The JSON-file backend (db_*.py) and several TTS / playback services
build filesystem paths from identifiers that *should* be server-issued
(user_id from JWT, session_key from Python's id(), call_sid from
Twilio) but the scanner can't prove that. To eliminate the file-
inclusion class entirely we:

  1. Reject any identifier that contains a path separator, a null byte,
     or a parent-traversal segment.
  2. Resolve the final path and verify it lives under an allowed base
     directory. This catches symlink games and absolute-path injection.

The helpers are designed to be the single funnel for "build a file
path from a string" in this repo. Callers do:

    from STT_server.utils.safe_path import safe_join, sanitize_id

    fname = safe_join(DATA_DIR, "settings", sanitize_id(user_id) + ".json")
"""
from __future__ import annotations

import os
import re
from pathlib import Path


# Conservative charset for user_id / session_key: alnum, dash, underscore,
# dot. Anything else gets rejected. This kills "../", "..\\", null bytes,
# absolute paths ("/etc/passwd") and shell metacharacters in one shot.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class UnsafePathError(ValueError):
    """Raised when a path or identifier fails the path-traversal check."""


def sanitize_id(value: str, *, field: str = "id") -> str:
    """Validate a user-derived identifier (user_id, session_key, etc.).

    Returns the string unchanged if it matches the conservative charset.
    Raises UnsafePathError otherwise.

    The check is intentionally strict — if a real identifier ever
    contains an exotic char (Unicode name, etc.) it should be hashed
    or escaped at the point of generation, not smuggled through the
    filesystem.
    """
    if not isinstance(value, str):
        raise UnsafePathError(f"{field} must be a string")
    if not _SAFE_ID.fullmatch(value):
        raise UnsafePathError(
            f"{field} contains forbidden characters: {value!r}"
        )
    if value in (".", ".."):
        raise UnsafePathError(f"{field} is a traversal segment: {value!r}")
    return value


def safe_join(base: str | os.PathLike, *parts: str) -> str:
    """Join `parts` under `base` and verify the result is inside `base`.

    Returns the absolute resolved path as a string. Raises
    UnsafePathError if the resolved path escapes `base`.
    """
    base_path = Path(base).resolve()
    target = base_path.joinpath(*parts).resolve()
    try:
        target.relative_to(base_path)
    except ValueError as exc:
        raise UnsafePathError(
            f"Path escapes base directory: {target} not under {base_path}"
        ) from exc
    return str(target)
