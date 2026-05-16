from __future__ import annotations

from typing import Any


def get_nested(obj: dict, path: str, default: Any = None) -> Any:
    """Resolve a dot-notation path into a nested dict.

    Tries the full path as a literal key first, then falls back to
    step-by-step navigation so both flat keys like "a.b.c" and nested
    dicts like {"a": {"b": {"c": ...}}} are handled.
    """
    if isinstance(obj, dict) and path in obj:
        return obj[path]
    current = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current
