from __future__ import annotations

from typing import Any


def get_device() -> str:
    """Return 'cuda' or 'cpu'. GPU is only used when allowGPU: true is set in config."""
    from app.config import is_gpu_allowed
    import torch
    return ("cuda" if torch.cuda.is_available() else "cpu") if is_gpu_allowed() else "cpu"


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
