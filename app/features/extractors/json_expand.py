from __future__ import annotations

import json
from typing import Any

from app.features.extractors.base import FieldHandler, SubFields


def _flatten_dict(prefix: str, obj: dict) -> dict[str, tuple[Any, None]]:
    """Recursively flatten a nested dict into dot-notation leaf entries."""
    result: dict[str, tuple[Any, None]] = {}
    for k, v in obj.items():
        full_key = f"{prefix}.{k}"
        if isinstance(v, dict):
            result.update(_flatten_dict(full_key, v))
        else:
            result[full_key] = (v, None)
    return result


class JsonExpandHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> SubFields:
        if not isinstance(value, str):
            raise TypeError(f"{key}: expected str, got {type(value).__name__}")
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"{key}: cannot parse {value!r} as JSON")
        if not isinstance(parsed, dict):
            raise ValueError(f"{key}: expected JSON object, got {type(parsed).__name__}")
        return SubFields(_flatten_dict(key, parsed))
