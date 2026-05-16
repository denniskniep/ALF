from __future__ import annotations

import re
from typing import Any

from app.config import FieldConfig
from app.features.extractors.base import FieldHandler, SubFields


class SemverHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> SubFields:
        if not isinstance(value, str):
            raise TypeError(f"{key}: expected str, got {type(value).__name__}")
        m = re.match(r"(\d+)(?:\.(\d+)(?:\.(\d+))?)?", value.strip())
        if not m:
            raise ValueError(f"{key}: cannot parse {value!r} as semver")
        items: dict[str, tuple[Any, FieldConfig]] = {
            f"{key}.major": (float(m.group(1)), FieldConfig(type="numeric")),
        }
        if m.group(2):
            items[f"{key}.minor"] = (float(m.group(2)), FieldConfig(type="numeric"))
        if m.group(3):
            items[f"{key}.patch"] = (float(m.group(3)), FieldConfig(type="numeric"))
        return SubFields(items)
