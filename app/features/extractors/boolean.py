from __future__ import annotations

from typing import Any

from app.features.extractors.base import FieldHandler


class BooleanHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> float:
        if not isinstance(value, bool):
            raise TypeError(f"{key}: expected bool, got {type(value).__name__}")
        return 1.0 if value else 0.0
