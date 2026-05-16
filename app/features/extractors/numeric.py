from __future__ import annotations

from typing import Any

from app.features.extractors.base import FieldHandler


class NumericHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> float:
        # bool is a subclass of int in Python — reject it explicitly
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key}: expected int or float, got {type(value).__name__}")
        return float(value)
