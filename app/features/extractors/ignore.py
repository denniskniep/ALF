from __future__ import annotations

from typing import Any

from app.features.extractors.base import FieldHandler


class IgnoreHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> None:
        return None
