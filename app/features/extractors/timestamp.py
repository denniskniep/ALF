from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import FieldConfig
from app.features.extractors.base import FieldHandler, SubFields


class TimestampHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> SubFields:
        if not isinstance(value, str):
            raise TypeError(f"{key}: expected str, got {type(value).__name__}")
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return SubFields({
                f"{key}.hour":              (float(ts.hour),                                  FieldConfig(type="numeric")),
                f"{key}.dayofweek":         (float(ts.weekday()),                             FieldConfig(type="numeric")),
                f"{key}.is_weekend":        (float(ts.weekday() >= 5),                        FieldConfig(type="numeric")),
            })
        except (ValueError, TypeError):
            raise ValueError(f"{key}: cannot parse {value!r} as datetime")
