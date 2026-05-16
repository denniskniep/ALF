from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo


class PassThrough:
    """Value is already a float e.g. Boolean: value is already 0/1."""

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        return {FieldInfo(key, preprocessor="PassThrough"): float(value)}
