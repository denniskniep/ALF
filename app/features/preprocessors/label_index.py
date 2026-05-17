from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo


class LabelIndex:
    """Encodes categorical string values as incrementally-assigned integer indices.
       assigned values start at 1.
    """

    def __init__(self, expected_categories: int = 200) -> None:
        self._vocab: dict[str, int] = {}
        self._next: int = 1
        self._expected_categories = expected_categories

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, list[float]]:
        s = str(value)
        if s not in self._vocab:
            self._vocab[s] = self._next
            self._next += 1
        limits = (0.0, float(self._expected_categories))
        return {FieldInfo(original=key, limits=limits, preprocessor="LabelIndex"): [float(self._vocab[s])]}

    @property
    def vocab_size(self) -> int:
        return self._next
