from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo


class FrequencyEncoder:
    """Maps string → observed frequency in [0, 1].

    Unseen values map to 0.0 — a direct novelty signal.
    Default for str_categorical (finite, low-cardinality sets like country codes).
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    # we need to do the counter increases post transform, because otherwise new values would not get 0!
    def learn_post_transform(self, key: str, value: Any) -> None:
        s = str(value)
        self._counts[s] = self._counts.get(s, 0) + 1

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        s = str(value)
        value = self._counts.get(s, 0)
        if value == 0:
            return {FieldInfo(key, preprocessor="FrequencyEncoder"): float(0)}

        total = sum(self._counts.values())
        if total == 0:
            raise ValueError(f"Total can not be 0 (devide by 0)")

        return {FieldInfo(key, preprocessor="FrequencyEncoder"): float(value / total)}
