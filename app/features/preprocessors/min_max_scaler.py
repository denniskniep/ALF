from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo


class MinMaxScaler:
    """Online min-max normalizer.

    learn_pre_transform() updates the observed min/max

    Before the first event the range is undefined;

    When min == max (only one distinct value seen) transform returns 0.0.

    Values outside the seen range produce outputs outside [0, 1];
    this is intentional — it signals a distribution shift to the downstream model.
    """

    def __init__(self) -> None:
        self._min: float | None = None
        self._max: float | None = None

    def learn_pre_transform(self, key: str, value: Any) -> None:
        v = float(value)
        if self._min is None:
            self._min = v
            self._max = v
        else:
            if v < self._min:
                self._min = v
            if v > self._max:
                self._max = v

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        v = float(value)
        if self._min is None:
            return {FieldInfo(original=key, preprocessor="MinMaxScaler"): float("nan")}
        if self._max == self._min:
            return {FieldInfo(original=key, preprocessor="MinMaxScaler"): 0.0}
        return {FieldInfo(original=key, preprocessor="MinMaxScaler"): (v - self._min) / (self._max - self._min)}
