from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo


class OneHotEncoder:
    """Online one-hot encoder with a fixed number of buckets.

    On first encounter of a value, the next free bucket index is reserved for it.

    transform() produces n_categories binary features:
    1.0 in the bucket for the active value,
    0.0 in all others.

    Values that arrive after the vocabulary is full throw an error.
    """

    def __init__(self, n_categories: int = 10) -> None:
        self._n = n_categories
        self._vocab: dict[str, int] = {}
        self._next: int = 1

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        s = str(value)
        if s not in self._vocab:
            if self._next > self._n:
                raise ValueError(
                    f"For '{key}' are only {self._n} categories available. Value: '{value}' can not be processed!")
            self._vocab[s] = self._next
            self._next += 1

        active = self._vocab.get(str(value))
        return {
            FieldInfo(original=key, bucket=str(i), preprocessor="OneHotEncoder"): 1.0 if i == active else 0.0
            for i in range(1, self._n + 1)
        }
