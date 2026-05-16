from __future__ import annotations

import math
from typing import Any

from app.features.preprocessors.base import FieldInfo

_CLIP = (-5.0, 5.0)


class StandardScaler:
    """Online z-score standardizer using Welford's incremental algorithm.

    learn_pre_transform() updates the running mean and variance so that
    transform() immediately reflects the current event — matching the

    Returns 0.0 when fewer than 2 samples have been seen (no variance yet).

    Output is clipped to (-5, 5) so that extreme outliers don't land outside


    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # Welford's sum-of-squared-deviations

    def learn_pre_transform(self, key: str, value: Any) -> None:
        v = float(value)
        self._n += 1
        delta = v - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (v - self._mean)  # uses post-update mean

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        v = float(value)
        if self._n < 2 or self._m2 == 0.0:
            z = 0.0
        else:
            std = math.sqrt(self._m2 / self._n)
            z = (v - self._mean) / std
        lo, hi = _CLIP
        return {FieldInfo(original=key, limits=_CLIP, preprocessor="StandardScaler"): max(lo, min(hi, z))}
