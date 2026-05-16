from __future__ import annotations

import hashlib
from typing import Any

from app.features.preprocessors.base import FieldInfo


class OneHotHashEncoder:
    """Maps high-cardinality string values to a fixed-width sparse vector via the hashing trick.

    String values are hashed as f"{key}={value}".
    Use seed via MD5 for cross-session determinism.

    transform() produces one FieldInfo containing bucket=hash(value) % n_features with value 1.0.
    """

    def __init__(self, n_features: int = 1_048_576, seed: int = 0) -> None:
        self._n = n_features
        self._seed_prefix = f"{seed}:".encode()

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        feature = f"{key}={value}" if isinstance(value, str) else key
        bucket = self._hash(feature)
        return {FieldInfo(original=key, bucket=str(bucket), preprocessor="OneHotHashEncoder"): 1.0}

    def _hash(self, feature: str) -> int:
        data = self._seed_prefix + feature.encode()
        return int.from_bytes(hashlib.md5(data).digest(), "little") % self._n
