from __future__ import annotations

import hashlib
from typing import Any

from app.features.preprocessors.base import FieldInfo


class HashIndex:
    """Encodes categorical string values as hash-bucketed integer indices.

    Like LabelIndex but stateless: the index is deterministic hash(value) % n_features
    rather than an incrementally assigned integer. Useful for high-cardinality unbounded
    fields where the full vocabulary is unknown — no vocabulary dict is maintained.

    String values are hashed as f"{key}={value}".
    Use seed via MD5 for cross-session determinism.
    """

    def __init__(self, n_features: int = 2_000, seed: int = 0) -> None:
        self._n = n_features
        self._seed_prefix = f"{seed}:".encode()

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, float]:
        feature = f"{key}={value}" if isinstance(value, str) else key
        idx = self._hash(feature)
        return {FieldInfo(original=key, limits=(0.0, float(self._n)), preprocessor="HashIndex"): float(idx)}

    def _hash(self, feature: str) -> int:
        data = self._seed_prefix + feature.encode()
        return int.from_bytes(hashlib.md5(data).digest(), "little") % self._n

    @property
    def vocab_size(self) -> int:
        return self._n
