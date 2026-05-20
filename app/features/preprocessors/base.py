from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FieldInfo:
    """Tracks both the original config field name and its preprocessed key.

    For single-output preprocessors (MinMaxScaler, StandardScaler, LabelIndex, …) bucket
    is empty and unique_key returns "original".  For multi-output preprocessors (OneHotEncoder,
    OneHotHashEncoder) bucket holds the string bucket id and unique_key returns
    "original__bucket".

    limits carries the preprocessor's expected output range so ModelSlice can lazily
    register it with HalfSpaceTrees before the first learn_one call.

    preprocessor is the class name of the FieldPreprocessor that produced this value
    (e.g. "LabelIndex", "StandardScaler").  Downstream models use it to route features
    — AutoencoderDetector checks fi.preprocessor to decide embedding vs numeric pathway.

    raw is optional human-readable metadata set by preprocessors.
    """
    original: str
    bucket: str = ""
    limits: tuple[float, float] | None = None
    preprocessor: str = ""
    raw: str | None = field(default=None, compare=False, hash=False)

    @property
    def unique_key(self) -> str:
        return f"{self.original}__{self.bucket}" if self.bucket else self.original


@runtime_checkable
class FieldPreprocessor(Protocol):
    def learn_pre_transform(self, key: str, value: Any) -> None: ...
    def learn_post_transform(self, key: str, value: Any) -> None: ...
    def transform(self, key: str, value: Any) -> dict[FieldInfo, list[float]]: ...
