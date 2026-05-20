from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any

from app.features.preprocessors import FieldInfo


@dataclass
class FeatureResult:
    field: str
    value: Any
    preprocessed: dict[str, list[float]]
    raw: dict[str, str] = dc_field(default_factory=dict)


@dataclass
class FieldContribution:
    field: str
    value: Any
    delta: float | None
    preprocessed: dict[str, list[float]]
    raw: dict[str, str] = dc_field(default_factory=dict)


@dataclass
class CohortExplanation:
    features: list[FieldContribution]
    baseline_score: float | None = 0.0


@dataclass
class DetectorResult:
    score: float | None
    properties: dict[str, Any] = dc_field(default_factory=dict)
    explanation: CohortExplanation | None = None
    score_label: str = ""


class BaseModel(ABC):
    """Abstract base class for anomaly detection algorithm implementations.

    Subclasses declare PREPROCESSOR_TYPE_DEFAULTS to control which preprocessor
    is selected for each feature type.  The Detector owns the preprocessing
    pipeline and calls these methods with already-preprocessed features.
    """

    PREPROCESSOR_TYPE_DEFAULTS: dict[str, str] = {}

    @abstractmethod
    def train(self, features: dict[FieldInfo, list[float]], n_learned: int) -> None:
        """Update the model with one preprocessed feature vector.

        n_learned is the Detector's total event count after this sample —
        models that use staggered slices (sliding_steps > 1) use it to
        determine which slices are active.
        """
        ...

    @abstractmethod
    def score(
        self,
        features: dict[FieldInfo, list[float]],
        flat: dict[str, Any],
        explain: bool,
    ) -> DetectorResult:
        """Score a preprocessed feature vector and return a DetectorResult."""
        ...

    @abstractmethod
    def get_state(self) -> Any:
        """Return picklable model-specific state (no lock needed — Detector serialises)."""
        ...

    @abstractmethod
    def set_state(self, state: Any) -> None:
        """Restore from the value returned by get_state."""
        ...
