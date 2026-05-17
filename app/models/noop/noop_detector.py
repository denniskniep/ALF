from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo
from app.models.base import BaseModel, DetectorResult
from app.orchestration import labels


class NoOpDetector(BaseModel):
    """No-operation model for testcases that exercise feature extraction or preprocessing only."""

    PREPROCESSOR_TYPE_DEFAULTS: dict[str, str] = {
        "numeric":         "NoOp",
        "boolean":         "NoOp",
        "str_categorical": "NoOp",
        "str_identifier":  "NoOp",
        "str_text":        "SentenceTransformerEncoder",
    }

    def train(self, features: dict[FieldInfo, list[float]], n_learned: int) -> None:
        pass

    def score(
        self,
        features: dict[FieldInfo, list[float]],
        flat: dict[str, Any],
        explain: bool,
    ) -> DetectorResult:
        return DetectorResult(score=None, score_label=labels.INSUFFICIENT_DATA)

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any) -> None:
        pass
