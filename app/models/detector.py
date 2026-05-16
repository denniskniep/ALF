from __future__ import annotations

import pickle
import threading
from typing import Any

from app.config import FeatureConfig
from app.features.extractor import extract
from app.features.preprocessor import Preprocessor
from app.models.base import (
    BaseModel,
    CohortExplanation,
    DetectorResult,
    FieldContribution,
    FeatureResult,
)
from app.orchestration import labels


class Detector:
    """Orchestrates anomaly detection for one cohort.

    Owns the preprocessing pipeline and delegates model-specific train/score
    logic to a BaseModel instance.

    Threading: self._lock protects the Preprocessor and n_learned counter.
    The BaseModel manages its own locking for model-internal state.
    """

    def __init__(
        self,
        model: BaseModel,
        name: str = "",
        feature_cfg: FeatureConfig | None = None,
    ) -> None:
        self._model = model
        self.name = name
        self.feature_cfg = feature_cfg or FeatureConfig()
        self._preprocessor = Preprocessor(model.PREPROCESSOR_TYPE_DEFAULTS)
        self._n_learned: int = 0
        self._lock = threading.Lock()

    @property
    def sample_count(self) -> int:
        return self._n_learned

    def learn_one(self, payload: dict[str, Any]) -> list[FeatureResult]:
        extracted = extract(payload, self.feature_cfg)
        with self._lock:
            final = self._preprocessor.process(extracted, is_learning=True)
            self._n_learned += 1
            n_learned = self._n_learned
        if final:
            self._model.train(final, n_learned)
        return [
            FeatureResult(
                field=field,
                value=value,
                preprocessed={fi.unique_key: final[fi] for fi in final if fi.original == field},
            )
            for field, (value, _) in extracted.items()
        ]

    def score(self, payload: dict[str, Any], explain: bool = False) -> DetectorResult:
        extracted = extract(payload, self.feature_cfg)
        flat = {k: v for k, (v, _) in extracted.items()}
        with self._lock:
            final = self._preprocessor.process(extracted, is_learning=False)
        result = self._model.score(final, flat, explain)
        result.score_label = labels.score_label(result.score)
        if result.explanation is None:
            result.explanation = CohortExplanation(
                features=[FieldContribution(field=k, value=v, delta=None, preprocessed={}) for k, v in flat.items()],
                baseline_score=None,
            )
        return result

    def get_state(self) -> bytes:
        return pickle.dumps({
            "preprocessor": self._preprocessor,
            "n_learned": self._n_learned,
            "model_state": self._model.get_state(),
        })

    def set_state(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        self._preprocessor = state["preprocessor"]
        self._n_learned = state["n_learned"]
        self._model.set_state(state["model_state"])
