from __future__ import annotations

import pickle
import threading
from typing import Any

from app.config import FeatureConfig, FieldConfig
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

    Warmup phase (when warmup_count > 0): raw payloads are buffered without any
    preprocessing or model training. Once warmup_count events have been collected,
    the buffer is flushed via preprocessor.process_batch.

    Threading: self._lock protects the Preprocessor, counters, and warmup buffer.
    The BaseModel manages its own locking for model-internal state.
    """

    def __init__(
        self,
        model: BaseModel,
        name: str = "",
        feature_cfg: FeatureConfig | None = None,
        warmup_count: int = 0,
    ) -> None:
        self._model = model
        self.name = name
        self.feature_cfg = feature_cfg or FeatureConfig()
        self._preprocessor = Preprocessor(model.PREPROCESSOR_TYPE_DEFAULTS, warmup_count=warmup_count)
        self._n_learned: int = 0
        self._lock = threading.Lock()
        self._warmup_count: int = warmup_count
        self._warmup_buffer: list[dict[str, tuple[Any, FieldConfig]]] = []
        self._warmed_up: bool = warmup_count == 0

    @property
    def sample_count(self) -> int:
        if not self._warmed_up:
            return len(self._warmup_buffer)
        return self._n_learned

    def learn_one(self, payload: dict[str, Any]) -> list[FeatureResult]:
        extracted = extract(payload, self.feature_cfg)
        with self._lock:
            if not self._warmed_up:
                self._warmup_buffer.append(extracted)
                if len(self._warmup_buffer) < self._warmup_count:
                    return [
                        FeatureResult(field=field, value=value, preprocessed={})
                        for field, (value, _) in extracted.items()
                    ]
                extracted_to_process = list(self._warmup_buffer)
                self._warmup_buffer.clear()
                self._warmed_up = True
            else:
                extracted_to_process = [extracted]

        with self._lock:
            preprocessed = self._preprocessor.process_batch(extracted_to_process, is_learning=True)


        for i, final in enumerate(preprocessed):
            self._n_learned += 1
            self._model.train(final, self._n_learned)

        last_preprocessed = preprocessed[-1]
        return [
            FeatureResult(
                field=field,
                value=value,
                preprocessed={fi.unique_key: last_preprocessed[fi] for fi in last_preprocessed if fi.original == field},
            )
            for field, (value, _) in extracted.items()
        ]

    def score(self, payload: dict[str, Any], explain: bool = False) -> DetectorResult:
        extracted = extract(payload, self.feature_cfg)
        flat = {k: v for k, (v, _) in extracted.items()}
        with self._lock:
            if not self._warmed_up:
                return DetectorResult(
                    score=None,
                    score_label=labels.INSUFFICIENT_DATA,
                    explanation=CohortExplanation(
                        features=[FieldContribution(field=k, value=v, delta=None, preprocessed={}) for k, v in flat.items()],
                        baseline_score=None,
                    ),
                )
            final = self._preprocessor.process_batch([extracted], is_learning=False)[0]
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
            "warmup_count": self._warmup_count,
            "warmup_buffer": self._warmup_buffer,
            "warmed_up": self._warmed_up,
            "model_state": self._model.get_state(),
        })

    def set_state(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        self._preprocessor = state["preprocessor"]
        self._n_learned = state["n_learned"]
        self._warmup_count = state.get("warmup_count", 0)
        self._warmup_buffer = state.get("warmup_buffer", [])
        self._warmed_up = state.get("warmed_up", True)
        self._model.set_state(state["model_state"])

