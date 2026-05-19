from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from app.features.preprocessors import FieldInfo
from app.models.base import (
    BaseModel,
    CohortExplanation,
    DetectorResult,
    FieldContribution,
)
from app.models.half_space_trees.model_slice import ModelSlice
from app.models.half_space_trees.stats_window import StatsWindow

class HalfSpaceTreesDetector(BaseModel):
    """
    River HalfSpaceTrees anomaly detection model.

    With sliding_steps=1 (default) a single ModelSlice is used — the original
    two-buffer batch behaviour.

    With sliding_steps=N, N ModelSlices are maintained in staggered phases, each
    offset by window_size // N events.  Every window_size / N events some slice
    just completed a pivot, so the freshest available reference is at most
    window_size / N events old instead of window_size.
    """

    PREPROCESSOR_TYPE_DEFAULTS: dict[str, str] = {
        "numeric":         "StandardScaler",
        "boolean":         "PassThrough",
        "str_categorical": "OneHotEncoder",
        "str_identifier":  "OneHotHashEncoder",
        "str_text":        "NotSupported"
    }

    def __init__(
        self,
        n_trees: int = 25,
        height: int | None = None,
        window_size: int = 250,
        sliding_steps: int = 1,
        warmup_count: int = 0,
    ) -> None:
        if sliding_steps < 1 or sliding_steps > window_size:
            raise ValueError(f"sliding_steps must be between 1 and window_size ({window_size})")

        self.n_trees = n_trees
        self.height = height if height is not None else self._default_height(window_size)
        self.window_size = window_size
        self.sliding_steps = sliding_steps
        self._phase_step = window_size // sliding_steps

        self._slices: list[ModelSlice] = [
            ModelSlice(n_trees=n_trees, height=self.height, window_size=window_size, seed=42 + k)
            for k in range(sliding_steps)
        ]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # BaseModel                                                            #
    # ------------------------------------------------------------------ #

    def train(self, features: dict[FieldInfo, list[float]], n_learned: int) -> None:
        with self._lock:
            for k, slc in enumerate(self._slices):
                if n_learned > k * self._phase_step:
                    slc.learn_one(features)

    def score(
        self,
        features: dict[FieldInfo, list[float]],
        flat: dict[str, Any],
        explain: bool,
    ) -> DetectorResult:
        if not self._is_trained():
            return DetectorResult(score=None)

        with self._lock:
            raw = self._recent_model_slice().score_one(features)
        s = float(np.clip(raw * 100.0, 0.0, 100.0))

        active_slice = self._recent_model_slice()
        explanation = self._build_explanation(features, flat, active_slice.r_window, s) if explain else None
        return DetectorResult(
            score=s,
            properties={"reference_age": active_slice.recency},
            explanation=explanation,
        )

    def get_state(self) -> Any:
        return self._slices

    def set_state(self, state: Any) -> None:
        self._slices = state

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _is_trained(self) -> bool:
        return self._slices[0].is_ready

    def _build_explanation(
        self,
        features: dict[FieldInfo, list[float]],
        flat: dict[str, Any],
        r_window: StatsWindow,
        original_score: float,
    ) -> CohortExplanation:
        groups: dict[str, list[FieldInfo]] = {}
        for fi in features:
            groups.setdefault(fi.original, []).append(fi)

        baseline: dict[FieldInfo, list[float]] = {
            fi: [r_window.mean(f"{fi.unique_key}__{i}") for i in range(len(values))]
            for fi, values in features.items()
        }
        baseline_score = self._score_features(baseline)

        contributors = []
        for original_field, fis in groups.items():
            isolated = dict(baseline)
            for fi in fis:
                isolated[fi] = features[fi]
            delta = self._score_features(isolated) - baseline_score
            contributors.append(FieldContribution(
                field=original_field,
                value=flat.get(original_field),
                delta=delta,
                preprocessed={fi.unique_key: features[fi] for fi in fis},
            ))
        contributors.sort(key=lambda c: abs(c.delta), reverse=True)
        return CohortExplanation(features=contributors, baseline_score=round(baseline_score, 4))

    def _score_features(self, features: dict[FieldInfo, list[float]]) -> float:
        with self._lock:
            raw = self._recent_model_slice().score_one(features)
        return float(np.clip(raw * 100.0, 0.0, 100.0))

    def _recent_model_slice(self) -> ModelSlice | None:
        return min(
            (slc for slc in self._slices if slc.is_ready),
            key=lambda slc: slc.recency,
            default=None,
        )

    def _default_height(self, window_size: int) -> int:
        """Max height where average leaf mass stays above River's size_limit threshold.

        Derived from: window_size / 2^height >= 0.1 * window_size → 2^height <= 10.
        """
        return max(1, int(math.log2(window_size / 10)))