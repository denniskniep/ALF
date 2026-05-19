from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from dataclasses import dataclass, field

from app.config import AppConfig, CohortConfig
from app.utils import get_nested
from app.models.detector import Detector
from app.models.base import FeatureResult, CohortExplanation
from app.models import factory
from app.orchestration import labels

logger = logging.getLogger(__name__)

_MODEL_TYPE = "cohort"

@dataclass
class IngestResult:
    name: str
    features: list[FeatureResult]

@dataclass
class ScoringResult:
    composite_score: float | None
    status: str
    cohort_scores: list[CohortScoreResult]

@dataclass
class CohortScoreResult:
    name: str
    key: dict[str, str]
    weight: float
    score: float | None
    score_label: str
    properties: dict[str, Any] = field(default_factory=dict)
    explanation: CohortExplanation | None = field(default=None)


def _cohort_key(cohort_cfg: CohortConfig, payload: dict[str, Any]) -> str:
    from app.features.extractor import get_nested
    parts = [str(get_nested(payload, f, "__missing__")) for f in cohort_cfg.fields]
    return f"{cohort_cfg.name}::{'::'.join(parts)}"


class LRUCache:
    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Detector] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Detector | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, model: Detector) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = model
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class ModelRegistry:
    """
    Owns all model instances for every cohort (as per config.yml - including global, user, etc. ).

    Global  = CohortConfig(fields=[])       → single model, always in RAM
    User    = CohortConfig(lru_size=N)      → LRU-backed by SQLite
    Cohorts = CohortConfig(fields=[...])    → all instances in RAM
    """

    def __init__(self, config: AppConfig, store) -> None:
        self._config = config
        self._store = store
        self._lock = threading.Lock()

        self._lru_caches: dict[str, LRUCache] = {
            c.name: LRUCache(c.lru_size)
            for c in config.cohorts if c.lru_size is not None
        }
        self._cohorts: dict[str, Detector] = {}
        self._preload()

    # ------------------------------------------------------------------ #
    # Public: get models                                                   #
    # ------------------------------------------------------------------ #

    def get_all_cohort_detectors(
        self,
        payload: dict[str, Any],
    ) -> list[tuple[CohortConfig, str, Detector]]:
        return [
            (cfg, key := _cohort_key(cfg, payload), self._get_model(cfg, key))
            for cfg in self._config.cohorts
        ]

    # ------------------------------------------------------------------ #
    # Public: ingest                                                       #
    # ------------------------------------------------------------------ #

    def ingest(self, payload: dict[str, Any]) -> list[IngestResult]:
        results = []
        for cohort_cfg in self._config.cohorts:
            key = _cohort_key(cohort_cfg, payload)
            model = self._get_model(cohort_cfg, key)
            features = model.learn_one(payload)
            self._persist(model, key)
            results.append(IngestResult(name=cohort_cfg.name, features=features))
        return results

    def score(
            self,
            payload: dict[str, Any],
            explain: bool = False,
    ) -> ScoringResult:
        entries = self.get_all_cohort_detectors(payload)

        cohort_results = [
            (cohort_cfg, det, det.score(payload, explain=explain))
            for cohort_cfg, _key, det in entries
        ]

        cohort_scores = [
            CohortScoreResult(
                name=cfg.name,
                key={f: str(get_nested(payload, f, "__missing__")) for f in cfg.fields},
                weight=cfg.weight,
                score=r.score,
                score_label=r.score_label,
                properties=r.properties,
                explanation=r.explanation if explain else None,
            )
            for cfg, det, r in cohort_results
        ]

        trained = [(cs.weight, cs.score) for cs in cohort_scores if cs.score is not None]
        if not trained:
            return ScoringResult(composite_score=None, status="insufficient_data", cohort_scores=cohort_scores)

        total_weight = sum(w for w, _ in trained)
        composite = sum(w * s for w, s in trained) / total_weight
        status = labels.score_label(composite)

        return ScoringResult(composite_score=composite, status=status, cohort_scores=cohort_scores)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_model(self, cohort_cfg: CohortConfig, key: str) -> Detector:
        if cohort_cfg.lru_size is not None:
            lru = self._lru_caches[cohort_cfg.name]
            model = lru.get(key)
            if model is None:
                model = self._load_or_create(cohort_cfg, key)
                lru.put(key, model)
            return model
        with self._lock:
            if key not in self._cohorts:
                self._cohorts[key] = self._load_or_create(cohort_cfg, key)
        return self._cohorts[key]

    def _make_model(self, cohort_cfg: CohortConfig) -> Detector:
        feature_cfg = self._config.features.get(cohort_cfg.features_config)
        if feature_cfg is None:
            raise ValueError(f"Unknown features_config: {cohort_cfg.features_config!r}")
        return factory.create_model(
            cohort_cfg.model,
            feature_cfg,
            name=cohort_cfg.name,
            warmup_count=cohort_cfg.warmup_count,
        )

    def _load_or_create(self, cohort_cfg: CohortConfig, key: str) -> Detector:
        model = self._make_model(cohort_cfg)
        blob = self._store.load(_MODEL_TYPE, key)
        if blob is not None:
            model.set_state(blob)
            logger.debug("Loaded %s from store (%d samples)", key, model.sample_count)
        return model

    def _persist(self, model: Detector, key: str) -> None:
        self._store.save(_MODEL_TYPE, key, model.get_state(), model.sample_count)

    def _cohort_cfg_for_key(self, key: str) -> CohortConfig | None:
        for c in self._config.cohorts:
            if key.startswith(f"{c.name}::"):
                return c
        return None

    def _preload(self) -> None:
        for key in self._store.list_keys(_MODEL_TYPE):
            cohort_cfg = self._cohort_cfg_for_key(key)
            if cohort_cfg is not None and cohort_cfg.lru_size is None:
                self._cohorts[key] = self._load_or_create(cohort_cfg, key)
