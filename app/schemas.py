from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_serializer


class ScoreRequest(BaseModel):
    payload: dict[str, Any]


class IngestRequest(BaseModel):
    payload: dict[str, Any]


class FieldContributionOut(BaseModel):
    field: str
    value: Any
    delta: float | None
    preprocessed: dict[str, float]


class CohortExplanationOut(BaseModel):
    features: list[FieldContributionOut]
    baseline_score: float | None


class CohortScoreOut(BaseModel):
    name: str
    key: dict[str, str]
    weight: float
    anomaly_score: float | None
    score_label: str
    properties: dict[str, Any] = {}
    explanation: CohortExplanationOut | None = None


class ScoreResponse(BaseModel):
    accepted: bool = True
    composite_score: float | None = None
    status: str = ""
    identifiers: dict[str, Any] | None = None
    scores: list[CohortScoreOut] = []
    error: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        if d.get("error") is None:
            d.pop("error", None)
        if d.get("identifiers") is None:
            d.pop("identifiers", None)
        return d


class IngestFeatureOut(BaseModel):
    field: str
    value: Any
    preprocessed: dict[str, float]


class IngestCohortOut(BaseModel):
    name: str
    features: list[IngestFeatureOut]


class IngestResponse(BaseModel):
    accepted: bool
    identifiers: dict[str, Any] | None = None
    cohorts: list[IngestCohortOut] | None = None
    error: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        if d.get("error") is None:
            d.pop("error", None)
        if d.get("identifiers") is None:
            d.pop("identifiers", None)
        return d


class HealthResponse(BaseModel):
    status: str
    global_baseline_size: int
