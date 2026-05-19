from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Query

from app.config import load_config, AppConfig
from app.debug import TrainDetectLogger
from app.orchestration.registry import ModelRegistry
from app.orchestration.store import ModelStore, create_store
from app.schemas import (
    CohortExplanationOut,
    CohortScoreOut,
    FieldContributionOut,
    IngestCohortOut,
    IngestFeatureOut,
    IngestRequest,
    IngestResponse,
    ScoreRequest,
    ScoreResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_config: AppConfig = cast(AppConfig, cast(object, None))
_store: ModelStore = cast(ModelStore, cast(object, None))
_registry: ModelRegistry = cast(ModelRegistry, cast(object, None))
_logger: TrainDetectLogger = cast(TrainDetectLogger, cast(object, None))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _store, _registry, _logger
    _config = load_config()
    _store = create_store(_config.database)
    _registry = ModelRegistry(_config, _store)
    _logger = TrainDetectLogger(enabled=_config.debug.dump_results)
    logger.info("Anomaly detection service started")
    yield
    logger.info("Anomaly detection service stopped")


app = FastAPI(title="ALF - Anomaly Log Finder", version="0.1.0", lifespan=lifespan)


def _extract_identifiers(payload: dict, paths: list[str]) -> dict:
    result = {}
    for path in paths:
        val: Any = payload
        for part in path.split("."):
            val = val.get(part) if isinstance(val, dict) else None
        result[path] = val
    return result


def ensure_ready() -> None:
    if _registry is None or _store is None or _config is None:
        raise HTTPException(status_code=500, detail="Not ready yet")


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    include_features: bool = Query(default=False),
) -> IngestResponse:
    ensure_ready()
    try:
        identifiers = _extract_identifiers(request.payload, _config.identifiers) if _config.identifiers else None
        ingest_results = _registry.ingest(request.payload)
        cohorts_out = [
            IngestCohortOut(
                name=r.name,
                features=[
                    IngestFeatureOut(field=f.field, value=f.value, preprocessed=f.preprocessed)
                    for f in r.features
                ],
            )
            for r in ingest_results
        ]

        response = IngestResponse(
            accepted=True,
            identifiers=identifiers,
            cohorts=cohorts_out
        )

        _logger.log_train(response)

        if not include_features:
            response.cohorts = None

        return response
    except Exception as exc:
        response = IngestResponse(accepted=False, error=str(exc))
        _logger.log_train(response)
        return response



@app.post("/score", response_model=ScoreResponse)
def score(
    request: ScoreRequest,
    explain: bool = Query(default=False),
) -> ScoreResponse:
    ensure_ready()
    try:
        identifiers = _extract_identifiers(request.payload, _config.identifiers) if _config.identifiers else None
        result = _registry.score(request.payload, explain=explain)
        response = ScoreResponse(
            accepted=True,
            composite_score=round(result.composite_score, 2) if result.composite_score is not None else None,
            status=result.status,
            identifiers=identifiers,
            scores=[
                CohortScoreOut(
                    name=r.name,
                    key=r.key,
                    weight=r.weight,
                    anomaly_score=r.score,
                    score_label=r.score_label,
                    properties=r.properties,
                    explanation=CohortExplanationOut(
                        features=[
                            FieldContributionOut(field=c.field, value=c.value, delta=c.delta, preprocessed=c.preprocessed)
                            for c in r.explanation.features
                        ],
                        baseline_score=r.explanation.baseline_score,
                    ) if r.explanation is not None else None,
                )
                for r in result.cohort_scores
            ],
        )
        _logger.log_score(response)
        return response
    except Exception as exc:
        response = ScoreResponse(accepted=False, error=str(exc))
        _logger.log_score(response)
        return response
