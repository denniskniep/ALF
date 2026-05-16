from __future__ import annotations

from app.config import FeatureConfig, ModelConfig
from app.models.autoencoder import AutoencoderDetector
from app.models.base import BaseModel
from app.models.detector import Detector
from app.models.half_space_trees import HalfSpaceTreesDetector
from app.models.noop import NoOpDetector

_REGISTRY: dict[str, type[BaseModel]] = {
    "HalfSpaceTrees": HalfSpaceTreesDetector,
    "Autoencoder":    AutoencoderDetector,
    "NoOp":           NoOpDetector,
}


def create_model(model_cfg: ModelConfig, feature_cfg: FeatureConfig, name: str) -> Detector:
    cls = _REGISTRY.get(model_cfg.name)
    if cls is None:
        raise ValueError(
            f"Unknown model: {model_cfg.name!r}. Available: {list(_REGISTRY.keys())}"
        )
    model = cls(**model_cfg.params)
    return Detector(model=model, name=name, feature_cfg=feature_cfg)
