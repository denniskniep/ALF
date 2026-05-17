from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PreprocessorConfig:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class CohortConfig:
    name: str
    fields: list[str]
    weight: float
    model: ModelConfig
    features_config: str
    lru_size: int | None = None


@dataclass
class DatabaseConfig:
    type: str = "memory"
    path: str | None = None


@dataclass
class FieldConfig:
    type: str
    preprocessor: PreprocessorConfig | None = None
    source: str | None = None


@dataclass
class FeatureConfig:
    fields: dict[str, FieldConfig] = field(default_factory=dict)


@dataclass
class DebugConfig:
    dump_results: bool = False


@dataclass
class AppConfig:
    cohorts: list[CohortConfig]
    database: DatabaseConfig
    features: dict[str, FeatureConfig]
    identifiers: list[str] = field(default_factory=list)
    debug: DebugConfig = field(default_factory=DebugConfig)
    allow_gpu: bool = False


_allow_gpu: bool = False


def is_gpu_allowed() -> bool:
    return _allow_gpu



def _parse_field_cfgs(raw_fields: dict) -> FeatureConfig:
    field_cfgs: dict[str, FieldConfig] = {}
    for fpath, cfg in raw_fields.items():
        if isinstance(cfg, str):
            fc = FieldConfig(type=cfg, source=fpath)
        else:
            preprocessor = None
            if "preprocessor" in cfg:
                p = cfg["preprocessor"]
                preprocessor = PreprocessorConfig(name=p["name"], params=p.get("params", {}))

            source = fpath
            if "source" in cfg:
                source = cfg["source"]

            fc = FieldConfig(type=cfg["type"], preprocessor=preprocessor, source=source)
        field_cfgs[fpath] = fc

    return FeatureConfig(fields=field_cfgs)


def load_config(path: str | None = None) -> AppConfig:
    global _allow_gpu
    _default = Path(__file__).parent.parent / "config.yml"
    config_path = path or os.environ.get("CONFIG_PATH", str(_default))
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    database = DatabaseConfig(**raw["database"])

    features: dict[str, FeatureConfig] = {
        name: _parse_field_cfgs(fields_dict)
        for name, fields_dict in raw.get("features", {}).items()
    }

    cohorts = [
        CohortConfig(
            name=c["name"],
            fields=c["fields"],
            weight=c["weight"],
            model=ModelConfig(
                name=c["model"]["name"],
                params=c["model"].get("params", {}),
            ),
            lru_size=c.get("lru_size"),
            features_config=c.get("features_config"),
        )
        for c in raw["cohorts"]
    ]

    for c in cohorts:
        if c.features_config is not None and c.features_config not in features:
            raise ValueError(
                f"Cohort '{c.name}' references unknown features_config '{c.features_config}'. "
                f"Available: {list(features.keys())}"
            )

    debug = DebugConfig(**raw.get("debug", {}))
    identifiers = raw.get("identifiers", [])
    _allow_gpu = bool(raw.get("allowGPU", False))

    return AppConfig(
        cohorts=cohorts,
        database=database,
        features=features,
        identifiers=identifiers,
        debug=debug,
        allow_gpu=_allow_gpu,
    )
