from __future__ import annotations

from typing import Callable

from app.config import FieldConfig
from app.features.preprocessors.base import FieldInfo, FieldPreprocessor
from app.features.preprocessors.one_hot_hash_encoder import OneHotHashEncoder
from app.features.preprocessors.frequency_encoder import FrequencyEncoder
from app.features.preprocessors.hash_index import HashIndex
from app.features.preprocessors.label_index import LabelIndex
from app.features.preprocessors.min_max_scaler import MinMaxScaler
from app.features.preprocessors.noop import NoOp
from app.features.preprocessors.pass_through import PassThrough
from app.features.preprocessors.standard_scaler import StandardScaler
from app.features.preprocessors.one_hot_encoder import OneHotEncoder
from app.features.preprocessors.sentence_transformer_encoder import SentenceTransformerEncoder
from app.features.preprocessors.drain3_encoder import Drain3Encoder

_FACTORIES: dict[str, Callable[[dict, int], FieldPreprocessor]] = {
    "NoOp":                         lambda p, wc: NoOp(),
    "PassThrough":                  lambda p, wc: PassThrough(),
    "FrequencyEncoder":             lambda p, wc: FrequencyEncoder(),
    "HashIndex":                    lambda p, wc: HashIndex(**{"seed": 0, **p}),
    "LabelIndex":                   lambda p, wc: LabelIndex(**p),
    "MinMaxScaler":                 lambda p, wc: MinMaxScaler(),
    "StandardScaler":               lambda p, wc: StandardScaler(),
    "OneHotEncoder":                lambda p, wc: OneHotEncoder(**p),
    "OneHotHashEncoder":            lambda p, wc: OneHotHashEncoder(**{"seed": 0, **p}),
    "SentenceTransformerEncoder":   lambda p, wc: SentenceTransformerEncoder(warmup_count=wc, **p),
    "Drain3Encoder":                lambda p, wc: Drain3Encoder(**p),
}


def make_field_preprocessor(
    field_cfg: FieldConfig,
    type_defaults: dict[str, str] | None = None,
    warmup_count: int = 0,
) -> FieldPreprocessor:
    if type_defaults is None:
        raise ValueError("type_defaults are not set!")
    name = field_cfg.preprocessor.name if field_cfg.preprocessor else type_defaults.get(field_cfg.type, "PassThrough")
    params = field_cfg.preprocessor.params if field_cfg.preprocessor else {}
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown preprocessor: {name!r}")
    return factory(params, warmup_count)


__all__ = ["FieldInfo", "FieldPreprocessor", "make_field_preprocessor"]
