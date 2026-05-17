from __future__ import annotations

from typing import Any

from app.config import FieldConfig
from app.features.preprocessors import FieldInfo, FieldPreprocessor, make_field_preprocessor


class Preprocessor:
    """Stateful preprocessing pipeline.

    Lazily creates and caches a FieldPreprocessor per flat key, applying
    the learn_pre / transform / learn_post cycle on each event.

    Picklable so Detector can include it in its state blob.
    """

    def __init__(self, type_defaults: dict[str, str]) -> None:
        self._type_defaults = type_defaults
        self._preprocessors: dict[str, FieldPreprocessor] = {}

    def process(
        self,
        extracted: dict[str, tuple[Any, FieldConfig]],
        is_learning: bool,
    ) -> dict[FieldInfo, list[float]]:
        result: dict[FieldInfo, list[float]] = {}
        for key, (value, field_cfg) in extracted.items():
            pp = self._get_or_create(key, field_cfg)
            if is_learning:
                pp.learn_pre_transform(key, value)
            field_result = pp.transform(key, value)
            if is_learning:
                pp.learn_post_transform(key, value)
            result.update(field_result)
        return result

    def _get_or_create(self, flat_key: str, field_cfg: FieldConfig) -> FieldPreprocessor:
        if flat_key not in self._preprocessors:
            self._preprocessors[flat_key] = make_field_preprocessor(
                field_cfg, self._type_defaults
            )
        return self._preprocessors[flat_key]
