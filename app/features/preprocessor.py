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

    def __init__(self, type_defaults: dict[str, str], warmup_count: int = 0) -> None:
        self._type_defaults = type_defaults
        self._warmup_count = warmup_count
        self._preprocessors: dict[str, FieldPreprocessor] = {}

    def process_batch(
        self,
        extracted_list: list[dict[str, tuple[Any, FieldConfig]]],
        is_learning: bool,
    ) -> list[dict[FieldInfo, list[float]]]:
        """Process a batch of extracted events for learning.

        Pass 1: learn_pre_transform for every event before any transform — ensures
        vocabularies and stats are primed across the full batch before encoding begins.
        Pass 2: transform + learn_post_transform for each event.

        Used for both warmup flush (many events) and normal ingestion (one event),
        giving a single code path in Detector.
        """
        if is_learning:
            for extracted in extracted_list:
                for key, (value, field_cfg) in extracted.items():
                    pp = self._get_or_create(key, field_cfg)
                    pp.learn_pre_transform(key, value)
        results = []
        for extracted in extracted_list:
            result: dict[FieldInfo, list[float]] = {}
            for key, (value, field_cfg) in extracted.items():
                pp = self._get_or_create(key, field_cfg)
                result.update(pp.transform(key, value))
                if is_learning:
                    pp.learn_post_transform(key, value)
            results.append(result)
        return results

    def _get_or_create(self, flat_key: str, field_cfg: FieldConfig) -> FieldPreprocessor:
        if flat_key not in self._preprocessors:
            self._preprocessors[flat_key] = make_field_preprocessor(
                field_cfg, self._type_defaults, warmup_count=self._warmup_count
            )
        return self._preprocessors[flat_key]
