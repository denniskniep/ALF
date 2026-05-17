from __future__ import annotations

from typing import Any

from app.features.preprocessors.base import FieldInfo


class NoOp:
    """NoOp for testing!"""

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, list[float]]:
        return {FieldInfo(key, preprocessor="NoOp"): [float(0.0)]}
