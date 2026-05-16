from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.config import FieldConfig


@dataclass
class SubFields:
    """Returned by handlers that expand one field into named sub-fields.

    Each entry maps a fully-qualified dot-notation key, value pair to Subfields.

    FieldConfig may be None
    (e.g. for json_expand. In this case config must be defined by user, else silently dropped)
    """
    items: dict[str, tuple[Any, FieldConfig | None]]


class FieldHandler(ABC):
    @abstractmethod
    def handle(self, key: str, value: Any) -> SubFields | Any: ...
