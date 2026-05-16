from __future__ import annotations

from typing import Any

from app.config import FeatureConfig, FieldConfig
from app.features.extractors import HANDLERS, SubFields
from app.utils import get_nested

_DEFAULT_FEATURE_CONFIG = FeatureConfig()
_MISSING = object()

# ---------------------------------------------------------------------------
# Stage 1: extraction
# ---------------------------------------------------------------------------

def extract(
    obj: Any,
    feature_cfg: FeatureConfig,
) -> dict[str, tuple[Any, FieldConfig]]:
    """Resolve each configured feature from obj via its source path.

    Iterates over feature_cfg.fields; for each entry resolves field_cfg.source
    as a dot-notation path in obj.

    Compound handlers (timestamp, ip_address, semver, json_expand) return SubFields which are
    expanded via _expand_subfields.

    Fields whose source path is not found in the payload are silently skipped.

    Returns a flat dict mapping feature name → (value, FieldConfig).
    The FieldConfig travels with its value so downstream stages can select the
    right preprocessor without any further lookup.
    """
    if not isinstance(obj, dict):
        return {}
    out: dict[str, tuple[Any, FieldConfig]] = {}
    raw_values: dict[str, Any] = {}
    for feature_name, field_cfg in feature_cfg.fields.items():
        value = get_nested(obj, field_cfg.source, default=_MISSING)
        if value is _MISSING:
            value = raw_values.get(field_cfg.source, _MISSING)
        if value is _MISSING:
            continue
        raw_values[feature_name] = value
        result = _apply_handler(feature_name, value, field_cfg)
        if isinstance(result, SubFields):
            sub_out = _expand_subfields(result, raw_values)
            out.update(sub_out)
        elif result is not None:
            out[feature_name] = (result, field_cfg)
        else:
            out.pop(feature_name, None)
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _apply_handler(key: str, value: Any, field_cfg: FieldConfig) -> SubFields | Any:
    """Dispatch to the handler for field_cfg.type.

    Returns SubFields for compound types, a scalar for simple types,
    or None when the field should be skipped (ignore / unknown type).
    """
    handler = HANDLERS.get(field_cfg.type)
    return handler.handle(key, value) if handler else None


# ---------------------------------------------------------------------------
# Sub-field expansion
# ---------------------------------------------------------------------------

def _expand_subfields(
    subfields: SubFields,
    raw_values: dict[str, Any],
) -> dict[str, tuple[Any, FieldConfig]]:
    """Recursively process a SubFields result from a handler.

    Stores the pre-handler value for every sub-field into raw_values so that
    later iterations of the main extract() loop can resolve source aliases
    referencing values produced by expansion.

    Sub-fields with a handler-provided default FieldConfig are processed and
    added to the output. Sub-fields with no default config (json_expand)
    are stored in raw_values only — the main loop handles them when their
    feature name is reached in the config.

    Returns a flat dict mapping each key to (value, resolved_FieldConfig).
    """
    out: dict[str, tuple[Any, FieldConfig]] = {}
    for sub_key, (sub_v, default_cfg) in subfields.items.items():
        raw_values[sub_key] = sub_v
        if default_cfg is not None:
            result = _apply_handler(sub_key, sub_v, default_cfg)
            if isinstance(result, SubFields):
                out.update(_expand_subfields(result, raw_values))
            elif result is not None:
                out[sub_key] = (result, default_cfg)
    return out
