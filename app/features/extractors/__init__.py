from app.features.extractors.base import FieldHandler, SubFields
from app.features.extractors.boolean import BooleanHandler
from app.features.extractors.ignore import IgnoreHandler
from app.features.extractors.ip_address import IpAddressHandler
from app.features.extractors.json_expand import JsonExpandHandler
from app.features.extractors.numeric import NumericHandler
from app.features.extractors.semver import SemverHandler
from app.features.extractors.string import StringHandler
from app.features.extractors.timestamp import TimestampHandler

HANDLERS: dict[str, FieldHandler] = {
    "timestamp":       TimestampHandler(),
    "numeric":         NumericHandler(),
    "boolean":         BooleanHandler(),
    "str_categorical": StringHandler(),
    "str_identifier":  StringHandler(),
    "str_text":        StringHandler(),
    "ip_address":      IpAddressHandler(),
    "semver":          SemverHandler(),
    "json_expand":     JsonExpandHandler(),
    "ignore":          IgnoreHandler(),
}

__all__ = ["FieldHandler", "SubFields", "HANDLERS"]
