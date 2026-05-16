from __future__ import annotations

import hashlib
import ipaddress
from typing import Any

from app.config import FieldConfig
from app.features.extractors.base import FieldHandler, SubFields


def _hash_string(s: str) -> float:
    """Stable SHA-256 hash of a string, normalised to [0, 1]."""
    digest = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


class IpAddressHandler(FieldHandler):
    def handle(self, key: str, value: Any) -> SubFields:
        if not isinstance(value, str):
            raise TypeError(f"{key}: expected str, got {type(value).__name__}")
        try:
            ip = ipaddress.ip_address(value)
            if isinstance(ip, ipaddress.IPv4Address):
                octets = value.split(".")
                return SubFields({
                    f"{key}.is_private": (float(ip.is_private), FieldConfig(type="numeric")),
                    f"{key}.o1":         (float(octets[0]),     FieldConfig(type="numeric")),
                    f"{key}.o2":         (float(octets[1]),     FieldConfig(type="numeric")),
                    f"{key}.o3":         (float(octets[2]),     FieldConfig(type="numeric")),
                    f"{key}.o4":         (float(octets[3]),     FieldConfig(type="numeric")),
                })
            else:
                subnet = ipaddress.ip_network(f"{ip}/48", strict=False)
                return SubFields({
                    f"{key}.is_private":  (float(ip.is_private),      FieldConfig(type="numeric")),
                    f"{key}.subnet_hash": (_hash_string(str(subnet)), FieldConfig(type="numeric")),
                })
        except ValueError:
            raise ValueError(f"{key}: cannot parse {value!r} as IP address")
