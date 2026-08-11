"""Canonical JSON encoding + sha256 identity (mirrors the d052 pattern).

Deterministic, fail-closed canonical serialization used for window
hashes, spec hashes and ledger digests. Pure standard library.

Rules:
* dicts require string keys and are key-sorted;
* tuples are encoded as lists;
* bool/int/str/None pass through; floats must be FINITE (NaN/Inf would
  make the encoding implementation-defined, so they fail closed);
* any other type fails closed with a greppable code.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .schemas import E1Code, E1SchemaError


def _check_and_normalize(obj: Any, path: str) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise E1SchemaError(
                    E1Code.CANONICAL_UNSUPPORTED_TYPE,
                    f"non-string dict key at {path} ({type(key).__name__})",
                )
            out[key] = _check_and_normalize(value, f"{path}.{key}")
        return out
    if isinstance(obj, (list, tuple)):
        return [
            _check_and_normalize(item, f"{path}[{i}]")
            for i, item in enumerate(obj)
        ]
    if isinstance(obj, bool) or obj is None or isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise E1SchemaError(
                E1Code.CANONICAL_UNSUPPORTED_TYPE,
                f"non-finite float at {path}: {obj!r}",
            )
        return obj
    if isinstance(obj, str):
        return obj
    raise E1SchemaError(
        E1Code.CANONICAL_UNSUPPORTED_TYPE,
        f"unsupported type {type(obj).__name__} at {path}",
    )


def canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON string (sorted keys, tight separators)."""
    normalized = _check_and_normalize(obj, "$")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(obj: Any) -> str:
    """sha256 hexdigest of the canonical JSON encoding of ``obj``."""
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def sha256_hex(data: bytes) -> str:
    """sha256 hexdigest of raw bytes."""
    return hashlib.sha256(data).hexdigest()
