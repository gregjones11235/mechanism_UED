"""Stable pytree / payload hashing for E3-litesim gates."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import jax
import numpy as np


def hash_pytree(params: Any, *, prefix: str = "") -> str:
    """Stable sha256 over a pytree: sorted keypath|dtype|shape|bytes."""
    digest = hashlib.sha256()
    entries = []
    for keypath, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        entries.append((jax.tree_util.keystr(keypath), leaf))
    for path, leaf in sorted(entries, key=lambda item: item[0]):
        arr = np.ascontiguousarray(np.asarray(leaf))
        digest.update(f"{prefix}/{path}|{arr.dtype}|{arr.shape}".encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    """sha256 of a JSON-serializable payload (sorted, compact)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        .encode("utf-8")
    ).hexdigest()


def hash_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()