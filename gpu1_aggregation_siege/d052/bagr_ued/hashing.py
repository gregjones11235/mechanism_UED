"""Deterministic hashing helpers for BA-BAGR-UED.

Every hash in this subpackage is a content hash over CANONICAL JSON
(sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str), so
identical inputs reproduce identical hashes across runs and machines. Detector
provenance additionally carries the sha256 of the detector's own source text
(detector_source_sha256, task section 4) so a detector's behaviour cannot drift
undetected from its recorded identity.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Canonical JSON serialization used for EVERY content hash in this package."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def canonical_sha256(obj: Any) -> str:
    """sha256 over the canonical JSON of ``obj`` (content hash)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    """sha256 of a UTF-8 text blob (prompt/response identity)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_sha256(obj: Any) -> str:
    """sha256 of the source text of ``obj`` (class/function provenance).

    Used for detector_version-independent provenance: detector_source_sha256
    binds an anomaly candidate to the exact detector code that produced it.
    """
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()


def verify_content_hash(payload: dict, *, hash_field: str, carried: str,
                        kind: str) -> str:
    """C14 / P1-5: RECOMPUTE the canonical content hash and compare verbatim.

    An externally carried content hash is NEVER accepted as-is: the content
    (``payload`` minus ``hash_field``) is re-serialized through the canonical
    JSON and hashed again; if the carried value is non-empty and differs from
    the recomputation, the object was tampered with (or serialized through a
    non-canonical encoding) and this fails CLOSED with
    ``CONTENT_HASH_MISMATCH``. Returns the recomputed hash so callers can
    stamp it with ``object.__setattr__``.
    """
    body = {k: v for k, v in payload.items() if k != hash_field}
    recomputed = canonical_sha256(body)
    if carried and carried != recomputed:
        raise ValueError(
            f"CONTENT_HASH_MISMATCH: {kind} carried {hash_field}="
            f"{carried!r} but its content recomputes to {recomputed!r} — "
            "the object was tampered with or serialized through a "
            "non-canonical encoding")
    return recomputed
