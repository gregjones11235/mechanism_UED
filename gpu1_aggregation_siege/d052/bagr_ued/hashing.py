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
