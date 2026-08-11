"""Canonical JSON + sha256 helpers for the counterfactual package.

Uses the SAME serialization convention as the rest of canonical_v2 (sorted keys,
compact separators, UTF-8) so every hash here is deterministic and reproducible
bit-for-bit across runs and machines.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    """sha256 over the canonical JSON of ``obj`` (lowercase 64-char hex)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
