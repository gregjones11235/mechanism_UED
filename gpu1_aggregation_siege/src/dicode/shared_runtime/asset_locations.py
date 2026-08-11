"""Deployment-bound real asset locations (sha256-verified at use).

The paths come from ``configs/production_asset_locations.json`` in the
siege root; every consumer re-verifies the declared sha256 before use
(fail closed). No path is ever guessed or defaulted.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

_SIEGE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_LOCATIONS_PATH = os.path.join(
    _SIEGE_ROOT, "configs", "production_asset_locations.json"
)

_cache: Dict[str, Any] = {}


class AssetLocationError(RuntimeError):
    """Fail-closed asset location violation."""


def siege_root() -> str:
    return _SIEGE_ROOT


def asset_locations() -> Dict[str, Any]:
    if not _cache:
        if not os.path.isfile(_LOCATIONS_PATH):
            raise AssetLocationError(
                "ASSET_LOCATIONS_MISSING: the production asset-location "
                f"manifest {_LOCATIONS_PATH!r} does not exist (no "
                "guessing, no defaults)"
            )
        with open(_LOCATIONS_PATH, "r", encoding="utf-8") as handle:
            _cache.update(json.load(handle))
    return _cache


def student_locations() -> Dict[str, Any]:
    return asset_locations()["student"]


def resolve_repo_relative(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_SIEGE_ROOT, path)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str, expected_sha256: str, what: str) -> str:
    """Verify a real asset file exists and matches its declared sha256."""
    if not os.path.isfile(path):
        raise AssetLocationError(
            f"ASSET_MISSING: {what} not found at {path!r} (fail closed)"
        )
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise AssetLocationError(
            f"ASSET_SHA_MISMATCH: {what} at {path!r} has sha256 "
            f"{actual!r} != declared {expected_sha256!r} (fail closed)"
        )
    return path
