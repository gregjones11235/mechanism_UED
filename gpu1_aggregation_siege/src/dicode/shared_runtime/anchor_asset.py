"""The REAL frozen shared anchor manifest.

The FROZEN manifest (``configs/e1_formal_ued_anchor_manifest.FROZEN.json``)
is produced by ``scripts/freeze_e1_anchor_manifest.py`` from the DRAFT +
the supervisor-frozen curriculum config. Its hash identities are derived
from the frozen config bytes (tamper-evident), never hand-written.
"""
from __future__ import annotations

import json
import os
from typing import Any

from . import asset_locations as AL

FROZEN_MANIFEST_RELATIVE = os.path.join(
    "configs", "e1_formal_ued_anchor_manifest.FROZEN.json")


class AnchorAssetError(RuntimeError):
    """Fail-closed anchor asset violation."""


class FrozenAnchorManifestHandle:
    """The consumed FROZEN SharedAnchorManifest with a stable identity."""

    def __init__(self, manifest: Any, manifest_sha256: str):
        self._manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.object_identity_hash = manifest_sha256
        self.registry_identity = manifest_sha256

    @property
    def manifest(self):
        return self._manifest

    @property
    def anchors(self):
        return self._manifest.anchors

    @property
    def is_frozen(self) -> bool:
        return self._manifest.is_frozen


def real_anchor_manifest() -> FrozenAnchorManifestHandle:
    from dicode.teachers.e1_formal import anchor_manifest as AM

    path = AL.resolve_repo_relative(FROZEN_MANIFEST_RELATIVE)
    if not os.path.isfile(path):
        raise AnchorAssetError(
            "ANCHOR_MANIFEST_NOT_FROZEN: the FROZEN anchor manifest "
            f"{path!r} does not exist; run "
            "scripts/freeze_e1_anchor_manifest.py (supervisor act)")
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    manifest = AM.consume_anchor_manifest(
        mapping, "shared_runtime.anchor_manifest")
    if not manifest.is_frozen:
        raise AnchorAssetError(
            "ANCHOR_MANIFEST_NOT_FROZEN: the production path consumes a "
            "FROZEN anchor manifest only")
    return FrozenAnchorManifestHandle(
        manifest, str(mapping.get("manifest_sha256", "")))
