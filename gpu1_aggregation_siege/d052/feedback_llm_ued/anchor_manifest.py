"""C13: the shared frozen anchor-manifest seam.

The four standard-reset anchors must be consumed from the cross-direction
shared FROZEN manifest — a local constant may never silently pose as one.
This module is the ONLY seam:

* it accepts an EXPLICITLY INJECTED manifest (nothing in this package
  fabricates or fetches one);
* ``resolve()`` RECOMPUTES the canonical manifest hash and fails closed on
  any mismatch, refuses unfrozen manifests, and refuses absence;
* verified fact for this worktree: NO shared frozen manifest exists. The
  loop therefore runs on the scaffold placeholder — the same four anchor
  ids, explicitly labeled ``SCAFFOLD_PLACEHOLDER_NOT_SHARED``, budget
  unchanged (12 dynamic + 4 anchors) in all three modes — and
  ``SHARED_ANCHOR_MANIFEST_BOUND`` stays False. Formal retention may only
  count manifest-bound anchor probes; this round's placeholder anchors are
  reported as exactly that.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel

#: explicit label: anchors are scaffold placeholders, NOT a shared binding
SCAFFOLD_PLACEHOLDER_NOT_SHARED = "SCAFFOLD_PLACEHOLDER_NOT_SHARED"
#: label a run carries once a real frozen manifest IS bound (unreachable
#: this round — kept here so the two states are the only two states)
SHARED_MANIFEST_BOUND_LABEL = "SHARED_ANCHOR_MANIFEST_BOUND"


class AnchorManifestBlocked(RuntimeError):
    """Fail-closed: no consumable shared FROZEN anchor manifest."""


class SharedAnchorManifest(CanonicalModel):
    """Read-only contract of a cross-direction shared anchor manifest."""

    manifest_id: str = Field(min_length=1)
    anchors: List[str] = Field(default_factory=list)
    frozen: bool = False
    manifest_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "SharedAnchorManifest":
        if len(self.anchors) != C.GLOBAL_ANCHOR_SLOTS:
            raise ValueError(
                f"ILLEGAL_ANCHOR_SLOT_COUNT: {len(self.anchors)} anchor(s); "
                f"the shared manifest must bind exactly "
                f"{C.GLOBAL_ANCHOR_SLOTS}")
        if len(set(self.anchors)) != len(self.anchors):
            raise ValueError("DUPLICATE_ANCHOR_ID")
        for anchor in self.anchors:
            if not isinstance(anchor, str) or not anchor:
                raise ValueError(f"ILLEGAL_ANCHOR_ID: {anchor!r}")
        if not self.manifest_hash:
            payload = self.model_dump()
            payload.pop("manifest_hash", None)
            object.__setattr__(self, "manifest_hash",
                               canonical_sha256(payload))
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("manifest_hash", None)
        return canonical_sha256(payload)


class AnchorManifestSource:
    """The ONLY surface through which the loop touches anchor provenance.

    Explicit injection only. ``resolve()`` fails closed on:

    * no manifest injected          -> AnchorManifestBlocked
      (BLOCKED_SHARED_ANCHOR_MANIFEST);
    * manifest not frozen           -> AnchorManifestBlocked
      (BLOCKED_SHARED_ANCHOR_MANIFEST, naming the manifest);
    * carried hash != recomputed    -> ValueError
      (ANCHOR_MANIFEST_HASH_MISMATCH — tamper or non-canonical encoding).
    """

    def __init__(self,
                 manifest: Optional[SharedAnchorManifest] = None) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> Optional[SharedAnchorManifest]:
        return self._manifest

    def resolve(self) -> Tuple[str, ...]:
        if self._manifest is None:
            raise AnchorManifestBlocked(
                f"{C.BLOCKED_SHARED_ANCHOR_MANIFEST}: no cross-direction "
                "shared FROZEN anchor manifest was injected into this run; "
                "anchors must fall back to the labeled scaffold placeholder")
        if not self._manifest.frozen:
            raise AnchorManifestBlocked(
                f"{C.BLOCKED_SHARED_ANCHOR_MANIFEST}: manifest "
                f"{self._manifest.manifest_id!r} is NOT frozen; only a "
                "FROZEN shared manifest may bind the standard-reset anchors")
        recomputed = self._manifest.rehash()
        if recomputed != self._manifest.manifest_hash:
            raise ValueError(
                f"ANCHOR_MANIFEST_HASH_MISMATCH: carried "
                f"{self._manifest.manifest_hash!r} != recomputed "
                f"{recomputed!r} — the manifest was tampered with or "
                "serialized through a non-canonical encoding")
        return tuple(self._manifest.anchors)

    def scaffold_placeholder(self) -> Tuple[str, ...]:
        """The round's honest fallback: the four canonical anchor ids,
        explicitly NOT a shared-manifest binding (budget stays 12 + 4)."""
        return tuple(C.GLOBAL_CANONICAL_ANCHOR_IDS)
