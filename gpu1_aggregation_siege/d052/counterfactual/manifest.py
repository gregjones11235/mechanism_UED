"""Matched-counterfactual manifest (Phase 2.5).

Task §matched counterfactual manifest. This is the single audit artifact that binds
a matched B/C run end-to-end: the shared frozen pool, the judgment cache, both arms'
prompt contracts, the strict-match verification (gate 1), both arms' selection
results + selection hashes (gate 2 replayable), the B<->C selection delta
(MODELER_CANONICAL_SELECTION_CHANGE = x/8), the execution-mapping certificate
attestation (gate 4), the canonical-target firewall attestation (gate 3), and the
zero-training attestation (gate 8). Its ``manifest_hash`` is a content hash over all
of it, so the manifest cannot be altered without detection.
"""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import Field, model_validator

from d052.counterfactual._hash import sha256_hex
from d052.counterfactual.protocol import (
    CounterfactualArm,
    MatchedVerification,
)
from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.selector import SelectionResult


def selection_change(selected_b: List[str], selected_c: List[str]) -> Dict[str, object]:
    """Compute the B<->C selection delta (same k => |in| == |out|)."""
    sb, sc = list(selected_b), list(selected_c)
    changed_in = sorted(set(sc) - set(sb))     # ids C swapped IN
    changed_out = sorted(set(sb) - set(sc))    # ids C swapped OUT
    return {
        "changed_in": changed_in,
        "changed_out": changed_out,
        "count": len(changed_in),
        "over": len(sb),
    }


class MatchedCounterfactualManifest(CanonicalModel):
    """Audit-grade record of one matched B/C counterfactual selection."""

    manifest_version: str = "canonical_v2.phase25.manifest.v1"
    pool_id: str = Field(min_length=1)
    pool_hash: str
    arm_b: CounterfactualArm
    arm_c: CounterfactualArm
    matched_verification: MatchedVerification
    selection_b: SelectionResult
    selection_c: SelectionResult
    changed_in: List[str] = Field(default_factory=list)
    changed_out: List[str] = Field(default_factory=list)
    modeler_canonical_selection_change: int = Field(ge=0)
    selection_change_over: int = Field(ge=0)
    certificates_b_count: int = Field(ge=0)
    certificates_c_count: int = Field(ge=0)
    all_certificates_executed_as_intended: bool
    canonical_target_firewall: str = "PASS"     # gate 3 attestation
    training_timesteps: Literal[0] = 0          # gate 8: zero training this phase
    no_training_attestation: Dict[str, object] = Field(default_factory=dict)
    manifest_hash: str

    @model_validator(mode="after")
    def _verify(self) -> "MatchedCounterfactualManifest":
        validate_sha256_hex(self.pool_hash, "pool_hash")
        validate_sha256_hex(self.manifest_hash, "manifest_hash")
        # the recorded delta must agree with the two selection results
        delta = selection_change(self.selection_b.selected_ids,
                                 self.selection_c.selected_ids)
        if delta["changed_in"] != self.changed_in or \
                delta["changed_out"] != self.changed_out:
            raise ValueError("MANIFEST_DELTA_MISMATCH: changed_in/out disagree with selections")
        if delta["count"] != self.modeler_canonical_selection_change:
            raise ValueError("MANIFEST_CHANGE_COUNT_MISMATCH")
        if delta["over"] != self.selection_change_over:
            raise ValueError("MANIFEST_CHANGE_OVER_MISMATCH")
        if self.manifest_hash != _manifest_payload_hash(self):
            raise ValueError("MANIFEST_HASH_MISMATCH")
        return self


def _manifest_payload(m: "MatchedCounterfactualManifest") -> dict:
    return {
        "manifest_version": m.manifest_version,
        "pool_id": m.pool_id,
        "pool_hash": m.pool_hash,
        "arm_b": m.arm_b.model_dump(mode="json"),
        "arm_c": m.arm_c.model_dump(mode="json"),
        "matched_verification": m.matched_verification.model_dump(mode="json"),
        "selection_b": m.selection_b.model_dump(mode="json"),
        "selection_c": m.selection_c.model_dump(mode="json"),
        "changed_in": m.changed_in,
        "changed_out": m.changed_out,
        "modeler_canonical_selection_change": m.modeler_canonical_selection_change,
        "selection_change_over": m.selection_change_over,
        "certificates_b_count": m.certificates_b_count,
        "certificates_c_count": m.certificates_c_count,
        "all_certificates_executed_as_intended": m.all_certificates_executed_as_intended,
        "canonical_target_firewall": m.canonical_target_firewall,
        "training_timesteps": m.training_timesteps,
        "no_training_attestation": m.no_training_attestation,
    }


def _manifest_payload_hash(m: "MatchedCounterfactualManifest") -> str:
    return sha256_hex(_manifest_payload(m))


def build_manifest(*, pool_id: str, pool_hash: str, arm_b: CounterfactualArm,
                   arm_c: CounterfactualArm, verification: MatchedVerification,
                   selection_b: SelectionResult, selection_c: SelectionResult,
                   certificates_b_count: int, certificates_c_count: int,
                   all_certificates_executed_as_intended: bool,
                   canonical_target_firewall: str = "PASS",
                   no_training_attestation: Dict[str, object]) -> MatchedCounterfactualManifest:
    """Assemble + content-hash the matched manifest (hash computed then verified)."""
    delta = selection_change(selection_b.selected_ids, selection_c.selected_ids)
    stub = MatchedCounterfactualManifest.model_construct(
        protocol_version="canonical_v2",
        pool_id=pool_id, pool_hash=pool_hash, arm_b=arm_b, arm_c=arm_c,
        matched_verification=verification, selection_b=selection_b,
        selection_c=selection_c, changed_in=delta["changed_in"],
        changed_out=delta["changed_out"],
        modeler_canonical_selection_change=delta["count"],
        selection_change_over=delta["over"],
        certificates_b_count=certificates_b_count,
        certificates_c_count=certificates_c_count,
        all_certificates_executed_as_intended=all_certificates_executed_as_intended,
        canonical_target_firewall=canonical_target_firewall,
        training_timesteps=0,
        no_training_attestation=no_training_attestation,
        manifest_hash="0" * 64,
    )
    mh = _manifest_payload_hash(stub)
    return MatchedCounterfactualManifest.model_validate(
        {**stub.model_dump(mode="json"), "manifest_hash": mh})
