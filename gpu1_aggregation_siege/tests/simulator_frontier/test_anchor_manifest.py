"""Schema/binding tests for the shared anchor manifest interface (condition 5).

Every fixture here is SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT: the tests
exercise the schema and fail-closed binding rules ONLY.  Anchor science
(distributions/worlds/seeds) comes exclusively from the 总控 manifest, which
has not arrived this round -> SHARED_ANCHOR_MANIFEST_BOUND stays False with
status BLOCKED_SHARED_ANCHOR_MANIFEST.
"""

from __future__ import annotations

import pytest

from dicode.simulator_frontier import (
    ANCHOR_SLOT_COUNT,
    BLOCKED_SHARED_ANCHOR_MANIFEST,
    DYNAMIC_DISTRIBUTION_COUNT,
    SHARED_ANCHOR_MANIFEST_BOUND,
    AnchorDefinition,
    AnchorManifest,
    RetentionContract,
    bind_anchor_manifest,
    manifest_hash_of,
    unbound_status,
    validate_anchor_manifest,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.provenance import ProvenanceViolationError

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True  # labeling discipline


def _synthetic_anchors(n: int = ANCHOR_SLOT_COUNT) -> tuple:
    # SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT: ids/refs below are placeholders.
    return tuple(
        AnchorDefinition(
            anchor_id=f"SYNTH_ANCHOR_{i}",
            world_set_ref=f"synthetic-world-set-ref-{i}",
            reset_protocol="STANDARD_RESET",
            seed_policy_ref=f"synthetic-seed-policy-{i}")
        for i in range(n))


def _synthetic_manifest(**overrides) -> AnchorManifest:
    anchors = overrides.pop("anchors", _synthetic_anchors())
    manifest_id = overrides.pop("manifest_id", "SYNTHETIC_FIXTURE_MANIFEST")
    signature = overrides.pop("controller_signature_ref",
                              "SYNTHETIC_CONTROLLER_SIGNATURE_REF")
    manifest_hash = overrides.pop("manifest_hash", None) or manifest_hash_of(
        manifest_id, signature, anchors)
    return AnchorManifest(
        manifest_id=manifest_id, controller_signature_ref=signature,
        frozen=overrides.pop("frozen", True), anchors=anchors,
        manifest_hash=manifest_hash)


def _retention(**overrides) -> RetentionContract:
    base = dict(dynamic_distribution_count=DYNAMIC_DISTRIBUTION_COUNT,
                anchor_slot_count=ANCHOR_SLOT_COUNT, anchor_ratio=0.25)
    base.update(overrides)
    return RetentionContract(**base)


class TestRoundStatus:
    def test_no_controller_manifest_means_unbound(self):
        assert SHARED_ANCHOR_MANIFEST_BOUND is False
        status = unbound_status()
        assert status["bound"] is False
        assert status["status"] == BLOCKED_SHARED_ANCHOR_MANIFEST
        assert status["schema_ready"] is True


class TestValidateAnchorManifest:
    def test_valid_synthetic_fixture_validates(self):
        validate_anchor_manifest(_synthetic_manifest())  # raises nothing

    def test_missing_controller_signature_raises(self):
        with pytest.raises(InvalidEvidenceError):
            validate_anchor_manifest(_synthetic_manifest(controller_signature_ref=""))

    def test_unfrozen_manifest_raises(self):
        with pytest.raises(InvalidEvidenceError):
            validate_anchor_manifest(_synthetic_manifest(frozen=False))

    @pytest.mark.parametrize("slot_count", [0, 3, 5])
    def test_wrong_slot_count_raises(self, slot_count):
        with pytest.raises(InvalidEvidenceError):
            validate_anchor_manifest(_synthetic_manifest(
                anchors=_synthetic_anchors(slot_count)))

    def test_duplicate_anchor_ids_raise(self):
        anchors = _synthetic_anchors()
        # slot 0 reuses slot 1's anchor_id -> genuine duplicate
        dup = (AnchorDefinition(anchors[1].anchor_id, "w", "STANDARD_RESET", "s"),) + anchors[1:]
        with pytest.raises(InvalidEvidenceError):
            validate_anchor_manifest(_synthetic_manifest(anchors=dup))

    def test_non_standard_reset_protocol_raises(self):
        anchors = _synthetic_anchors()
        bad = (AnchorDefinition(anchors[0].anchor_id, "w", "FRONTIER_START", "s"),) + anchors[1:]
        with pytest.raises(InvalidEvidenceError):
            validate_anchor_manifest(_synthetic_manifest(anchors=bad))

    def test_hash_mismatch_raises(self):
        manifest = _synthetic_manifest()
        tampered = AnchorManifest(manifest.manifest_id,
                                  manifest.controller_signature_ref, True,
                                  manifest.anchors, "0" * 64)
        with pytest.raises(InvalidEvidenceError):
            validate_anchor_manifest(tampered)

    def test_anchor_definition_requires_all_fields(self):
        with pytest.raises(InvalidEvidenceError):
            AnchorDefinition("", "w", "STANDARD_RESET", "s")


class TestBinding:
    def test_bind_valid_manifest_into_retention_contract(self):
        manifest = _synthetic_manifest()
        record = bind_anchor_manifest(manifest, _retention())
        assert record["bound"] is True
        assert record["manifest_id"] == manifest.manifest_id
        assert record["manifest_hash"] == manifest.manifest_hash
        assert record["status"] == "SHARED_ANCHOR_MANIFEST_BOUND"
        assert len(record["anchor_ids"]) == ANCHOR_SLOT_COUNT

    @pytest.mark.parametrize("ratio", [0.0, -0.1])
    def test_anchor_ratio_zero_or_negative_raises(self, ratio):
        with pytest.raises(InvalidEvidenceError):
            bind_anchor_manifest(_synthetic_manifest(), _retention(anchor_ratio=ratio))

    def test_wrong_distribution_shape_raises(self):
        with pytest.raises(InvalidEvidenceError):
            bind_anchor_manifest(_synthetic_manifest(),
                                 _retention(dynamic_distribution_count=11))
        with pytest.raises(InvalidEvidenceError):
            bind_anchor_manifest(_synthetic_manifest(),
                                 _retention(anchor_slot_count=3))

    def test_formal_banks_in_curriculum_raise(self):
        with pytest.raises(ProvenanceViolationError):
            bind_anchor_manifest(
                _synthetic_manifest(),
                _retention(formal_banks_in_online_curriculum=True))

    def test_invalid_manifest_is_not_bound(self):
        with pytest.raises(InvalidEvidenceError):
            bind_anchor_manifest(_synthetic_manifest(frozen=False), _retention())
