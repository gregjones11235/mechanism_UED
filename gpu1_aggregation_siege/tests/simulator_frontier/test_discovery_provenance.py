"""Stage 2 tests (review condition 2): TRAINING_DISCOVERY capture provenance.

Frontier collection must carry TRAINING_DISCOVERY provenance and stay
structurally isolated from the frozen formal evaluation banks/worlds: any
FORMAL bank or formal world identifier entering a capture request raises.
Contract-level isolation — no real collection runs this round.
"""

from __future__ import annotations

import pytest

from dicode.simulator_frontier.discovery_provenance import (
    CaptureProvenance,
    DiscoveryProvenance,
    assert_not_formal,
    discovery_source_for,
    validate_capture_provenance,
)
from dicode.simulator_frontier.errors import ProvenanceViolationError
from dicode.simulator_frontier.provenance import DataSource


def _cap(**overrides):
    base = dict(
        provenance=DiscoveryProvenance.TRAINING_DISCOVERY,
        rollout_protocol_id="standard-reset-v1",
        world_set_hash="w" * 64,
    )
    base.update(overrides)
    return CaptureProvenance(**base)


class TestLegalProvenance:
    def test_training_discovery_passes(self):
        validate_capture_provenance(_cap())  # must not raise

    def test_string_provenance_is_normalized(self):
        cap = _cap(provenance="TRAINING_DISCOVERY")
        assert cap.provenance is DiscoveryProvenance.TRAINING_DISCOVERY
        validate_capture_provenance(cap)

    def test_synthetic_fixture_requires_explicit_opt_in(self):
        cap = _cap(provenance=DiscoveryProvenance.SYNTHETIC_FIXTURE)
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(cap)
        validate_capture_provenance(cap, allow_synthetic_fixture=True)

    def test_missing_protocol_or_world_hash_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(rollout_protocol_id=""))
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_hash=""))


class TestFormalIsolation:
    @pytest.mark.parametrize("marker", [
        "FORMAL_FRONT_BANK_A", "FORMAL_BACK_BANK_B", "FORMAL_FULL",
        "FORMAL_WORLD_SET", "FORMAL_EVAL_WORLD_3",
    ])
    def test_formal_bank_marker_in_bank_refs_raises(self, marker):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(bank_refs=(marker,)))

    def test_exact_formal_datasource_ref_raises_via_guard(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(bank_refs=(DataSource.FORMAL_FRONT.value,)))

    def test_formal_world_id_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_id="FORMAL_WORLD_ice"))

    def test_formal_marker_in_notes_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(notes={"source": "mixed with FORMAL_BACK data"}))

    def test_non_formal_refs_pass(self):
        validate_capture_provenance(_cap(
            bank_refs=("training_capture_bank",),
            world_set_id="discovery_worlds_v1",
            notes={"purpose": "frontier collection"}))


class TestSourceMapping:
    def test_discovery_source_mapping(self):
        assert discovery_source_for(DiscoveryProvenance.TRAINING_DISCOVERY) \
            is DataSource.TRAINING_FRONTIER_CAPTURE
        assert discovery_source_for(DiscoveryProvenance.SYNTHETIC_FIXTURE) \
            is DataSource.SYNTHETIC_TEST

    def test_assert_not_formal_bridge(self):
        assert_not_formal(DataSource.TRAINING_FRONTIER_CAPTURE, consumer="FrontierArchive")
        with pytest.raises(ProvenanceViolationError):
            assert_not_formal(DataSource.FORMAL_FULL, consumer="FrontierArchive")
        with pytest.raises(ProvenanceViolationError):
            assert_not_formal("FORMAL_FRONT", consumer="curriculum")
