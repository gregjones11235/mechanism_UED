"""Stage 2 tests: student identity binding for archive entries and outcomes.

Every missing/placeholder binding must raise (fail closed); an unbound entry
can never be silently treated as bound.
"""

from __future__ import annotations

import pytest

from dicode.simulator_frontier.archive_schema import FrontierArchiveEntry
from dicode.simulator_frontier.errors import SimulatorFrontierError
from dicode.simulator_frontier.memory_modes import MemoryRestoreMode, MemoryRestoreRequest
from dicode.simulator_frontier.search_statistics import BranchOutcome
from dicode.simulator_frontier.student_binding import (
    REQUIRED_ENTRY_BINDING_FIELDS,
    UNBOUND_STUDENT,
    assert_entry_bound,
    assert_outcome_bound,
    bind_branch_outcome,
    bind_capture_entry,
    check_bound_entry_memory_request,
    identity_mapping_hash_fields,
)


def _entry(**overrides):
    base = dict(
        state_id="s1", source_checkpoint_id="ckpt-A", source_episode_id="e1",
        source_seed=0, source_timestep=10, capture_reason="frontier",
        floor=3, gate_progress=0.5, health_band="ok", threat_band="low",
        resource_band="mid", inventory_stage="early",
        achievement_snapshot={"a": 1}, terminal=False, memory_mode="PERSISTENT",
        encoded_state_ref="ref", state_hash="h", provenance_hash="p",
        created_at="2026-08-03T00:00:00Z",
    )
    base.update(overrides)
    return FrontierArchiveEntry(**base)


def _outcome(**overrides):
    base = dict(
        branch_id="b1", state_id="s1", search_source="frontier", rng_seed=0,
        horizon=32, transitions_used=32, success=True, progress=0.9,
        terminal_event=None, failure_category=None, memory_mode="PERSISTENT",
        outcome_hash="o",
    )
    base.update(overrides)
    return BranchOutcome(**base)


BINDING = dict(
    student_identity_hash="i" * 64,
    parameter_hash="p" * 64,
    memory_spec_hash="m" * 64,
    capture_student_id="PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    discovery_provenance="TRAINING_DISCOVERY",
)


class TestEntryBinding:
    def test_fresh_entry_is_unbound_and_assert_raises(self):
        entry = _entry()
        for name in REQUIRED_ENTRY_BINDING_FIELDS:
            assert getattr(entry, name) == ""
        with pytest.raises(SimulatorFrontierError):
            assert_entry_bound(entry)

    def test_bind_then_assert_passes(self):
        bound = bind_capture_entry(_entry(), **BINDING)
        assert_entry_bound(bound)
        assert bound.source_student_identity_hash == BINDING["student_identity_hash"]
        assert bound.discovery_provenance == "TRAINING_DISCOVERY"

    @pytest.mark.parametrize("field", sorted(BINDING))
    def test_binding_missing_or_placeholder_raises(self, field):
        for bad in ("", "   ", "PENDING", "UNKNOWN", "TODO"):
            kwargs = dict(BINDING)
            kwargs[field] = bad
            with pytest.raises(SimulatorFrontierError):
                bind_capture_entry(_entry(), **kwargs)


class TestOutcomeBinding:
    def test_fresh_outcome_is_unbound(self):
        outcome = _outcome()
        assert outcome.capture_student_id == UNBOUND_STUDENT
        assert outcome.memory_compatibility_status == "UNSPECIFIED"
        with pytest.raises(SimulatorFrontierError):
            assert_outcome_bound(outcome)

    def test_same_student_binding_derives_no_cross_policy(self):
        bound = bind_branch_outcome(
            _outcome(),
            capture_student_id="STUDENT_X", search_student_id="STUDENT_X",
            train_student_id="STUDENT_X", memory_compatibility_status="COMPATIBLE")
        assert bound.cross_policy_search is False
        assert_outcome_bound(bound)

    def test_different_ids_derive_cross_policy(self):
        bound = bind_branch_outcome(
            _outcome(),
            capture_student_id="STUDENT_X", search_student_id="STUDENT_Y",
            train_student_id="STUDENT_X", memory_compatibility_status="COMPATIBLE")
        assert bound.cross_policy_search is True

    def test_unspecified_status_rejected_even_with_ids(self):
        bound = bind_branch_outcome(
            _outcome(),
            capture_student_id="A", search_student_id="A", train_student_id="A",
            memory_compatibility_status="UNSPECIFIED")
        with pytest.raises(SimulatorFrontierError):
            assert_outcome_bound(bound)


class TestIdentityMappingAndMemoryBridge:
    def test_identity_mapping_hash_fields(self):
        out = identity_mapping_hash_fields(
            {"identity_hash": "i" * 64, "params_sha256": "p" * 64, "memory_spec_hash": "m" * 64})
        assert set(out) == {"identity_hash", "params_sha256", "memory_spec_hash"}
        with pytest.raises(SimulatorFrontierError):
            identity_mapping_hash_fields({"identity_hash": "", "params_sha256": "p" * 64,
                                          "memory_spec_hash": "m" * 64})

    def test_bound_entry_memory_request_compatible(self):
        entry = bind_capture_entry(_entry(), **BINDING)
        request = MemoryRestoreRequest(
            mode=MemoryRestoreMode.SAVED_POLICY_MEMORY,
            policy_architecture_id="RMT16",
            checkpoint_id=entry.source_checkpoint_id,
            memory_tree_structure=("memories", "mem_mask", "mem_idx"),
        )
        report = check_bound_entry_memory_request(entry, request)
        assert report.compatible is True, report.reasons

    def test_bound_entry_memory_request_checkpoint_mismatch(self):
        entry = bind_capture_entry(_entry(), **BINDING)
        request = MemoryRestoreRequest(
            mode=MemoryRestoreMode.SAVED_POLICY_MEMORY,
            policy_architecture_id="RMT16",
            checkpoint_id="SOME_OTHER_CHECKPOINT",
            memory_tree_structure=("memories",),
        )
        report = check_bound_entry_memory_request(entry, request)
        assert report.compatible is False and any("checkpoint" in r for r in report.reasons)

    def test_unbound_entry_cannot_reach_memory_bridge(self):
        request = MemoryRestoreRequest(
            mode=MemoryRestoreMode.ZERO_MEMORY, policy_architecture_id="RMT16")
        with pytest.raises(SimulatorFrontierError):
            check_bound_entry_memory_request(_entry(), request)
