# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-10): the 12 dynamic frontier distributions are a pure,
deterministic function of the TYPED planner output plus the minted selection
evidence — never a caller-supplied list.  The compiler re-binds the plan to
the measured evidence hash, stamps the verified evidence onto every
distribution, validates every eligible state against the archive, and
rejects unbound/tampered/empty/malformed inputs fail closed.
"""

import dataclasses

import pytest

from dicode.simulator_frontier.archive_schema import FrontierArchiveEntry
from dicode.simulator_frontier.branch_search_runner import (
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
)
from dicode.simulator_frontier.discovery_provenance import (
    DiscoveryProvenance,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.evidence_selector import (
    MIN_BUCKET_DIVERSITY,
    mint_selection_evidence_from_outcomes,
)
from dicode.simulator_frontier.feasibility_classifier import FrontierClass
from dicode.simulator_frontier.frontier_archive import FrontierArchive
from dicode.simulator_frontier.frontier_distributions import (
    DISTRIBUTION_SLOT_IDS,
    compile_planner_to_frontier_distributions,
)
from dicode.simulator_frontier.llm_contracts import (
    LLMContractError,
    PlannerOutput,
    compute_planner_hash,
)
from dicode.simulator_frontier.search_statistics import BranchOutcome
from dicode.simulator_frontier.state_codec import StateBundle, StateCodec

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

SHA = "a" * 64
_BUCKET = (1, "mid", "low", "mid", "stone", False)


def _outcome(branch_id: str) -> BranchOutcome:
    return BranchOutcome(
        branch_id=branch_id, state_id="s", search_source=SEARCH_SOURCE_STUDENT_STOCHASTIC,
        rng_seed=0, horizon=8, transitions_used=4, success=True,
        progress=0.6, terminal_event=None, failure_category=None,
        memory_mode="SAVED_POLICY_MEMORY", outcome_hash="b" * 64,
        memory_compatibility_status="SAVED_POLICY_MEMORY_VERIFIED")


def _evidence() -> object:
    return mint_selection_evidence_from_outcomes(
        state_id="s", frontier_class=FrontierClass.LEARNABLE_FRONTIER,
        outcomes=(_outcome("b0"), _outcome("b1")),
        retention_ok=True, anchor_coverage_ok=True,
        bucket_diversity=MIN_BUCKET_DIVERSITY)


def _archive(states=("s1", "s2", "s3", "s4"), *, discovery=None,
             bucket=_BUCKET) -> FrontierArchive:
    import jax
    codec = StateCodec()
    archive = FrontierArchive(capacity=64, per_bucket_quota=64)
    for index, state_id in enumerate(states):
        state = {"x": float(index), "t": index}
        encoded = codec.encode(StateBundle(state, jax.random.PRNGKey(index), 0, 0, {}))
        entry = FrontierArchiveEntry(
            state_id=state_id,
            source_checkpoint_id="ckpt-1",
            source_episode_id="e1",
            source_seed=index,
            source_timestep=index,
            capture_reason="fixture",
            floor=bucket[0], gate_progress=0.5,
            health_band=bucket[1], threat_band=bucket[2],
            resource_band=bucket[3], inventory_stage=bucket[4],
            achievement_snapshot={}, terminal=bucket[5],
            memory_mode="SAVED_POLICY_MEMORY",
            encoded_state_ref=encoded.payload_hash,
            state_hash=encoded.payload_hash,
            provenance_hash="p" * 64,
            created_at="2026-08-03T00:00:00Z",
            discovery_provenance=(discovery
                                  if discovery is not None
                                  else DiscoveryProvenance.TRAINING_DISCOVERY.value),
        )
        assert archive.add(entry, encoded)
    return archive


def _plan(evidence_hash: str, *, memory_mode: str = "SAVED_POLICY_MEMORY",
          slots=None) -> PlannerOutput:
    start_distribution = {}
    for slot in DISTRIBUTION_SLOT_IDS:
        start_distribution[slot] = slots[slot] if slots else {"s1": 0.5, "s2": 0.5}
    fields = {
        "plan_id": "plan-001",
        "based_on_diagnosis_hash": "c" * 64,
        "bucket_modifications": {"b": 1},
        "start_distribution": start_distribution,
        "taskparam_ranges": {"max_timesteps": 128},
        "seed_distribution": {"kind": "canonical", "base": 7},
        "stochasticity_distribution": {"epsilon": 0.1},
        "search_source": SEARCH_SOURCE_STUDENT_STOCHASTIC,
        "actual_n": 6,
        "horizon": 8,
        "memory_mode": memory_mode,
        "anchor_ratio": 0.25,
        "retention_constraints": ("anchor_ratio>=0.250000",),
        "reason": "fixture",
    }
    plan_hash = compute_planner_hash(fields, evidence_hash=evidence_hash)
    return PlannerOutput(**fields, plan_hash=plan_hash)


class TestCompilerPositive:
    def test_compiles_exactly_12_distributions(self):
        evidence = _evidence()
        archive = _archive()
        compilation = compile_planner_to_frontier_distributions(
            _plan(evidence.evidence_hash),
            plan_evidence_hash=evidence.evidence_hash,
            selection_evidence=evidence,
            archive=archive)
        assert len(compilation.distributions) == 12
        assert [d.distribution_id for d in compilation.distributions] == [
            f"plan-001::{slot}" for slot in DISTRIBUTION_SLOT_IDS]
        # The evidence hash is stamped onto every distribution.
        for distribution in compilation.distributions:
            assert distribution.evidence_hash == evidence.evidence_hash
            assert distribution.memory_mode == "SAVED_POLICY_MEMORY"

    def test_states_share_the_measured_archive_bucket(self):
        evidence = _evidence()
        archive = _archive(states=("s1", "s2", "s3", "s4"))
        compilation = compile_planner_to_frontier_distributions(
            _plan(evidence.evidence_hash),
            plan_evidence_hash=evidence.evidence_hash,
            selection_evidence=evidence,
            archive=archive)
        for distribution in compilation.distributions:
            assert tuple(distribution.bucket) == _BUCKET

    def test_compilation_is_deterministic(self):
        evidence = _evidence()
        archive = _archive()
        a = compile_planner_to_frontier_distributions(
            _plan(evidence.evidence_hash), plan_evidence_hash=evidence.evidence_hash,
            selection_evidence=evidence, archive=archive)
        b = compile_planner_to_frontier_distributions(
            _plan(evidence.evidence_hash), plan_evidence_hash=evidence.evidence_hash,
            selection_evidence=evidence, archive=archive)
        assert a.compilation_hash == b.compilation_hash


class TestCompilerFailClosed:
    def _compile(self, plan, evidence, archive):
        return compile_planner_to_frontier_distributions(
            plan, plan_evidence_hash=evidence.evidence_hash,
            selection_evidence=evidence, archive=archive)

    def test_mapping_plan_refused(self):
        with pytest.raises(InvalidEvidenceError):
            self._compile({"plan_id": "x"}, _evidence(), _archive())

    def test_foreign_plan_refused(self):
        with pytest.raises(InvalidEvidenceError):
            self._compile("plan", _evidence(), _archive())

    def test_plan_bound_to_wrong_evidence_hash_refused(self):
        evidence = _evidence()
        plan = _plan("b" * 64)  # plan binds a DIFFERENT evidence hash
        with pytest.raises(LLMContractError):
            self._compile(plan, evidence, _archive())

    def test_tampered_selection_evidence_refused(self):
        evidence = _evidence()
        tampered = dataclasses.replace(evidence)
        object.__setattr__(tampered, "bucket_diversity", 99)
        with pytest.raises(InvalidEvidenceError):
            self._compile(_plan(evidence.evidence_hash), tampered, _archive())

    def test_missing_slot_refused(self):
        evidence = _evidence()
        plan = _plan(evidence.evidence_hash)
        plan = dataclasses.replace(plan, start_distribution={
            k: v for k, v in plan.start_distribution.items() if k != "D00"})
        with pytest.raises(InvalidEvidenceError):
            self._compile(plan, evidence, _archive())

    def test_extra_slot_refused(self):
        evidence = _evidence()
        plan = _plan(evidence.evidence_hash)
        plan = dataclasses.replace(plan, start_distribution={
            **plan.start_distribution, "EXTRA": {"s1": 1.0}})
        with pytest.raises(InvalidEvidenceError):
            self._compile(plan, evidence, _archive())

    def test_zero_memory_plan_refused(self):
        evidence = _evidence()
        with pytest.raises(InvalidEvidenceError):
            self._compile(
                _plan(evidence.evidence_hash, memory_mode="ZERO_MEMORY"),
                evidence, _archive())

    def test_empty_seed_distribution_refused(self):
        evidence = _evidence()
        plan = _plan(evidence.evidence_hash)
        plan = dataclasses.replace(plan, seed_distribution={})
        with pytest.raises(InvalidEvidenceError):
            self._compile(plan, evidence, _archive())

    def test_state_not_in_archive_refused(self):
        evidence = _evidence()
        archive = _archive(states=("s1", "s2", "s3", "s4"))
        plan = _plan(evidence.evidence_hash)
        plan = dataclasses.replace(plan, start_distribution={
            slot: {"missing-state": 1.0} for slot in DISTRIBUTION_SLOT_IDS})
        with pytest.raises(InvalidEvidenceError):
            self._compile(plan, evidence, archive)

    def test_state_not_training_discovery_refused(self):
        evidence = _evidence()
        archive = _archive(discovery=DiscoveryProvenance.SYNTHETIC_FIXTURE.value)
        with pytest.raises(InvalidEvidenceError):
            self._compile(_plan(evidence.evidence_hash), evidence, archive)

    def test_multi_bucket_slot_refused(self):
        evidence = _evidence()
        archive = _archive()
        # Give the FIRST archive entry a different bucket so slot D00 (s1+s2)
        # spans two buckets.
        entry, _encoded = archive.get("s1")
        new_entry = dataclasses.replace(entry, floor=99)
        archive._entries["s1"] = new_entry
        with pytest.raises(InvalidEvidenceError):
            self._compile(_plan(evidence.evidence_hash), evidence, archive)
