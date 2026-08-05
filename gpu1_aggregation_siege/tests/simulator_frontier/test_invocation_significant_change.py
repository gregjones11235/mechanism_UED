# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-9): the 0-or-2 LLM decision is DERIVED from measured
evidence change, never from the mere presence of a previous plan reference.
A stale or tampered previous plan always forces a revision; only a typed
plan that genuinely re-binds the previous evidence hash can be reused.
"""

import pytest

from dicode.simulator_frontier.llm_contracts import (
    LLMContractError,
    PlannerOutput,
    assert_planner_output_bound,
    compute_planner_hash,
    derive_invocation_from_evidence,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

SHA_A = "a" * 64
SHA_B = "b" * 64


def _plan(evidence_hash: str, *, actual_n: int = 6, horizon: int = 8,
          memory_mode: str = "SAVED_POLICY_MEMORY",
          plan_id: str = "plan-001") -> PlannerOutput:
    fields = {
        "plan_id": plan_id,
        "based_on_diagnosis_hash": "c" * 64,
        "bucket_modifications": {"b": 1},
        "start_distribution": {"D00": {"s1": 1.0}},
        "taskparam_ranges": {"t": 1},
        "seed_distribution": {"k": "x"},
        "stochasticity_distribution": {"epsilon": 0.1},
        "search_source": "STUDENT_STOCHASTIC",
        "actual_n": actual_n,
        "horizon": horizon,
        "memory_mode": memory_mode,
        "anchor_ratio": 0.25,
        "retention_constraints": ("anchor_ratio>=0.250000",),
        "reason": "fixture",
    }
    plan_hash = compute_planner_hash(fields, evidence_hash=evidence_hash)
    return PlannerOutput(**fields, plan_hash=plan_hash)


class TestEvidenceChangeInvocation:
    def test_no_previous_plan_forces_revision(self):
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_A, previous_plan=None,
            previous_evidence_hash="", requested_n=6, horizon=8,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 2
        assert "NO_PREVIOUS_PLAN" in reasons

    def test_missing_previous_evidence_hash_forces_revision(self):
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_A, previous_plan=_plan(SHA_A),
            previous_evidence_hash="", requested_n=6, horizon=8,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 2
        assert "NO_PREVIOUS_EVIDENCE_HASH" in reasons

    def test_unchanged_evidence_allows_reuse(self):
        plan = _plan(SHA_A)
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_A, previous_plan=plan,
            previous_evidence_hash=SHA_A, requested_n=6, horizon=8,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 0
        assert reasons == ()
        assert decision.reuse_plan_ref == plan.plan_id

    def test_changed_evidence_hash_forces_revision(self):
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_B, previous_plan=_plan(SHA_A),
            previous_evidence_hash=SHA_A, requested_n=6, horizon=8,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 2
        assert "EVIDENCE_HASH_CHANGED" in reasons

    def test_stale_actual_n_forces_revision(self):
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_A, previous_plan=_plan(SHA_A, actual_n=6),
            previous_evidence_hash=SHA_A, requested_n=9, horizon=8,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 2
        assert "PLAN_ACTUAL_N_STALE" in reasons

    def test_stale_horizon_forces_revision(self):
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_A, previous_plan=_plan(SHA_A, horizon=8),
            previous_evidence_hash=SHA_A, requested_n=6, horizon=16,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 2
        assert "PLAN_HORIZON_STALE" in reasons

    def test_stale_memory_mode_forces_revision(self):
        decision, reasons = derive_invocation_from_evidence(
            current_evidence_hash=SHA_A,
            previous_plan=_plan(SHA_A, memory_mode="HISTORY_BURN_IN"),
            previous_evidence_hash=SHA_A, requested_n=6, horizon=8,
            memory_mode="SAVED_POLICY_MEMORY")
        assert decision.llm_calls == 2
        assert "PLAN_MEMORY_MODE_STALE" in reasons


class TestUntypedOrTamperedPlan:
    def test_mapping_previous_plan_refused(self):
        with pytest.raises(LLMContractError):
            derive_invocation_from_evidence(
                current_evidence_hash=SHA_A, previous_plan={"plan_id": "x"},
                previous_evidence_hash=SHA_A, requested_n=6, horizon=8,
                memory_mode="SAVED_POLICY_MEMORY")

    def test_foreign_previous_plan_refused(self):
        with pytest.raises(LLMContractError):
            derive_invocation_from_evidence(
                current_evidence_hash=SHA_A, previous_plan="plan",
                previous_evidence_hash=SHA_A, requested_n=6, horizon=8,
                memory_mode="SAVED_POLICY_MEMORY")

    def test_tampered_previous_plan_raises_hash_mismatch(self):
        plan = _plan(SHA_A)
        import dataclasses
        tampered = dataclasses.replace(plan, actual_n=99)  # plan_hash now stale
        with pytest.raises(LLMContractError) as exc:
            derive_invocation_from_evidence(
                current_evidence_hash=SHA_A, previous_plan=tampered,
                previous_evidence_hash=SHA_A, requested_n=6, horizon=8,
                memory_mode="SAVED_POLICY_MEMORY")
        assert "PREVIOUS_PLAN_HASH_MISMATCH" in str(exc.value)

    def test_assert_bound_rejects_mapping_and_foreign(self):
        with pytest.raises(LLMContractError):
            assert_planner_output_bound({"plan_id": "x"}, evidence_hash=SHA_A)
        with pytest.raises(LLMContractError):
            assert_planner_output_bound("plan", evidence_hash=SHA_A)
