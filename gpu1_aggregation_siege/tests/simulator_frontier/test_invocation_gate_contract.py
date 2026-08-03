"""CONTRACT tests for the InvocationGate 0-or-2 rule (fake_client only).

Naming discipline (review condition 3): every LLM double in this file is a
``FakeLLMClient``; nothing here touches a real API.  TWO_LLM_GATE_CONTRACT_ONLY
means CONTRACT_AND_FAKE_CLIENT_TEST_READY and nothing more;
REAL_TWO_LLM_CALL_EXECUTED stays False this round.
"""

from __future__ import annotations

import pytest

from dicode.simulator_frontier import (
    LLM_ROLE_SEQUENCE,
    BranchOutcome,
    DataSource,
    FeasibilityEstimate,
    FakeLLMClient,
    InvocationContractError,
    InvocationDecision,
    InvocationReason,
    assert_never_exactly_one_call,
    build_aggregate_evidence,
    decide_invocation,
    deterministic_select,
    estimate_feasibility,
    evidence_hash_of,
    run_two_llm_gate,
)
from dicode.simulator_frontier.errors import (
    InvalidEvidenceError,
    ProvenanceViolationError,
)
from dicode.simulator_frontier.invocation_gate import (
    REAL_TWO_LLM_CALL_EXECUTED,
    TWO_LLM_GATE_CONTRACT_ONLY,
)


def _outcomes() -> list[BranchOutcome]:
    rows = []
    for i in range(8):
        rows.append(BranchOutcome(
            branch_id=f"b{i}", state_id="S1", search_source="actual_n",
            rng_seed=i, horizon=32, transitions_used=32,
            success=(i % 2 == 0), progress=0.25 * (i % 4),
            terminal_event=None, failure_category=None,
            memory_mode="SAVED_POLICY_MEMORY", outcome_hash=f"h{i}"))
    return rows


def _feasibility() -> FeasibilityEstimate:
    return estimate_feasibility(_outcomes(), state_id="S1")


class TestZeroOrTwoRule:
    def test_no_significant_change_is_zero_calls_and_needs_reuse_ref(self):
        decision = decide_invocation(InvocationReason.NO_SIGNIFICANT_CHANGE,
                                     reuse_plan_ref="plan:v7")
        assert decision.llm_calls == 0
        assert decision.reuse_plan_ref == "plan:v7"
        assert decision.planned_roles == ()

    def test_no_significant_change_without_reuse_ref_raises(self):
        with pytest.raises(InvocationContractError):
            decide_invocation(InvocationReason.NO_SIGNIFICANT_CHANGE)

    def test_revision_required_is_exactly_two_in_fixed_order(self):
        decision = decide_invocation(InvocationReason.REVISION_REQUIRED)
        assert decision.llm_calls == 2
        assert decision.planned_roles == LLM_ROLE_SEQUENCE
        assert decision.reuse_plan_ref is None

    def test_attempts_equal_one_is_always_a_violation(self):
        with pytest.raises(InvocationContractError):
            InvocationDecision(InvocationReason.REVISION_REQUIRED, 1, None,
                               LLM_ROLE_SEQUENCE)
        with pytest.raises(InvocationContractError):
            assert_never_exactly_one_call([0, 1, 2])

    def test_zero_call_decision_cannot_schedule_roles(self):
        with pytest.raises(InvocationContractError):
            InvocationDecision(InvocationReason.NO_SIGNIFICANT_CHANGE, 0,
                               "plan:v7", ("curriculum_search_planner",))

    def test_two_call_decision_rejects_reuse_ref(self):
        with pytest.raises(InvocationContractError):
            InvocationDecision(InvocationReason.REVISION_REQUIRED, 2, "plan:v7",
                               LLM_ROLE_SEQUENCE)

    def test_two_call_decision_requires_exact_role_order(self):
        with pytest.raises(InvocationContractError):
            InvocationDecision(InvocationReason.REVISION_REQUIRED, 2, None,
                               tuple(reversed(LLM_ROLE_SEQUENCE)))

    def test_round_status_constants_are_honest(self):
        assert TWO_LLM_GATE_CONTRACT_ONLY is True
        assert REAL_TWO_LLM_CALL_EXECUTED is False


class TestAggregateEvidenceBoundary:
    def test_aggregate_evidence_builds_from_feasibility_only(self):
        evidence = build_aggregate_evidence(
            _feasibility(), {"entries": 3, "buckets": 2},
            data_source=DataSource.TRAINING_BRANCH_SEARCH.value)
        assert evidence["feasibility"]["total_actual_branches"] == 8
        assert evidence["feasibility"]["successes"] == 4
        assert evidence["archive_summary"] == {"entries": 3, "buckets": 2}

    @pytest.mark.parametrize("formal_source", [
        DataSource.FORMAL_FRONT.value, DataSource.FORMAL_BACK.value,
        DataSource.FORMAL_FULL.value])
    def test_formal_sources_never_feed_the_gate(self, formal_source):
        with pytest.raises(ProvenanceViolationError):
            build_aggregate_evidence(_feasibility(), {}, data_source=formal_source)

    @pytest.mark.parametrize("forbidden_key", [
        "action_sequence", "waypoint", "route", "logits", "hidden_states",
        "expert_trajectory", "successful_actions", "action"])
    def test_forbidden_fields_in_archive_summary_raise(self, forbidden_key):
        with pytest.raises(ProvenanceViolationError):
            build_aggregate_evidence(
                _feasibility(), {"summary": {forbidden_key: [1, 2, 3]}},
                data_source=DataSource.TRAINING_BRANCH_SEARCH.value)

    def test_aggregate_keys_like_action_count_stay_legal(self):
        evidence = build_aggregate_evidence(
            _feasibility(), {"action_count": 43},
            data_source=DataSource.TRAINING_BRANCH_SEARCH.value)
        assert evidence["archive_summary"]["action_count"] == 43


class TestFakeClientGateExecution:
    def _decision_two(self):
        return decide_invocation(InvocationReason.REVISION_REQUIRED)

    def _decision_zero(self):
        return decide_invocation(InvocationReason.NO_SIGNIFICANT_CHANGE,
                                 reuse_plan_ref="plan:v7")

    def _evidence(self):
        return build_aggregate_evidence(
            _feasibility(), {"entries": 1},
            data_source=DataSource.TRAINING_BRANCH_SEARCH.value)

    def test_fake_client_zero_call_run_never_invokes_clients(self):
        diag = FakeLLMClient(LLM_ROLE_SEQUENCE[0])
        plan = FakeLLMClient(LLM_ROLE_SEQUENCE[1])
        result = run_two_llm_gate(self._decision_zero(), self._evidence(),
                                  diagnostician_client=diag, planner_client=plan)
        assert result["llm_calls"] == 0
        assert result["reuse_plan_ref"] == "plan:v7"
        assert diag.calls == [] and plan.calls == []

    def test_fake_client_two_call_run_uses_fixed_role_order(self):
        diag = FakeLLMClient(LLM_ROLE_SEQUENCE[0], {"diagnosis": "stall", "severity": 2})
        plan = FakeLLMClient(LLM_ROLE_SEQUENCE[1],
                             {"plan_id": "p1", "curriculum_ref": "c1",
                              "priority_score": 0.5})
        result = run_two_llm_gate(self._decision_two(), self._evidence(),
                                  diagnostician_client=diag, planner_client=plan)
        assert result["llm_calls"] == 2
        assert result["role_order"] == LLM_ROLE_SEQUENCE
        assert len(diag.calls) == 1 and len(plan.calls) == 1
        # planner saw the diagnostician aggregate summary (and nothing else)
        assert plan.calls[0]["evidence"]["diagnostician_summary"]["diagnosis"] == "stall"

    def test_fake_client_diagnostician_output_with_actions_raises(self):
        diag = FakeLLMClient(LLM_ROLE_SEQUENCE[0], {"successful_actions": [1, 2]})
        plan = FakeLLMClient(LLM_ROLE_SEQUENCE[1])
        with pytest.raises(ProvenanceViolationError):
            run_two_llm_gate(self._decision_two(), self._evidence(),
                             diagnostician_client=diag, planner_client=plan)

    def test_fake_client_planner_output_with_route_raises(self):
        diag = FakeLLMClient(LLM_ROLE_SEQUENCE[0], {"diagnosis": "ok"})
        plan = FakeLLMClient(LLM_ROLE_SEQUENCE[1], {"route": ["a", "b"]})
        with pytest.raises(ProvenanceViolationError):
            run_two_llm_gate(self._decision_two(), self._evidence(),
                             diagnostician_client=diag, planner_client=plan)

    def test_evidence_with_forbidden_field_is_rejected_upfront(self):
        diag = FakeLLMClient(LLM_ROLE_SEQUENCE[0])
        plan = FakeLLMClient(LLM_ROLE_SEQUENCE[1])
        bad = dict(self._evidence())
        bad["waypoint"] = "x"
        with pytest.raises(ProvenanceViolationError):
            run_two_llm_gate(self._decision_two(), bad,
                             diagnostician_client=diag, planner_client=plan)
        assert diag.calls == [] and plan.calls == []


class TestDeterministicSelector:
    def _evidence_hash(self):
        return evidence_hash_of({"feasibility": {"success_rate": 0.5}})

    def test_valid_candidates_select_by_score_then_id(self):
        candidates = [
            {"plan_id": "p_b", "curriculum_ref": "c1", "priority_score": 0.9},
            {"plan_id": "p_a", "curriculum_ref": "c2", "priority_score": 0.9},
            {"plan_id": "p_c", "curriculum_ref": "c3", "priority_score": 0.1},
        ]
        result = deterministic_select(candidates, evidence_hash=self._evidence_hash())
        assert result.chosen_plan_id == "p_a"  # tie broken by smallest plan_id
        assert result.rejected == ()

    def test_selection_hash_is_reproducible(self):
        candidates = [{"plan_id": "p_a", "curriculum_ref": "c1", "priority_score": 1.0}]
        r1 = deterministic_select(candidates, evidence_hash=self._evidence_hash())
        r2 = deterministic_select(candidates, evidence_hash=self._evidence_hash())
        assert r1.selection_hash == r2.selection_hash and len(r1.selection_hash) == 64

    def test_llm_output_smuggling_actions_is_vetoed(self):
        candidates = [
            {"plan_id": "evil", "curriculum_ref": "c1", "priority_score": 99.0,
             "action_sequence": [3, 4, 5]},
            {"plan_id": "good", "curriculum_ref": "c2", "priority_score": 0.1},
        ]
        result = deterministic_select(candidates, evidence_hash=self._evidence_hash())
        assert result.chosen_plan_id == "good"  # high-score smuggler vetoed
        assert "evil" in result.rejected
        assert "FORBIDDEN_ACTION_GUIDANCE_FIELD" in result.reason_codes

    def test_invalid_candidates_are_rejected_with_reasons(self):
        candidates = [
            {"plan_id": "", "curriculum_ref": "c", "priority_score": 1.0},
            {"plan_id": "p", "curriculum_ref": "c", "priority_score": "high"},
            {"plan_id": "valid", "curriculum_ref": "c", "priority_score": 0.2},
        ]
        result = deterministic_select(candidates, evidence_hash=self._evidence_hash())
        assert result.chosen_plan_id == "valid"
        assert len(result.rejected) == 2
        assert "MISSING_OR_INVALID_REQUIRED_FIELDS" in result.reason_codes

    def test_all_invalid_candidates_fail_closed(self):
        candidates = [
            {"plan_id": "", "curriculum_ref": "c", "priority_score": 1.0},
            {"plan_id": "p", "curriculum_ref": "c", "priority_score": "high"},
        ]
        with pytest.raises(InvalidEvidenceError):
            deterministic_select(candidates, evidence_hash=self._evidence_hash())

    def test_missing_evidence_hash_raises(self):
        with pytest.raises(InvalidEvidenceError):
            deterministic_select(
                [{"plan_id": "p", "curriculum_ref": "c", "priority_score": 1.0}],
                evidence_hash="")
