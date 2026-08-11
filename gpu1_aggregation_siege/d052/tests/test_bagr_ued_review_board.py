"""Review board + reconciler tests (sections 5-10)."""
import json

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.behavior_clip_selector import BehaviorClipSelector
from d052.bagr_ued.event_extractor import DeterministicEventExtractor
from d052.bagr_ued.mock_llm_backend import DeterministicMockBackend
from d052.bagr_ued.review_board import ReviewBoard
from d052.bagr_ued.review_reconciler import ReviewBoardReconciler
from d052.bagr_ued.synthetic_traces import (
    TEST_VOCABULARY,
    build_unsafe_rest_raw_rollout,
)
from d052.bagr_ued.training_trace_adapter import TrainingTrajectoryEvidenceAdapter
from d052.bagr_ued.trajectory_evidence import (
    EvidenceSource,
    MockSymbolicAdapter,
)
from d052.bagr_ued.trajectory_supervision_guard import GuardViolation


def _evidence():
    adapter = TrainingTrajectoryEvidenceAdapter(MockSymbolicAdapter(TEST_VOCABULARY))
    bundle = adapter.adapt(build_unsafe_rest_raw_rollout(), bundle_id="t",
                           source=EvidenceSource.SYNTHETIC_TEST_TRACE)
    anomalies = DeterministicEventExtractor().extract(bundle)
    clips, _ = BehaviorClipSelector().select(bundle, anomalies)
    return bundle, anomalies, clips


def _board_output(backend=None):
    bundle, anomalies, clips = _evidence()
    backend = backend or DeterministicMockBackend()
    board = ReviewBoard(backend)
    manifest = DeterministicEventExtractor().detector_manifest()
    return board, board.run(bundle, anomalies, clips, manifest)


def test_all_six_roles_run_in_fixed_order():
    _, out = _board_output()
    assert [e.role for e in out.envelopes] == list(C.REVIEW_BOARD_ROLES)
    assert [e.sequence for e in out.envelopes] == [0, 1, 2, 3, 4, 5]
    assert out.supervision_guard_status == "PASS"
    assert out.leakage_guard_status == "PASS"
    assert out.real_llm_calls == 0
    for e in out.envelopes:
        assert len(e.request_hash) == 64 and len(e.response_hash) == 64
        assert e.backend_id == C.MOCK_BACKEND_ID


def test_auditor_outputs_no_causes_proposals_or_advice():
    _, out = _board_output()
    auditor = next(e for e in out.envelopes
                   if e.role == C.ROLE_BEHAVIOR_AUDITOR)
    for f in auditor.parsed_json["behavior_findings"]:
        # protocol_version is the d052 CanonicalModel identity field, not
        # content; everything beyond it + protocol_version would mean the
        # auditor leaked causes / proposals / advice into its schema.
        assert set(f) <= {"finding_id", "behavior_pattern", "severity",
                          "recurrence", "evidence_span_ids",
                          "supporting_fields", "counter_evidence",
                          "confidence", "protocol_version"}
    unsafe = [f for f in auditor.parsed_json["behavior_findings"]
              if f["behavior_pattern"] == "unsafe_rest_near_hostile"]
    assert unsafe and unsafe[0]["recurrence"] >= 1


def test_analyst_multi_competing_causes_within_vocabulary():
    _, out = _board_output()
    analyst = next(e for e in out.envelopes
                   if e.role == C.ROLE_CAUSAL_FAILURE_ANALYST)
    by_finding = {}
    for h in analyst.parsed_json["causal_hypotheses"]:
        assert h["cause_category"] in C.CAUSE_CATEGORIES
        for a in h["required_counterfactual_variables"]:
            assert a in C.MUTATION_AXES
        assert h["testable_prediction"]
        by_finding.setdefault(h["finding_id"], set()).add(h["cause_category"])
    for fid, cats in by_finding.items():
        assert len(cats) >= 2, f"single-cause attribution for {fid}"


def test_tutor_axes_legal_with_control_groups():
    _, out = _board_output()
    tutor = next(e for e in out.envelopes
                 if e.role == C.ROLE_INTERVENTION_TUTOR)
    axes_all = set()
    for itv in tutor.parsed_json["intervention_hypotheses"]:
        assert set(itv["mutation_axes"]) <= set(C.MUTATION_AXES)
        assert "control" in itv["counterfactual_groups"]
        axes_all.update(itv["mutation_axes"])
    # the required unsafe_rest axes must be collectively covered
    assert {"threat_distance_grading", "safe_rest_area_availability",
            "rest_need_pressure", "threat_count", "view_occlusion"} <= axes_all


def test_explorer_proposes_families_different_from_tutor():
    _, out = _board_output()
    tutor = next(e for e in out.envelopes
                 if e.role == C.ROLE_INTERVENTION_TUTOR)
    explorer = next(e for e in out.envelopes if e.role == C.ROLE_EXPLORER)
    tutor_axes = {a for i in tutor.parsed_json["intervention_hypotheses"]
                  for a in i["mutation_axes"]}
    fams = {p["environment_family"]
            for p in explorer.parsed_json["alternative_environment_proposals"]}
    # on this trace the Tutor already covers 8 of the 11 mutation axes, so the
    # disjoint family remainder is exactly {multi-monster interference, global
    # task conflict}; section 15 requires alternative families DIFFERENT from
    # the Tutor's, not a fixed count.
    assert fams and len(fams) >= 2
    # no proposed family may be one the tutor already covers
    from d052.bagr_ued.explorer import FAMILY_PRIMARY_AXES
    for fam in fams:
        assert not (FAMILY_PRIMARY_AXES[fam] & tutor_axes), fam


def test_critic_keeps_rules_pending_and_evidence_separate():
    _, out = _board_output()
    critic = next(e for e in out.envelopes if e.role == C.ROLE_CRITIC_SKEPTIC)
    cj = critic.parsed_json
    assert cj["real_canonical_critic_reject_derivation_rule"] == "PENDING"
    assert cj["real_canonical_critic_selection_policy"] == "PENDING"
    # the two evidence blocks are kept DISTINCT (never merged)
    assert cj["reject_derivation_evidence"] != \
        cj["selection_recommendation_evidence"]
    assert set(cj["reject_derivation_evidence"]) != \
        set(cj["selection_recommendation_evidence"])
    dims = {i["dimension"] for i in cj["critique_items"]}
    assert {"evidence_sufficiency", "action_guidance_leakage",
            "formal_info_usage", "tier3_only_bias",
            "counterfactual_controls", "falsifiability"} <= dims


def test_critic_schema_refuses_frozen_rules():
    from d052.bagr_ued.critic_skeptic import CriticSkepticOutput
    with pytest.raises(Exception, match="CRITIC_RULE_FROZEN_FORBIDDEN"):
        CriticSkepticOutput(
            real_canonical_critic_reject_derivation_rule="FROZEN_BY_CC3",
            real_canonical_critic_selection_policy="PENDING")


def test_reconciler_binds_provenance_and_is_deterministic():
    _, out = _board_output()
    r1 = ReviewBoardReconciler().reconcile(out)
    r2 = ReviewBoardReconciler().reconcile(out)
    assert r1.reconciliation_hash == r2.reconciliation_hash
    items = (r1.accepted_behavior_findings + r1.supported_causal_hypotheses +
             r1.accepted_intervention_hypotheses)
    assert items
    for it in items:
        assert it.bound_role_output_hashes
        assert all(len(h) == 64 for h in it.bound_role_output_hashes.values())
        assert it.prompt_versions and it.backend_id and it.model_id
        assert it.reconciliation_rule_version == C.RECONCILIATION_RULE_VERSION
    assert r1.required_counterfactual_tests
    assert r1.supported_causal_hypotheses


class _AdviceInjectingBackend(DeterministicMockBackend):
    """Delegates everything, except the Analyst emits direct action advice."""

    def complete(self, role, prompt):
        if role == C.ROLE_CAUSAL_FAILURE_ANALYST:
            return json.dumps(dict(causal_hypotheses=[
                dict(hypothesis_id="hyp:bad:1", finding_id="finding:x",
                     cause_category="value_or_risk_misestimation",
                     causal_statement="You should flee from the monster and "
                                      "never sleep.",
                     supporting_evidence=["s"], contradicting_evidence=["c"],
                     alternative_explanations=["unknown"], confidence=0.5,
                     testable_prediction="p",
                     required_counterfactual_variables=["threat_count"]),
                dict(hypothesis_id="hyp:bad:2", finding_id="finding:x",
                     cause_category="unknown",
                     causal_statement="Don't sleep near hostiles.",
                     supporting_evidence=["s"], contradicting_evidence=["c"],
                     alternative_explanations=["exploration_noise"],
                     confidence=0.4, testable_prediction="p",
                     required_counterfactual_variables=[]),
            ]))
        return super().complete(role, prompt)


def test_board_fails_closed_on_direct_action_advice():
    with pytest.raises(GuardViolation) as ei:
        _board_output(backend=_AdviceInjectingBackend())
    assert ei.value.code == GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN
