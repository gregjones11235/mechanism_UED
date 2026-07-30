"""End-to-end unsafe_rest dry-run test (section 15) + legality boundaries."""
import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.controller import BAGRUEdController
from d052.bagr_ued.formal_evaluation_leakage_guard import FormalLeakageViolation
from d052.bagr_ued.synthetic_traces import (
    build_unsafe_rest_raw_rollout,
)
from d052.bagr_ued.training_trace_adapter import TrainingTrajectoryEvidenceAdapter
from d052.bagr_ued.trajectory_evidence import (
    EvidenceSource,
    MockSymbolicAdapter,
)


@pytest.fixture(scope="module")
def result():
    return BAGRUEdController().run_dry_run(build_unsafe_rest_raw_rollout())


def test_e2e_full_chain(result):
    d = result.model_dump()
    patterns = {a["behavior_pattern"] for a in d["anomalies"]}

    # EventExtractor identifies the anomaly
    assert "unsafe_rest_near_hostile" in patterns
    # BehaviorAuditor marks unsafe rest
    findings = [f for f in
                next(e for e in d["board"]["envelopes"]
                     if e["role"] == C.ROLE_BEHAVIOR_AUDITOR)["parsed_json"]
                ["behavior_findings"]
                if f["behavior_pattern"] == "unsafe_rest_near_hostile"]
    assert findings and findings[0]["confidence"] >= 0.5
    # CausalAnalyst: multiple competing causes
    rec = d["reconciliation"]
    assert rec["supported_causal_hypotheses"]
    cats = {h["item_id"].rsplit(":", 1)[-1]
            for h in rec["supported_causal_hypotheses"]}
    assert len(cats) >= 2
    # Tutor: environment counterfactual induction (unsafe_rest axes covered)
    assert "threat_distance_grading" in rec["required_counterfactual_tests"]
    assert "safe_rest_area_availability" in rec["required_counterfactual_tests"]
    # Explorer: alternative env families, disjoint from the Tutor's axes
    exp = next(e for e in d["board"]["envelopes"]
               if e["role"] == C.ROLE_EXPLORER)["parsed_json"]
    assert len(exp["alternative_environment_proposals"]) >= 2
    # Reconciler keeps verifiable hypotheses (bound provenance)
    for it in rec["supported_causal_hypotheses"]:
        assert it["bound_role_output_hashes"]
    # Proposer: mock Global TaskParams candidates, all legal
    assert d["descriptors"] and not d["rejected_descriptors"]
    for desc in d["descriptors"]:
        assert desc["real_adapter_status"] == C.REAL_TASKPARAMS_ADAPTER
    # Soft Copeland receives the behavioral gap dimension
    comps = d["copeland_ranking"]["entries"][0]["components"]
    assert "behavioral_gap" in comps
    # Budget: 12 UED + 4 anchors
    bp = d["budget_plan"]
    assert len(bp["ued_slots"]) == 12 and len(bp["anchor_slots"]) == 4


def test_e2e_no_guidance_no_shaping_no_bank(result):
    cert = result.dry_run_certificate
    assert cert["run_class"] == "ENGINEERING_DRY_RUN"
    assert cert["training_authorized"] is False
    assert cert["performance_claim_authorized"] is False
    assert cert["real_llm_calls"] == 0
    assert cert["real_environment_rollouts"] == 0
    assert cert["formal_front_bank_used"] is False
    assert cert["formal_back_bank_used"] is False
    assert cert["student_action_guidance_emitted"] is False
    assert cert["reward_shaping_emitted"] is False
    assert cert["gpu_used"] is False
    assert cert["push_performed"] is False
    assert cert["training_started"] is False
    assert cert["formal_evaluation_started"] is False
    assert result.ued_nature_assertions["no_action_guidance_to_student"]
    assert result.ued_nature_assertions["no_reward_shaping_emitted"]


def test_e2e_global_not_tier3_only(result):
    cert = result.dry_run_certificate
    assert cert["training_scope"] == "GLOBAL"
    assert cert["tier3_only_training"] is False
    assert cert["dry_run_env_count"] == 16
    assert cert["dry_run_transitions"] == 2048
    assert cert["review_interval_transitions"] == 8192
    patterns = {a["behavior_pattern"] for a in result.model_dump()["anomalies"]}
    # GLOBAL patterns beyond the threat axis
    assert patterns & {"resource_neglect", "repeated_no_effect"}


def test_formal_front_input_fails_closed():
    adapter = TrainingTrajectoryEvidenceAdapter(
        MockSymbolicAdapter(__import__(
            "d052.bagr_ued.synthetic_traces", fromlist=["TEST_VOCABULARY"]
        ).TEST_VOCABULARY))
    with pytest.raises(FormalLeakageViolation):
        adapter.adapt(build_unsafe_rest_raw_rollout(), bundle_id="x",
                      source=EvidenceSource.FORMAL_FRONT)


def test_generative_training_source_allowed():
    from d052.bagr_ued.synthetic_traces import TEST_VOCABULARY
    adapter = TrainingTrajectoryEvidenceAdapter(MockSymbolicAdapter(TEST_VOCABULARY))
    bundle = adapter.adapt(build_unsafe_rest_raw_rollout(), bundle_id="ok",
                           source=EvidenceSource.GENERATIVE_TRAINING_ENV)
    assert bundle.leakage_guard_status == "PASS"
    assert bundle.source is EvidenceSource.GENERATIVE_TRAINING_ENV


def test_legality_gate_rejects_invented_fields():
    from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
    # extra=forbid (pydantic "Extra inputs are not permitted") and the
    # UNAUTHORIZED_DESCRIPTOR_FIELD whitelist validator are BOTH fail-closed
    # paths against guessed real TaskParams fields; either message proves the
    # invention was refused.
    with pytest.raises(Exception,
                       match=r"Extra inputs are not permitted|"
                             r"UNAUTHORIZED_DESCRIPTOR_FIELD"):
        TaskParamsDescriptor(
            descriptor_id="tpd:bad",
            mock_env_family="f", mock_variant_id="v", mock_variant_kind="k",
            mob_spawn_rate=3.0)   # guessed real field -> hard error
