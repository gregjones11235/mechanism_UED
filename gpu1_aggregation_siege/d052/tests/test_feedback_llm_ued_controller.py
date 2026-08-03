"""End-to-end: plan_k -> probe -> comparison -> LLM feedback -> plan_{k+1}.

Runs all three §5 comparison modes on the deterministic mock backend +
symbolic probe runner and asserts the loop is genuinely feedback-driven:
static never invokes the LLM, normal revises on real probe feedback, and
shuffling the candidate<->feedback binding changes the resulting plans.
"""
import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import (
    BOOTSTRAP_PLAN_ID,
    FeedbackUEDController,
)
from d052.feedback_llm_ued.plan_revision import FEEDBACK_DRIVEN_LABEL
from d052.feedback_llm_ued.simulator_probe import (
    DeterministicSymbolicProbeRunner,
)

WINDOWS = 6
TRANSITIONS_PER_PROBED_WINDOW = 61440      # 64 fast + 24 full probes


@pytest.fixture(scope="module")
def runs():
    controllers, summaries = {}, {}
    for mode in (C.MODE_STATIC_LLM, C.MODE_NORMAL_FEEDBACK,
                 C.MODE_SHUFFLED_FEEDBACK):
        ctl = FeedbackUEDController(mode)
        summaries[mode] = ctl.run(max_windows=WINDOWS)
        controllers[mode] = ctl
    return controllers, summaries


class TestAuthorizationPosture:
    def test_all_flags_false_this_round(self):
        assert C.TRAINING_AUTHORIZED is False
        assert C.FORMAL_EVALUATION_AUTHORIZED is False
        assert C.REAL_LLM_CALLS_AUTHORIZED is False
        assert C.REAL_SIMULATOR_PROBE_AUTHORIZED is False
        assert C.REAL_SIMULATOR_PROBE_STATUS == "BLOCKED_NO_LOCAL_CRAFTAX"

    def test_controller_refuses_any_true_flag(self, monkeypatch):
        monkeypatch.setattr(C, "TRAINING_AUTHORIZED", True)
        with pytest.raises(RuntimeError,
                           match="AUTHORIZATION_POSTURE_VIOLATED"):
            FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="UNKNOWN_MODE"):
            FeedbackUEDController("self_training")


class TestStaticBaseline:
    def test_never_reads_feedback_never_calls_llm(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_STATIC_LLM]
        assert s.n_llm_calls == 0
        assert s.revision_rate == 0.0
        assert len(set(s.plan_signature_hashes)) == 1      # plan stays fixed
        assert s.total_simulator_transitions == \
            TRANSITIONS_PER_PROBED_WINDOW                  # only window 0
        for w in s.windows[1:]:
            assert w["reused_previous_plan"] is True
            assert w["invoked_llm"] is False
            assert w["revision_label"] == "REUSED"
            assert w["n_feedback_records"] == 0
            assert w["funnel_stats"] == {}

    def test_bootstrap_window_shape(self, runs):
        _ctls, sums = runs
        w0 = sums[C.MODE_STATIC_LLM].windows[0]
        assert w0["plan_id"] == BOOTSTRAP_PLAN_ID
        assert w0["revision_label"] == C.EXPLORATION_LABEL
        assert w0["invoked_llm"] is False
        assert w0["n_candidates"] == 64
        assert w0["n_feedback_records"] == 64
        fs = w0["funnel_stats"]
        assert fs["raw"] == 64 and fs["stage1_probed"] == 64
        assert fs["stage1_survivors"] == 24
        assert fs["stage2_probed"] == 24
        assert fs["stage2_selected"] == 12
        assert fs["final_batch"] == 16
        assert fs["total_simulator_transitions"] == \
            TRANSITIONS_PER_PROBED_WINDOW


class TestNormalFeedbackLoop:
    def test_llm_invocations_and_revision_rate(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_NORMAL_FEEDBACK]
        # every adaptive window triggered (opposite-probe risk fires the
        # reviewer each time): 5 windows x 3 calls
        assert s.n_llm_calls == 15
        assert s.revision_rate == round(5 / WINDOWS, 4)
        for w in s.windows[1:]:
            assert w["invoked_llm"] is True
            assert w["n_llm_calls"] == 3
            assert w["revision_label"] == FEEDBACK_DRIVEN_LABEL
            assert w["reviewer_invoked"] is True
            assert C.RISK_PROBE_OPPOSITE_DIRECTION in w["risk_triggers"]
            assert w["n_feedback_records"] == 64

    def test_verdict_driven_retention_and_retirement(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_NORMAL_FEEDBACK]
        assert s.supported_retention_rate == 1.0
        assert s.refuted_retirement_rate == 1.0
        dist = s.decision_distribution
        assert dist.get(C.DECISION_RETAIN, 0) > 0
        assert dist.get(C.DECISION_MUTATE, 0) > 0
        assert dist.get(C.DECISION_RETIRE, 0) > 0
        assert s.feedback_citation_coverage > 0.5

    def test_ledger_states_moved_by_bound_feedback(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        statuses = ctl.ledger.by_status()
        assert statuses[C.HYPOTHESIS_SUPPORTED]          # >=1 supported
        assert statuses[C.HYPOTHESIS_REFUTED]            # >=1 refuted
        for rec in ctl.ledger.all():
            assert rec.status in C.HYPOTHESIS_STATUSES
            assert rec.revision_history                  # verdict recorded
            for entry in rec.revision_history:
                assert len(entry["previous_record_hash"]) == 64
        # verdicts cite real feedback records from the store
        all_ids = set(ctl.store.ids())
        for rec in ctl.ledger.all():
            for fid in rec.supporting_feedback_ids + \
                    rec.contradicting_feedback_ids:
                assert fid in all_ids

    def test_revisions_only_cite_existing_feedback(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        all_ids = set(ctl.store.ids())
        assert len(all_ids) == 64 * WINDOWS
        for rev in ctl.revisions:
            if rev.revision_id == "rev-w00-bootstrap":
                assert rev.label == C.EXPLORATION_LABEL
                assert rev.based_on_feedback_ids == []
                continue
            if rev.label == FEEDBACK_DRIVEN_LABEL:
                assert rev.based_on_feedback_ids       # honesty: real cites
            for fid in rev.based_on_feedback_ids:
                assert fid in all_ids
            for mod in rev.modifications:
                if mod.is_exploration:
                    assert mod.based_on_feedback_ids == []

    def test_envelopes_are_hash_bound_mock_calls(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        assert len(ctl.envelopes) == 15
        roles = [e.role for e in ctl.envelopes]
        assert roles.count(C.ROLE_FEEDBACK_DIAGNOSTICIAN) == 5
        assert roles.count(C.ROLE_ADAPTIVE_ENVIRONMENT_DESIGNER) == 5
        assert roles.count(C.ROLE_ADVERSARIAL_REVIEWER) == 5
        for e in ctl.envelopes:
            assert len(e.request_hash) == 64
            assert len(e.response_hash) == 64
            assert e.backend_id == C.MOCK_BACKEND_ID
        assert ctl.backend.real_calls == 0
        assert ctl.runner.real_simulator is False

    def test_simulator_cost_accounting(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_NORMAL_FEEDBACK]
        assert s.total_simulator_transitions == \
            TRANSITIONS_PER_PROBED_WINDOW * WINDOWS
        assert s.transitions_per_useful_environment > 0

    def test_feedback_records_are_normal_binding(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        for rec in ctl.store.all():
            assert rec.provenance["binding"] == "normal"
            assert rec.provenance["real_adapter_status"] == \
                C.REAL_SIMULATOR_PROBE_STATUS


class TestShuffledFeedback:
    def test_binding_labelled_shuffled(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for rec in ctl.store.all():
            assert rec.provenance["binding"] == "shuffled"

    def test_shuffling_changes_plans(self, runs):
        _ctls, sums = runs
        normal, shuffled = (sums[C.MODE_NORMAL_FEEDBACK],
                            sums[C.MODE_SHUFFLED_FEEDBACK])
        assert normal.plan_signature_hashes != shuffled.plan_signature_hashes
        assert normal.decision_distribution != shuffled.decision_distribution
        comparison = FeedbackUEDController.compare_summaries(
            normal, shuffled, sums[C.MODE_STATIC_LLM])
        assert comparison["feedback_binding_matters"] is True
        assert comparison["plan_difference_windows"] >= 1
        assert comparison["static_llm_calls"] == 0
        assert comparison["static_revision_rate"] == 0.0
        assert comparison["static_plan_difference_vs_normal"] >= 1


class TestDeterminism:
    def test_two_runs_byte_identical(self, runs):
        _ctls, sums = runs
        first = json.dumps(sums[C.MODE_NORMAL_FEEDBACK].to_dict(),
                           sort_keys=True)
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        again = json.dumps(ctl.run(max_windows=WINDOWS).to_dict(),
                           sort_keys=True)
        assert first == again

    def test_probe_runner_transitions_match_funnel(self, runs):
        ctls, sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        assert ctl.runner.total_transitions == \
            sums[C.MODE_NORMAL_FEEDBACK].total_simulator_transitions
        assert isinstance(ctl.runner, DeterministicSymbolicProbeRunner)
