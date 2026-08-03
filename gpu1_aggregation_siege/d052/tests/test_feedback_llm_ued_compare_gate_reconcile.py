"""ExpectedObservedComparator + FeedbackInvocationGate + DeterministicReconciler."""
import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.deterministic_reconciler import (
    EXPLORATION_SLOT_CAP,
    DeterministicReconciler,
)
from d052.feedback_llm_ued.expected_observed import (
    ExpectedObservedComparator,
    relative_gap,
)
from d052.feedback_llm_ued.feedback_contracts import FamilyAllocation
from d052.feedback_llm_ued.feedback_invocation_gate import (
    GateInput,
    evaluate_gate,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

FAM = C.ENVIRONMENT_FAMILIES[0]
FAM2 = C.ENVIRONMENT_FAMILIES[1]
FAM3 = C.ENVIRONMENT_FAMILIES[2]
FAM4 = C.ENVIRONMENT_FAMILIES[3]


# --------------------------------------------------------------- comparator
class TestComparator:
    def test_relative_gap(self):
        assert relative_gap(0.5, 0.6) == pytest.approx(0.2)
        # zero expected does not divide
        assert relative_gap(0.0, 0.1) > 100

    def test_agree_neutral_opposite_thresholds(self):
        cmp_ = ExpectedObservedComparator()            # tol = 0.25
        obs = {"student_success_rate": 0.6}            # gap 0.20 -> agree
        out = cmp_.compare({"student_success_rate": 0.5}, obs)
        assert out["overall"] == C.MATCH_DIRECTION_AGREE
        obs = {"student_success_rate": 0.7}            # gap 0.40 -> neutral
        out = cmp_.compare({"student_success_rate": 0.5}, obs)
        assert out["overall"] == C.MATCH_DIRECTION_NEUTRAL
        obs = {"student_success_rate": 0.95}           # gap 0.90 -> opposite
        out = cmp_.compare({"student_success_rate": 0.5}, obs)
        assert out["overall"] == C.MATCH_DIRECTION_OPPOSITE

    def test_counter_semantics(self):
        cmp_ = ExpectedObservedComparator()
        exp = {"student_success_rate": 0.8}
        cnt = {"student_success_rate": 0.2}
        # observation clearly closer to the prediction -> agree
        out = cmp_.compare(exp, {"student_success_rate": 0.75}, counter=cnt)
        assert out["per_metric"]["student_success_rate"]["direction"] == \
            C.MATCH_DIRECTION_AGREE
        # observation clearly closer to the counter -> opposite
        out = cmp_.compare(exp, {"student_success_rate": 0.25}, counter=cnt)
        assert out["per_metric"]["student_success_rate"]["direction"] == \
            C.MATCH_DIRECTION_OPPOSITE
        # equidistant -> neutral
        out = cmp_.compare(exp, {"student_success_rate": 0.5}, counter=cnt)
        assert out["per_metric"]["student_success_rate"]["direction"] == \
            C.MATCH_DIRECTION_NEUTRAL

    def test_overall_majority_and_tie(self):
        cmp_ = ExpectedObservedComparator()
        expected = {"a": 0.5, "b": 0.5, "c": 0.5}
        observed = {"a": 0.5, "b": 0.5, "c": 0.95}     # 2 agree, 1 opposite
        # metric names must be observable-ish but compare() is generic
        out = cmp_.compare(expected, observed)
        assert out["n_agree"] == 2 and out["n_opposite"] == 1
        assert out["overall"] == C.MATCH_DIRECTION_AGREE
        observed = {"a": 0.95, "b": 0.5}               # 1 opposite, 1 agree
        out = cmp_.compare(expected, observed)
        assert out["overall"] == C.MATCH_DIRECTION_NEUTRAL   # tie
        out = cmp_.compare({"a": 0.5}, {})             # no overlap
        assert out["no_overlap"] is True
        assert out["overall"] == C.MATCH_DIRECTION_NEUTRAL
        assert out["skipped_expected_metrics"] == ["a"]

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ValueError, match="ILLEGAL_COMPARATOR_TOLERANCE"):
            ExpectedObservedComparator(tol=0.0)

    def test_grade_record_binds_and_rehashes(self):
        store = SimulatorFeedbackStore()
        cand = synthetic_candidate(candidate_id="cand-1", family=FAM)
        rec = store.add(synthetic_feedback_record(
            feedback_id="fb-1", candidate=cand, plan_id="p", window=0,
            student_success_rate=0.5,
            expected_signature={"student_success_rate": 0.5}))
        h0 = rec.record_hash
        detail = ExpectedObservedComparator().grade_record(store, "fb-1")
        assert detail["overall"] == C.MATCH_DIRECTION_AGREE
        rec = store.get("fb-1")
        assert rec.expected_observed_match == C.MATCH_DIRECTION_AGREE
        assert rec.record_hash != h0

    def test_grade_record_without_metrics_is_neutral(self):
        store = SimulatorFeedbackStore()
        cand = synthetic_candidate(candidate_id="cand-1", family=FAM)
        rec = SimulatorFeedbackRecord(
            feedback_id="fb-nm", candidate_id=cand.candidate_id,
            candidate_hash=cand.candidate_hash, source_plan_id="p",
            window=0, environment_family=FAM,
            expected_signature={"student_success_rate": 0.5})
        store.add(rec)
        detail = ExpectedObservedComparator().grade_record(store, "fb-nm")
        assert detail["overall"] == C.MATCH_DIRECTION_NEUTRAL
        assert detail["reason"] == "NO_PROBE_METRICS"


# --------------------------------------------------------------------- gate
def _gate(**over):
    base = dict(window=1, has_prior_diagnosis=True)
    base.update(over)
    return evaluate_gate(GateInput(**base))


class TestGate:
    def test_no_condition_no_invoke(self):
        out = _gate(valid_candidate_count=C.STAGE1_KEEP)
        assert out["invoke_llm"] is False
        assert out["conditions"] == ()
        assert "reuse previous diagnosis" in out["reason"]

    def test_each_condition_individually(self):
        cases = [
            (dict(has_prior_diagnosis=False), C.GATE_FIRST_WINDOW),
            (dict(window=0), C.GATE_FIRST_WINDOW),
            (dict(new_detector_types=["det_x"]), C.GATE_NEW_DETECTOR_TYPE),
            (dict(core_behavior_rate_change=0.25),
             C.GATE_CORE_BEHAVIOR_RATE_SHIFT),
            (dict(front_stalled_windows=2),
             C.GATE_FRONT_STALLED_TWO_WINDOWS),
            (dict(global_retention_delta=-0.05),
             C.GATE_GLOBAL_RETENTION_REGRESSION),
            (dict(previous_plan_exhausted=True),
             C.GATE_PREVIOUS_PLAN_EXHAUSTED),
            (dict(valid_candidate_count=10, required_candidate_count=24),
             C.GATE_INSUFFICIENT_VALID_CANDIDATES),
            (dict(cached_plan_age_windows=4), C.GATE_CACHED_PLAN_AGE),
        ]
        for kwargs, expected in cases:
            kwargs.setdefault("valid_candidate_count", C.STAGE1_KEEP)
            out = _gate(**kwargs)
            assert out["invoke_llm"] is True, kwargs
            assert expected in out["conditions"], kwargs

    def test_thresholds_are_strict(self):
        out = _gate(core_behavior_rate_change=0.24,
                    front_stalled_windows=1,
                    global_retention_delta=-0.04,
                    cached_plan_age_windows=3,
                    valid_candidate_count=C.STAGE1_KEEP)
        assert out["invoke_llm"] is False

    def test_all_non_first_conditions_fire_together(self):
        out = _gate(new_detector_types=["d"], core_behavior_rate_change=0.3,
                    front_stalled_windows=3, global_retention_delta=-0.1,
                    previous_plan_exhausted=True, valid_candidate_count=1,
                    required_candidate_count=24, cached_plan_age_windows=5)
        assert len(out["conditions"]) == 7          # first_window excluded
        # deterministic order == GATE_MUST_INVOKE_CONDITIONS order
        order = [c for c in C.GATE_MUST_INVOKE_CONDITIONS
                 if c != C.GATE_FIRST_WINDOW]
        assert list(out["conditions"]) == order

    def test_window_zero_can_fire_all_eight(self):
        out = _gate(window=0, new_detector_types=["d"],
                    core_behavior_rate_change=0.3, front_stalled_windows=3,
                    global_retention_delta=-0.1, previous_plan_exhausted=True,
                    valid_candidate_count=1, required_candidate_count=24,
                    cached_plan_age_windows=5)
        assert len(out["conditions"]) == 8


# --------------------------------------------------------------- reconciler
def _alloc(family, decision, slots, cited=(), exploration=False):
    return FamilyAllocation(environment_family=family, slots=slots,
                            decision=decision,
                            based_on_feedback_ids=list(cited),
                            reason=f"test {decision}",
                            is_exploration=exploration)


class TestReconciler:
    def setup_method(self):
        self.rec = DeterministicReconciler()
        self.known = {"fb-1", "fb-2", "fb-3"}

    def test_budget_fills_exactly_12_and_is_deterministic(self):
        props = [_alloc(FAM, C.DECISION_RETAIN, 4, ["fb-1"]),
                 _alloc(FAM2, C.DECISION_MUTATE, 4, ["fb-2"]),
                 _alloc(FAM3, C.DECISION_EXPAND_BUDGET, 4, ["fb-3"])]
        rc1 = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                 proposals=props, known_feedback_ids=self.known)
        rc2 = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                 proposals=props, known_feedback_ids=self.known)
        total = sum(a.slots for a in rc1.plan.allocations)
        assert total == C.DYNAMIC_UED_SLOTS == 12
        assert rc1.plan.plan_id == rc2.plan.plan_id
        assert rc1.plan.plan_id.startswith("plan-0001-")
        assert rc1.stats["anchor_ids"] == list(C.GLOBAL_CANONICAL_ANCHOR_IDS)
        assert rc1.stats["final_batch"] == C.FINAL_BATCH

    def test_leftover_tops_up_highest_priority_core(self):
        props = [_alloc(FAM, C.DECISION_MUTATE, 3, ["fb-1"]),
                 _alloc(FAM2, C.DECISION_RETAIN, 4, ["fb-2"])]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        slots = {a.environment_family: a.slots for a in rc.plan.allocations}
        assert slots[FAM2] == 9          # RETAIN priority wins the top-up
        assert slots[FAM] == 3
        assert any(e["rule"] == "leftover_top_up" for e in rc.log)

    def test_exploration_reserves_slice_despite_core_demand(self):
        core = [_alloc(f, C.DECISION_RETAIN, 4, ["fb-1"])
                for f in (FAM, FAM2, FAM3, FAM4)]
        props = core + [_alloc(C.ENVIRONMENT_FAMILIES[4], C.DECISION_MUTATE,
                               2, exploration=True)]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        slots = {a.environment_family: a.slots for a in rc.plan.allocations}
        assert slots[C.ENVIRONMENT_FAMILIES[4]] == EXPLORATION_SLOT_CAP
        assert sum(slots.values()) == C.DYNAMIC_UED_SLOTS
        assert rc.stats["exploration_slots"] == EXPLORATION_SLOT_CAP

    def test_exploration_cap_two_proposals_two_slots(self):
        props = [_alloc(f, C.DECISION_MUTATE, 1, exploration=True)
                 for f in (FAM, FAM2, FAM3)]
        props.append(_alloc(FAM4, C.DECISION_RETAIN, 4, ["fb-1"]))
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        assert any(e["rule"] == "exploration_cap_dropped" for e in rc.log)
        expl = [a for a in rc.plan.allocations if a.is_exploration]
        assert len(expl) == C.MAX_EXPLORATION_PROPOSALS
        assert sum(a.slots for a in expl) <= EXPLORATION_SLOT_CAP

    def test_unknown_feedback_id_fails_closed(self):
        props = [_alloc(FAM, C.DECISION_RETAIN, 4, ["fb-ghost"])]
        with pytest.raises(ValueError, match="UNKNOWN_FEEDBACK_ID"):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=props, known_feedback_ids=self.known)

    def test_uncited_retire_rejected(self):
        props = [_alloc(FAM, C.DECISION_RETIRE, 0, exploration=True)]
        with pytest.raises(ValueError, match="RETIRE_REQUIRES_FEEDBACK"):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=props, known_feedback_ids=self.known)

    def test_uncited_non_exploration_decision_rejected(self):
        props = [_alloc(FAM, C.DECISION_RETAIN, 2)]
        with pytest.raises(ValueError, match="EXPLORATION_DECISION_ONLY"):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=props, known_feedback_ids=self.known)

    def test_uncited_mutate_forced_to_exploration(self):
        props = [_alloc(FAM, C.DECISION_MUTATE, 1),
                 _alloc(FAM2, C.DECISION_RETAIN, 4, ["fb-1"])]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        assert any(e["rule"] == "forced_exploration_label" for e in rc.log)
        fam_alloc = next(a for a in rc.plan.allocations
                         if a.environment_family == FAM)
        assert fam_alloc.is_exploration is True

    def test_masquerade_forbidden(self):
        props = [_alloc(FAM, C.DECISION_RETAIN, 2, ["fb-1"], exploration=True)]
        with pytest.raises(ValueError, match="MASQUERADE_FORBIDDEN"):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=props, known_feedback_ids=self.known)

    def test_retire_removes_family_and_overrides_active(self):
        props = [_alloc(FAM, C.DECISION_RETIRE, 0, ["fb-1"]),
                 _alloc(FAM, C.DECISION_RETAIN, 3, ["fb-2"]),
                 _alloc(FAM2, C.DECISION_RETAIN, 3, ["fb-3"])]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known,
                                previous_slots={FAM: 3, FAM2: 3})
        families = {a.environment_family for a in rc.plan.allocations}
        assert FAM not in families
        assert FAM in rc.plan.retired_families
        assert any(e["rule"] == "retire_overrides_active" for e in rc.log)
        retire_mod = next(m for m in rc.modifications
                          if m["decision"] == C.DECISION_RETIRE)
        assert retire_mod["new_slots"] == 0
        assert retire_mod["old_slots"] == 3
        assert retire_mod["based_on_feedback_ids"] == ["fb-1"]

    def test_request_control_zero_budget_and_flag(self):
        props = [_alloc(FAM, C.DECISION_REQUEST_CONTROL, 0, ["fb-1"]),
                 _alloc(FAM2, C.DECISION_RETAIN, 3, ["fb-2"])]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        assert rc.request_control is True
        assert rc.stats["request_control"] is True
        assert any(e["rule"] == "request_control_escalation" for e in rc.log)
        assert all(a.environment_family != FAM for a in rc.plan.allocations)

    def test_duplicate_family_first_by_priority_wins(self):
        props = [_alloc(FAM, C.DECISION_MUTATE, 1, ["fb-1"]),
                 _alloc(FAM, C.DECISION_RETAIN, 2, ["fb-2"])]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        assert any(e["rule"] == "duplicate_dropped" for e in rc.log)
        fam_alloc = next(a for a in rc.plan.allocations
                         if a.environment_family == FAM)
        assert fam_alloc.decision == C.DECISION_RETAIN   # higher priority

    def test_empty_proposals_fail_closed(self):
        with pytest.raises(ValueError, match="INSUFFICIENT_DYNAMIC_ALLOCATION"):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=[], known_feedback_ids=self.known)

    def test_malformed_dict_proposal_hard_error(self):
        with pytest.raises(ValueError):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=[{"environment_family": FAM,
                                           "slots": 2, "decision": "RETAIN",
                                           "bogus_field": 1}],
                               known_feedback_ids=self.known)
        with pytest.raises(ValueError, match="ILLEGAL_PROPOSAL_TYPE"):
            self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               proposals=[42], known_feedback_ids=self.known)

    def test_only_exploration_falls_back_top_up(self):
        # documented fallback: with NO core allocation funded, leftover tops up
        # the first grant so the dynamic budget never ships half-empty
        props = [_alloc(FAM, C.DECISION_MUTATE, 1, exploration=True),
                 _alloc(FAM2, C.DECISION_MUTATE, 1, exploration=True)]
        rc = self.rec.reconcile(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                                proposals=props, known_feedback_ids=self.known)
        assert sum(a.slots for a in rc.plan.allocations) == C.DYNAMIC_UED_SLOTS
        assert any(e["rule"] == "leftover_top_up" for e in rc.log)
