"""P0-12 (§19 seam coverage): the no-feedback control is the SHAPE-MATCHED
mask — NOT the structurally empty view.

Contract under test:

* ``MaskedFeedbackView`` presents EXACTLY the window k-1 record set the
  normal mode sees: the same item count, the same prompt field set, the
  same deterministic order — every value replaced by its controlled
  NULL/MASK value (ids/family masked, axes/signature/hypothesis bindings
  emptied, match state NEUTRAL, rates zero);
* the evidence layer is masked consistently with the prompt layer (same
  masked ids, identity masked, controlled-null gaps/severity) and carries
  no channel the prompt layer does not;
* ``resolve_citation`` fails closed — the control can never act on
  feedback;
* every mode matches: six-role call count (6/window), EnvCoder budget
  (1/window), feedback item count (64/window), lifecycle query (the
  static mode runs the SAME retirement-state query — frozen-empty),
  candidates / episodes / transitions (identical totals), anchors
  (12 dynamic + 4), training steps (none authorized) and checkpoint
  cadence (none executed);
* the board prompt field set is identical between the masked and normal
  views.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: mock backends
and the deterministic symbolic runner — NO real LLM call, NO simulator
episode, and NO passing test flips a REAL_* flag.
"""
from __future__ import annotations

import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import SEVERITY_NONE
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.feedback_view import (
    MASKED_IDENTITY,
    MaskedFeedbackView,
    NormalFeedbackView,
    record_payload,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

WINDOWS = 2
MODES = (C.MODE_NORMAL_FEEDBACK, C.MODE_STATIC_LLM,
         C.MODE_SHUFFLED_FEEDBACK)


def _records(window: int):
    return [
        synthetic_feedback_record(
            feedback_id=f"fb-mask-w{window}-{i:03d}",
            candidate=synthetic_candidate(
                candidate_id=f"c-mask-w{window}-{i:03d}",
                family=C.ENVIRONMENT_FAMILIES[i % 7],
                axes=["threat_distance_grading"]),
            plan_id=f"plan-mask-w{window}", window=window,
            student_success_rate=0.42,
            expected_signature={"student_success_rate": 0.45},
            distinguishes_hypothesis_ids=["hyp-00"])
        for i in range(3)]


@pytest.fixture
def runs():
    ctls = {mode: FeedbackUEDController(mode) for mode in MODES}
    sums = {mode: ctl.run(max_windows=WINDOWS)
            for mode, ctl in ctls.items()}
    return ctls, sums


class TestMaskedViewContract:
    def test_shape_matches_normal_view(self):
        records = _records(1)
        masked = MaskedFeedbackView(records, window_scope=1, board_window=2)
        normal = NormalFeedbackView(records, window_scope=1)
        assert len(masked.records()) == len(normal.records()) == 3
        assert len(masked.to_prompt_payload()) == 3
        assert len(masked.behavior_evidence()) == 3
        #: same deterministic order (sorted by feedback_id)
        assert [r.feedback_id for r in masked.records()] == \
            [r.feedback_id for r in normal.records()]
        #: same prompt field set
        normal_keys = set(record_payload(records[0]))
        for item in masked.to_prompt_payload():
            assert set(item) == normal_keys

    def test_all_values_are_controlled_mask(self):
        records = _records(1)
        masked = MaskedFeedbackView(records, window_scope=1, board_window=2)
        payload = masked.to_prompt_payload()
        assert [p["feedback_id"] for p in payload] == \
            [f"masked-w02-{i:03d}" for i in range(3)]
        for p in payload:
            assert p["candidate_id"] == MASKED_IDENTITY
            assert p["environment_family"] == MASKED_IDENTITY
            assert p["mutation_axes"] == []
            assert p["axis_values"] == {}
            assert p["held_constant_axes"] == {}
            assert p["distinguishes_hypothesis_ids"] == []
            assert p["expected_signature"] == {}
            assert p["expected_observed_match"] \
                == C.MATCH_DIRECTION_NEUTRAL
            assert p["student_success_rate"] == 0.0
            assert p["reference_success_rate"] == 0.0
            assert p["window"] == 1
        #: no real id / candidate / rate value anywhere
        serialized = json.dumps(payload, sort_keys=True)
        for r in records:
            assert r.feedback_id not in serialized
            assert r.candidate_id not in serialized
            assert "0.42" not in serialized

    def test_evidence_is_consistently_masked(self):
        records = _records(1)
        masked = MaskedFeedbackView(records, window_scope=1, board_window=2)
        payload = masked.to_prompt_payload()
        evidence = masked.behavior_evidence()
        assert len(evidence) == len(payload)
        for item, e in zip(payload, evidence):
            assert e.feedback_id == item["feedback_id"]
            assert e.candidate_id == MASKED_IDENTITY
            assert e.environment_family == MASKED_IDENTITY
            assert e.reference_gap == 0.0
            assert e.return_shortfall == 0.0
            assert e.behavior_activation_gap == 0.0
            assert e.front_progress_gap == 0.0
            assert e.severity == SEVERITY_NONE
            assert e.expected_observed_match == C.MATCH_DIRECTION_NEUTRAL
            assert e.student_success_rate == 0.0
            assert e.reference_success_rate == 0.0

    def test_determinism(self):
        records = _records(1)
        a = MaskedFeedbackView(records, window_scope=1, board_window=2)
        b = MaskedFeedbackView(records, window_scope=1, board_window=2)
        assert a.to_prompt_payload() == b.to_prompt_payload()
        assert ([e.model_dump() for e in a.behavior_evidence()]
                == [e.model_dump() for e in b.behavior_evidence()])

    def test_resolve_citation_fails_closed(self):
        records = _records(1)
        masked = MaskedFeedbackView(records, window_scope=1, board_window=2)
        with pytest.raises(ValueError,
                           match="MASKED_VIEW_CITATION_NOT_RESOLVABLE"):
            masked.resolve_citation("masked-w02-000")
        with pytest.raises(ValueError,
                           match="MASKED_VIEW_CITATION_NOT_RESOLVABLE"):
            masked.resolve_citation(records[0].feedback_id)

    def test_window_zero_mask_is_empty(self):
        masked = MaskedFeedbackView([], window_scope=0, board_window=0)
        assert masked.to_prompt_payload() == []
        assert masked.behavior_evidence() == []
        assert masked.records() == []


class TestComputeMatchAcrossModes:
    def test_six_role_and_envcoder_call_counts_match(self, runs):
        sums = runs[1]
        for mode, s in sums.items():
            assert s.n_llm_calls == 7 * WINDOWS, mode
            for w in s.windows:
                assert w["board_call_count"] == C.BOARD_CALLS_PER_WINDOW == 6
                assert w["env_coder_call_count"] == 1, (mode, w["window"])

    def test_feedback_item_counts_match(self, runs):
        ctls, _sums = runs
        for mode, ctl in ctls.items():
            for window in range(WINDOWS):
                assert len(list(ctl.store.for_window(window))) == 64, \
                    (mode, window)

    def test_episodes_and_transitions_match(self, runs):
        sums = runs[1]
        first = sums[MODES[0]].total_simulator_transitions
        for mode, s in sums.items():
            assert s.total_simulator_transitions == first, mode

    def test_anchors_budget_matches(self, runs):
        sums = runs[1]
        for mode, s in sums.items():
            for w in s.windows:
                assert w["funnel_stats"]["anchors"] == C.GLOBAL_ANCHOR_SLOTS
                assert w["funnel_stats"]["dynamic_selected"] == 12
                assert w["funnel_stats"]["final_batch"] == 16

    def test_lifecycle_query_runs_in_static(self, runs):
        #: P0-12: the static mode runs the SAME retirement-state query —
        #: the masked view resolves no citation, so the registry can never
        #: become non-empty and the query returns the frozen-empty partition
        ctls, _sums = runs
        ctl = ctls[C.MODE_STATIC_LLM]
        assert ctl._retired_at == {}
        assert ctl._retirement_state(1) == ([], [], ())
        #: every static revision is exploration (no feedback content)
        assert all(rev.based_on_feedback_ids == []
                   for rev in ctl.revisions)

    def test_training_and_checkpoint_cadence_match(self, runs):
        #: no training is authorized this round: every mode records the
        #: same SKIPPED_UNAUTHORIZED sequence and executes no checkpoint
        ctls, sums = runs
        statuses = {mode: [t.status for t in ctl.training_log]
                    for mode, ctl in ctls.items()}
        assert statuses[C.MODE_NORMAL_FEEDBACK] == \
            statuses[C.MODE_STATIC_LLM] == statuses[C.MODE_SHUFFLED_FEEDBACK]
        assert statuses[C.MODE_NORMAL_FEEDBACK] == \
            ["SKIPPED_UNAUTHORIZED"] * WINDOWS
        assert all(s.checkpoint_round_trip_pass is False
                   for mode in MODES for s in ctls[mode].training_log)
        assert sums[C.MODE_STATIC_LLM].request_control_stopped is False

    def test_board_prompt_field_set_matches_masked_vs_normal(self, runs):
        ctls, _sums = runs
        normal = ctls[C.MODE_NORMAL_FEEDBACK]
        masked = ctls[C.MODE_STATIC_LLM]
        for role in C.BOARD_ROLES:
            n_prompt = next(e.prompt for e in normal.envelopes
                            if e.role == role and e.window == 1)
            m_prompt = next(e.prompt for e in masked.envelopes
                            if e.role == role and e.window == 1)
            n_ctx = extract_context(n_prompt)
            m_ctx = extract_context(m_prompt)
            #: same top-level context keys
            assert set(n_ctx) == set(m_ctx), role
            #: same feedback field set and item count in the prompt
            assert (set(n_ctx["feedback"][0]) == set(m_ctx["feedback"][0])
                    == set(record_payload(
                        normal.store.for_window(0)[0])))
            assert len(n_ctx["feedback"]) == len(m_ctx["feedback"]) == 64
            #: the masked items carry no real identity
            for item in m_ctx["feedback"]:
                assert item["candidate_id"] == MASKED_IDENTITY


class TestPosture:
    def test_real_capability_flags_stay_false(self, runs):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name

    def test_controller_windows_report_masked_label(self, runs):
        sums = runs[1]
        for w in sums[C.MODE_STATIC_LLM].windows:
            assert w["feedback_view_label"] == "masked"
