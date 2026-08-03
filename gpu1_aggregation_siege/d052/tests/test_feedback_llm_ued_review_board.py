"""C6: six-role Review Board — contract, citation discipline, honesty.

Positive tests (full board runs on the deterministic mock backend) plus the
negative citation tests the director's REQUEST_CHANGES review mandated:
unknown / future / duplicate feedback citations all fail closed, and the
static-mode NullFeedbackView is structurally empty.
"""
import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import critic_skeptic
from d052.feedback_llm_ued.axis_directive import (
    ROLE_CONTROL,
    ROLE_TREATMENT,
    assert_directive_batch_legal,
)
from d052.feedback_llm_ued.behavior_failure import (
    SEVERITY_HIGH,
    assemble_board_context,
)
from d052.feedback_llm_ued.causal_failure_analyst import BoardHypothesisVerdict
from d052.feedback_llm_ued.critic_skeptic import WIDE_CI
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.feedback_view import (
    FeedbackView,
    NormalFeedbackView,
    NullFeedbackView,
)
from d052.feedback_llm_ued.hypothesis_ledger import HypothesisRecord
from d052.feedback_llm_ued.llm_backend import DeterministicMockFeedbackBackend
from d052.feedback_llm_ued.review_board import (
    BoardOutput,
    run_review_board,
    validate_citations,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    MATCH_UNGRADED,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

FAM_A = "threat_distance_family"
FAM_B = "resource_pressure_family"


def make_record(i, *, window=0, family=FAM_A, student_sr=0.4,
                distinguishes=None, feedback_id=None):
    cand = synthetic_candidate(candidate_id=f"c-rb-{window}-{i}",
                               family=family)
    return synthetic_feedback_record(
        feedback_id=feedback_id or f"fb-rb-w{window}-{i}",
        candidate=cand, plan_id=f"plan-{window}", window=window,
        student_success_rate=student_sr,
        expected_signature={"student_success_rate": 0.5},
        distinguishes_hypothesis_ids=list(distinguishes or []))


def make_hyp(hid, family=FAM_A, signature=None):
    return HypothesisRecord(
        hypothesis_id=hid, source_window=0,
        target_behavior=f"behavior under {family}",
        predicted_signature=signature or {"student_success_rate": 0.5},
        environment_family=family, confidence=0.5)


def _case(*, mode=C.MODE_NORMAL_FEEDBACK, board_window=1):
    """One graded opposite record + one agreeing record (window 0) and one
    ledger hypothesis; the board runs at ``board_window``."""
    store = SimulatorFeedbackStore()
    store.add(make_record(0, distinguishes=["hyp-1"]))
    store.add(make_record(1, family=FAM_B, student_sr=0.6))
    store.bind_match("fb-rb-w0-0", direction="opposite")
    store.bind_match("fb-rb-w0-1", direction="agree")
    ctx = assemble_board_context(store, window=board_window - 1, mode=mode)
    view = NormalFeedbackView.from_store(store,
                                         max_window=board_window - 1)
    return store, ctx, view


def run_board(*, mode=C.MODE_NORMAL_FEEDBACK, board_window=1,
              hypotheses=None, view=None, ctx=None):
    store, ctx_d, view_d = _case(mode=mode, board_window=board_window)
    backend = DeterministicMockFeedbackBackend()
    out = run_review_board(
        window=board_window, mode=mode,
        board_context=ctx if ctx is not None else ctx_d,
        view=view if view is not None else view_d,
        hypotheses=(hypotheses if hypotheses is not None
                    else [make_hyp("hyp-1")]),
        backend=backend, sequence_start=0)
    return out, backend


class TestBoardContract:
    def test_exactly_six_calls_in_fixed_role_order(self):
        out, backend = run_board()
        assert backend.usage.mock_calls == C.BOARD_CALLS_PER_WINDOW == 6
        assert backend.usage.real_calls == 0
        assert out.board_call_count == 6
        assert [e.role for e in out.envelopes] == list(C.BOARD_ROLES)

    def test_envelopes_carry_window_and_contiguous_sequence(self):
        out, _ = run_board(board_window=3)
        assert all(e.window == 3 for e in out.envelopes)
        assert [e.sequence for e in out.envelopes] == [0, 1, 2, 3, 4, 5]

    def test_evidence_status_is_engineering_scaffold(self):
        out, _ = run_board()
        assert out.evidence_status == C.ENGINEERING_SCAFFOLD
        dump = out.model_dump()
        dump["evidence_status"] = "REAL_EVIDENCE"
        with pytest.raises(ValidationError, match="ILLEGAL_EVIDENCE_STATUS"):
            BoardOutput(**dump)

    def test_call_count_mismatch_rejected(self):
        out, _ = run_board()
        dump = out.model_dump()
        dump["board_call_count"] = 5
        with pytest.raises(ValidationError,
                           match="BOARD_CALL_COUNT_MISMATCH"):
            BoardOutput(**dump)

    def test_role_sequence_mismatch_rejected(self):
        out, _ = run_board()
        dump = out.model_dump()
        dump["envelopes"][0], dump["envelopes"][1] = \
            dump["envelopes"][1], dump["envelopes"][0]
        with pytest.raises(ValidationError,
                           match="BOARD_ROLE_SEQUENCE_MISMATCH"):
            BoardOutput(**dump)

    def test_deliverable_tamper_rejected(self):
        out, _ = run_board()
        dump = out.model_dump()
        dump["verdicts"] = []                    # detach from role output
        with pytest.raises(ValidationError,
                           match="BOARD_OUTPUT_INCONSISTENT"):
            BoardOutput(**dump)

    def test_board_run_is_deterministic(self):
        a, _ = run_board()
        b, _ = run_board()
        assert a.model_dump() == b.model_dump()
        assert a.board_hash == b.board_hash
        assert len(a.board_hash) == 64
        assert a.rehash() == a.board_hash

    def test_board_hash_reflects_content(self):
        a, _ = run_board(board_window=1)
        b, _ = run_board(board_window=2)
        assert a.board_hash != b.board_hash


class TestWindowZeroAndNullView:
    def test_window0_empty_view_still_runs_full_board(self):
        ctx = assemble_board_context(SimulatorFeedbackStore(), window=0,
                                     mode=C.MODE_STATIC_LLM)
        backend = DeterministicMockFeedbackBackend()
        out = run_review_board(window=0, mode=C.MODE_STATIC_LLM,
                               board_context=ctx, view=NullFeedbackView(),
                               hypotheses=[], backend=backend,
                               sequence_start=0)
        assert backend.usage.mock_calls == 6
        assert out.verdicts == []                # nothing to cite yet
        # only bounded exploration directives, no treatment of evidence
        assert all("exploration" in d.directive_id for d in out.directives)
        assert out.request_control is False

    def test_null_view_is_structurally_empty(self):
        view = NullFeedbackView()
        assert isinstance(view, FeedbackView)
        assert view.records() == []
        assert view.to_prompt_payload() == []
        # structural: the instance carries NO state that could hold feedback
        assert vars(view) == {}

    def test_static_mode_prompts_carry_zero_feedback_payload(self):
        ctx = assemble_board_context(SimulatorFeedbackStore(), window=0,
                                     mode=C.MODE_STATIC_LLM)

        class PromptSpy:
            """Wraps the mock backend and captures every prompt sent."""

            def __init__(self, inner):
                self._inner = inner
                self.prompts = []
                self.backend_id = inner.backend_id
                self.model_id = inner.model_id
                self.usage = inner.usage

            def complete(self, role, prompt):
                self.prompts.append((role, prompt))
                return self._inner.complete(role, prompt)

        spy = PromptSpy(DeterministicMockFeedbackBackend())
        run_review_board(window=0, mode=C.MODE_STATIC_LLM,
                         board_context=ctx, view=NullFeedbackView(),
                         hypotheses=[], backend=spy, sequence_start=0)
        assert len(spy.prompts) == 6
        for role, prompt in spy.prompts:
            assert extract_context(prompt)["feedback"] == []


class TestCitationDiscipline:
    def test_verdicts_cite_only_visible_earlier_feedback(self):
        out, _ = run_board(board_window=1)
        _, _, view = _case(board_window=1)
        visible = {p["feedback_id"]: p for p in view.to_prompt_payload()}
        assert out.verdicts                      # this case produces one
        for v in out.verdicts:
            for fid in v.cited_feedback_ids:
                assert fid in visible
                assert visible[fid]["window"] <= 0     # <= board window - 1
        verdict = out.verdicts[0]
        assert verdict.verdict == C.HYPOTHESIS_REFUTED
        assert verdict.cited_feedback_ids == ["fb-rb-w0-0"]
        assert verdict.cited_prediction_signature == {
            "student_success_rate": 0.5}

    def test_stale_verdict_without_citations_is_legal(self):
        store, ctx, view = _case()
        backend = DeterministicMockFeedbackBackend()
        # hyp-2 exists in the ledger but no visible record distinguishes it
        out = run_review_board(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               board_context=ctx, view=view,
                               hypotheses=[make_hyp("hyp-1"),
                                           make_hyp("hyp-2", family=FAM_B)],
                               backend=backend, sequence_start=0)
        by_id = {v.hypothesis_id: v for v in out.verdicts}
        assert by_id["hyp-2"].verdict == C.HYPOTHESIS_STALE
        assert by_id["hyp-2"].cited_feedback_ids == []

    def test_unknown_feedback_id_rejected(self):
        _, _, view = _case()
        visible = {p["feedback_id"]: p for p in view.to_prompt_payload()}
        bad = BoardHypothesisVerdict(
            hypothesis_id="hyp-1", verdict=C.HYPOTHESIS_SUPPORTED,
            new_confidence=0.6, cited_feedback_ids=["fb-does-not-exist"],
            cited_prediction_signature={"student_success_rate": 0.5})
        with pytest.raises(ValueError, match="UNKNOWN_FEEDBACK_ID"):
            validate_citations([bad], visible, 1, frozenset({"hyp-1"}))

    def test_future_feedback_id_rejected_end_to_end(self):
        # a window-1 board handed a view that (wrongly) contains a window-1
        # record must refuse to cite it — same-window feedback is future
        store = SimulatorFeedbackStore()
        store.add(make_record(0))                       # evidence only
        store.add(make_record(9, window=1, distinguishes=["hyp-1"]))
        ctx = assemble_board_context(store, window=0,
                                     mode=C.MODE_NORMAL_FEEDBACK)
        view = NormalFeedbackView.from_store(store, max_window=1)
        backend = DeterministicMockFeedbackBackend()
        with pytest.raises(ValueError, match="FUTURE_FEEDBACK_ID"):
            run_review_board(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                             board_context=ctx, view=view,
                             hypotheses=[make_hyp("hyp-1")],
                             backend=backend, sequence_start=0)

    def test_duplicate_feedback_citation_rejected(self):
        _, _, view = _case()
        visible = {p["feedback_id"]: p for p in view.to_prompt_payload()}
        bad = BoardHypothesisVerdict(
            hypothesis_id="hyp-1", verdict=C.HYPOTHESIS_SUPPORTED,
            new_confidence=0.6,
            cited_feedback_ids=["fb-rb-w0-1", "fb-rb-w0-1"],
            cited_prediction_signature={"student_success_rate": 0.5})
        with pytest.raises(ValueError, match="DUPLICATE_FEEDBACK_CITATION"):
            validate_citations([bad], visible, 1, frozenset({"hyp-1"}))

    def test_duplicate_hypothesis_verdict_rejected(self):
        _, _, view = _case()
        visible = {p["feedback_id"]: p for p in view.to_prompt_payload()}
        v1 = BoardHypothesisVerdict(
            hypothesis_id="hyp-1", verdict=C.HYPOTHESIS_SUPPORTED,
            new_confidence=0.6, cited_feedback_ids=[],
            cited_prediction_signature={"student_success_rate": 0.5})
        v2 = BoardHypothesisVerdict(
            hypothesis_id="hyp-1", verdict=C.HYPOTHESIS_REFUTED,
            new_confidence=0.3, cited_feedback_ids=[],
            cited_prediction_signature={"student_success_rate": 0.5})
        with pytest.raises(ValueError, match="DUPLICATE_HYPOTHESIS_VERDICT"):
            validate_citations([v1, v2], visible, 1, frozenset({"hyp-1"}))

    def test_unknown_hypothesis_id_rejected(self):
        _, _, view = _case()
        visible = {p["feedback_id"]: p for p in view.to_prompt_payload()}
        bad = BoardHypothesisVerdict(
            hypothesis_id="hyp-ghost", verdict=C.HYPOTHESIS_STALE,
            new_confidence=0.5, cited_feedback_ids=[],
            cited_prediction_signature={})
        with pytest.raises(ValueError, match="UNKNOWN_HYPOTHESIS_ID"):
            validate_citations([bad], visible, 1, frozenset({"hyp-1"}))


class TestDeliverables:
    def test_directives_are_legal_batch_and_window_bound(self):
        out, _ = run_board(board_window=2)
        assert_directive_batch_legal(out.directives)     # no raise
        assert out.directives
        for d in out.directives:
            assert d.source_window == 2
            assert len(d.directive_hash) == 64
        # at most one treatment per family/axis; controls hold their level
        treatments = [d for d in out.directives
                      if d.experiment_control_role == ROLE_TREATMENT]
        keys = [(d.environment_family, d.axis) for d in treatments]
        assert len(keys) == len(set(keys))
        for d in out.directives:
            if d.experiment_control_role == ROLE_CONTROL:
                assert d.new_level == d.old_level

    def test_exploration_directives_only_for_unevidenced_families(self):
        out, _ = run_board()
        evidenced = {p["environment_family"]
                     for p in NormalFeedbackView.from_store(
                         _case()[0], max_window=0).to_prompt_payload()}
        for d in out.directives:
            if "exploration" in d.directive_id:
                assert d.environment_family not in evidenced

    def test_family_proposals_keep_the_honesty_invariant(self):
        out, _ = run_board()
        assert out.family_proposals
        for p in out.family_proposals:
            cited = bool(p.based_on_feedback_ids or p.based_on_hypothesis_ids)
            if cited:
                assert not p.is_exploration
            else:
                assert p.is_exploration
                assert p.decision == C.DECISION_MUTATE
        # the refuted hypothesis's family is proposed for RETIRE, cited
        retire = [p for p in out.family_proposals
                  if p.decision == C.DECISION_RETIRE]
        assert retire and retire[0].environment_family == FAM_A
        assert retire[0].based_on_feedback_ids == ["fb-rb-w0-0"]

    def test_severe_but_precise_evidence_is_high_risk_without_stop(self):
        # C11: 3 records -> pooled 18 episodes -> CI half-width ~0.226
        # (< WIDE_CI), so the evidence is severe but PRECISE: HIGH risk and
        # not endorsed, but the loop does NOT halt — severe-and-certain is
        # exactly what the RETIRE / MUTATE curriculum actions are for.
        store = SimulatorFeedbackStore()
        for i in range(3):
            store.add(make_record(i, student_sr=0.4))
        store.bind_match("fb-rb-w0-0", direction="opposite")
        store.bind_match("fb-rb-w0-1", direction="opposite")
        store.bind_match("fb-rb-w0-2", direction="opposite")
        ctx = assemble_board_context(store, window=0,
                                     mode=C.MODE_NORMAL_FEEDBACK)
        view = NormalFeedbackView.from_store(store, max_window=0)
        backend = DeterministicMockFeedbackBackend()
        out = run_review_board(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               board_context=ctx, view=view,
                               hypotheses=[], backend=backend,
                               sequence_start=0)
        assert out.critic.global_risk == "HIGH"
        assert out.critic.endorsed is False
        assert out.critic.request_control is False
        assert out.request_control is False

    def test_ungraded_feedback_is_an_honesty_objection(self):
        store = SimulatorFeedbackStore()
        store.add(make_record(0, distinguishes=["hyp-1"]))   # never graded
        ctx = assemble_board_context(store, window=0,
                                     mode=C.MODE_NORMAL_FEEDBACK)
        view = NormalFeedbackView.from_store(store, max_window=0)
        backend = DeterministicMockFeedbackBackend()
        out = run_review_board(window=1, mode=C.MODE_NORMAL_FEEDBACK,
                               board_context=ctx, view=view,
                               hypotheses=[make_hyp("hyp-1")],
                               backend=backend, sequence_start=0)
        assert out.critic.honesty_check_passed is False
        assert out.critic.global_risk in ("MEDIUM", "HIGH")
        assert any("never graded" in o for o in out.critic.objections)
        # C11: an honesty violation ALWAYS escalates — the loop must stop
        # for human review when feedback was never graded
        assert out.critic.request_control is True
        assert out.request_control is True


class TestCriticEscalationRule:
    """C11: ``request_control`` HALTS the whole loop (HumanDecisionArtifact,
    no execution batch), so the mock critic escalates only where autonomous
    continuation is indefensible: an honesty violation (ungraded feedback)
    or HIGH risk built from THIN evidence (CI half-width >= WIDE_CI).
    Severe-but-precise evidence stays HIGH risk without halting — the risk
    grading itself is unchanged."""

    @staticmethod
    def _ctx(*, high_sev=0, opposite=0, ungraded=0, ci=0.0):
        evidence = [{"severity": SEVERITY_HIGH} for _ in range(high_sev)]
        feedback = (
            [dict(feedback_id=f"fb-opp-{i}",
                  expected_observed_match=C.MATCH_DIRECTION_OPPOSITE)
             for i in range(opposite)] +
            [dict(feedback_id=f"fb-ungraded-{i}",
                  expected_observed_match=MATCH_UNGRADED)
             for i in range(ungraded)])
        return dict(window=1,
                    board_context=dict(behavior_evidence=evidence,
                                       student_success_rate_ci=ci),
                    feedback=feedback)

    def test_high_risk_with_precise_evidence_does_not_halt(self):
        out = critic_skeptic.mock_rule(self._ctx(high_sev=3, ci=0.1))
        assert out["global_risk"] == "HIGH"
        assert out["endorsed"] is False
        assert out["request_control"] is False

    def test_high_risk_with_thin_evidence_halts(self):
        out = critic_skeptic.mock_rule(self._ctx(high_sev=3, ci=0.6))
        assert out["global_risk"] == "HIGH"
        assert out["request_control"] is True

    def test_ungraded_feedback_halts_even_without_high_risk(self):
        out = critic_skeptic.mock_rule(self._ctx(ungraded=1, ci=0.1))
        assert out["global_risk"] == "MEDIUM"
        assert out["honesty_check_passed"] is False
        assert out["request_control"] is True

    def test_clean_evidence_does_not_halt(self):
        out = critic_skeptic.mock_rule(self._ctx(ci=0.1))
        assert out["global_risk"] == "LOW"
        assert out["request_control"] is False

    def test_wide_ci_boundary_is_inclusive(self):
        wide = critic_skeptic.mock_rule(self._ctx(opposite=2, ci=WIDE_CI))
        assert wide["global_risk"] == "HIGH"
        assert wide["request_control"] is True
        tight = critic_skeptic.mock_rule(
            self._ctx(opposite=2, ci=WIDE_CI - 0.01))
        assert tight["global_risk"] == "HIGH"
        assert tight["request_control"] is False


class TestFeedbackView:
    def test_normal_view_snapshot_is_scoped_sorted_immutable(self):
        store = SimulatorFeedbackStore()
        store.add(make_record(1, window=0, feedback_id="fb-z"))
        store.add(make_record(0, window=0, feedback_id="fb-a"))
        store.add(make_record(2, window=1, feedback_id="fb-w1"))
        view = NormalFeedbackView.from_store(store, max_window=0)
        assert view.window_scope == 0
        assert [p["feedback_id"] for p in view.to_prompt_payload()] == \
            ["fb-a", "fb-z"]                      # sorted, window <= scope
        view.records().clear()                    # copies: snapshot intact
        assert len(view.records()) == 2

    def test_view_window_scope_must_be_non_negative(self):
        with pytest.raises(ValueError, match="ILLEGAL_VIEW_WINDOW_SCOPE"):
            NormalFeedbackView([], window_scope=-1)

    def test_record_payload_carries_no_forbidden_carriers(self):
        store, _, view = _case()
        for payload in view.to_prompt_payload():
            leak = set(payload) & set(C.REFERENCE_FORBIDDEN_CARRIERS)
            assert leak == set()
            assert "student_success_rate" in payload
            assert "axis_values" in payload       # env-level, allowed
