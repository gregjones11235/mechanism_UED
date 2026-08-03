"""End-to-end: the double-window state machine (C8).

Per window k: EVIDENCE -> BOARD (always all six roles) -> REVISION (verdicts
+ plan_k, citing ONLY feedback from windows <= k-1) -> PROBING (EnvCoder +
gates + staged funnel probe -> staged feedback_k) -> atomic FREEZE. After the
freeze, ANY same-window verdict application or plan change raises
SAME_WINDOW_REVISION_FORBIDDEN; feedback_k can only be acted on by window
k+1's complete six-role board (NEXT_WINDOW_REVISION_ONLY /
SAME_WINDOW_REVISION_REJECTED).

Runs all three comparison modes on the deterministic mock backend + symbolic
probe runner and asserts the loop is genuinely feedback-driven and
compute-matched: every mode spends the SAME 7 LLM-family calls and the SAME
simulator transitions per window; static never cites feedback, normal revises
on honest probe feedback, and shuffling the candidate<->feedback binding
changes the resulting plans.

C9 isolation: the FeedbackStore is HONEST in every mode; the isolation lives
in the view the board receives — static reads the structurally empty
NullFeedbackView (zero feedback payload in every board prompt), shuffled
reads a frozen recomputable PermutedFeedbackView (anonymized ids, identity
side channels masked; only the controller resolves citations back to store
ids), normal reads the honest snapshot.

C9 re-baseline (6 windows, deterministic mock): 42 LLM-family calls and
368640 simulator transitions per mode; normal coverage 0.6667 with
{MUTATE: 7, RETIRE: 13} and 5 unique plan signatures; shuffled coverage
0.8047 with {MUTATE: 6, RETIRE: 12, RETAIN: 2}, six unique plan signatures
and anon-citation resolution into honest ledger/revision ids; static keeps a
single plan signature and 0.0 coverage.
"""
import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import assemble_board_context
from d052.feedback_llm_ued.causal_failure_analyst import BoardHypothesisVerdict
from d052.feedback_llm_ued.controller import (
    PHASE_BOARD,
    PHASE_EVIDENCE,
    PHASE_FROZEN,
    FeedbackUEDController,
    SameWindowRevisionForbidden,
    StateMachineViolation,
)
from d052.feedback_llm_ued.plan_revision import FEEDBACK_DRIVEN_LABEL
from d052.feedback_llm_ued.review_board import (
    build_board_prompt_context,
    normalize_hypothesis_inputs,
)
from d052.feedback_llm_ued.simulator_probe import (
    DeterministicSymbolicProbeRunner,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

WINDOWS = 6

#: per-window probe cost: 64 fast probes x (2+1) episodes + 24 full probes
#: x (8+4) episodes, all at ROLLOUT_LENGTH transitions
TRANSITIONS_PER_PROBED_WINDOW = (
    C.RAW_CANDIDATES * (C.STAGE1_STUDENT_EPISODES
                        + C.STAGE1_REFERENCE_EPISODES) * C.ROLLOUT_LENGTH
    + C.STAGE1_KEEP * (C.STAGE2_STUDENT_EPISODES_MAX
                       + C.STAGE2_REFERENCE_EPISODES_MAX) * C.ROLLOUT_LENGTH)
assert TRANSITIONS_PER_PROBED_WINDOW == 61440

#: every window, every mode: the identical funnel shape (compute-matched)
EXPECTED_FUNNEL = dict(
    raw=64, static_rejects=0, duplicates=0, stage1_probed=64,
    stage1_survivors=24, stage2_probed=24, stage2_selected=12,
    dynamic_selected=12, anchors=4, final_batch=16,
    total_simulator_transitions=TRANSITIONS_PER_PROBED_WINDOW)

FAM0 = C.ENVIRONMENT_FAMILIES[0]          # hyp-00's family
FAM1 = C.ENVIRONMENT_FAMILIES[1]          # hyp-01's family


@pytest.fixture(scope="module")
def runs():
    controllers, summaries = {}, {}
    for mode in (C.MODE_STATIC_LLM, C.MODE_NORMAL_FEEDBACK,
                 C.MODE_SHUFFLED_FEEDBACK):
        ctl = FeedbackUEDController(mode)
        summaries[mode] = ctl.run(max_windows=WINDOWS)
        controllers[mode] = ctl
    return controllers, summaries


def _records_of(ctl, window):
    return [r for r in ctl.store.all() if r.window == window]


def _verdict(hypothesis_id, fids, verdict=C.HYPOTHESIS_SUPPORTED,
             confidence=0.6):
    return BoardHypothesisVerdict(
        hypothesis_id=hypothesis_id, verdict=verdict,
        new_confidence=confidence, cited_feedback_ids=list(fids),
        reason="test verdict")


# ------------------------------------------------------------------ posture
class TestAuthorizationPosture:
    def test_all_flags_false_this_round(self):
        assert C.TRAINING_AUTHORIZED is False
        assert C.FORMAL_EVALUATION_AUTHORIZED is False
        assert C.REAL_LLM_CALLS_AUTHORIZED is False
        assert C.REAL_SIMULATOR_PROBE_AUTHORIZED is False
        assert C.REAL_SIMULATOR_PROBE_STATUS == "BLOCKED_NO_LOCAL_CRAFTAX"
        # round status flags (director review board): every real capability
        # flag stays False; the implementation is ENGINEERING_SCAFFOLD
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.SOTA_INTEGRATION_READY is False
        assert C.REAL_CHECKPOINT_LOADED is False
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False
        assert C.REAL_ENVCODER_USED is False
        assert C.REAL_SIMULATOR_PROBE is False
        assert C.SHARED_ANCHOR_MANIFEST_BOUND is False
        # C8 double-window flags are ON
        assert C.NEXT_WINDOW_REVISION_ONLY is True
        assert C.SAME_WINDOW_REVISION_REJECTED is True
        # C9 isolation flags are ON
        assert C.STATIC_FEEDBACK_STRUCTURALLY_HIDDEN is True
        assert C.SHUFFLE_PERMUTATION_FROZEN is True
        assert len(C.SEED_SCHEDULE_HASH) == 64

    def test_controller_refuses_any_true_flag(self, monkeypatch):
        monkeypatch.setattr(C, "TRAINING_AUTHORIZED", True)
        with pytest.raises(RuntimeError,
                           match="AUTHORIZATION_POSTURE_VIOLATED"):
            FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)

    def test_controller_refuses_any_never_true_status_flag(self, monkeypatch):
        for name in ("SOTA_INTEGRATION_READY", "REAL_CHECKPOINT_LOADED",
                     "REAL_TRAINING_UPDATE_EXECUTED", "REAL_ENVCODER_USED",
                     "REAL_SIMULATOR_PROBE"):
            monkeypatch.setattr(C, name, True)
            with pytest.raises(RuntimeError,
                               match="AUTHORIZATION_POSTURE_VIOLATED"):
                FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
            monkeypatch.setattr(C, name, False)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="UNKNOWN_MODE"):
            FeedbackUEDController("self_training")


# ------------------------------------------- double-window state machine
class TestDoubleWindowStateMachine:
    """Fail-closed negative tests: once feedback_k is frozen, window k is
    CLOSED — verdict application and plan revision are only legal during the
    window's REVISION phase, and only window k+1's board may cite
    feedback_k."""

    def test_apply_verdicts_after_freeze_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        for window in (0, 3, WINDOWS - 1):
            with pytest.raises(SameWindowRevisionForbidden,
                               match="SAME_WINDOW_REVISION_FORBIDDEN"):
                ctl.apply_board_verdicts(window, [])

    def test_revise_plan_after_freeze_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        for window in (0, WINDOWS - 1):
            with pytest.raises(SameWindowRevisionForbidden,
                               match="SAME_WINDOW_REVISION_FORBIDDEN"):
                ctl.revise_plan(window, ctl.boards[window])

    def test_all_windows_end_frozen(self, runs):
        ctls, _sums = runs
        for mode, ctl in ctls.items():
            for window in range(WINDOWS):
                assert ctl.phase_of(window) == PHASE_FROZEN, (mode, window)

    def test_phase_regression_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        with pytest.raises(StateMachineViolation,
                           match="STATE_MACHINE_PHASE_REGRESSION"):
            ctl._set_phase(0, PHASE_EVIDENCE)
        with pytest.raises(StateMachineViolation,
                           match="STATE_MACHINE_PHASE_REGRESSION"):
            ctl._set_phase(3, PHASE_BOARD)
        with pytest.raises(StateMachineViolation, match="UNKNOWN_PHASE"):
            ctl._set_phase(0, "REVIEW")

    def test_double_window_lag_every_citation_lags_one_window(self, runs):
        """The defining invariant: every feedback record a revision cites was
        frozen in a STRICTLY EARLIER window (revision lags feedback by one
        window); window 0 cites nothing at all."""
        ctls, _sums = runs
        for mode, ctl in ctls.items():
            for rev in ctl.revisions:
                for fid in rev.based_on_feedback_ids:
                    rec = ctl.store.get(fid)
                    assert rec.window <= rev.window - 1, (mode, rev.window,
                                                          fid, rec.window)
            assert ctl.revisions[0].based_on_feedback_ids == []
            # same for ledger verdicts: bound feedback is always older than
            # the verdict window
            for rec in ctl.ledger.all():
                for entry in rec.revision_history:
                    for fid in entry["feedback_ids"]:
                        fb = ctl.store.get(fid)
                        assert fb.window <= int(entry["window"]) - 1

    # -- citation validator (pure, phase-independent P0-6 guard) -----------
    def test_legal_next_window_citation_passes(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        # window 1 citing a window-0 record that distinguishes hyp-00: legal
        ctl.validate_verdict_citations(
            1, [_verdict("hyp-00", [fid])])

    def test_same_window_feedback_is_future(self, runs):
        """Citing feedback produced in the VERY window being revised is the
        double-window violation the state machine exists to prevent."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        with pytest.raises(ValueError, match="FUTURE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(0, [_verdict("hyp-00", [fid])])

    def test_later_window_feedback_is_future(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 2)[0].feedback_id
        with pytest.raises(ValueError, match="FUTURE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(1, [_verdict("hyp-00", [fid])])

    def test_unknown_feedback_id_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        with pytest.raises(ValueError, match="UNKNOWN_FEEDBACK_ID"):
            ctl.validate_verdict_citations(
                1, [_verdict("hyp-00", ["fb-ghost"])])

    def test_duplicate_feedback_citation_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        with pytest.raises(ValueError, match="DUPLICATE_FEEDBACK_CITATION"):
            ctl.validate_verdict_citations(
                1, [_verdict("hyp-00", [fid, fid])])

    def test_duplicate_hypothesis_verdict_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        with pytest.raises(ValueError, match="DUPLICATE_HYPOTHESIS_VERDICT"):
            ctl.validate_verdict_citations(
                1, [_verdict("hyp-00", [fid]), _verdict("hyp-00", [fid])])

    def test_unknown_hypothesis_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        with pytest.raises(ValueError, match="UNKNOWN_HYPOTHESIS_ID"):
            ctl.validate_verdict_citations(
                1, [_verdict("hyp-99", [])])

    def test_binding_mismatch_refused(self, runs):
        """A window-0 record distinguishes hyp-00 (FAM0) — hyp-01 (FAM1) may
        not cite it."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        with pytest.raises(ValueError, match="FEEDBACK_BINDING_MISMATCH"):
            ctl.validate_verdict_citations(
                1, [_verdict("hyp-01", [fid])])

    def test_unknown_source_plan_refused(self):
        """A record minted by a plan this run never generated is refused —
        even when its window/binding are otherwise legal."""
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        ctl._seed()
        cand = synthetic_candidate(candidate_id="cand-ghost", family=FAM0)
        rec = synthetic_feedback_record(
            feedback_id="fb-ghost-plan", candidate=cand, plan_id="plan-ghost",
            window=0, student_success_rate=0.5,
            expected_signature={"student_success_rate": 0.5},
            distinguishes_hypothesis_ids=["hyp-00"])
        ctl.store.add(rec)
        with pytest.raises(ValueError, match="UNKNOWN_SOURCE_PLAN"):
            ctl.validate_verdict_citations(
                1, [_verdict("hyp-00", ["fb-ghost-plan"])])

    def test_illegal_revision_window_refused(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        with pytest.raises(ValueError, match="ILLEGAL_REVISION_WINDOW"):
            ctl.validate_verdict_citations(-1, [])


# ------------------------------------------------------ compute-matched
class TestComputeMatched:
    """P0-4: the three modes differ ONLY in what feedback the board sees —
    never in roles, EnvCoder, probe schedule, seeds or budget."""

    def test_seven_llm_family_calls_per_window_all_modes(self, runs):
        _ctls, sums = runs
        for mode, s in sums.items():
            assert s.n_llm_calls == 7 * WINDOWS, mode
            for w in s.windows:
                assert w["n_llm_calls"] == 7, (mode, w["window"])
                assert w["board_call_count"] == C.BOARD_CALLS_PER_WINDOW == 6
                assert w["env_coder_call_count"] == 1

    def test_equal_call_and_transition_budgets_across_modes(self, runs):
        _ctls, sums = runs
        modes = list(sums)
        for other in modes[1:]:
            assert sums[other].n_llm_calls == sums[modes[0]].n_llm_calls
            assert sums[other].total_simulator_transitions == \
                sums[modes[0]].total_simulator_transitions

    def test_simulator_cost_per_window_all_modes(self, runs):
        _ctls, sums = runs
        for mode, s in sums.items():
            assert s.total_simulator_transitions == \
                TRANSITIONS_PER_PROBED_WINDOW * WINDOWS, mode
            for w in s.windows:
                assert w["funnel_stats"] == EXPECTED_FUNNEL, (mode,
                                                              w["window"])
                assert w["n_candidates"] == C.RAW_CANDIDATES == 64
                assert w["n_feedback_records"] == 64
                assert w["gate_passed"] is True
                assert w["evidence_window"] == max(0, w["window"] - 1)

    def test_envelope_role_sequence_per_window(self, runs):
        ctls, _sums = runs
        expected_roles = list(C.BOARD_ROLES) + [C.ROLE_ENV_CODER]
        for mode, ctl in ctls.items():
            assert len(ctl.envelopes) == 7 * WINDOWS, mode
            for window in range(WINDOWS):
                roles = [e.role for e in
                         ctl.envelopes[7 * window:7 * window + 7]]
                assert roles == expected_roles, (mode, window, roles)

    def test_training_seam_identical_and_unauthorized(self, runs):
        ctls, sums = runs
        for mode, s in sums.items():
            assert all(w["training_step_status"] == "SKIPPED_UNAUTHORIZED"
                       for w in s.windows), mode
        assert all(len(ctl.training_log) == WINDOWS for ctl in ctls.values())

    def test_store_size_and_runner_honesty(self, runs):
        ctls, _sums = runs
        for mode, ctl in ctls.items():
            assert len(list(ctl.store.ids())) == 64 * WINDOWS, mode
            assert isinstance(ctl.runner, DeterministicSymbolicProbeRunner)
            assert ctl.runner.real_simulator is False
            assert ctl.runner.total_transitions == \
                TRANSITIONS_PER_PROBED_WINDOW * WINDOWS
            assert ctl.backend.usage.real_calls == 0
            assert ctl.backend.usage.total_calls == 7 * WINDOWS


# ------------------------------------------------------------- static mode
class TestStaticBaseline:
    def test_structurally_never_sees_feedback(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_STATIC_LLM]
        assert s.feedback_citation_coverage == 0.0
        assert s.supported_retention_rate == 0.0
        assert s.refuted_retirement_rate == 0.0
        for w in s.windows:
            assert w["feedback_view_label"] == "null"

    def test_every_revision_is_exploration(self, runs):
        ctls, sums = runs
        s = sums[C.MODE_STATIC_LLM]
        assert all(w["revision_label"] == C.EXPLORATION_LABEL
                   for w in s.windows)
        ctl = ctls[C.MODE_STATIC_LLM]
        assert all(rev.label == C.EXPLORATION_LABEL for rev in ctl.revisions)
        assert all(rev.based_on_feedback_ids == [] for rev in ctl.revisions)
        assert s.decision_distribution == {C.DECISION_MUTATE: WINDOWS}

    def test_plan_never_changes(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_STATIC_LLM]
        assert len(set(s.plan_signature_hashes)) == 1

    def test_no_verdict_ever_cites_anything(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_STATIC_LLM]
        for window, board in ctl.boards.items():
            for v in board.verdicts:
                assert v.verdict == C.HYPOTHESIS_STALE, (window, v)
                assert v.cited_feedback_ids == []
        statuses = ctl.ledger.by_status()
        assert statuses[C.HYPOTHESIS_SUPPORTED] == []
        assert statuses[C.HYPOTHESIS_REFUTED] == []

    def test_assembled_board_context_carries_a_zero_feedback_payload(self,
                                                                      runs):
        """STATIC_FEEDBACK_STRUCTURALLY_HIDDEN: the store DOES accumulate
        feedback in the static run (64 records per window), yet the board
        context assembled by the SAME path run_review_board uses carries an
        EMPTY feedback array under the null view. The isolation is
        structural (the view holds nothing), not prompt discipline."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_STATIC_LLM]
        assert len(list(ctl.store.ids())) == 64 * WINDOWS   # store NOT empty…
        for window in range(WINDOWS):
            view = ctl._feedback_view(window)
            assert view.label == "null"
            assert view.records() == []
            assert view.to_prompt_payload() == []
        # windows >= 1: rebuild the exact context run_review_board assembles
        # (window 0's board ran before any record existed, so its evidence
        # slice is not rebuildable after the fact)
        for window in range(1, WINDOWS):
            view = ctl._feedback_view(window)
            context = build_board_prompt_context(
                window=window, mode=C.MODE_STATIC_LLM,
                board_context=assemble_board_context(
                    ctl.store, window=window - 1,
                    mode=C.MODE_STATIC_LLM,
                    feedback_view_label=view.label),
                view=view,
                hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))
            assert context["feedback"] == []          # …but the board sees 0
            assert context["feedback_view_label"] == "null"


# ------------------------------------------------------------- normal mode
class TestNormalFeedbackLoop:
    def test_revision_labels_and_rate(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_NORMAL_FEEDBACK]
        assert s.revision_rate == 1.0
        assert s.windows[0]["revision_label"] == C.EXPLORATION_LABEL
        assert all(w["revision_label"] == FEEDBACK_DRIVEN_LABEL
                   for w in s.windows[1:])

    def test_revisions_cite_only_frozen_feedback(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        all_ids = set(ctl.store.ids())
        assert len(all_ids) == 64 * WINDOWS
        for rev in ctl.revisions:
            for fid in rev.based_on_feedback_ids:
                assert fid in all_ids
            for mod in rev.modifications:
                if mod.is_exploration:
                    assert mod.based_on_feedback_ids == []
        # C8 re-baseline: windows 0-3 records cited, later ones not yet
        assert ctl.revisions[0].label == C.EXPLORATION_LABEL
        assert all(rev.label == FEEDBACK_DRIVEN_LABEL
                   for rev in ctl.revisions[1:])

    def test_rebaselined_summary_numbers(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_NORMAL_FEEDBACK]
        assert s.feedback_citation_coverage == 0.6667
        assert s.decision_distribution == {C.DECISION_MUTATE: 7,
                                           C.DECISION_RETIRE: 13}
        assert s.supported_retention_rate == 0.0    # no SUPPORTED this run
        assert s.refuted_retirement_rate == 1.0     # every REFUTE retires
        assert len(set(s.plan_signature_hashes)) == 5
        assert [w["global_risk"] for w in s.windows] == \
            ["MEDIUM"] + ["HIGH"] * (WINDOWS - 1)
        assert [w["request_control"] for w in s.windows] == \
            [False] + [True] * (WINDOWS - 1)

    def test_ledger_moved_by_bound_feedback(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        statuses = ctl.ledger.by_status()
        assert statuses[C.HYPOTHESIS_REFUTED]          # >=1 refuted line
        all_ids = set(ctl.store.ids())
        for rec in ctl.ledger.all():
            assert rec.status in C.HYPOTHESIS_STATUSES
            # board-born hypotheses are registered at the freeze with no
            # verdict yet (they only get one in a LATER window), so the
            # revision-history invariant applies to the seeded lines of inquiry
            if rec.hypothesis_id.startswith("hyp-w"):
                continue
            assert rec.revision_history
            for entry in rec.revision_history:
                assert len(entry["previous_record_hash"]) == 64
            for fid in rec.supporting_feedback_ids + \
                    rec.contradicting_feedback_ids:
                assert fid in all_ids
        # the board's new hypothesis was registered at the window-5 freeze:
        # PENDING, with a predicted signature, never retroactively bound
        new_hyps = [h for h in ctl.ledger.all()
                    if h.hypothesis_id.startswith("hyp-w")]
        assert len(new_hyps) == 1
        assert new_hyps[0].status == C.HYPOTHESIS_PENDING
        assert new_hyps[0].source_window == WINDOWS - 1
        assert new_hyps[0].predicted_signature

    def test_feedback_records_are_normal_binding(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        for rec in ctl.store.all():
            assert rec.provenance["binding"] == "normal"
            assert rec.provenance["real_adapter_status"] == \
                C.REAL_SIMULATOR_PROBE_STATUS

    def test_envelopes_are_hash_bound_mock_calls(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        for e in ctl.envelopes:
            assert len(e.request_hash) == 64
            assert len(e.response_hash) == 64
            assert e.backend_id == C.MOCK_BACKEND_ID


# ---------------------------------------------------------- shuffled mode
class TestShuffledFeedback:
    def test_store_stays_honest_only_the_view_is_permuted(self, runs):
        """C9: the permutation never touches the store — every record keeps
        its HONEST candidate<->feedback binding; only the board's view is
        permuted + anonymized (label 'permuted:<seed[:16]>', one frozen
        permutation seed per board window)."""
        ctls, sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for rec in ctl.store.all():
            assert rec.provenance["binding"] == "normal"
        labels = [w["feedback_view_label"]
                  for w in sums[C.MODE_SHUFFLED_FEEDBACK].windows]
        assert len(labels) == WINDOWS
        for label in labels:
            prefix, seed16 = label.split(":")
            assert prefix == "permuted"
            assert len(seed16) == 16
            int(seed16, 16)                          # hex
        # frozen per board window: six distinct permutation seeds
        assert len(set(labels)) == WINDOWS

    def test_board_citations_resolve_to_honest_store_ids(self, runs):
        """The six roles cite ANONYMIZED ids; the controller resolves them
        back to store ids — so every ledger binding and revision citation is
        a real SimulatorFeedbackStore id, and every cited record still lags
        its verdict by at least one window."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        all_ids = set(ctl.store.ids())
        cited_any = False
        for rev in ctl.revisions:
            for fid in rev.based_on_feedback_ids:
                assert fid in all_ids
                assert not fid.startswith("anon-")
                cited_any = True
        assert cited_any
        for rec in ctl.ledger.all():
            for entry in rec.revision_history:
                for fid in entry["feedback_ids"]:
                    assert fid in all_ids
                    assert ctl.store.get(fid).window <= int(entry["window"]) - 1

    def test_rebaselined_shuffled_numbers(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_SHUFFLED_FEEDBACK]
        assert s.feedback_citation_coverage == 0.8047
        assert s.decision_distribution == {C.DECISION_MUTATE: 6,
                                           C.DECISION_RETIRE: 12,
                                           C.DECISION_RETAIN: 2}
        assert s.supported_retention_rate == 1.0
        assert s.refuted_retirement_rate == 1.0
        assert len(set(s.plan_signature_hashes)) == 6
        assert [w["global_risk"] for w in s.windows] == \
            ["MEDIUM"] + ["HIGH"] * (WINDOWS - 1)
        assert [w["request_control"] for w in s.windows] == \
            [False] + [True] * (WINDOWS - 1)

    def test_shuffling_changes_plans(self, runs):
        _ctls, sums = runs
        normal, shuffled = (sums[C.MODE_NORMAL_FEEDBACK],
                            sums[C.MODE_SHUFFLED_FEEDBACK])
        assert normal.plan_signature_hashes != shuffled.plan_signature_hashes
        comparison = FeedbackUEDController.compare_summaries(
            normal, shuffled, sums[C.MODE_STATIC_LLM])
        assert comparison["feedback_binding_matters"] is True
        assert comparison["plan_difference_windows"] >= 1
        # compute-matched even inside the comparison artifact
        assert comparison["normal_llm_calls"] == \
            comparison["shuffled_llm_calls"] == \
            comparison["static_llm_calls"] == 7 * WINDOWS
        assert comparison["static_revision_rate"] == 1.0
        assert comparison["static_plan_difference_vs_normal"] >= 1


# -------------------------------------------------------------- determinism
class TestDeterminism:
    def test_two_runs_byte_identical(self, runs):
        _ctls, sums = runs
        first = json.dumps(sums[C.MODE_NORMAL_FEEDBACK].to_dict(),
                           sort_keys=True)
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        again = json.dumps(ctl.run(max_windows=WINDOWS).to_dict(),
                           sort_keys=True)
        assert first == again

    def test_shuffled_runs_are_reproducible(self, runs):
        """SHUFFLE_PERMUTATION_FROZEN: a second shuffled run reproduces the
        first one byte-for-byte — the permutation is derived from the frozen
        seed schedule, never from runtime randomness."""
        _ctls, sums = runs
        first = json.dumps(sums[C.MODE_SHUFFLED_FEEDBACK].to_dict(),
                           sort_keys=True)
        ctl = FeedbackUEDController(C.MODE_SHUFFLED_FEEDBACK)
        again = json.dumps(ctl.run(max_windows=WINDOWS).to_dict(),
                           sort_keys=True)
        assert first == again

    def test_probe_runner_transitions_match_funnel(self, runs):
        ctls, sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        assert ctl.runner.total_transitions == \
            sums[C.MODE_NORMAL_FEEDBACK].total_simulator_transitions
