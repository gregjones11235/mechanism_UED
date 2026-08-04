"""End-to-end: the double-window state machine (C8).

Per window k: EVIDENCE -> BOARD (always all six roles) -> REVISION (verdicts
+ plan_k, citing ONLY feedback from EXACTLY window k-1 — CC3 C9 gate;
older/current/future records fail closed as STALE_FEEDBACK_ID) -> PROBING
(EnvCoder + gates + staged funnel probe -> staged feedback_k) -> atomic
FREEZE. After the
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

C10 RETIRE lifecycle: a retired family enters a RETIRE_COOLDOWN_WINDOWS=3
cooldown (hard block, FAMILY_IN_COOLDOWN fail closed at the Reconciler),
then STAYS retired until reopened (human_reopen_families or all-new
evidence) — so each family retires at most ONCE per run, a STALE verdict
can never resurrect it, and the board context carries the blocked lists so
the six roles skip cooldown/retired families by construction.

CC4 C9 gate round-two re-baseline (6 windows, deterministic mock, exact k-1
lag, shuffled numeric side channel removed): 42 LLM-family calls and 368640
simulator transitions per mode; normal coverage 0.8047 with {MUTATE: 7,
RETIRE: 4, RETAIN: 3} — threat_distance retired@1, day_night_rest_need@4,
visibility@4, resource_pressure@5 (the normal view is honest and untouched);
shuffled coverage 0.8047 with {MUTATE: 9, RETIRE: 3, RETAIN: 1} —
threat_distance@1, resource_pressure@2, day_night_rest_need@5 (a DIFFERENT
retirement set and decision mix than normal, so feedback_binding_matters
stays True). The shuffled shift vs the CC3 baseline ({MUTATE: 7, RETIRE: 3,
RETAIN: 3}) is expected: the shuffled view now publishes ONLY family-level
window aggregates instead of exact per-candidate rates/gaps (which were
candidate-hash fingerprints), so the mock roles rank and retire from coarser
numbers. Both modes keep six unique plan signatures and anon-citation
resolution into honest ledger/revision ids; static keeps {MUTATE: 6}, a
single plan signature and 0.0 coverage.

C11 REQUEST_CONTROL blocking: a board that requests human control (critic
escalation and/or a tutor REQUEST_CONTROL proposal) halts the loop right
after phase B — no verdicts, no plan, no probe, no freeze, NO execution
batch. The halt is recorded as a hash-bound HumanDecisionArtifact in the
RunSummary (tutor citations resolved to real store ids), the stopped window
stays closed to revision (phase BOARD), and the LaunchGate final_batch
verdict is final=False. The mock critic escalates only on honesty
violations or severe-but-THIN evidence, so the deterministic baselines
(severe but PRECISE from window 1 on) never halt; scripted backends prove
both trigger paths and the halt semantics.
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
from d052.feedback_llm_ued.execution_mode import FeedbackLaunchGate
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.human_decision import HumanDecisionArtifact
from d052.feedback_llm_ued.llm_backend import DeterministicMockFeedbackBackend
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
        # CC4 C9 GATE ROUND TWO (2026-08-04), EARNED AGAIN: both director
        # findings fixed and locked by negative tests
        # (test_feedback_llm_ued_c9_gate.py, 18 cases): (1) the shuffled
        # view's identity-correlated NUMERIC side channels (exact probe
        # rates / exact evidence gaps — deterministic per-candidate-hash
        # fingerprints) are consistently anonymized at the prompt layer AND
        # the BehaviorFailureEvidence layer (family-level window aggregates
        # only; family-grain predicted signature dropped); (2) the STATIC
        # phase-A assembly no longer touches the RETIRE lifecycle query
        # (whose reopen gate reads the raw store) — static uses the frozen
        # empty lifecycle and the query itself fails closed, so the static
        # BoardContext and all six prompts are a pure function of the
        # non-feedback state (byte-identical under foreign-store pollution).
        assert C.STATIC_FEEDBACK_STRUCTURALLY_HIDDEN is True
        assert C.SHUFFLE_PERMUTATION_FROZEN is True
        assert len(C.SEED_SCHEDULE_HASH) == 64
        # C10 RETIRE lifecycle constant
        assert C.RETIRE_COOLDOWN_WINDOWS == 3
        # C16: the plan-alignment and feedback-binding engineering flags are
        # ON (ENGINEERING_SCAFFOLD evidence level; every REAL_* flag above
        # remains False)
        assert C.SIX_ROLE_BOARD_IMPLEMENTED is True
        assert C.E2_FORMAL_PLAN_ALIGNED is True
        assert C.FEEDBACK_REVISION_BOUND is True

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

    def test_double_window_lag_every_citation_lags_exactly_one_window(self,
                                                                      runs):
        """The defining invariant (tightened by the CC3 C9 gate): every
        feedback record a revision cites was frozen in EXACTLY the previous
        window — rec.window == revision window - 1. Older records are
        stale, current/future records cannot be cited at all; window 0
        cites nothing."""
        ctls, _sums = runs
        for mode, ctl in ctls.items():
            for rev in ctl.revisions:
                for fid in rev.based_on_feedback_ids:
                    rec = ctl.store.get(fid)
                    assert rec.window == rev.window - 1, (mode, rev.window,
                                                          fid, rec.window)
            assert ctl.revisions[0].based_on_feedback_ids == []
            # same for ledger verdicts: bound feedback lags the verdict by
            # EXACTLY one window
            for rec in ctl.ledger.all():
                for entry in rec.revision_history:
                    for fid in entry["feedback_ids"]:
                        fb = ctl.store.get(fid)
                        assert fb.window == int(entry["window"]) - 1

    # -- citation validator (pure, phase-independent P0-6 guard) -----------
    def test_legal_next_window_citation_passes(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        # window 1 citing a window-0 record that distinguishes hyp-00: legal
        ctl.validate_verdict_citations(
            1, [_verdict("hyp-00", [fid])])

    def test_same_window_feedback_is_stale(self, runs):
        """Citing feedback produced in the VERY window being revised is the
        double-window violation the state machine exists to prevent (CC3 C9
        gate: current-window records fail closed as STALE_FEEDBACK_ID)."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(0, [_verdict("hyp-00", [fid])])

    def test_later_window_feedback_is_stale(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 2)[0].feedback_id
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(1, [_verdict("hyp-00", [fid])])

    def test_older_window_feedback_is_stale(self, runs):
        """CC3 C9 gate negative test: the lag is EXACTLY one window — a
        window-3 revision citing a window-0 record (older than k-1) fails
        closed as STALE_FEEDBACK_ID, not merely 'still legal'."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid = _records_of(ctl, 0)[0].feedback_id
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(3, [_verdict("hyp-00", [fid])])

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
        # (CC3 C9 gate: assembly consumes the VIEW only — never the raw
        # store, which here is provably full)
        for window in range(1, WINDOWS):
            view = ctl._feedback_view(window)
            board_context = assemble_board_context(
                view, window=window - 1, mode=C.MODE_STATIC_LLM)
            context = build_board_prompt_context(
                window=window, mode=C.MODE_STATIC_LLM,
                board_context=board_context,
                view=view,
                hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))
            assert context["feedback"] == []          # …but the board sees 0
            assert context["feedback_view_label"] == "null"
            # the whole evidence layer is structurally empty too: no
            # behavior evidence, no pooled SR/CI, no candidate ids
            assert board_context.behavior_evidence == []
            assert board_context.pooled_episodes == 0
            assert board_context.pooled_student_success_rate == 0.0
            assert board_context.student_success_rate_ci == 1.0


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
        assert s.feedback_citation_coverage == 0.8047
        # CC3 C9 gate re-baseline (EXACT k-1 lag): each window's board sees
        # ONLY the previous window's 64 records, so refutations arrive with
        # fresh citable evidence and every refuted line retires (4 of 4)
        assert s.decision_distribution == {C.DECISION_MUTATE: 7,
                                           C.DECISION_RETIRE: 4,
                                           C.DECISION_RETAIN: 3}
        assert s.supported_retention_rate == 1.0
        assert s.refuted_retirement_rate == 1.0
        assert len(set(s.plan_signature_hashes)) == 6
        assert [w["global_risk"] for w in s.windows] == \
            ["MEDIUM"] + ["HIGH"] * (WINDOWS - 1)
        # C11: the loop never escalates — the baseline evidence is severe
        # but PRECISE (pooled CI is tiny from window 1 on), and C11 halts
        # only on honesty violations or severe-but-THIN evidence
        assert [w["request_control"] for w in s.windows] == \
            [False] * WINDOWS
        assert s.request_control_stopped is False
        assert s.stopped_window is None
        assert s.human_decision_artifact is None
        # …but even a completed loop ships no FINAL batch this round:
        # training is unauthorized (MOCK_DRY_RUN)
        assert s.final_batch["final"] is False
        assert "TRAINING_NOT_ALLOWED" in s.final_batch["reason"]
        assert s.final_batch["loop_completed"] is True

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
                    # CC3 C9 gate: the lag is EXACTLY one window
                    assert ctl.store.get(fid).window == \
                        int(entry["window"]) - 1

    def test_rebaselined_shuffled_numbers(self, runs):
        _ctls, sums = runs
        s = sums[C.MODE_SHUFFLED_FEEDBACK]
        assert s.feedback_citation_coverage == 0.8047
        # CC4 C9 gate round-two re-baseline (numeric side-channel
        # hardening): the shuffled view now publishes ONLY family-level
        # window aggregates (exact rates/gaps were per-candidate-hash
        # fingerprints), so the mock roles see coarser numbers and the
        # shuffled decision distribution shifts again — but the two modes
        # STILL differ (3 vs 4 retirements, different decision mix), which
        # is the point of the comparison
        assert s.decision_distribution == {C.DECISION_MUTATE: 9,
                                           C.DECISION_RETIRE: 3,
                                           C.DECISION_RETAIN: 1}
        assert s.supported_retention_rate == 1.0
        assert s.refuted_retirement_rate == 1.0
        assert len(set(s.plan_signature_hashes)) == 6
        assert [w["global_risk"] for w in s.windows] == \
            ["MEDIUM"] + ["HIGH"] * (WINDOWS - 1)
        # C11: same as normal — severe but precise evidence never halts
        assert [w["request_control"] for w in s.windows] == \
            [False] * WINDOWS
        assert s.request_control_stopped is False
        assert s.stopped_window is None
        assert s.human_decision_artifact is None
        assert s.final_batch["final"] is False

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


# --------------------------------------------------------- C10 RETIRE lifecycle
class TestRetireLifecycle:
    """RETIRE is a lifecycle, not a per-window event: cooldown block ->
    retired-until-reopened -> reopen only via human authorization or
    all-new distinguishing evidence. STALE verdicts cannot resurrect."""

    def test_each_family_retires_at_most_once(self, runs):
        ctls, _sums = runs
        for mode in (C.MODE_NORMAL_FEEDBACK, C.MODE_SHUFFLED_FEEDBACK):
            ctl = ctls[mode]
            retired_ever = {}
            for window in sorted(ctl._plans_by_window):
                plan = ctl._plans_by_window[window]
                for fam in plan.retired_families:
                    assert fam not in retired_ever, (mode, fam, window)
                    retired_ever[fam] = window
            # no reopen happened in the base runs: registry == history
            assert retired_ever == ctl._retired_at

    def test_retirement_partition_over_the_run(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fam = FAM0                                   # retired at window 1
        assert ctl._retired_at[fam] == 1
        for window in range(2, 5):                   # cooldown windows 2..4
            in_cooldown, blocked_retired, reopened = \
                ctl._retirement_state(window)
            assert fam in in_cooldown
            assert fam not in blocked_retired
            assert fam not in reopened
        in_cooldown, blocked_retired, reopened = ctl._retirement_state(5)
        assert fam not in in_cooldown                # cooldown over at w5...
        assert fam in blocked_retired                # ...but NOT reopened
        assert reopened == ()
        # the loop never funded a blocked family in any window
        for window in range(WINDOWS):
            plan = ctl._plans_by_window[window]
            funded = {a.environment_family for a in plan.allocations}
            in_cooldown, blocked_retired, _r = ctl._retirement_state(window)
            assert not funded & (set(in_cooldown) | set(blocked_retired))

    def test_board_skips_blocked_families_entirely(self, runs):
        """From the retirement window onwards the six roles emit NO proposal
        and NO directive for the blocked family (tutor + explorer skip)."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fam = FAM0                                   # retired at window 1
        for window in range(2, WINDOWS):
            board = ctl.boards[window]
            assert all(p.environment_family != fam
                       for p in board.family_proposals), window
            assert all(d.environment_family != fam
                       for d in board.directives), window

    def test_stale_verdict_cannot_resurrect_retired_family(self):
        """Unit: a blocked family whose hypothesis has NO visible records
        (the STALE branch) receives no exploration proposal — resurrection
        is structurally impossible."""
        from d052.feedback_llm_ued import intervention_tutor
        fam = FAM0
        context = dict(window=3,
                       hypotheses=[{"hypothesis_id": "hyp-00",
                                    "environment_family": fam}],
                       feedback=[],                  # nothing visible: STALE
                       board_context={"retired_families": [],
                                      "families_in_cooldown": [fam]})
        out = intervention_tutor.mock_rule(context)
        fams = [p["environment_family"] for p in out["family_proposals"]]
        assert fam not in fams
        assert "skipped retired/cooldown families" in out["rationale"]
        # identical context WITHOUT the block does propose exploration
        context["board_context"] = {"retired_families": [],
                                    "families_in_cooldown": []}
        out = intervention_tutor.mock_rule(context)
        fams = [p["environment_family"] for p in out["family_proposals"]]
        assert fam in fams

    def test_explorer_emits_no_directives_for_blocked_families(self):
        from d052.feedback_llm_ued import explorer
        fam, fam2 = FAM0, FAM1
        rec = {"student_success_rate": 0.4, "reference_success_rate": 0.9,
               "axis_values": {}, "distinguishes_hypothesis_ids": []}
        context = dict(window=3,
                       feedback=[dict(rec, feedback_id="fb-x",
                                      environment_family=fam),
                                 dict(rec, feedback_id="fb-y",
                                      environment_family=fam2)],
                       board_context={"retired_families": [fam],
                                      "families_in_cooldown": []})
        out = explorer.mock_rule(context)
        fams = {d["environment_family"] for d in out["directives"]}
        assert fam not in fams
        assert fam2 in fams

    def test_board_context_validates_retirement_lists(self):
        from d052.feedback_llm_ued.behavior_failure import BoardContext
        with pytest.raises(ValueError, match="UNKNOWN_ENVIRONMENT_FAMILY"):
            BoardContext(window=0, mode=C.MODE_NORMAL_FEEDBACK,
                         retired_families=["not_a_family"])
        with pytest.raises(ValueError, match="DUPLICATE_FAMILY_IN"):
            BoardContext(window=0, mode=C.MODE_NORMAL_FEEDBACK,
                         families_in_cooldown=[FAM0, FAM0])

    def test_human_reopen_authorizes_a_comeback_window(self):
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                    human_reopen_families=[FAM0])
        s = ctl.run(max_windows=WINDOWS)
        # CC3 C9 gate re-baseline (EXACT k-1 lag): FAM0 retires at w1
        # (cited verdict), is blocked through w2..w4, and is human-reopened
        # at w5. Under the strict one-window lag the w5 board sees NO FAM0
        # feedback at all (the blocked windows produced none), so the
        # comeback proposal is an uncited EXPLORATION MUTATE; the family is
        # funded, the reopen is CONSUMED, and FAM0 leaves the retirement
        # registry — a re-retirement would need fresh refuting evidence,
        # which only a later window could bring.
        assert FAM0 not in ctl._retired_at
        props = [p for p in ctl.boards[WINDOWS - 1].family_proposals
                 if p.environment_family == FAM0]
        assert len(props) == 1
        assert props[0].decision == C.DECISION_MUTATE
        assert props[0].is_exploration
        assert props[0].based_on_feedback_ids == []
        funded = {a.environment_family
                  for a in ctl._plans_by_window[WINDOWS - 1].allocations}
        assert FAM0 in funded
        # the retirement itself happened at w1 and WAS cited (a real
        # verdict, not exploration)
        retire_at_w1 = [p for p in ctl.boards[1].family_proposals
                        if p.environment_family == FAM0]
        assert len(retire_at_w1) == 1
        assert retire_at_w1[0].decision == C.DECISION_RETIRE
        assert retire_at_w1[0].based_on_feedback_ids
        # the intermediate windows kept the family fully blocked
        for window in range(2, WINDOWS - 1):
            assert all(p.environment_family != FAM0
                       for p in ctl.boards[window].family_proposals)
        # lifecycle invariants hold under reopen: compute-matched + the
        # run's RETIRE decisions are the w1 FAM0 retirement plus the three
        # later families
        assert s.n_llm_calls == 7 * WINDOWS
        assert s.total_simulator_transitions == \
            WINDOWS * TRANSITIONS_PER_PROBED_WINDOW
        assert s.decision_distribution[C.DECISION_RETIRE] == 4

    def test_illegal_human_reopen_family_refused(self):
        with pytest.raises(ValueError, match="UNKNOWN_ENVIRONMENT_FAMILY"):
            FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                  human_reopen_families=["not_a_family"])

    def test_reopen_eligibility_rules(self):
        """The reopen gate: human authorization OR genuinely new evidence —
        an empty or stale evidence set authorizes nothing."""
        retired = {FAM0: 1}
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        ctl._seed()                                   # hyp-00 -> FAM0
        # (a) no evidence + no authorization -> stays retired
        assert ctl._reopen_eligible(5, retired) == ()
        # (b) human authorization reopens regardless of evidence state
        ctl.human_reopen_families = frozenset({FAM0})
        assert ctl._reopen_eligible(5, retired) == (FAM0,)
        ctl.human_reopen_families = frozenset()
        # (c) OLD distinguishing evidence blocks the evidence path
        cand = synthetic_candidate(candidate_id="cand-r1", family=FAM0)
        ctl.store.add(synthetic_feedback_record(
            feedback_id="fb-r-old", candidate=cand, plan_id="plan-x",
            window=0, student_success_rate=0.4,
            expected_signature={"student_success_rate": 0.47},
            distinguishes_hypothesis_ids=["hyp-00"]))
        assert ctl._reopen_eligible(5, retired) == ()
        # (d) ALL distinguishing evidence strictly after the retirement
        # window reopens the family
        fresh = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        fresh._seed()
        for i, window in enumerate((2, 3)):
            cand = synthetic_candidate(candidate_id=f"cand-r{i + 2}",
                                       family=FAM0)
            fresh.store.add(synthetic_feedback_record(
                feedback_id=f"fb-r-new{i}", candidate=cand, plan_id="plan-x",
                window=window, student_success_rate=0.4,
                expected_signature={"student_success_rate": 0.47},
                distinguishes_hypothesis_ids=["hyp-00"]))
        assert fresh._reopen_eligible(5, retired) == (FAM0,)
        # ...but never inside the cooldown, new evidence or not
        assert fresh._reopen_eligible(3, retired) == ()


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


# ---------------------------------------- C11 scripted REQUEST_CONTROL backends
class _EscalatingCriticBackend(DeterministicMockFeedbackBackend):
    """Mock backend whose Nth Critic/Skeptic call demands human control.
    Deterministic: identical runs produce identical escalations."""

    def __init__(self, escalate_on_critic_call: int):
        super().__init__()
        self._target = escalate_on_critic_call
        self._critic_calls = 0

    def complete(self, role, prompt):
        raw = super().complete(role, prompt)
        if role != C.ROLE_CRITIC_SKEPTIC:
            return raw
        self._critic_calls += 1
        if self._critic_calls != self._target:
            return raw
        dump = json.loads(raw)
        dump["request_control"] = True
        dump["endorsed"] = False
        dump["critique_summary"] += " | C11 test: human control requested"
        return json.dumps(dump, sort_keys=True, ensure_ascii=False)


class _RequestControlTutorBackend(DeterministicMockFeedbackBackend):
    """Mock backend whose Nth InterventionTutor call injects a legal,
    feedback-cited REQUEST_CONTROL proposal for the first visible family
    (citation taken from the prompt context between the CONTRACT markers)."""

    def __init__(self, escalate_on_tutor_call: int):
        super().__init__()
        self._target = escalate_on_tutor_call
        self._tutor_calls = 0

    def complete(self, role, prompt):
        raw = super().complete(role, prompt)
        if role != C.ROLE_INTERVENTION_TUTOR:
            return raw
        self._tutor_calls += 1
        if self._tutor_calls != self._target:
            return raw
        visible = sorted(extract_context(prompt)["feedback"],
                         key=lambda p: p["feedback_id"])
        assert visible, "the escalation window must have visible feedback"
        first = visible[0]
        dump = json.loads(raw)
        dump["family_proposals"].append(dict(
            environment_family=first["environment_family"],
            decision=C.DECISION_REQUEST_CONTROL,
            based_on_feedback_ids=[first["feedback_id"]],
            based_on_hypothesis_ids=[],
            reason="C11 test: human control requested on cited evidence",
            is_exploration=False))
        return json.dumps(dump, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------- C11 REQUEST_CONTROL block
class TestRequestControlBlocking:
    """C11: a REQUEST_CONTROL board HALTS the loop right after phase B. The
    stopped window produces NO execution batch (no verdict application, no
    plan, no probe, no freeze), a HumanDecisionArtifact lands in the
    RunSummary, the stopped window is closed to revision exactly like a
    frozen one, and the LaunchGate's final_batch verdict is final=False."""

    def test_critic_escalation_stops_the_loop(self):
        ctl = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            backend=_EscalatingCriticBackend(escalate_on_critic_call=3))
        s = ctl.run(max_windows=WINDOWS)
        stop = 2                     # the 3rd critic call belongs to window 2
        assert s.n_windows == stop + 1
        assert s.request_control_stopped is True
        assert s.stopped_window == stop
        # budget: windows 0..1 ran fully (7 calls each) + only the six
        # board calls of the stopped window; only two windows ever probed
        assert s.n_llm_calls == 7 * stop + C.BOARD_CALLS_PER_WINDOW
        assert s.total_simulator_transitions == \
            TRANSITIONS_PER_PROBED_WINDOW * stop
        assert len(list(ctl.store.ids())) == 64 * stop

        rec = s.windows[stop]
        assert rec["request_control"] is True
        assert rec["phase"] == PHASE_BOARD          # halted right after B
        assert rec["board_call_count"] == C.BOARD_CALLS_PER_WINDOW
        assert rec["env_coder_call_count"] == 0
        assert rec["n_llm_calls"] == C.BOARD_CALLS_PER_WINDOW
        assert rec["plan_id"] == ""                 # NO execution batch
        assert rec["plan_signature_hash"] == ""
        assert rec["revision_label"] == \
            C.REVISION_LABEL_REQUEST_CONTROL_STOPPED
        assert rec["gate_passed"] is False
        assert rec["n_candidates"] == 0
        assert rec["n_feedback_records"] == 0
        assert rec["funnel_stats"] == {}
        assert rec["window_aggregates"] == {}
        assert rec["training_step_status"] == "NOT_EXECUTED_REQUEST_CONTROL"

        # the artifact: critic-triggered, hash-bound to the escalated board
        board = ctl.boards[stop]
        art = s.human_decision_artifact
        assert art is not None
        assert art["window"] == stop
        assert art["mode"] == C.MODE_NORMAL_FEEDBACK
        assert art["trigger_sources"] == [C.ROLE_CRITIC_SKEPTIC]
        assert art["global_risk"] == board.critic.global_risk
        assert art["critic_objections"] == list(board.critic.objections)
        assert art["board_hash"] == board.board_hash
        assert art["artifact_id"] == \
            f"hda-w{stop:02d}-{board.board_hash[:16]}"
        assert art["request_control_families"] == []   # critic-only stop
        assert art["cited_feedback_ids"] == []
        assert len(art["artifact_hash"]) == 64

        # LaunchGate: a stopped loop never ships a final batch
        assert s.final_batch["final"] is False
        assert s.final_batch["request_control_stopped"] is True
        assert s.final_batch["loop_completed"] is False
        assert "REQUEST_CONTROL_STOPPED" in s.final_batch["reason"]

        # the stopped window is closed to revision exactly like a frozen one
        with pytest.raises(SameWindowRevisionForbidden):
            ctl.apply_board_verdicts(stop, [])
        with pytest.raises(SameWindowRevisionForbidden):
            ctl.revise_plan(stop, board)

        # deterministic: an identically scripted run reproduces everything
        again = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            backend=_EscalatingCriticBackend(escalate_on_critic_call=3))
        assert json.dumps(s.to_dict(), sort_keys=True) == \
            json.dumps(again.run(max_windows=WINDOWS).to_dict(),
                       sort_keys=True)

    def test_tutor_request_control_proposal_stops_the_loop(self):
        ctl = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            backend=_RequestControlTutorBackend(escalate_on_tutor_call=2))
        s = ctl.run(max_windows=WINDOWS)
        stop = 1                     # the 2nd tutor call belongs to window 1
        assert s.n_windows == stop + 1
        assert s.request_control_stopped is True
        assert s.stopped_window == stop
        assert s.n_llm_calls == 7 * stop + C.BOARD_CALLS_PER_WINDOW
        assert s.total_simulator_transitions == \
            TRANSITIONS_PER_PROBED_WINDOW * stop

        art = s.human_decision_artifact
        assert art is not None
        # the critic did NOT escalate (severe-but-precise evidence from
        # window 1 on) — the tutor's cited REQUEST_CONTROL proposal alone
        # halts the loop
        assert art["trigger_sources"] == [C.ROLE_INTERVENTION_TUTOR]
        assert art["request_control_families"]
        for fam in art["request_control_families"]:
            assert fam in C.ENVIRONMENT_FAMILIES
        # citations resolved to REAL store ids (window-0 records)
        all_ids = set(ctl.store.ids())
        assert art["cited_feedback_ids"]
        for fid in art["cited_feedback_ids"]:
            assert fid in all_ids
            assert not fid.startswith("anon-")
            assert ctl.store.get(fid).window == 0
        assert s.final_batch["final"] is False
        assert s.final_batch["request_control_stopped"] is True

    def test_critic_escalation_stops_static_mode_at_window_zero(self):
        # even the structurally feedback-blind mode honors the halt
        ctl = FeedbackUEDController(
            C.MODE_STATIC_LLM,
            backend=_EscalatingCriticBackend(escalate_on_critic_call=1))
        s = ctl.run(max_windows=WINDOWS)
        assert s.n_windows == 1
        assert s.stopped_window == 0
        assert s.n_llm_calls == C.BOARD_CALLS_PER_WINDOW
        assert s.total_simulator_transitions == 0
        assert s.human_decision_artifact["trigger_sources"] == \
            [C.ROLE_CRITIC_SKEPTIC]
        assert s.final_batch["final"] is False

    def test_final_batch_gate_units(self):
        gate = FeedbackLaunchGate()
        done = gate.evaluate_final_batch(loop_completed=True,
                                         request_control_stopped=False)
        assert done.final is False          # training unauthorized this round
        assert done.loop_completed is True
        assert "TRAINING_NOT_ALLOWED" in done.reason
        stopped = gate.evaluate_final_batch(loop_completed=False,
                                            request_control_stopped=True)
        assert stopped.final is False
        assert stopped.request_control_stopped is True
        assert "REQUEST_CONTROL_STOPPED" in stopped.reason
        assert "LOOP_NOT_COMPLETED" in stopped.reason
        assert "TRAINING_NOT_ALLOWED" in stopped.reason

    def test_artifact_validation_is_fail_closed(self):
        board_hash = "ab" * 32
        base = dict(artifact_id=f"hda-w01-{board_hash[:16]}", window=1,
                    mode=C.MODE_NORMAL_FEEDBACK,
                    trigger_sources=[C.ROLE_CRITIC_SKEPTIC],
                    global_risk="HIGH", board_hash=board_hash)
        art = HumanDecisionArtifact(**base)
        assert len(art.artifact_hash) == 64
        assert art.rehash() == art.artifact_hash
        with pytest.raises(ValueError, match="UNKNOWN_MODE"):
            HumanDecisionArtifact(**{**base, "mode": "bogus"})
        with pytest.raises(ValueError, match="EMPTY_TRIGGER_SOURCES"):
            HumanDecisionArtifact(**{**base, "trigger_sources": []})
        with pytest.raises(ValueError, match="ILLEGAL_TRIGGER_SOURCE"):
            HumanDecisionArtifact(**{**base, "trigger_sources": ["explorer"]})
        with pytest.raises(ValueError, match="ILLEGAL_BOARD_HASH"):
            HumanDecisionArtifact(**{**base, "board_hash": "short"})
        with pytest.raises(ValueError,
                           match="TUTOR_TRIGGER_WITHOUT_PROPOSALS"):
            HumanDecisionArtifact(**{
                **base, "trigger_sources": [C.ROLE_INTERVENTION_TUTOR]})
        with pytest.raises(ValueError, match="ARTIFACT_ID_MISMATCH"):
            HumanDecisionArtifact(**{**base, "artifact_id": "hda-w99-xxxx"})
        # tamper: content changed but the old hash kept -> rehash diverges
        dump = art.model_dump()
        dump["global_risk"] = "LOW"
        tampered = HumanDecisionArtifact(**dump)
        assert tampered.rehash() != tampered.artifact_hash
