"""CC3 C9 GATE — targeted bypass and lag tests (2026-08-04).

The director's CC3 gate re-opened the two C9 isolation flags until THIS
file's targeted tests pass again. The gate's two mandates, tested here
against real three-mode runs (deterministic mock backend + symbolic probe):

BYPASS — the BoardContext is built ONLY from the window's FeedbackView:

* the raw SimulatorFeedbackStore is refused by the assembly itself
  (BOARD_CONTEXT_STORE_FORBIDDEN, unit-tested in
  test_feedback_llm_ued_evidence.py);
* STATIC: even with a FULL store (64 frozen records per completed window),
  every board context is structurally empty — no behavior evidence, zero
  pooled episodes, zero pooled SR, maximal-uncertainty CI — and neither
  the BoardContext nor the assembled prompt context contains a single real
  feedback id or candidate id;
* SHUFFLED: the board context AND the assembled prompt context carry ONLY
  anonymized identity — no real feedback id, no real candidate id anywhere
  (evidence layer included); the evidence items' anonymized ids are the
  SAME ids, in the SAME order, as the prompt payload (evidence<->payload
  consistency); the permutation presents EXACTLY the honest window k-1
  record set and only ``resolve_citation`` maps back.

LAG — the double-window lag is EXACTLY one window (rec.window == window-1):

* a window-k view presents EXACTLY window k-1's 64 frozen records (window
  0: none), for normal and shuffled alike, even though the store already
  holds every older window;
* constructing a Normal/Permuted view from mixed-window records fails
  closed (STALE_FEEDBACK_ID);
* the citation validators refuse older/current/future records as
  STALE_FEEDBACK_ID and accept exactly k-1;
* end-to-end: every revision citation and every ledger verdict citation
  lags its window by EXACTLY one, in all three modes.
"""
import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import assemble_board_context
from d052.feedback_llm_ued.causal_failure_analyst import BoardHypothesisVerdict
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.feedback_view import (
    MASKED_IDENTITY,
    NormalFeedbackView,
    NullFeedbackView,
    PermutedFeedbackView,
)
from d052.feedback_llm_ued.review_board import (
    build_board_prompt_context,
    normalize_hypothesis_inputs,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

WINDOWS = 4          # enough for the full older/current/future lag matrix


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
        reason="gate verdict")


def _synthetic_record(i, *, window):
    cand = synthetic_candidate(candidate_id=f"c-gate-w{window}-{i}",
                               family=C.ENVIRONMENT_FAMILIES[i % 7])
    return synthetic_feedback_record(
        feedback_id=f"fb-gate-w{window}-{i}", candidate=cand,
        plan_id="plan-gate", window=window, student_success_rate=0.4,
        expected_signature={"student_success_rate": 0.47},
        distinguishes_hypothesis_ids=[f"hyp-{i % 7:02d}"])


# --------------------------------------------------------- BYPASS: static
class TestStaticBypassSealed:
    def test_full_store_yields_a_structurally_empty_board_context(self,
                                                                  runs):
        """CC3 C9 gate: the static store is HONEST and FULL (64 records per
        completed window), yet every board context assembled from the
        static view is structurally empty — the store's content is
        unreachable through the view by construction."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_STATIC_LLM]
        assert len(list(ctl.store.ids())) == 64 * WINDOWS
        real_feedback_ids = set(ctl.store.ids())
        real_candidate_ids = {r.candidate_id for r in ctl.store.all()}
        assert real_feedback_ids and real_candidate_ids
        for window in range(1, WINDOWS):
            view = ctl._feedback_view(window)
            assert isinstance(view, NullFeedbackView)
            assert view.records() == []
            assert view.to_prompt_payload() == []
            assert view.behavior_evidence() == []
            ctx = assemble_board_context(
                view, window=window - 1, mode=C.MODE_STATIC_LLM)
            # empty evidence, zero pooled SR/episodes, maximal uncertainty
            assert ctx.behavior_evidence == []
            assert ctx.pooled_episodes == 0
            assert ctx.pooled_student_success_rate == 0.0
            assert ctx.student_success_rate_ci == 1.0
            assert ctx.feedback_view_label == "null"
            # no real identity / history anywhere in the assembled context
            context = build_board_prompt_context(
                window=window, mode=C.MODE_STATIC_LLM, board_context=ctx,
                view=view,
                hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))
            assert context["feedback"] == []
            serialized = json.dumps(context, sort_keys=True)
            for fid in real_feedback_ids:
                assert fid not in serialized, (window, fid)
            for cid in real_candidate_ids:
                assert cid not in serialized, (window, cid)

    def test_static_never_cites_despite_the_full_store(self, runs):
        ctls, _sums = runs
        ctl = ctls[C.MODE_STATIC_LLM]
        assert all(rev.based_on_feedback_ids == []
                   for rev in ctl.revisions)


# ------------------------------------------------------- BYPASS: shuffled
class TestShuffledBypassSealed:
    def test_context_carries_only_anonymized_identity(self, runs):
        """CC3 C9 gate: the shuffled board context + prompt payload contain
        NO real feedback id and NO real candidate id — at the prompt layer
        AND at the evidence layer (no identity side channel)."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        real_feedback_ids = set(ctl.store.ids())
        real_candidate_ids = {r.candidate_id for r in ctl.store.all()}
        for window in range(1, WINDOWS):
            view = ctl._feedback_view(window)
            assert isinstance(view, PermutedFeedbackView)
            payload = view.to_prompt_payload()
            assert len(payload) == 64
            anon_ids = [p["feedback_id"] for p in payload]
            assert len(set(anon_ids)) == 64
            for p in payload:
                assert p["feedback_id"].startswith(f"anon-w{window:02d}-")
                assert p["candidate_id"] == MASKED_IDENTITY
                assert p["mutation_axes"] == []
                assert p["axis_values"] == {}
                assert p["held_constant_axes"] == {}
            ctx = assemble_board_context(
                view, window=window - 1, mode=C.MODE_SHUFFLED_FEEDBACK)
            evidence = ctx.behavior_evidence
            assert len(evidence) == 64
            # evidence<->payload consistency: the SAME anonymized ids, in
            # the SAME order; candidate id masked at the evidence layer
            assert [e.feedback_id for e in evidence] == anon_ids
            assert all(e.candidate_id == MASKED_IDENTITY for e in evidence)
            context = build_board_prompt_context(
                window=window, mode=C.MODE_SHUFFLED_FEEDBACK,
                board_context=ctx, view=view,
                hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))
            serialized = json.dumps(context, sort_keys=True)
            for fid in real_feedback_ids:
                assert fid not in serialized, (window, fid)
            for cid in real_candidate_ids:
                assert cid not in serialized, (window, cid)

    def test_permutation_presents_exactly_the_honest_record_set(self, runs):
        """The permutation changes the BINDING only: the presented records
        are exactly window k-1's honest records, and resolve_citation is
        the only path back (real ids are not citable through the view)."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for window in range(1, WINDOWS):
            view = ctl._feedback_view(window)
            anon_ids = [p["feedback_id"] for p in view.to_prompt_payload()]
            resolved = {view.resolve_citation(a) for a in anon_ids}
            honest = {r.feedback_id for r in _records_of(ctl, window - 1)}
            assert resolved == honest
            # the permutation is a bijection over the honest set — nothing
            # injected, nothing dropped
            assert len(resolved) == len(honest) == 64
            # a real store id is NOT a citation the view accepts
            with pytest.raises(ValueError,
                               match="UNKNOWN_ANONYMIZED_CITATION"):
                view.resolve_citation(sorted(honest)[0])

    def test_permutation_is_frozen_and_recomputable_including_evidence(
            self):
        """Two independent constructions over the same inputs are
        bit-identical — payload AND evidence layer (frozen permutation,
        no runtime randomness)."""
        ctl = FeedbackUEDController(C.MODE_SHUFFLED_FEEDBACK)
        ctl.run(max_windows=2)
        a = ctl._feedback_view(1)
        b = ctl._feedback_view(1)
        assert a.to_prompt_payload() == b.to_prompt_payload()
        assert [e.model_dump() for e in a.behavior_evidence()] == \
            [e.model_dump() for e in b.behavior_evidence()]
        assert a.permutation_seed == b.permutation_seed
        assert a.label == b.label


# -------------------------------------------------------- LAG: exactly k-1
class TestExactOneWindowLag:
    def test_views_present_exactly_the_previous_window(self, runs):
        """A window-k view presents EXACTLY window k-1's 64 frozen records
        (window 0: none) even though the store holds every older window —
        older records are unreachable through the view."""
        ctls, _sums = runs
        for mode in (C.MODE_NORMAL_FEEDBACK, C.MODE_SHUFFLED_FEEDBACK):
            ctl = ctls[mode]
            view0 = ctl._feedback_view(0)
            assert view0.records() == []
            assert view0.to_prompt_payload() == []
            assert view0.behavior_evidence() == []
            for window in range(1, WINDOWS):
                view = ctl._feedback_view(window)
                assert view.window_scope == window - 1
                records = view.records()
                assert len(records) == 64, (mode, window)
                assert all(r.window == window - 1 for r in records)
                # the store already holds ALL windows — the view still
                # presents exactly one of them
                assert len(list(ctl.store.ids())) == 64 * WINDOWS
        # static: structurally empty at every window
        static = ctls[C.MODE_STATIC_LLM]
        for window in range(WINDOWS):
            assert isinstance(static._feedback_view(window),
                              NullFeedbackView)

    def test_mixed_window_view_construction_fails_closed(self):
        """CC3 C9 gate defense in depth: handing a Normal/Permuted view a
        record from ANY other window raises STALE_FEEDBACK_ID at
        construction time."""
        rec_w0 = _synthetic_record(0, window=0)
        rec_w1 = _synthetic_record(1, window=1)
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            NormalFeedbackView([rec_w0, rec_w1], window_scope=0)
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            NormalFeedbackView([rec_w0], window_scope=1)
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            PermutedFeedbackView(
                [rec_w0], window_scope=1, board_window=2,
                mode=C.MODE_SHUFFLED_FEEDBACK,
                seed_schedule_hash=C.SEED_SCHEDULE_HASH)

    def test_citation_validator_matrix(self, runs):
        """older / current / future citations all fail closed as
        STALE_FEEDBACK_ID; exactly k-1 passes."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_NORMAL_FEEDBACK]
        fid_w0 = _records_of(ctl, 0)[0].feedback_id      # distinguishes hyp-00
        fid_w2 = _records_of(ctl, 2)[0].feedback_id
        # older than k-1: window 3 citing a window-0 record
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(3, [_verdict("hyp-00", [fid_w0])])
        # current window: window 0 citing its own record
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(0, [_verdict("hyp-00", [fid_w0])])
        # future: window 1 citing a window-2 record
        with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
            ctl.validate_verdict_citations(1, [_verdict("hyp-00", [fid_w2])])
        # exactly k-1: legal
        ctl.validate_verdict_citations(1, [_verdict("hyp-00", [fid_w0])])
        recs_w1 = [r for r in _records_of(ctl, 1)
                   if "hyp-01" in r.distinguishes_hypothesis_ids]
        assert recs_w1, "window 1 must still probe hyp-01's family"
        ctl.validate_verdict_citations(
            2, [_verdict("hyp-01", [recs_w1[0].feedback_id])])

    def test_prompt_context_refuses_wrong_window_evidence_and_context(
            self):
        """build_board_prompt_context enforces the exact lag twice: every
        evidence item must be from window k-1, and the BoardContext window
        must equal the evidence window."""
        rec = _synthetic_record(0, window=0)
        view = NormalFeedbackView([rec], window_scope=0)
        ctx = assemble_board_context(view, window=0,
                                     mode=C.MODE_NORMAL_FEEDBACK)
        # window-2 board handed window-0 evidence: refused
        with pytest.raises(ValueError,
                           match="BOARD_CONTEXT_WINDOW_MISMATCH"):
            build_board_prompt_context(
                window=2, mode=C.MODE_NORMAL_FEEDBACK, board_context=ctx,
                view=view, hypotheses=[])
        # board-context window check (empty evidence, wrong ctx window):
        null_ctx = assemble_board_context(NullFeedbackView(), window=0,
                                          mode=C.MODE_STATIC_LLM)
        with pytest.raises(ValueError,
                           match="BOARD_CONTEXT_WINDOW_MISMATCH"):
            build_board_prompt_context(
                window=2, mode=C.MODE_STATIC_LLM, board_context=null_ctx,
                view=NullFeedbackView(), hypotheses=[])

    def test_every_citation_lags_exactly_one_window_end_to_end(self, runs):
        """The gate's defining invariant, end to end: in ALL three modes,
        every revision citation and every ledger verdict citation lags its
        revision/verdict window by EXACTLY one window."""
        ctls, _sums = runs
        for mode, ctl in ctls.items():
            cited = 0
            for rev in ctl.revisions:
                for fid in rev.based_on_feedback_ids:
                    rec = ctl.store.get(fid)
                    assert rec.window == rev.window - 1, \
                        (mode, rev.window, fid, rec.window)
                    cited += 1
            for hyp in ctl.ledger.all():
                for entry in hyp.revision_history:
                    for fid in entry["feedback_ids"]:
                        fb = ctl.store.get(fid)
                        assert fb.window == int(entry["window"]) - 1, \
                            (mode, hyp.hypothesis_id, entry["window"], fid)
            if mode == C.MODE_STATIC_LLM:
                assert cited == 0          # structurally nothing to cite
            else:
                assert cited > 0           # the loop really cites feedback
