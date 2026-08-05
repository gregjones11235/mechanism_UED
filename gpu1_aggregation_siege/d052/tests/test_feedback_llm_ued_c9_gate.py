"""C9 GATE — targeted bypass, lag, re-identification and byte-parity tests.

The director's CC3 gate (and the CC4 round-two gate, 2026-08-04) keep the
two C9 isolation flags open until THIS file's targeted tests pass. The
mandates, tested here against real three-mode runs (deterministic mock
backend + symbolic probe):

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

RE-IDENTIFICATION (CC4 round two) — no numeric side channel either:

* exact probe rates/gaps are deterministic per-candidate-hash fingerprints,
  so the shuffled view publishes ONLY per-family window aggregates, the
  SAME numbers at the prompt layer and the evidence layer (consistent
  anonymization); the family-grain predicted signature is dropped;
* a store-joining adversary cannot narrow any presented item below its
  public (window, family, match) class — the view adds NO identifying
  power, and in this gate's windows every such class holds >= 2 records
  (uniqueness negative test);
* no exact per-record metric appears anywhere in the serialized prompt
  context (every published float is checked against the honest store).

BYTE PARITY (CC4 round two) — the frozen permutation is byte-reproducible:
two INDEPENDENT full runs assemble byte-identical prompt contexts (view
payload + board context + hypotheses, canonical JSON) at every window.

STATIC INDEPENDENCE (CC4 round two, director REQUEST_CHANGES) — the static
leak: phase A used to derive ``families_in_cooldown``/``retired_families``
from the RETIRE lifecycle query, whose reopen gate reads the raw
SimulatorFeedbackStore — feedback-independent only BY COINCIDENCE (the
static registry is empty). The fix makes it STRUCTURAL: the static mode
uses the frozen empty lifecycle, and the store-reading query itself fails
closed (STATIC_MODE_HAS_NO_RETIREMENT_LIFECYCLE). Proven here: two static
runs whose stores differ ONLY in feedback records assemble byte-identical
board contexts and all six board prompts at every window.

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
from collections import Counter

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import (
    BehaviorFailureEvidence,
    assemble_board_context,
)
from d052.feedback_llm_ued.causal_failure_analyst import BoardHypothesisVerdict
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.feedback_view import (
    MASKED_IDENTITY,
    MaskedFeedbackView,
    NormalFeedbackView,
    NullFeedbackView,
    PermutedFeedbackView,
    family_level_metrics,
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
    def test_full_store_yields_a_content_masked_board_context(self, runs):
        """CC3 C9 gate + P0-12: the static store is HONEST and FULL (64
        records per completed window), yet every board context assembled
        from the static view is shape-matched AND content-masked — the
        same feedback item count as the normal mode, every value a
        controlled NULL/MASK; no real feedback id / candidate id / value
        ever reaches a prompt."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_STATIC_LLM]
        assert len(list(ctl.store.ids())) == 64 * WINDOWS
        real_feedback_ids = set(ctl.store.ids())
        real_candidate_ids = {r.candidate_id for r in ctl.store.all()}
        assert real_feedback_ids and real_candidate_ids
        for window in range(1, WINDOWS):
            view = ctl._feedback_view(window)
            assert isinstance(view, MaskedFeedbackView)
            payload = view.to_prompt_payload()
            evidence = view.behavior_evidence()
            #: shape-matched: the SAME item count the normal mode presents
            assert len(payload) == len(evidence) == 64
            assert len(view.records()) == 64
            #: content-masked: every payload value is the controlled mask
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
            ctx = assemble_board_context(
                view, window=window - 1, mode=C.MODE_STATIC_LLM)
            assert ctx.feedback_view_label == "masked"
            assert len(ctx.behavior_evidence) == 64
            for e in ctx.behavior_evidence:
                assert e.candidate_id == MASKED_IDENTITY
                assert e.environment_family == MASKED_IDENTITY
                assert e.reference_gap == 0.0
                assert e.severity == "none"
                assert e.expected_observed_match \
                    == C.MATCH_DIRECTION_NEUTRAL
            # no real identity / content anywhere in the assembled context
            context = build_board_prompt_context(
                window=window, mode=C.MODE_STATIC_LLM, board_context=ctx,
                view=view,
                hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))
            assert len(context["feedback"]) == 64
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


# ---------------------- RE-IDENTIFICATION: no numeric side channel (CC4)
def _all_floats(obj):
    """Every float reachable in a JSON-serializable structure."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _all_floats(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _all_floats(value)


class TestShuffledNoReidentification:
    """CC4 C9 gate round two: the exact probe rates/gaps are deterministic
    per-candidate-hash fingerprints, so the shuffled view must publish only
    per-family window aggregates — consistent at the prompt layer and the
    evidence layer — and a store-joining adversary must not be able to
    recover the true candidate<->feedback pairing."""

    def test_published_numbers_are_public_family_aggregates_both_layers(
            self, runs):
        """Every numeric field at BOTH layers equals the adversarially
        recomputable family-level window aggregate; the family-grain
        predicted signature is dropped. The view publishes nothing finer
        than the public (window, family) partition."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for window in range(1, WINDOWS):
            honest = _records_of(ctl, window - 1)
            coarse = family_level_metrics(honest)
            view = ctl._feedback_view(window)
            evidence = {e.feedback_id: e for e in view.behavior_evidence()}
            for payload in view.to_prompt_payload():
                rec = ctl.store.get(view.resolve_citation(
                    payload["feedback_id"]))
                fam = coarse[rec.environment_family]
                assert payload["expected_signature"] == {}
                assert payload["student_success_rate"] == \
                    fam["student_success_rate"], window
                assert payload["reference_success_rate"] == \
                    fam["reference_success_rate"], window
                ev = evidence[payload["feedback_id"]]
                assert ev.student_success_rate == fam["student_success_rate"]
                assert ev.reference_success_rate == \
                    fam["reference_success_rate"]
                assert ev.return_shortfall == fam["return_shortfall"]
                assert ev.behavior_activation_gap == \
                    fam["behavior_activation_gap"]
                assert ev.front_progress_gap == fam["front_progress_gap"]
                assert ev.reference_gap == fam["reference_gap"]
                assert ev.severity == fam["severity"]
                assert ev.expected_observed_match == \
                    rec.expected_observed_match

    def test_store_joining_adversary_cannot_narrow_to_a_singleton(self,
                                                                    runs):
        """Uniqueness negative test: the adversary joins a presented item
        against the honest store on EVERY visible field (family, match,
        distinguished hypotheses, all published numbers). Because every
        published number is a family-level aggregate, the candidate set is
        exactly the item's public (family, match) class — which in this
        gate's windows always holds >= 2 records: the real pairing is never
        uniquely recoverable."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for window in range(1, WINDOWS):
            honest = _records_of(ctl, window - 1)
            classes = Counter((r.environment_family,
                               r.expected_observed_match) for r in honest)
            assert min(classes.values()) >= 2, \
                (window, min(classes, key=classes.get))
            coarse = family_level_metrics(honest)
            view = ctl._feedback_view(window)
            for payload in view.to_prompt_payload():
                real_id = view.resolve_citation(payload["feedback_id"])
                candidates = [
                    r for r in honest
                    if r.environment_family == payload["environment_family"]
                    and r.expected_observed_match
                    == payload["expected_observed_match"]
                    and tuple(r.distinguishes_hypothesis_ids)
                    == tuple(payload["distinguishes_hypothesis_ids"])
                    and r.window == payload["window"]
                    # the published numbers join too — they are family-level
                    # aggregates, so they narrow nothing below the class
                    and coarse[r.environment_family]["student_success_rate"]
                    == payload["student_success_rate"]
                    and coarse[r.environment_family][
                        "reference_success_rate"]
                    == payload["reference_success_rate"]]
                assert len(candidates) >= 2, (window, real_id)
                # the adversary's set always contains a record OTHER than
                # the true one — the pairing itself is unrecoverable
                assert any(r.feedback_id != real_id for r in candidates)

    def test_no_exact_per_record_metric_in_the_serialized_context(self,
                                                                  runs):
        """Full serialized-context scan: no exact per-record rate or gap
        (any value differing from every family's published aggregate) occurs
        anywhere in the shuffled prompt context — payload layer or evidence
        layer."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for window in range(1, WINDOWS):
            honest = _records_of(ctl, window - 1)
            coarse = family_level_metrics(honest)
            public_values = set()
            for fam in coarse.values():
                public_values.update(fam.values())
            view = ctl._feedback_view(window)
            ctx = assemble_board_context(
                view, window=window - 1, mode=C.MODE_SHUFFLED_FEEDBACK)
            context = build_board_prompt_context(
                window=window, mode=C.MODE_SHUFFLED_FEEDBACK,
                board_context=ctx, view=view,
                hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))
            published = set(_all_floats(context))
            # the schema clamp bounds are universal public constants (they
            # bound EVERY conforming value and appear in the context even
            # with zero records) — never per-record fingerprints
            clamp_constants = {0.0, 1.0}
            for rec in honest:
                ev = BehaviorFailureEvidence.from_record(rec)
                exact = {ev.student_success_rate,
                         ev.reference_success_rate,
                         ev.return_shortfall,
                         ev.behavior_activation_gap,
                         ev.front_progress_gap,
                         ev.reference_gap}
                for value in exact - public_values - clamp_constants:
                    assert value not in published, \
                        (window, rec.feedback_id, value)


# ------------------------------------- BYTE PARITY: frozen permutation (CC4)
def _prompt_context_of(ctl, window):
    """The EXACT full prompt context the window-k board assembles (same
    construction path as the controller's phase A + board). The static
    branch mirrors the controller: the static lifecycle is the frozen empty
    one — the retirement-state query reads the store and is refused."""
    view = ctl._feedback_view(window)
    if ctl.mode == C.MODE_STATIC_LLM:
        in_cooldown, blocked_retired = [], []
    else:
        in_cooldown, blocked_retired, _reopened = \
            ctl._retirement_state(window)
    ctx = assemble_board_context(
        view, window=max(0, window - 1), mode=ctl.mode,
        families_in_cooldown=in_cooldown,
        retired_families=blocked_retired)
    return build_board_prompt_context(
        window=window, mode=ctl.mode, board_context=ctx, view=view,
        hypotheses=normalize_hypothesis_inputs(ctl.ledger.all()))


class TestFrozenPromptByteParity:
    """CC4 C9 gate round two: the frozen deterministic permutation must be
    byte-reproducible at the FULL prompt-context level, not just the view
    internals."""

    def test_independent_runs_are_byte_identical_at_every_window(self):
        """Two fully independent shuffled runs assemble byte-identical
        prompt contexts (payload + board context + hypotheses, canonical
        JSON) at every window — the permutation has no runtime randomness."""
        first = FeedbackUEDController(C.MODE_SHUFFLED_FEEDBACK)
        first.run(max_windows=WINDOWS)
        second = FeedbackUEDController(C.MODE_SHUFFLED_FEEDBACK)
        second.run(max_windows=WINDOWS)
        for window in range(WINDOWS):
            a = json.dumps(_prompt_context_of(first, window),
                           sort_keys=True)
            b = json.dumps(_prompt_context_of(second, window),
                           sort_keys=True)
            assert a == b, window

    def test_independent_normal_runs_are_byte_identical_too(self):
        first = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        first.run(max_windows=WINDOWS)
        second = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        second.run(max_windows=WINDOWS)
        for window in range(WINDOWS):
            a = json.dumps(_prompt_context_of(first, window),
                           sort_keys=True)
            b = json.dumps(_prompt_context_of(second, window),
                           sort_keys=True)
            assert a == b, window

    def test_reassembly_inside_one_run_is_byte_identical(self, runs):
        """Re-assembling the same window's context twice (fresh view, fresh
        BoardContext) reproduces it byte-for-byte."""
        ctls, _sums = runs
        ctl = ctls[C.MODE_SHUFFLED_FEEDBACK]
        for window in range(WINDOWS):
            a = json.dumps(_prompt_context_of(ctl, window), sort_keys=True)
            b = json.dumps(_prompt_context_of(ctl, window), sort_keys=True)
            assert a == b, window


# ------------------------- STATIC INDEPENDENCE: no store read in phase A
class TestStaticContextIsFeedbackIndependent:
    """CC4 C9 gate round two (director REQUEST_CHANGES) + P0-12: the
    static (no-feedback control) mask's CONTENT is feedback-independent —
    every value is a fixed controlled NULL/MASK regardless of what the
    store holds; only the SHAPE (item count) follows the window k-1
    record set (exactly like the normal mode, keeping the modes
    compute-matched). The retirement lifecycle stays frozen-empty: a
    RETIRE decision cites feedback, and the masked view resolves no
    citation, so the registry can never become non-empty."""

    @staticmethod
    def _pollute(ctl):
        """Add foreign feedback records to the controller's store BEFORE
        the run — exactly the shape ``_reopen_eligible`` consumes: records
        distinguishing live ledger hypotheses, spread over every board
        window. If ANY static production path consulted the store, these
        records would be visible there."""
        for window in range(WINDOWS):
            for i in range(4):
                ctl.store.add(synthetic_feedback_record(
                    feedback_id=f"fb-junk-w{window}-{i}",
                    candidate=synthetic_candidate(
                        candidate_id=f"c-junk-w{window}-{i}",
                        family=C.ENVIRONMENT_FAMILIES[i % 7]),
                    plan_id="plan-junk", window=window,
                    student_success_rate=0.31,
                    expected_signature={"student_success_rate": 0.42},
                    distinguishes_hypothesis_ids=["hyp-00", "hyp-01"]))

    def test_static_mask_content_is_feedback_independent_under_foreign_store(
            self):
        """CC4 C9 gate + P0-12: the static (no-feedback control) mask's
        CONTENT is feedback-independent — every payload value is the fixed
        controlled NULL/MASK regardless of what the store holds; the SHAPE
        (item count) follows the window k-1 record set exactly like the
        normal mode's view, so the two modes stay compute-matched under
        any store. Junk feedback ids/candidates never reach a prompt."""
        clean = FeedbackUEDController(C.MODE_STATIC_LLM)
        polluted = FeedbackUEDController(C.MODE_STATIC_LLM)
        self._pollute(polluted)
        # the two stores genuinely differ in feedback records BEFORE the
        # runs start (and the difference survives them)
        junk_ids = {f"fb-junk-w{w}-{i}"
                    for w in range(WINDOWS) for i in range(4)}
        assert set(polluted.store.ids()) - set(clean.store.ids()) == junk_ids
        clean.run(max_windows=WINDOWS)
        polluted.run(max_windows=WINDOWS)
        assert set(polluted.store.ids()) - set(clean.store.ids()) == junk_ids

        def _freeze(value):
            if isinstance(value, dict):
                return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
            if isinstance(value, list):
                return tuple(_freeze(v) for v in value)
            return value

        def _strip(item):
            content = dict(item)
            content.pop("feedback_id")     #: positional masked id only
            return frozenset((k, _freeze(v)) for k, v in content.items())

        for window in range(WINDOWS):
            v_clean = clean._feedback_view(window)
            v_polluted = polluted._feedback_view(window)
            assert isinstance(v_clean, MaskedFeedbackView)
            assert isinstance(v_polluted, MaskedFeedbackView)
            # shape follows the record set: polluted carries 4 extra junk
            # records per window -> 4 extra masked items
            assert (len(v_polluted.records())
                    == len(v_clean.records()) + (4 if window >= 1 else 0))
            # window 0 has no frozen feedback under ANY mode -> empty mask
            if window == 0:
                assert v_clean.to_prompt_payload() == []
                assert v_polluted.to_prompt_payload() == []
                continue
            # content is invariant: every masked value is the FIXED mask
            assert {_strip(item)
                    for item in v_clean.to_prompt_payload()} == \
                {_strip(item) for item in v_polluted.to_prompt_payload()}
            expected = frozenset({
                ("axis_values", ()), ("candidate_id", MASKED_IDENTITY),
                ("distinguishes_hypothesis_ids", ()),
                ("environment_family", MASKED_IDENTITY),
                ("expected_observed_match", C.MATCH_DIRECTION_NEUTRAL),
                ("expected_signature", ()), ("held_constant_axes", ()),
                ("mutation_axes", ()), ("reference_success_rate", 0.0),
                ("student_success_rate", 0.0), ("window", window - 1)})
            for item in v_clean.to_prompt_payload():
                assert _strip(item) == expected

        # no junk id or candidate ever reaches any of the six prompts
        junk_candidates = {f"c-junk-w{w}-{i}"
                           for w in range(WINDOWS) for i in range(4)}
        for ctl in (clean, polluted):
            for e in ctl.envelopes:
                if e.role in C.BOARD_ROLES:
                    assert all(j not in e.prompt for j in junk_ids)
                    assert all(j not in e.prompt for j in junk_candidates)
                    #: windows >= 1 carry the shape-matched mask (window 0
                    #: has no frozen feedback under ANY mode)
                    if e.window >= 1:
                        assert MASKED_IDENTITY in e.prompt

    def test_retirement_state_is_frozen_empty_in_static(self):
        """P0-12: every mode runs the SAME lifecycle query; the static
        mode's masked view resolves no citation, so the retirement
        registry can never become non-empty and the query returns the
        frozen empty partition by construction."""
        ctl = FeedbackUEDController(C.MODE_STATIC_LLM)
        ctl.run(max_windows=2)
        assert ctl._retired_at == {}
        assert ctl._retirement_state(1) == ([], [], ())


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
        # static: shape-matched mask at every window — the SAME item count
        # the normal mode's view presents for the same store
        static = ctls[C.MODE_STATIC_LLM]
        normal = ctls[C.MODE_NORMAL_FEEDBACK]
        for window in range(WINDOWS):
            s_view = static._feedback_view(window)
            assert isinstance(s_view, MaskedFeedbackView)
            assert s_view.window_scope == max(0, window - 1)
            assert (len(s_view.records())
                    == len(normal._feedback_view(window).records()))
            assert (len(s_view.to_prompt_payload())
                    == len(s_view.records()))
            assert (len(s_view.behavior_evidence())
                    == len(s_view.records()))

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
