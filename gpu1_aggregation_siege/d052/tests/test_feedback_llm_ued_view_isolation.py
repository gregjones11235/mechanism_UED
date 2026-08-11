"""C9 three-mode comparison isolation — the FeedbackView layer.

* ``NullFeedbackView`` is STRUCTURALLY empty: the type cannot even be handed
  a store (constructor takes no arguments), so the static mode's board can
  never reach feedback by construction — not by prompt discipline.
* ``PermutedFeedbackView`` is a FROZEN, recomputable permutation of the
  frozen records, presented under anonymized ids with every identity side
  channel removed or consistently anonymized: candidate id / mutation axes /
  axis values / held-constant axes masked, the family-grain predicted
  signature dropped, and the exact probe rates / evidence gaps (which are
  deterministic per-candidate-hash fingerprints) published ONLY as per-
  family window aggregates, identical at the prompt layer and the evidence
  layer. Two views built from the same inputs are bit-identical, the real
  candidate<->feedback pairing never appears in any board-visible payload,
  and de-anonymization is only possible through ``resolve_citation`` —
  which fails closed on anything the view did not present (including the
  real store ids themselves).
"""
import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_view import (
    MASKED_IDENTITY,
    VIEW_LABEL_NULL,
    VIEW_LABEL_PERMUTED,
    NormalFeedbackView,
    NullFeedbackView,
    PermutedFeedbackView,
    family_level_metrics,
    record_payload,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

FAM = C.ENVIRONMENT_FAMILIES[0]
N_RECORDS = 12


def _record(i):
    cand = synthetic_candidate(candidate_id=f"cand-iso-{i:02d}", family=FAM)
    return synthetic_feedback_record(
        feedback_id=f"fb-iso-{i:02d}", candidate=cand, plan_id="plan-iso",
        window=0, student_success_rate=0.3 + 0.01 * i,
        expected_signature={"student_success_rate": 0.47},
        distinguishes_hypothesis_ids=["hyp-00"])


def _records():
    return [_record(i) for i in range(N_RECORDS)]


def _view(records, *, board_window=1, scope=0,
          seed=C.SEED_SCHEDULE_HASH):
    return PermutedFeedbackView(
        records, window_scope=scope, board_window=board_window,
        mode=C.MODE_SHUFFLED_FEEDBACK, seed_schedule_hash=seed)


def _anon(board_window, slot):
    return f"anon-w{board_window:02d}-{slot:03d}"


# ------------------------------------------------- NullFeedbackView (static)
class TestNullFeedbackViewStructural:
    def test_constructor_takes_no_arguments(self):
        """Type-level isolation: the view defines no constructor state — it
        cannot even be HANDED a store — and rejects any argument."""
        assert NullFeedbackView.__init__ is object.__init__
        with pytest.raises(TypeError):
            NullFeedbackView(object())
        with pytest.raises(TypeError):
            NullFeedbackView(records=[])

    def test_payload_is_zero_and_unresolvable(self):
        view = NullFeedbackView()
        assert view.label == VIEW_LABEL_NULL == "null"
        assert view.window_scope == -1
        assert view.records() == []
        assert view.to_prompt_payload() == []
        with pytest.raises(ValueError, match="NULL_VIEW_HAS_NO_FEEDBACK"):
            view.resolve_citation("fb-w00-anything")

    def test_normal_view_resolution_identity_and_fail_closed(self):
        recs = _records()
        view = NormalFeedbackView(recs, window_scope=0)
        for r in recs:
            assert view.resolve_citation(r.feedback_id) == r.feedback_id
        with pytest.raises(ValueError, match="UNKNOWN_FEEDBACK_CITATION"):
            view.resolve_citation("fb-ghost")


# ------------------------------------- PermutedFeedbackView (shuffled, C9)
class TestPermutedFeedbackViewFrozen:
    def test_recomputable_bit_identical(self):
        """The view is a pure function of its inputs: reconstructing it from
        the same records must reproduce the identical payload + mapping."""
        v1 = _view(_records())
        v2 = _view(_records())
        assert v1.label == v2.label
        assert v1.to_prompt_payload() == v2.to_prompt_payload()
        for slot in range(N_RECORDS):
            anon = _anon(1, slot)
            assert v1.resolve_citation(anon) == v2.resolve_citation(anon)

    def test_input_order_does_not_matter(self):
        """Records are canonically sorted by feedback_id first, so feeding
        them in any order reproduces the same frozen presentation."""
        v1 = _view(_records())
        v2 = _view(list(reversed(_records())))
        assert v1.label == v2.label
        assert v1.to_prompt_payload() == v2.to_prompt_payload()

    def test_permutation_is_frozen_per_window_and_seed(self):
        """Derived ONLY from (mode, board window, scope, seed schedule):
        different board windows or a different seed schedule freeze a
        different permutation; identical inputs freeze an identical one."""
        base = _view(_records())
        other_window = _view(_records(), board_window=2)
        other_seed = _view(_records(), seed="a" * 64)
        assert base.permutation_seed != other_window.permutation_seed
        assert base.permutation_seed != other_seed.permutation_seed
        assert base.label != other_window.label
        assert base.label != other_seed.label
        assert base.label == _view(_records()).label

    def test_label_is_permuted_prefix_plus_seed_prefix(self):
        view = _view(_records())
        prefix, seed16 = view.label.split(":")
        assert prefix == VIEW_LABEL_PERMUTED == "permuted"
        assert seed16 == view.permutation_seed[:16]
        assert len(seed16) == 16

    def test_presentation_is_a_permutation_of_the_input_records(self):
        """Anonymized slots de-anonymize to EXACTLY the input record set —
        nothing added, nothing dropped. What moves verbatim with the record
        is loop-essential COARSE content (window, family, distinguished
        hypotheses, match state); every identity side channel is removed or
        consistently anonymized (CC4 C9 gate round two): ids/axes masked,
        family-grain predicted signature dropped, exact per-candidate rates
        replaced by the public family-level window aggregates."""
        recs = _records()
        view = _view(recs)
        by_id = {r.feedback_id: r for r in recs}
        payloads = view.to_prompt_payload()
        assert len(payloads) == N_RECORDS
        resolved = [view.resolve_citation(p["feedback_id"]) for p in payloads]
        assert sorted(resolved) == sorted(by_id)
        assert len(set(resolved)) == N_RECORDS
        coarse = family_level_metrics(recs)
        for payload in payloads:
            rec = by_id[view.resolve_citation(payload["feedback_id"])]
            expected = record_payload(rec)
            for key in ("window", "environment_family",
                        "distinguishes_hypothesis_ids",
                        "expected_observed_match"):
                assert payload[key] == expected[key], key
            fam = coarse[rec.environment_family]
            assert payload["student_success_rate"] == \
                fam["student_success_rate"]
            assert payload["reference_success_rate"] == \
                fam["reference_success_rate"]
            assert payload["expected_signature"] == {}
            assert payload["candidate_id"] == MASKED_IDENTITY
            assert payload["mutation_axes"] == []
            assert payload["axis_values"] == {}
            assert payload["held_constant_axes"] == {}

    def test_permutation_is_not_the_identity(self):
        """The frozen permutation genuinely re-binds slots to records (this
        fixture: zero slots keep their sorted-order record)."""
        sorted_ids = sorted(r.feedback_id for r in _records())
        view = _view(_records())
        displaced = sum(
            1 for slot in range(N_RECORDS)
            if view.resolve_citation(_anon(1, slot)) != sorted_ids[slot])
        assert displaced >= 1
        assert displaced == N_RECORDS       # full derangement for this set

    def test_empty_and_singleton_views(self):
        empty = _view([])
        assert empty.to_prompt_payload() == []
        assert empty.records() == []
        with pytest.raises(ValueError, match="UNKNOWN_ANONYMIZED_CITATION"):
            empty.resolve_citation(_anon(1, 0))
        single = _view([_record(0)])
        payload = single.to_prompt_payload()
        assert len(payload) == 1
        assert payload[0]["feedback_id"] == _anon(1, 0)
        assert single.resolve_citation(_anon(1, 0)) == "fb-iso-00"


class TestIdentitySideChannelBlocked:
    """Negative: the board-visible payload must make the real
    candidate<->feedback pairing UNRECOVERABLE."""

    def test_no_real_identity_anywhere_in_the_payload(self):
        recs = _records()
        serialized = json.dumps(_view(recs).to_prompt_payload(),
                                sort_keys=True)
        for rec in recs:
            assert rec.feedback_id not in serialized
            assert rec.candidate_id not in serialized
            assert rec.candidate_hash not in serialized
        for payload in _view(recs).to_prompt_payload():
            assert payload["feedback_id"].startswith("anon-w01-")
            assert payload["candidate_id"] == MASKED_IDENTITY

    def test_real_store_ids_are_not_valid_citations(self):
        """Citing a REAL feedback id at the shuffled view fails closed — the
        only legal citations are the anonymized ids the view presented."""
        recs = _records()
        view = _view(recs)
        for rec in recs:
            with pytest.raises(ValueError,
                               match="UNKNOWN_ANONYMIZED_CITATION"):
                view.resolve_citation(rec.feedback_id)

    def test_unknown_anonymized_citation_refused(self):
        view = _view(_records())
        with pytest.raises(ValueError, match="UNKNOWN_ANONYMIZED_CITATION"):
            view.resolve_citation(_anon(1, N_RECORDS))      # slot out of range
        with pytest.raises(ValueError, match="UNKNOWN_ANONYMIZED_CITATION"):
            view.resolve_citation("anon-w99-000")           # foreign window


class TestPermutedViewFailClosed:
    def test_illegal_window_scope(self):
        with pytest.raises(ValueError, match="ILLEGAL_VIEW_WINDOW_SCOPE"):
            _view(_records(), scope=-1)

    def test_illegal_board_window(self):
        with pytest.raises(ValueError, match="ILLEGAL_VIEW_BOARD_WINDOW"):
            _view(_records(), board_window=-1)

    def test_refuses_non_shuffled_mode(self):
        with pytest.raises(ValueError,
                           match="PERMUTED_VIEW_REQUIRES_SHUFFLED_MODE"):
            PermutedFeedbackView(_records(), window_scope=0, board_window=1,
                                 mode=C.MODE_NORMAL_FEEDBACK,
                                 seed_schedule_hash=C.SEED_SCHEDULE_HASH)

    def test_refuses_illegal_seed_schedule_hash(self):
        with pytest.raises(ValueError, match="ILLEGAL_SEED_SCHEDULE_HASH"):
            _view(_records(), seed="not-a-sha256")

    def test_seed_schedule_hash_constant_is_sha256(self):
        assert len(C.SEED_SCHEDULE_HASH) == 64
        int(C.SEED_SCHEDULE_HASH, 16)                       # hex
