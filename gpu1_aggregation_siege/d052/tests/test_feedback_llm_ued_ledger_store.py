"""HypothesisLedger + SimulatorFeedbackStore invariants."""
import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.formal_isolation import FormalIsolationError
from d052.feedback_llm_ued.hypothesis_ledger import (
    HypothesisLedger,
    HypothesisRecord,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    MATCH_UNGRADED,
    SimulatorFeedbackRecord,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

FAM = C.ENVIRONMENT_FAMILIES[0]
FAM2 = C.ENVIRONMENT_FAMILIES[1]


def _hyp(hid="hyp-01", family=FAM, **over):
    base = dict(hypothesis_id=hid, source_window=0,
                target_behavior="probe_rest_need",
                predicted_signature={"student_success_rate": 0.5},
                environment_family=family, confidence=0.5)
    base.update(over)
    return HypothesisRecord(**base)


class TestHypothesisLedger:
    def test_register_and_defaults(self):
        ledger = HypothesisLedger()
        rec = ledger.register(_hyp())
        assert rec.status == C.HYPOTHESIS_PENDING
        assert len(rec.record_hash) == 64
        assert ledger.ids() == ["hyp-01"]
        assert ledger.by_status()[C.HYPOTHESIS_PENDING] == ["hyp-01"]

    def test_duplicate_id_rejected(self):
        ledger = HypothesisLedger()
        ledger.register(_hyp())
        with pytest.raises(ValueError, match="DUPLICATE_HYPOTHESIS_ID"):
            ledger.register(_hyp())

    def test_illegal_status_and_family_rejected(self):
        with pytest.raises(ValueError, match="ILLEGAL_HYPOTHESIS_STATUS"):
            _hyp(status="PROVEN")
        with pytest.raises(ValueError, match="UNKNOWN_ENVIRONMENT_FAMILY"):
            _hyp(family="not_a_family")

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError, match="UNKNOWN_HYPOTHESIS_ID"):
            HypothesisLedger().get("nope")

    def test_bind_feedback_rehashes_and_buckets(self):
        ledger = HypothesisLedger()
        rec = ledger.register(_hyp())
        h0 = rec.record_hash
        ledger.bind_feedback("hyp-01", "fb-1", agrees=True)
        ledger.bind_feedback("hyp-01", "fb-2", agrees=False)
        ledger.bind_feedback("hyp-01", "fb-1", agrees=True)   # idempotent
        rec = ledger.get("hyp-01")
        assert rec.supporting_feedback_ids == ["fb-1"]
        assert rec.contradicting_feedback_ids == ["fb-2"]
        assert rec.record_hash != h0
        assert rec.record_hash == rec.rehash()

    def test_apply_verdict_lifecycle_and_history_chain(self):
        ledger = HypothesisLedger()
        ledger.register(_hyp())
        ledger.bind_feedback("hyp-01", "fb-1", agrees=True)
        h0 = ledger.get("hyp-01").record_hash      # post-binding hash
        ledger.apply_verdict("hyp-01", status=C.HYPOTHESIS_SUPPORTED,
                             window=1, reason="probe agreed",
                             feedback_ids=["fb-1"], confidence=0.6)
        rec = ledger.get("hyp-01")
        assert rec.status == C.HYPOTHESIS_SUPPORTED
        assert rec.confidence == 0.6
        assert len(rec.revision_history) == 1
        entry = rec.revision_history[0]
        assert entry["previous_status"] == C.HYPOTHESIS_PENDING
        assert entry["new_status"] == C.HYPOTHESIS_SUPPORTED
        assert entry["window"] == 1
        assert entry["feedback_ids"] == ["fb-1"]
        assert entry["previous_record_hash"] == h0
        # chain continues on the next verdict
        ledger.apply_verdict("hyp-01", status=C.HYPOTHESIS_REFUTED,
                             window=2, reason="later probe contradicted")
        rec = ledger.get("hyp-01")
        assert rec.revision_history[1]["previous_status"] == \
            C.HYPOTHESIS_SUPPORTED
        assert ledger.by_status()[C.HYPOTHESIS_REFUTED] == ["hyp-01"]

    def test_apply_verdict_illegal_status_and_confidence(self):
        ledger = HypothesisLedger()
        ledger.register(_hyp())
        with pytest.raises(ValueError, match="ILLEGAL_HYPOTHESIS_STATUS"):
            ledger.apply_verdict("hyp-01", status="CERTAIN", window=1,
                                 reason="r")
        with pytest.raises(ValueError, match="CONFIDENCE_OUT_OF_RANGE"):
            ledger.apply_verdict("hyp-01", status=C.HYPOTHESIS_SUPPORTED,
                                 window=1, reason="r", confidence=1.5)

    def test_mark_stale(self):
        ledger = HypothesisLedger()
        ledger.register(_hyp())
        ledger.mark_stale("hyp-01", window=3, reason="no probe this window")
        assert ledger.get("hyp-01").status == C.HYPOTHESIS_STALE

    def test_dump_round_trip(self):
        ledger = HypothesisLedger()
        ledger.register(_hyp())
        dump = ledger.dump()
        assert dump[0]["hypothesis_id"] == "hyp-01"
        assert dump[0]["record_hash"] == ledger.get("hyp-01").record_hash


class TestSimulatorFeedbackStore:
    def _record(self, **over):
        cand = synthetic_candidate(candidate_id="cand-x", family=FAM)
        kwargs = dict(feedback_id="fb-x", candidate=cand, plan_id="plan-a",
                      window=0, student_success_rate=0.5,
                      expected_signature={"student_success_rate": 0.5},
                      distinguishes_hypothesis_ids=["hyp-01"])
        kwargs.update(over)
        return synthetic_feedback_record(**kwargs)

    def test_add_and_hash(self):
        store = SimulatorFeedbackStore()
        rec = store.add(self._record())
        assert len(rec.record_hash) == 64
        assert rec.expected_observed_match == MATCH_UNGRADED
        assert store.ids() == ["fb-x"]
        assert store.for_candidate("cand-x")[0].feedback_id == "fb-x"
        assert store.for_plan("plan-a")[0].feedback_id == "fb-x"
        assert store.for_window(0)[0].feedback_id == "fb-x"

    def test_duplicate_feedback_id_rejected(self):
        store = SimulatorFeedbackStore()
        store.add(self._record())
        with pytest.raises(ValueError, match="DUPLICATE_FEEDBACK_ID"):
            store.add(self._record())

    def test_forbidden_reference_field_rejected(self):
        with pytest.raises(ValueError, match="REFERENCE_FIELD_FORBIDDEN"):
            SimulatorFeedbackRecord(
                feedback_id="fb-bad", candidate_id="c",
                candidate_hash="a" * 64, source_plan_id="p", window=0,
                environment_family=FAM,
                reference_stats={"hidden_score": 0.5})

    def test_reference_carrier_in_provenance_rejected(self):
        with pytest.raises(FormalIsolationError,
                           match="REFERENCE_CARRIER_FORBIDDEN"):
            SimulatorFeedbackRecord(
                feedback_id="fb-bad", candidate_id="c",
                candidate_hash="a" * 64, source_plan_id="p", window=0,
                environment_family=FAM,
                provenance={"action_sequence": [1, 2, 3]})

    def test_formal_source_rejected_at_construction(self):
        with pytest.raises(FormalIsolationError,
                           match="FORMAL_SOURCE_FORBIDDEN"):
            SimulatorFeedbackRecord(
                feedback_id="fb-bad", candidate_id="c",
                candidate_hash="a" * 64, source_plan_id="p", window=0,
                environment_family=FAM,
                provenance={"source": C.SOURCE_FORMAL_FRONT})

    def test_candidate_hash_must_be_sha256(self):
        cand = synthetic_candidate(candidate_id="cand-x", family=FAM)
        with pytest.raises(ValueError, match="CANDIDATE_HASH_NOT_SHA256"):
            SimulatorFeedbackRecord(
                feedback_id="fb-bad", candidate_id=cand.candidate_id,
                candidate_hash="not-a-hash", source_plan_id="p", window=0,
                environment_family=FAM)

    def test_illegal_match_state_rejected(self):
        cand = synthetic_candidate(candidate_id="cand-x", family=FAM)
        with pytest.raises(ValueError, match="ILLEGAL_MATCH_STATE"):
            SimulatorFeedbackRecord(
                feedback_id="fb-bad", candidate_id=cand.candidate_id,
                candidate_hash=cand.candidate_hash, source_plan_id="p",
                window=0, environment_family=FAM,
                expected_observed_match="kinda_agree")

    def test_bind_match_rehashes_and_filters(self):
        store = SimulatorFeedbackStore()
        rec = store.add(self._record())
        h0 = rec.record_hash
        with pytest.raises(ValueError, match="ILLEGAL_MATCH_DIRECTION"):
            store.bind_match("fb-x", direction="mostly_agree")
        store.bind_match("fb-x", direction=C.MATCH_DIRECTION_AGREE,
                         detail={"overall": C.MATCH_DIRECTION_AGREE})
        rec = store.get("fb-x")
        assert rec.expected_observed_match == C.MATCH_DIRECTION_AGREE
        assert rec.record_hash != h0
        assert rec.record_hash == rec.rehash()
        assert [r.feedback_id for r in
                store.graded(C.MATCH_DIRECTION_AGREE)] == ["fb-x"]
        assert store.graded(C.MATCH_DIRECTION_OPPOSITE) == []
        with pytest.raises(KeyError, match="UNKNOWN_FEEDBACK_ID"):
            store.bind_match("nope", direction=C.MATCH_DIRECTION_AGREE)

    def test_simulator_transitions_sum(self):
        rec = self._record()
        # synthetic records carry only stage2 metrics
        assert rec.simulator_transitions == \
            rec.stage2_metrics.simulator_transitions
