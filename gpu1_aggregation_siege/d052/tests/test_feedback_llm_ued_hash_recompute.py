"""C14 / P1-5: externally provided content hashes are RECOMPUTED and
compared verbatim — mismatch fails closed (CONTENT_HASH_MISMATCH).

Covers all seven hash-bearing object types of the loop:

    CandidateEnvironment.candidate_hash   HypothesisRecord.record_hash
    SimulatorFeedbackRecord.record_hash   CurriculumPlan.plan_hash
    PlanRevisionRecord.record_hash        FeedbackRoleEnvelope (request /
    response / prompt_sha256)             AxisDirective.directive_hash

For each type: an empty hash is computed; a carried hash equal to the
recomputation is accepted (self-dumps round-trip — the persistence
prerequisite); a carried hash that differs is rejected fail-closed. The
envelope additionally stores the prompt and recomputes ALL three hashes from
stored content, so a substituted prompt or response cannot parse.
"""
import pytest

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_INCREASE,
    LEVEL_NONE,
    ROLE_TREATMENT,
    AxisDirective,
)
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    CurriculumPlan,
    FeedbackRoleEnvelope,
)
from d052.feedback_llm_ued.hypothesis_ledger import HypothesisRecord
from d052.feedback_llm_ued.plan_revision import PlanRevisionRecord
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)

#: a syntactically valid sha256 hex that binds NO real content of any test
#: object — the canonical "externally provided but wrong" hash
WRONG_HASH = canonical_sha256({"tampered": True})

MISMATCH = "CONTENT_HASH_MISMATCH"


def _candidate(**over):
    base = dict(candidate_id="cand-hr-00",
                environment_family=C.ENVIRONMENT_FAMILIES[0],
                variant_id="var-hr-00", variant_kind="test")
    base.update(over)
    return CandidateEnvironment(**base)


def _hypothesis(**over):
    base = dict(hypothesis_id="hyp-hr-00", source_window=0,
                target_behavior="hash_recompute_probe",
                environment_family=C.ENVIRONMENT_FAMILIES[0],
                confidence=0.5)
    base.update(over)
    return HypothesisRecord(**base)


def _feedback(**over):
    base = dict(feedback_id="fb-hr-00", candidate_id="cand-hr-00",
                candidate_hash="a" * 64, source_plan_id="plan-hr",
                window=0, environment_family=C.ENVIRONMENT_FAMILIES[0])
    base.update(over)
    return SimulatorFeedbackRecord(**base)


def _plan(**over):
    base = dict(plan_id="plan-hr-00", window=0, mode=C.MODE_NORMAL_FEEDBACK)
    base.update(over)
    return CurriculumPlan(**base)


def _revision(**over):
    base = dict(revision_id="rev-hr-00", window=0,
                mode=C.MODE_NORMAL_FEEDBACK, new_plan_id="plan-hr-00")
    base.update(over)
    return PlanRevisionRecord(**base)


def _directive(**over):
    fam = C.ENVIRONMENT_FAMILIES[0]
    base = dict(directive_id="dir-hr-00", source_window=0,
                environment_family=fam, axis="threat_distance_grading",
                old_level=LEVEL_NONE, new_level="medium",
                direction=DIRECTION_INCREASE,
                experiment_control_role=ROLE_TREATMENT,
                expected_next_signature={"student_success_rate": 0.5})
    base.update(over)
    return AxisDirective(**base)


def _envelope(**over):
    base = dict(role=C.ROLE_STUDENT_MODELER,
                prompt_version="feedback_llm_ued.roles.v1.student_modeler",
                backend_id=C.MOCK_BACKEND_ID, model_id=C.MOCK_MODEL_ID,
                window=0, sequence=0, prompt="the prompt",
                raw_response='{"ok": true}', parsed_dump={"ok": True})
    base.update(over)
    return FeedbackRoleEnvelope.make(**base)


class TestEmptyHashIsComputed:
    def test_all_seven_types_stamp_their_hash(self):
        assert _candidate().candidate_hash
        assert _hypothesis().record_hash
        assert _feedback().record_hash
        assert _plan().plan_hash
        assert _revision().record_hash
        assert _directive().directive_hash
        env = _envelope()
        assert env.prompt_sha256 == text_sha256("the prompt")
        assert env.request_hash and env.response_hash


class TestCarriedHashMustMatchRecomputation:
    """The fail-closed negative matrix: a carried hash that does not
    reproduce byte-for-byte from the content is a hard error."""

    def test_candidate_hash_mismatch(self):
        with pytest.raises(ValueError, match=MISMATCH):
            _candidate(candidate_hash=WRONG_HASH)

    def test_hypothesis_record_hash_mismatch(self):
        with pytest.raises(ValueError, match=MISMATCH):
            _hypothesis(record_hash=WRONG_HASH)

    def test_feedback_record_hash_mismatch(self):
        with pytest.raises(ValueError, match=MISMATCH):
            _feedback(record_hash=WRONG_HASH)

    def test_plan_hash_mismatch(self):
        with pytest.raises(ValueError, match=MISMATCH):
            _plan(plan_hash=WRONG_HASH)

    def test_revision_record_hash_mismatch(self):
        with pytest.raises(ValueError, match=MISMATCH):
            _revision(record_hash=WRONG_HASH)

    def test_directive_hash_mismatch(self):
        with pytest.raises(ValueError, match=MISMATCH):
            _directive(directive_hash=WRONG_HASH)

    def test_envelope_prompt_sha256_mismatch(self):
        env = _envelope()
        dump = env.model_dump()
        dump["prompt_sha256"] = WRONG_HASH
        with pytest.raises(ValueError, match=MISMATCH):
            FeedbackRoleEnvelope(**dump)

    def test_envelope_request_hash_mismatch(self):
        env = _envelope()
        dump = env.model_dump()
        dump["request_hash"] = WRONG_HASH
        with pytest.raises(ValueError, match=MISMATCH):
            FeedbackRoleEnvelope(**dump)

    def test_envelope_response_hash_mismatch(self):
        env = _envelope()
        dump = env.model_dump()
        dump["response_hash"] = WRONG_HASH
        with pytest.raises(ValueError, match=MISMATCH):
            FeedbackRoleEnvelope(**dump)

    def test_envelope_substituted_prompt_cannot_parse(self):
        """A prompt swapped under a kept prompt_sha256/request_hash fails;
        swapping the hashes too still fails on the recomputation of the
        OTHER hash — one tampered field always trips a check."""
        env = _envelope()
        dump = env.model_dump()
        dump["prompt"] = "a substituted prompt"
        with pytest.raises(ValueError, match=MISMATCH):
            FeedbackRoleEnvelope(**dump)
        # fix prompt_sha256 only -> request_hash now mismatches
        dump["prompt_sha256"] = text_sha256(dump["prompt"])
        with pytest.raises(ValueError, match=MISMATCH):
            FeedbackRoleEnvelope(**dump)

    def test_envelope_substituted_response_cannot_parse(self):
        env = _envelope()
        dump = env.model_dump()
        dump["raw_response"] = '{"forged": true}'
        with pytest.raises(ValueError, match=MISMATCH):
            FeedbackRoleEnvelope(**dump)


class TestCorrectCarriedHashIsAccepted:
    """A carried hash EQUAL to the recomputation parses — this is exactly
    the round-trip persistence/replay relies on (load -> validate)."""

    def test_self_dump_round_trips_all_seven_types(self):
        for obj in (_candidate(), _hypothesis(), _feedback(), _plan(),
                    _revision(), _directive(), _envelope()):
            clone = type(obj)(**obj.model_dump())
            assert clone.model_dump() == obj.model_dump()

    def test_explicit_correct_hash_parses(self):
        cand = _candidate()
        again = _candidate(candidate_hash=cand.candidate_hash)
        assert again.candidate_hash == cand.candidate_hash
        hyp = _hypothesis()
        assert _hypothesis(record_hash=hyp.record_hash).record_hash == \
            hyp.record_hash
        d = _directive()
        assert _directive(directive_hash=d.directive_hash).directive_hash == \
            d.directive_hash


class TestPostConstructionTamperIsDetectable:
    """Audit path: mutating content behind the validators' back leaves the
    carried hash stale — ``rehash()`` exposes it (this is what a persistence
    hash-chain verification recomputes)."""

    def test_hypothesis_tamper_detected_by_rehash(self):
        rec = _hypothesis()
        object.__setattr__(rec, "confidence", 0.99)
        assert rec.rehash() != rec.record_hash

    def test_feedback_tamper_detected_by_rehash(self):
        rec = _feedback()
        object.__setattr__(rec, "window", 7)
        assert rec.rehash() != rec.record_hash

    def test_revision_tamper_detected_by_rehash(self):
        rec = _revision()
        object.__setattr__(rec, "new_plan_id", "plan-forged")
        assert rec.rehash() != rec.record_hash

    def test_directive_tamper_detected_by_rehash(self):
        rec = _directive()
        object.__setattr__(rec, "new_level", "high")
        assert rec.rehash() != rec.directive_hash


class TestEnvelopeReplayKeyConsistency:
    """prompt_sha256 must be the SAME key the replay corpus uses, so an
    envelope's prompt identity and the ReplayBackend lookup agree."""

    def test_prompt_sha256_equals_replay_key_definition(self):
        env = _envelope(prompt="some role prompt")
        assert env.prompt_sha256 == text_sha256("some role prompt")

    def test_request_hash_binds_role_and_prompt_version(self):
        env = _envelope()
        expected = canonical_sha256(
            {"role": env.role, "prompt_version": env.prompt_version,
             "prompt": env.prompt})
        assert env.request_hash == expected
