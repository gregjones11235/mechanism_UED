"""C2: CC4 StudentAdapter thin binding + feedback identity fields.

Direction two ONLY consumes the CC4 shared StudentAdapter; the adapter is
absent from this worktree, so every test here asserts the fail-closed or
honestly-labelled scaffold behaviour — never a real checkpoint load, never a
training update.
"""
from dataclasses import dataclass, replace

import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_MOCK_DRY_RUN,
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)
from d052.feedback_llm_ued.student_binding import (
    WEIGHTS_NOT_LOADED_LOCAL,
    WEIGHTS_REAL_CHECKPOINT,
    StudentBindingBlocked,
    StudentTrainingSeam,
    local_symbolic_binding,
    resolve_student_binding,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

GOOD_HASH = "a" * 64


@dataclass
class FakeContract:
    """Minimal stand-in for the CC4 shared StudentInitContract shape."""

    candidate_id: str = C.STRONG_STUDENT_CANDIDATE_ID
    architecture_family: str = "RMT16"
    memory_family: str = "RMT16_ORIGINAL"
    carry_mode: str = "PERSISTENT"
    parameter_tree_hash: str = GOOD_HASH
    checkpoint_global_step: int = 98304


class TestResolveStudentBinding:
    def test_missing_contract_fails_closed(self):
        with pytest.raises(StudentBindingBlocked,
                           match="STUDENT_INIT_CONTRACT_MISSING"):
            resolve_student_binding(None)

    def test_wrong_candidate_id_fails_closed(self):
        with pytest.raises(StudentBindingBlocked,
                           match="STUDENT_IDENTITY_MISMATCH"):
            resolve_student_binding(FakeContract(candidate_id="SOME_OTHER"))

    def test_bad_parameter_hash_fails_closed(self):
        with pytest.raises(StudentBindingBlocked,
                           match="STUDENT_IDENTITY_INCOMPLETE"):
            resolve_student_binding(FakeContract(parameter_tree_hash="xyz"))

    def test_bad_checkpoint_step_fails_closed(self):
        with pytest.raises(StudentBindingBlocked,
                           match="STUDENT_IDENTITY_INCOMPLETE"):
            resolve_student_binding(FakeContract(checkpoint_global_step=-1))

    def test_valid_contract_resolves_real_checkpoint_identity(self):
        identity = resolve_student_binding(FakeContract())
        assert identity.candidate_id == C.STRONG_STUDENT_CANDIDATE_ID
        assert identity.weights_status == WEIGHTS_REAL_CHECKPOINT
        assert identity.provenance_label == "CC4_SHARED_STUDENT_INIT_CONTRACT"
        assert len(identity.identity_hash) == 64
        # deterministic: same contract -> same identity hash
        assert identity.identity_hash == \
            resolve_student_binding(FakeContract()).identity_hash
        # identity hash is sensitive to every identity field
        altered = resolve_student_binding(
            FakeContract(checkpoint_global_step=1))
        assert altered.identity_hash != identity.identity_hash


class TestLocalSymbolicBinding:
    def test_honest_scaffold_posture(self):
        binding = local_symbolic_binding()
        assert binding.candidate_id == "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
        assert binding.weights_status == WEIGHTS_NOT_LOADED_LOCAL
        assert binding.provenance_label == C.ENGINEERING_SCAFFOLD
        assert binding.checkpoint_global_step == 0
        assert len(binding.parameter_tree_hash) == 64
        assert len(binding.identity_hash) == 64
        # determinism + honesty: recomputable, and NOT equal to a resolved
        # real-checkpoint identity
        assert binding.identity_hash == local_symbolic_binding().identity_hash
        assert binding.identity_hash != \
            resolve_student_binding(FakeContract()).identity_hash
        assert C.REAL_CHECKPOINT_LOADED is False


class TestStudentTrainingSeam:
    def _seam(self):
        gate = FeedbackLaunchGate(EXECUTION_MODE_MOCK_DRY_RUN)
        return StudentTrainingSeam(gate, local_symbolic_binding())

    def test_unauthorized_step_is_skipped_with_zero_transitions(self):
        rec = self._seam().execute_training_step(window=3)
        assert rec.status == "SKIPPED_UNAUTHORIZED"
        assert rec.student_training_transitions == 0
        assert "TRAINING_NOT_ALLOWED" in rec.reason
        assert "window=3" in rec.reason

    def test_even_authorized_still_needs_cc4_evidence(self, monkeypatch):
        # gate would allow training (REAL mode + flag flipped), but the seam
        # still demands real CC4 adapter evidence before any update may exist
        monkeypatch.setattr(C, "TRAINING_AUTHORIZED", True)
        gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
        seam = StudentTrainingSeam(gate, local_symbolic_binding())
        with pytest.raises(StudentBindingBlocked,
                           match="REAL_TRAINING_SEAM_NOT_IMPLEMENTED"):
            seam.execute_training_step(window=0)
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False


class TestFeedbackRecordIdentityFields:
    def _make(self, **overrides):
        cand = synthetic_candidate(candidate_id="c-bind",
                                   family=C.ENVIRONMENT_FAMILIES[0])
        base = dict(feedback_id="fb-bind", candidate_id=cand.candidate_id,
                    candidate_hash=cand.candidate_hash,
                    source_plan_id="plan-bind", window=0,
                    environment_family=cand.environment_family)
        base.update(overrides)
        return SimulatorFeedbackRecord(**base)

    def test_defaults_are_empty_and_legal(self):
        rec = self._make()
        assert rec.student_identity_hash == ""
        assert rec.student_roles == ()
        assert rec.student_checkpoint_step == 0
        assert rec.memory_compatibility_status == \
            C.MEMORY_COMPATIBILITY_NOT_APPLICABLE
        assert len(rec.record_hash) == 64

    def test_identity_hash_must_be_sha256(self):
        with pytest.raises(ValidationError,
                           match="STUDENT_IDENTITY_HASH_NOT_SHA256"):
            self._make(student_identity_hash="not-a-hash")

    def test_unknown_student_role_rejected(self):
        with pytest.raises(ValidationError, match="UNKNOWN_STUDENT_ROLE"):
            self._make(student_roles=("teleport",))

    def test_legal_binding_fields_accepted_and_hashed_in(self):
        binding = local_symbolic_binding()
        rec = self._make(student_identity_hash=binding.identity_hash,
                         student_roles=(C.STUDENT_ROLE_SEARCH,))
        assert rec.student_identity_hash == binding.identity_hash
        # the identity participates in the record hash
        bare = self._make()
        assert rec.record_hash != bare.record_hash

    def test_synthetic_factory_passes_identity_through(self):
        binding = local_symbolic_binding()
        cand = synthetic_candidate(candidate_id="c-syn",
                                   family=C.ENVIRONMENT_FAMILIES[1])
        rec = synthetic_feedback_record(
            feedback_id="fb-syn", candidate=cand, plan_id="p", window=0,
            student_success_rate=0.4,
            expected_signature={"student_success_rate": 0.47},
            student_identity_hash=binding.identity_hash)
        assert rec.student_identity_hash == binding.identity_hash


class TestControllerStampsBindingIdentity:
    def test_every_feedback_record_carries_the_binding(self):
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        ctl.run(max_windows=2)
        binding = ctl.student_binding
        assert binding.weights_status == WEIGHTS_NOT_LOADED_LOCAL
        records = ctl.store.all()
        assert records                                   # probed both windows
        for rec in records:
            assert rec.student_identity_hash == binding.identity_hash
            assert rec.student_parameter_tree_hash == \
                binding.parameter_tree_hash
            assert rec.student_checkpoint_step == 0
            assert rec.student_roles == (C.STUDENT_ROLE_SEARCH,)
            assert rec.memory_compatibility_status == \
                C.MEMORY_COMPATIBILITY_NOT_APPLICABLE
        # the training seam exists but never executed a real update
        assert isinstance(ctl.training_seam, StudentTrainingSeam)
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False

    def test_identity_hash_stable_across_independent_runs(self):
        ctl1 = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        ctl1.run(max_windows=1)
        ctl2 = FeedbackUEDController(C.MODE_SHUFFLED_FEEDBACK)
        ctl2.run(max_windows=1)
        # same symbolic binding across modes: identity is a loop constant,
        # not a function of the comparison mode
        assert ctl1.student_binding.identity_hash == \
            ctl2.student_binding.identity_hash
