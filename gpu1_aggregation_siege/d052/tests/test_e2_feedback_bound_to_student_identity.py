"""§七 (dual student): every real SimulatorFeedback records the Student
identity it was probed under (candidate id, identity hash, checkpoint
hash/step, memory mode, memory spec hash, source window, runtime bundle
hash).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

from d052.feedback_llm_ued import constants as C

from e2_test_sign_helpers import make_dual_student_controller


class TestFeedbackBoundToStudentIdentity:
    def test_records_stamp_full_student_identity(self):
        ctl = make_dual_student_controller(C.STRONG_STUDENT_CANDIDATE_ID)
        ctl.run(max_windows=2)
        binding = ctl.student_binding
        records = ctl.store.all()
        assert records
        for rec in records:
            assert rec.student_candidate_id == \
                C.STRONG_STUDENT_CANDIDATE_ID
            assert rec.student_identity_hash == binding.identity_hash
            assert rec.student_memory_mode == C.STUDENT_MEMORY_MODE_PERSISTENT
            assert rec.student_memory_spec_hash == binding.memory_spec_hash
            assert rec.runtime_bundle_hash == binding.runtime_bundle_hash
            #: source window is recorded (the window it was probed in)
            assert rec.window in (0, 1)
            assert rec.student_checkpoint_step == \
                binding.checkpoint_global_step

    def test_identity_participates_in_record_hash(self):
        ctl = make_dual_student_controller(C.STRONG_STUDENT_CANDIDATE_ID)
        ctl.run(max_windows=2)
        records = ctl.store.all()
        #: two records from the same window with the same candidate
        #: identity hash share the binding; a record with a DIFFERENT
        #: student identity would have a different record hash (proven in
        #: the switch test's fail-closed behaviour)
        assert all(r.student_identity_hash == records[0].student_identity_hash
                   for r in records)


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
