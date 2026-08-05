"""§七 (dual student): window k and window k+1 are bound to the SAME
director-selected Student — every feedback record stamps the selected
candidate's identity.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

from d052.feedback_llm_ued import constants as C

from e2_test_sign_helpers import make_dual_student_controller


class TestTwoWindowSameStudent:
    def test_all_records_carry_the_selected_student(self):
        ctl = make_dual_student_controller(C.STRONG_STUDENT_CANDIDATE_ID)
        summary = ctl.run(max_windows=2)
        assert summary.n_windows == 2
        assert summary.request_control_stopped is False
        binding = ctl.student_binding
        records = ctl.store.all()
        assert records
        for rec in records:
            #: the FULL Student identity is stamped on every record
            assert rec.student_candidate_id == \
                C.STRONG_STUDENT_CANDIDATE_ID
            assert rec.student_identity_hash == binding.identity_hash
            assert rec.student_memory_mode == C.STUDENT_MEMORY_MODE_PERSISTENT
            assert rec.student_memory_spec_hash == binding.memory_spec_hash
            assert rec.runtime_bundle_hash == binding.runtime_bundle_hash
        #: the two windows share the SAME binding (no switch)
        assert summary.windows[0]["window"] == 0
        assert summary.windows[1]["window"] == 1

    def test_reset128_records_carry_reset128_identity(self):
        ctl = make_dual_student_controller(C.RESET128_STUDENT_CANDIDATE_ID)
        summary = ctl.run(max_windows=2)
        assert summary.n_windows == 2
        for rec in ctl.store.all():
            assert rec.student_candidate_id == \
                C.RESET128_STUDENT_CANDIDATE_ID
            assert rec.student_memory_mode == C.STUDENT_MEMORY_MODE_RESET128


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
