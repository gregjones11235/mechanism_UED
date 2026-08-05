"""§七 (dual student): feedback produced under ONE Student can never be
reused by a run bound to the OTHER Student.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import pytest

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

from e2_test_sign_helpers import make_dual_student_controller

PERSISTENT = C.STRONG_STUDENT_CANDIDATE_ID
RESET128 = C.RESET128_STUDENT_CANDIDATE_ID


class TestRejectsCrossStudentFeedback:
    def test_persistent_run_rejects_reset128_feedback(self):
        ctl = make_dual_student_controller(PERSISTENT)
        reset128_hash = text_sha256("RESET128_STUDENT_IDENTITY_HASH")
        cand = synthetic_candidate(candidate_id="c-cross-r128",
                                   family=C.ENVIRONMENT_FAMILIES[1])
        ctl.store.add(synthetic_feedback_record(
            feedback_id="fb-cross-r128", candidate=cand, plan_id="p",
            window=0, student_success_rate=0.4,
            expected_signature={"student_success_rate": 0.45},
            student_identity_hash=reset128_hash))
        with pytest.raises(
                RuntimeError,
                match="E2_TWO_WINDOW_STUDENT_CONTINUITY_VIOLATION"):
            ctl.run(max_windows=2)

    def test_reset128_run_rejects_persistent_feedback(self):
        ctl = make_dual_student_controller(RESET128)
        persistent_hash = text_sha256("PERSISTENT_STUDENT_IDENTITY_HASH")
        cand = synthetic_candidate(candidate_id="c-cross-pers",
                                   family=C.ENVIRONMENT_FAMILIES[0])
        ctl.store.add(synthetic_feedback_record(
            feedback_id="fb-cross-pers", candidate=cand, plan_id="p",
            window=0, student_success_rate=0.4,
            expected_signature={"student_success_rate": 0.45},
            student_identity_hash=persistent_hash))
        with pytest.raises(
                RuntimeError,
                match="E2_TWO_WINDOW_STUDENT_CONTINUITY_VIOLATION"):
            ctl.run(max_windows=2)


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
