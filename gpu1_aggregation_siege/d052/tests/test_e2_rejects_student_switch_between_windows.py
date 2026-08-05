"""§七 (dual student): a Student SWITCH between windows is a continuity
violation — the loop stops and produces no feedback / trains nothing.

Window k+1 may only consume feedback produced under the SAME director-
selected Student; foreign-student feedback is refused
(E2_TWO_WINDOW_STUDENT_CONTINUITY_VIOLATION).

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


class TestRejectsStudentSwitch:
    def test_foreign_student_feedback_fails_closed(self):
        #: the run is bound to PERSISTENT; a window-0 record produced under
        #: the OTHER Student (RESET128) is present in the store -> window 1
        #: must refuse to consume it
        ctl = make_dual_student_controller(C.STRONG_STUDENT_CANDIDATE_ID)
        foreign_hash = text_sha256("RESET128_STUDENT_IDENTITY_HASH")
        cand = synthetic_candidate(
            candidate_id="c-foreign-student",
            family=C.ENVIRONMENT_FAMILIES[0])
        ctl.store.add(synthetic_feedback_record(
            feedback_id="fb-foreign-student",
            candidate=cand, plan_id="plan-x", window=0,
            student_success_rate=0.4,
            expected_signature={"student_success_rate": 0.45},
            student_identity_hash=foreign_hash))
        with pytest.raises(
                RuntimeError,
                match="E2_TWO_WINDOW_STUDENT_CONTINUITY_VIOLATION"):
            ctl.run(max_windows=2)
        #: nothing beyond the halt produced new feedback or trained
        assert all(t.status != "EXECUTED_ONE_UPDATE_CHECKPOINT_ROUNDTRIP"
                   for t in ctl.training_log)
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
