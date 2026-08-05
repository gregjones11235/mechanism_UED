"""§七 (dual student): there is NO default Student candidate — the
director must explicitly select one.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.student_binding import (
    StudentBindingBlocked,
    resolve_student_binding,
)

from e2_test_sign_helpers import student_contract


class TestNoDefaultStudent:
    def test_missing_selection_rejected(self):
        #: even a PERSISTENT-shaped contract cannot auto-select PERSISTENT
        with pytest.raises(StudentBindingBlocked,
                           match="E2_STUDENT_NO_DIRECTOR_SELECTION"):
            resolve_student_binding(
                student_contract(C.STRONG_STUDENT_CANDIDATE_ID),
                director_selected_candidate_id="")

    def test_controller_symbolic_path_needs_no_selection(self):
        #: the symbolic/non-real path uses local_symbolic_binding (no real
        #: Student is selected) — the mock controller constructs fine
        from d052.feedback_llm_ued.controller import FeedbackUEDController
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        assert ctl.student_binding.weights_status == "NOT_LOADED_LOCAL"


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
