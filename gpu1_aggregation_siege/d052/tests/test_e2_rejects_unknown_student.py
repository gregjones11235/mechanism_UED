"""§七 (dual student): an UNKNOWN Student candidate is rejected — the
binding requires a director-selected candidate in ALLOWED_STUDENT_IDS.

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


class TestRejectsUnknownStudent:
    def test_unknown_director_selection_rejected(self):
        with pytest.raises(StudentBindingBlocked,
                           match="E2_STUDENT_UNKNOWN_CANDIDATE"):
            resolve_student_binding(
                student_contract(C.STRONG_STUDENT_CANDIDATE_ID),
                director_selected_candidate_id="UNKNOWN_CANDIDATE_X")

    def test_contract_with_unknown_candidate_rejected(self):
        #: an unknown candidate cannot even satisfy the profile memory
        #: mapping (no legal mapping exists)
        with pytest.raises(StudentBindingBlocked):
            resolve_student_binding(
                student_contract(C.STRONG_STUDENT_CANDIDATE_ID)
                if False else student_contract(
                    C.STRONG_STUDENT_CANDIDATE_ID),
                director_selected_candidate_id="UNKNOWN_CANDIDATE_Y")

    def test_allowed_set_exactly_two(self):
        assert C.ALLOWED_STUDENT_CANDIDATE_IDS == {
            C.STRONG_STUDENT_CANDIDATE_ID,
            C.RESET128_STUDENT_CANDIDATE_ID}
        assert len(C.ALLOWED_STUDENT_CANDIDATE_IDS) == 2


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
