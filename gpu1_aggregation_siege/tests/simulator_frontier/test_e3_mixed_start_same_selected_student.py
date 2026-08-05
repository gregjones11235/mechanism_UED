# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: the mixed-start training handoff binds the SAME selected Student —
the two top Students are two independent experiment starting points, never a
Student/Reference pair.
"""

import pytest

from dicode.simulator_frontier.dual_student import (
    assert_reference_is_not_a_primary_student,
    assert_same_run_student,
)
from dicode.simulator_frontier.errors import ProvenanceViolationError

P = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
R = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def test_mixed_start_handoff_binds_same_student():
    assert_same_run_student(selected_candidate_id=P, capture_student_id=P,
                            search_student_id=P, train_student_id=P)


def test_mixed_start_cannot_train_other_arm():
    with pytest.raises(ProvenanceViolationError):
        assert_same_run_student(selected_candidate_id=P, capture_student_id=P,
                                search_student_id=P, train_student_id=R)


def test_second_student_is_not_a_reference():
    # The two top Students are two independent starting points, NOT a
    # Student/Reference pair: the Reset128 candidate can never be mounted as
    # the Reference.
    with pytest.raises(ProvenanceViolationError):
        assert_reference_is_not_a_primary_student(R)
