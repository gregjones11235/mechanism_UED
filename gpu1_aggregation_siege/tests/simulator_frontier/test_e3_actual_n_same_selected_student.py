# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: actual-N's Student deterministic and stochastic branches must use the
same selected Student + same checkpoint; the run can never switch Student.
"""

import pytest

from dicode.simulator_frontier.dual_student import (
    E3_FRONTIER_STUDENT_IDENTITY_MISMATCH,
    assert_same_run_student,
)
from dicode.simulator_frontier.errors import ProvenanceViolationError

P = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
R = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def test_actual_n_branches_bind_one_selected_student():
    assert_same_run_student(selected_candidate_id=P, capture_student_id=P,
                            search_student_id=P, train_student_id=P)


def test_actual_n_cannot_switch_student():
    with pytest.raises(ProvenanceViolationError) as exc:
        assert_same_run_student(selected_candidate_id=P, capture_student_id=P,
                                search_student_id=R, train_student_id=P)
    assert E3_FRONTIER_STUDENT_IDENTITY_MISMATCH in str(exc.value)
