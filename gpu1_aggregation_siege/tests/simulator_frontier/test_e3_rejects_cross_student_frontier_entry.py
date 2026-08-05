# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: a frontier entry captured by one arm can never be handed to the
other arm's training, and its memory can never be restored under the other
arm's carry rule.
"""

import pytest

from dicode.simulator_frontier.dual_student import (
    E3_FRONTIER_CARRY_MODE_MISMATCH,
    E3_FRONTIER_MEMORY_SPEC_MISMATCH,
    E3_FRONTIER_STUDENT_IDENTITY_MISMATCH,
    assert_memory_binding_for_student,
    assert_same_run_student,
)
from dicode.simulator_frontier.errors import ProvenanceViolationError

P = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
R = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def test_cross_student_capture_rejected():
    with pytest.raises(ProvenanceViolationError) as exc:
        assert_same_run_student(selected_candidate_id=P, capture_student_id=R,
                                search_student_id=P, train_student_id=P)
    assert E3_FRONTIER_STUDENT_IDENTITY_MISMATCH in str(exc.value)


def test_cross_student_train_rejected():
    with pytest.raises(ProvenanceViolationError):
        assert_same_run_student(selected_candidate_id=P, capture_student_id=P,
                                search_student_id=P, train_student_id=R)


def test_cross_student_memory_rejected():
    with pytest.raises(ProvenanceViolationError) as exc:
        assert_memory_binding_for_student(
            candidate_id=P, memory_mode="RESET128", carry_mode="RESET128",
            memory_spec_hash="a" * 64, expected_memory_spec_hash="a" * 64)
    assert E3_FRONTIER_CARRY_MODE_MISMATCH in str(exc.value)
    with pytest.raises(ProvenanceViolationError) as exc2:
        assert_memory_binding_for_student(
            candidate_id=P, memory_mode="PERSISTENT", carry_mode="PERSISTENT",
            memory_spec_hash="a" * 64, expected_memory_spec_hash="b" * 64)
    assert E3_FRONTIER_MEMORY_SPEC_MISMATCH in str(exc2.value)


def test_same_student_frontier_entry_passes():
    assert_same_run_student(selected_candidate_id=P, capture_student_id=P,
                            search_student_id=P, train_student_id=P)
