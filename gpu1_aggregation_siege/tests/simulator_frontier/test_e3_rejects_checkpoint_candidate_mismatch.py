# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: the memory/checkpoint binding for the selected Student is enforced —
a memory spec or carry mode from the OTHER arm never binds.
"""

import pytest

from dicode.simulator_frontier.dual_student import (
    E3_FRONTIER_CARRY_MODE_MISMATCH,
    E3_FRONTIER_MEMORY_SPEC_MISMATCH,
    assert_memory_binding_for_student,
)
from dicode.simulator_frontier.errors import ProvenanceViolationError

P = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
R = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def test_persistent_memory_spec_rejected_for_reset128():
    with pytest.raises(ProvenanceViolationError) as exc:
        assert_memory_binding_for_student(
            candidate_id=P, memory_mode="PERSISTENT", carry_mode="PERSISTENT",
            memory_spec_hash="r" * 64, expected_memory_spec_hash="p" * 64)
    assert E3_FRONTIER_MEMORY_SPEC_MISMATCH in str(exc.value)


def test_reset128_carry_rejected_for_persistent():
    with pytest.raises(ProvenanceViolationError) as exc:
        assert_memory_binding_for_student(
            candidate_id=P, memory_mode="PERSISTENT", carry_mode="RESET128",
            memory_spec_hash="p" * 64, expected_memory_spec_hash="p" * 64)
    assert E3_FRONTIER_CARRY_MODE_MISMATCH in str(exc.value)


def test_persistent_memory_rejected_for_reset128_candidate():
    with pytest.raises(ProvenanceViolationError):
        assert_memory_binding_for_student(
            candidate_id=R, memory_mode="PERSISTENT", carry_mode="PERSISTENT",
            memory_spec_hash="r" * 64, expected_memory_spec_hash="r" * 64)


def test_matching_binding_passes():
    assert_memory_binding_for_student(
        candidate_id=P, memory_mode="PERSISTENT", carry_mode="PERSISTENT",
        memory_spec_hash="p" * 64, expected_memory_spec_hash="p" * 64)
