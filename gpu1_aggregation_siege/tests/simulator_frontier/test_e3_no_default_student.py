# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: unknown or missing candidate selections FAIL CLOSED — the consumer
never defaults to the first candidate.
"""

import pytest

from dicode.simulator_frontier.dual_student import (
    ALLOWED_PRIMARY_STUDENT_IDS,
    PROFILE_NAME_TO_CANDIDATE_ID,
    candidate_from_profile_name,
    validate_primary_student_candidate,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError


def test_frozen_set_has_exactly_two():
    assert ALLOWED_PRIMARY_STUDENT_IDS == {
        "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        "RESET128_RMT16_ORIGINAL_VTRACE_98304",
    }


def test_no_default_first_candidate():
    for bad in ("", None, "UNKNOWN_CANDIDATE", "teacher", "first"):
        with pytest.raises(InvalidEvidenceError):
            validate_primary_student_candidate(bad)


def test_profile_map_is_exact():
    assert candidate_from_profile_name("rmt16_persistent_98304") == \
        "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
    assert candidate_from_profile_name("rmt16_reset128_98304") == \
        "RESET128_RMT16_ORIGINAL_VTRACE_98304"
    assert sorted(PROFILE_NAME_TO_CANDIDATE_ID) == sorted(
        ["rmt16_persistent_98304", "rmt16_reset128_98304"])
    with pytest.raises(InvalidEvidenceError):
        candidate_from_profile_name("unknown_profile")
