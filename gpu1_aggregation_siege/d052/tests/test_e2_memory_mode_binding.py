"""§七 (dual student): the memory-mode binding — each allowed Student has
a legal (memory_mode, carry_mode) profile and a distinct memory identity.

Persistent  -> PERSISTENT / persistent
Reset128    -> RESET128 / reset128

Any mismatch is E2_STUDENT_MEMORY_MODE_MISMATCH.

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

PERSISTENT = C.STRONG_STUDENT_CANDIDATE_ID
RESET128 = C.RESET128_STUDENT_CANDIDATE_ID


class TestMemoryModeBinding:
    @pytest.mark.parametrize("candidate", [PERSISTENT, RESET128])
    def test_legal_profile_resolves(self, candidate):
        memory_mode, carry_mode = C.STUDENT_PROFILE_MEMORY_MAP[candidate]
        identity = resolve_student_binding(
            student_contract(candidate), director_selected_candidate_id=candidate)
        assert identity.memory_mode == memory_mode
        assert identity.carry_mode == carry_mode

    def test_wrong_carry_mode_rejected(self):
        #: Persistent requires carry_mode='persistent'; 'reset128' is wrong
        contract = student_contract(PERSISTENT)
        contract.carry_mode = C.STUDENT_CARRY_MODE_RESET128
        with pytest.raises(StudentBindingBlocked,
                           match="E2_STUDENT_MEMORY_MODE_MISMATCH"):
            resolve_student_binding(contract,
                                    director_selected_candidate_id=PERSISTENT)

    def test_wrong_memory_mode_rejected(self):
        contract = student_contract(PERSISTENT)
        contract.memory_mode = C.STUDENT_MEMORY_MODE_RESET128
        with pytest.raises(StudentBindingBlocked,
                           match="E2_STUDENT_MEMORY_MODE_MISMATCH"):
            resolve_student_binding(contract,
                                    director_selected_candidate_id=PERSISTENT)

    def test_memory_spec_hash_required_sha256(self):
        contract = student_contract(PERSISTENT)
        contract.memory_spec_hash = "not-a-hash"
        with pytest.raises(StudentBindingBlocked,
                           match="E2_STUDENT_MEMORY_MODE_MISMATCH"):
            resolve_student_binding(contract,
                                    director_selected_candidate_id=PERSISTENT)

    def test_distinct_memory_identities(self):
        p = resolve_student_binding(student_contract(PERSISTENT),
                                    director_selected_candidate_id=PERSISTENT)
        r = resolve_student_binding(student_contract(RESET128),
                                    director_selected_candidate_id=RESET128)
        assert p.memory_spec_hash != r.memory_spec_hash \
            or p.memory_mode != r.memory_mode
        assert p.identity_hash != r.identity_hash


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
