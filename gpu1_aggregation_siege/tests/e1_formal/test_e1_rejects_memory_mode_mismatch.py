"""CC2-Student tests: memory-mode + adapter mismatches fail closed.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import student_contract as SC

_PERSISTENT = SC.PERSISTENT_STUDENT_CANDIDATE_ID


def _consume(**overrides):
    kwargs = dict(
        contract=SC.build_synthetic_student_contract(_PERSISTENT, "test"),
        director_selected_candidate_id=_PERSISTENT,
        runtime_bundle_hash="c0" * 32,
        ctx="test",
    )
    kwargs.update(overrides)
    return SC.consume_e1_student_contract(**kwargs)


class TestMemoryModeMismatch:
    def test_reset_memory_mode_under_persistent_selection_refused(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            _consume(director_memory_mode="RESET128")
        assert excinfo.value.code == SC.STUDENT_MEMORY_MODE_MISMATCH

    def test_memory_mode_is_mapped_explicitly(self):
        assert SC.STUDENT_MEMORY_MODE_BY_CANDIDATE[_PERSISTENT] == (
            "PERSISTENT"
        )
        assert (
            SC.STUDENT_MEMORY_MODE_BY_CANDIDATE[
                SC.RESET128_STUDENT_CANDIDATE_ID
            ]
            == "RESET128"
        )

    def test_matching_memory_mode_passes(self):
        mount = _consume(director_memory_mode="PERSISTENT")
        assert mount.memory_mode == "PERSISTENT"


class TestAdapterMismatch:
    def test_adapter_id_not_matching_profile_refused(self):
        contract = SC.build_synthetic_student_contract(
            _PERSISTENT, "test"
        )
        # the shared RMT16 adapter's identity must match the mapped
        # profile; a foreign adapter_id is refused
        forged = replace(contract, adapter_id="some_other_adapter")
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            _consume(contract=forged)
        assert excinfo.value.code == SC.STUDENT_ADAPTER_MISMATCH

    def test_adapter_identity_hash_is_bound_into_the_mount(self):
        mount = _consume(director_adapter_identity_hash="aa" * 32)
        assert mount.adapter_identity_hash == "aa" * 32
