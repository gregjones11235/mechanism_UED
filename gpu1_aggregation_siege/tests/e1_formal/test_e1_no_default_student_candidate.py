"""CC2-Student tests: the Student selection is NEVER defaulted.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import student_contract as SC


class TestNoDefaultStudentCandidate:
    def test_missing_selection_is_required(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.require_director_selection(None, "test")
        assert excinfo.value.code == SC.STUDENT_SELECTION_REQUIRED
        assert "NO default" in str(excinfo.value)

    def test_empty_selection_is_required(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.require_director_selection("", "test")
        assert excinfo.value.code == SC.STUDENT_SELECTION_REQUIRED

    def test_mount_without_bundle_selection_required(self):
        bundle = SimpleNamespace(
            bundle_hash="c0" * 32,
            student=None,  # no student block => no selection
        )
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.mount_student_from_director_bundle(
                bundle=bundle,
                director_selected_candidate_id=None,
                ctx="test",
            )
        assert excinfo.value.code == SC.STUDENT_SELECTION_REQUIRED

    def test_cli_cannot_override_bundle_selection(self):
        bundle = SimpleNamespace(
            bundle_hash="c0" * 32,
            student={
                "candidate_id": (
                    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
                ),
                "profile": "rmt16_persistent_98304",
                "memory_mode": "PERSISTENT",
                "expected_params_sha256": "aa" * 32,
            },
        )
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.mount_student_from_director_bundle(
                bundle=bundle,
                director_selected_candidate_id=(
                    "RESET128_RMT16_ORIGINAL_VTRACE_98304"
                ),
                ctx="test",
            )
        assert excinfo.value.code == SC.STUDENT_CONTRACT_MISMATCH
