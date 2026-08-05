"""CC2-Student tests: an unknown Student selection is refused.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
import pytest

from dicode.teachers.e1_formal import student_contract as SC


class TestRejectsUnknownStudent:
    def test_unknown_candidate_refused(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.require_director_selection(
                "SOME_OTHER_RMT16_98304", "test"
            )
        assert excinfo.value.code == SC.STUDENT_NOT_ALLOWED

    def test_bad_type_candidate_refused(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.require_director_selection(12345, "test")
        assert excinfo.value.code == SC.STUDENT_SELECTION_REQUIRED

    def test_consume_refuses_unknown_candidate(self):
        contract = SC.build_synthetic_student_contract(
            SC.PERSISTENT_STUDENT_CANDIDATE_ID, "test"
        )
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.consume_e1_student_contract(
                contract,
                director_selected_candidate_id="UNKNOWN_98304",
                runtime_bundle_hash="c0" * 32,
                ctx="test",
            )
        assert excinfo.value.code == SC.STUDENT_NOT_ALLOWED

    def test_allowed_set_is_exactly_the_two(self):
        assert SC.ALLOWED_STUDENT_CANDIDATE_IDS == {
            "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
            "RESET128_RMT16_ORIGINAL_VTRACE_98304",
        }
