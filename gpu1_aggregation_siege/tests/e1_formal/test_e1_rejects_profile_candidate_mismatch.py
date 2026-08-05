"""CC2-Student tests: profile/candidate mismatches fail closed.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
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


class TestProfileCandidateMismatch:
    def test_profile_for_reset128_under_persistent_selection_refused(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            _consume(director_profile="rmt16_reset128_98304")
        assert excinfo.value.code == SC.STUDENT_PROFILE_MISMATCH

    def test_profile_is_explicit_not_guessed(self):
        # the profile comes from the EXPLICIT map, never from
        # string-guessing the name
        assert SC.STUDENT_PROFILE_BY_CANDIDATE[_PERSISTENT] == (
            "rmt16_persistent_98304"
        )
        assert SC.STUDENT_CANDIDATE_BY_PROFILE[
            "rmt16_persistent_98304"
        ] == _PERSISTENT

    def test_contract_candidate_mismatch_refused(self):
        # a contract for RESET128 under a Persistent selection
        other = SC.build_synthetic_student_contract(
            SC.RESET128_STUDENT_CANDIDATE_ID, "test"
        )
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            _consume(contract=other)
        assert excinfo.value.code == SC.STUDENT_CONTRACT_MISMATCH

    def test_expected_params_mismatch_refused(self):
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            _consume(director_expected_params_sha256="ff" * 32)
        assert excinfo.value.code == SC.STUDENT_CHECKPOINT_MISMATCH
