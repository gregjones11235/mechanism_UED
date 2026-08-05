"""CC2-Student tests: switching Student mid-window fails closed.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import student_contract as SC

_PERSISTENT = SC.PERSISTENT_STUDENT_CANDIDATE_ID
_RESET128 = SC.RESET128_STUDENT_CANDIDATE_ID


def _mount(cid, **overrides):
    kwargs = dict(
        contract=SC.build_synthetic_student_contract(cid, "test"),
        director_selected_candidate_id=cid,
        runtime_bundle_hash="c0" * 32,
        ctx="test",
    )
    kwargs.update(overrides)
    return SC.consume_e1_student_contract(**kwargs)


class TestStudentSwitchMidWindow:
    def test_same_student_passes(self):
        first = _mount(_PERSISTENT)
        second = _mount(_PERSISTENT)
        SC.assert_same_student_mount(first, second, "test")

    def test_identity_switch_refused(self):
        first = _mount(_PERSISTENT)
        second = _mount(_RESET128)
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.assert_same_student_mount(first, second, "test")
        assert excinfo.value.code == SC.E1_STUDENT_IDENTITY_SWITCH

    def test_profile_switch_refused(self):
        first = _mount(_PERSISTENT)
        second = replace(_mount(_PERSISTENT), profile_id="rmt16_other")
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.assert_same_student_mount(first, second, "test")
        assert excinfo.value.code == SC.E1_STUDENT_PROFILE_SWITCH

    def test_memory_mode_switch_refused(self):
        first = _mount(_PERSISTENT)
        second = replace(_mount(_PERSISTENT), memory_mode="RESET128")
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.assert_same_student_mount(first, second, "test")
        assert excinfo.value.code == SC.E1_STUDENT_MEMORY_MODE_SWITCH

    def test_checkpoint_switch_refused(self):
        first = _mount(_PERSISTENT)
        second = replace(_mount(_PERSISTENT), params_sha256="ff" * 32)
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.assert_same_student_mount(first, second, "test")
        assert excinfo.value.code == SC.E1_STUDENT_CHECKPOINT_SWITCH

    def test_runtime_bundle_switch_refused(self):
        first = _mount(_PERSISTENT)
        second = _mount(_PERSISTENT, runtime_bundle_hash="ff" * 32)
        with pytest.raises(SC.StudentSelectionError) as excinfo:
            SC.assert_same_student_mount(first, second, "test")
        assert excinfo.value.code == SC.E1_STUDENT_IDENTITY_SWITCH
