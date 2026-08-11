"""CC2-Repair: production never synthesizes a Student contract."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import student_contract as SC


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestProductionRejectsSyntheticContract:
    def test_production_bundle_requires_a_real_contract(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        forged = replace(b, mode="PRODUCTION")
        with pytest.raises(SC.StudentSelectionError) as e:
            SC.mount_student_from_director_bundle(
                bundle=forged, director_selected_candidate_id=None, ctx="test")
        assert e.value.code == SC.STUDENT_SELECTION_REQUIRED

    def test_mount_never_reads_a_nonexistent_student_attribute(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        assert not hasattr(b, "student")
        mount = SC.mount_student_from_director_bundle(
            bundle=b, director_selected_candidate_id=None, ctx="test")
        assert mount.candidate_id == b.student_selection.selected_candidate_id
