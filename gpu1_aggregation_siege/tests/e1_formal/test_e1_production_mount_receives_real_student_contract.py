"""CC2-Repair-2: real contract to mount."""
"""PRODUCTION mount receives the REAL StudentInitContract only."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import student_contract as SC


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestMountReceivesRealContract:
    def test_production_mount_accepts_a_real_contract(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        forged = replace(b, mode="PRODUCTION")
        real = SC.build_synthetic_student_contract(
            forged.student_selection.selected_candidate_id, "test")
        # params must match the descriptor for production
        from dataclasses import replace as R2
        real2 = R2(real, parameter_tree_hash=(
            forged.student_selection.params_sha256))
        mount = SC.mount_student_from_director_bundle(
            bundle=forged, director_selected_candidate_id=None,
            ctx="test", contract=real2)
        assert mount.candidate_id == (
            forged.student_selection.selected_candidate_id)

    def test_production_mount_rejects_synthetic_when_no_contract(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        forged = replace(b, mode="PRODUCTION")
        with pytest.raises(SC.StudentSelectionError) as e:
            SC.mount_student_from_director_bundle(
                bundle=forged, director_selected_candidate_id=None,
                ctx="test", contract=None)
        assert e.value.code == SC.STUDENT_SELECTION_REQUIRED
