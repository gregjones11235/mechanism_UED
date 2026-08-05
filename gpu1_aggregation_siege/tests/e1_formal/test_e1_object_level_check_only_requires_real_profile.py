"""CC2-Repair: object-level check-only needs real objects."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestObjectLevelRequiresRealProfile:
    def test_missing_student_object_is_unbound(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        r = ROR.resolve_e1_runtime_objects(b, None, "test")
        assert r["all_bound"] is False
        assert "student_identity" in r["missing"]

    def test_string_object_is_never_a_real_profile(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        registry = {"student_identity": "student_identity_name"}
        r = ROR.resolve_e1_runtime_objects(b, registry, "test")
        assert r["resolutions"]["student_identity"].bound is False
        assert r["resolutions"]["student_identity"].code == (
            ROR.OBJ_RESOLUTION_STRING_PLACEHOLDER)
