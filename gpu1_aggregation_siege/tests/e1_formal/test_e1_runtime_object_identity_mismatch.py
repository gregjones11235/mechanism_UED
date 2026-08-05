"""CC2-Repair: object identity mismatch => fail closed."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


class _Wrong:
    identity_id = "WRONG"


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestRuntimeObjectIdentityMismatch:
    def test_wrong_object_is_refused(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        registry = {"student_adapter": _Wrong()}
        r = ROR.resolve_e1_runtime_objects(b, registry, "test")
        assert r["resolutions"]["student_adapter"].bound is False
        assert r["resolutions"]["student_adapter"].code == (
            ROR.OBJ_RESOLUTION_IDENTITY_MISMATCH)

    def test_mapping_impersonation_is_refused(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        r = ROR.resolve_e1_runtime_objects(
            b, {"anchor_manifest": {"kind": "fake"}}, "test")
        assert r["resolutions"]["anchor_manifest"].bound is False
        assert r["resolutions"]["anchor_manifest"].code == (
            ROR.OBJ_RESOLUTION_MAPPING_IMPERSONATION)
