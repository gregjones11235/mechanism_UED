"""CC2-Repair-2: object identity mismatch => fail closed."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class _Wrong:
    identity_id = "WRONG"


class TestRegistry:
    registry_identity = "test-only"
    registry_hash = "aa" * 32

    def __init__(self, assets):
        self._assets = assets

    def resolve_asset(self, *, contract, expected_identity):
        return self._assets.get(contract)

    def verify_implementation(self, *, contract, obj, expected_implementation_hash):
        return True


class TestRuntimeObjectIdentityMismatch:
    def test_wrong_object_is_refused(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        registry = TestRegistry({"student_adapter": _Wrong()})
        r = ROR.resolve_e1_runtime_objects(b, registry, "test")
        assert r["resolutions"]["student_adapter"].bound is False
        assert r["resolutions"]["student_adapter"].code == (
            ROR.OBJ_RESOLUTION_IDENTITY_MISMATCH)

    def test_mapping_registry_is_rejected(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        try:
            ROR.resolve_e1_runtime_objects(b, {"student_identity": "x"}, "test")
            raise AssertionError("expected MAPPING_IMPERSONATION")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_MAPPING_IMPERSONATION
