"""CC2-Repair-2: object-level needs a REAL registry + real objects."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestRegistry:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION registry."""
    registry_identity = "test-only"
    registry_hash = "aa" * 32

    def __init__(self, assets):
        self._assets = assets

    def resolve_asset(self, *, contract, expected_identity):
        return self._assets.get(contract)

    def verify_implementation(self, *, contract, obj, expected_implementation_hash):
        return True


class TestObjectLevelRequiresRealProfile:
    def test_no_registry_is_unbound(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        try:
            ROR.resolve_e1_runtime_objects(b, None, "test")
            raise AssertionError("expected FORMAL_ASSET_REGISTRY_UNBOUND")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_REGISTRY_UNBOUND

    def test_string_registry_is_rejected(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        try:
            ROR.resolve_e1_runtime_objects(b, "registry.json", "test")
            raise AssertionError("expected STRING_PLACEHOLDER")
        except ROR.RuntimeObjectResolutionError as e:
            assert e.code == ROR.OBJ_RESOLUTION_STRING_PLACEHOLDER

    def test_missing_student_object_is_unbound(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        registry = TestRegistry({})
        r = ROR.resolve_e1_runtime_objects(b, registry, "test")
        assert r["all_bound"] is False
        assert "student_identity" in r["missing"]
