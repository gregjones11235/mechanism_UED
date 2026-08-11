"""CC2-Repair-2: every REQUIRED object needs a DECLARED identity."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestRegistry:
    registry_identity = "test-only"
    registry_hash = "aa" * 32

    def resolve_asset(self, *, contract, expected_identity):
        return None

    def verify_implementation(self, *, contract, obj, expected_implementation_hash):
        return True


class TestObjectLevelRequiresAllObjects:
    def test_every_required_object_must_be_declared(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        registry = TestRegistry()
        r = ROR.resolve_e1_runtime_objects(b, registry, "test")
        assert r["all_bound"] is False
        # the three extra objects have NO declared identity in the
        # manifest => strict DECLARED_IDENTITY_MISSING, never skipped
        assert set(r["resolutions"]) == set(ROR.REQUIRED_RUNTIME_OBJECTS)
        assert "canonical_dicode_one_update_runtime" in r["missing"]
