"""CC2-Repair-2: missing declared identity."""
"""Missing declared identity => fail closed (never skipped)."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


class TestRegistry:
    registry_identity = "test-only"
    registry_hash = "aa" * 32

    def resolve_asset(self, *, contract, expected_identity):
        return None

    def verify_implementation(self, *, contract, obj, expected_implementation_hash):
        return True


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestMissingDeclaredIdentity:
    def test_extra_objects_without_declaration_fail_closed(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        r = ROR.resolve_e1_runtime_objects(b, TestRegistry(), "t")
        assert "canonical_dicode_one_update_runtime" in r["missing"]
        assert r["resolutions"][
            "canonical_dicode_one_update_runtime"].code == (
            ROR.OBJ_RESOLUTION_DECLARED_IDENTITY_MISSING)
