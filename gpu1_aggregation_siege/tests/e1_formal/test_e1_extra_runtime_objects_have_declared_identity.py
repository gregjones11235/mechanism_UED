"""CC2-Repair-2: extras declared identity."""
"""The extra runtime objects need a declared identity in the manifest."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestExtraObjectsDeclaredIdentity:
    def test_every_extra_contract_is_in_required_set(self):
        for c in ROR.EXTRA_RUNTIME_CONTRACTS:
            assert c in ROR.REQUIRED_RUNTIME_OBJECTS
            assert c not in RB.RUNTIME_CAPABILITY_CONTRACTS

    def test_manifest_has_no_extra_declarations_this_round(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        assert getattr(b, "declared_runtime_objects", None) is None
