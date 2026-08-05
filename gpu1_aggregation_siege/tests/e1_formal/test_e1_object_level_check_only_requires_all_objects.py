"""CC2-Repair: all nine runtime objects must be bound."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestObjectLevelRequiresAllObjects:
    def test_every_required_object_must_be_present(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        r = ROR.resolve_e1_runtime_objects(b, None, "test")
        assert r["all_bound"] is False
        expected = set(RB.RUNTIME_CAPABILITY_CONTRACTS) | set(
            ROR.EXTRA_RUNTIME_CONTRACTS)
        assert set(r["resolutions"]) == expected
