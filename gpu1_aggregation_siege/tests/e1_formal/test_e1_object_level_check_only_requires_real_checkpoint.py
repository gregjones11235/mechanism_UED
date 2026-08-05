"""CC2-Repair: checkpoint identity is verified, not assumed."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import runtime_object_resolution as ROR


class _FakeCheckpoint:
    candidate_id = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
    params_sha256 = "ff" * 32
    checkpoint_file_sha256 = "ff" * 32


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestObjectLevelRequiresRealCheckpoint:
    def test_checkpoint_identity_must_match_the_manifest(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        registry = {"student_identity": _FakeCheckpoint()}
        r = ROR.resolve_e1_runtime_objects(b, registry, "test")
        assert r["resolutions"]["student_identity"].bound is False
        assert r["resolutions"]["student_identity"].code == (
            ROR.OBJ_RESOLUTION_IDENTITY_MISMATCH)
