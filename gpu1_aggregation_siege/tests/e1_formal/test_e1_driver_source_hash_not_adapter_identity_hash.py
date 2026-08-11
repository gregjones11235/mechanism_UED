"""CC2-Repair: driver_source_sha256 is never the adapter identity."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB


class TestDriverSourceVsAdapterIdentity:
    def test_hashes_are_distinct_fields(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities={c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
                          for c in RB.RUNTIME_CAPABILITY_CONTRACTS})
        sel = b.student_selection
        assert sel.driver_source_sha256 != sel.adapter_identity_hash
        assert len(sel.adapter_identity_hash) == 64
        assert len(sel.adapter_implementation_hash) == 64
        assert len(sel.driver_source_sha256) == 64
