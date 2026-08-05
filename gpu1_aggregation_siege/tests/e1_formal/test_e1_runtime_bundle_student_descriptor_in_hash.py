"""CC2-Repair: student_selection is part of the bundle hash."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestStudentSelectionInHash:
    def test_different_selection_yields_different_bundle_hash(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        base = dict(b.object_identity_hashes)
        other = RB.compute_bundle_hash(
            bundle_id=b.bundle_id, mode=b.mode, source_commit=b.source_commit,
            signer_id=b.signer_id,
            authorization_grant_hash=b.authorization_grant_hash,
            object_identity_hashes=base,
            student_selection_hash="f" * 64)
        assert other != b.bundle_hash

    def test_manifest_roundtrip_preserves_the_selection(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        m = {"bundle_id": b.bundle_id, "mode": b.mode,
             "source_commit": b.source_commit, "signer_id": b.signer_id,
             "authorization_grant_hash": b.authorization_grant_hash,
             "object_identity_hashes": dict(b.object_identity_hashes),
             "student_selection": b.student_selection_mapping,
             "bundle_hash": b.bundle_hash}
        loaded = RB.load_verified_runtime_bundle(m, "test")
        assert loaded.student_selection.selected_candidate_id == (
            b.student_selection.selected_candidate_id)
