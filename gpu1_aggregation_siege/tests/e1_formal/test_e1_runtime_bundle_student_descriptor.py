"""CC2-Repair: StudentSelectionDescriptor in the runtime bundle."""
from types import SimpleNamespace
import pytest
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import student_contract as SC


def _caps():
    return {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS}


class TestBundleStudentDescriptor:
    def test_bundle_carries_a_student_selection(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        sel = b.student_selection
        assert sel.selected_candidate_id in SC.ALLOWED_STUDENT_CANDIDATE_IDS
        assert sel.profile_id == SC.STUDENT_PROFILE_BY_CANDIDATE[
            sel.selected_candidate_id]
        assert len(sel.descriptor_hash) == 64

    def test_missing_student_selection_fails_closed(self):
        with pytest.raises(RB.RuntimeBundleError) as e:
            RB.load_verified_runtime_bundle(
                {"bundle_id": "x", "mode": "TEST_ONLY",
                 "source_commit": "s", "signer_id": "SYNTHETIC_TEST_ONLY_SIGNER",
                 "authorization_grant_hash": "",
                 "object_identity_hashes": {c: "a" * 64 for c in RB.RUNTIME_CAPABILITY_CONTRACTS},
                 "bundle_hash": "b" * 64}, "test")
        assert e.value.code in (RB.RUNTIME_BUNDLE_STUDENT_SELECTION_MISSING,
                                RB.RUNTIME_BUNDLE_HASH_MISMATCH)

    def test_unknown_student_selection_fails_closed(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=_caps())
        block = dict(b.student_selection_mapping)
        block["selected_candidate_id"] = "UNKNOWN_98304"
        with pytest.raises(RB.RuntimeBundleError) as e:
            RB.parse_student_selection(block, "test")
        assert e.value.code == RB.RUNTIME_BUNDLE_STUDENT_SELECTION_BAD
