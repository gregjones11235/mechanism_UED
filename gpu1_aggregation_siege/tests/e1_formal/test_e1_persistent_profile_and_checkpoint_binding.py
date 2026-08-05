"""CC2-Repair-2: persistent profile binding."""
"""Persistent profile + checkpoint identity bind in the descriptor."""
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import student_contract as SC


class TestPersistentProfileBinding:
    def test_persistent_descriptor_fields(self):
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities={c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
                          for c in RB.RUNTIME_CAPABILITY_CONTRACTS})
        sel = b.student_selection
        assert sel.selected_candidate_id == (
            SC.PERSISTENT_STUDENT_CANDIDATE_ID)
        assert sel.profile_id == "rmt16_persistent_98304"
        assert sel.memory_mode == "PERSISTENT"
        assert sel.architecture_family.upper() == "RMT16"
        assert len(sel.checkpoint_file_sha256) == 64
        assert len(sel.params_sha256) == 64
        assert len(sel.adapter_identity_hash) == 64
