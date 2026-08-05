"""CC2-Student tests: the RESET128 Student is an allowed selection.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
from dicode.teachers.e1_formal import student_contract as SC

_RESET128 = SC.RESET128_STUDENT_CANDIDATE_ID


def _mount(**overrides):
    contract = SC.build_synthetic_student_contract(_RESET128, "test")
    kwargs = dict(
        contract=contract,
        director_selected_candidate_id=_RESET128,
        runtime_bundle_hash="c0" * 32,
        ctx="test",
    )
    kwargs.update(overrides)
    return SC.consume_e1_student_contract(**kwargs)


class TestAcceptsReset128Student:
    def test_reset128_is_in_the_allowed_set(self):
        assert _RESET128 in SC.ALLOWED_STUDENT_CANDIDATE_IDS

    def test_reset128_mounts_with_its_own_profile(self):
        mount = _mount()
        assert mount.candidate_id == _RESET128
        assert mount.profile_id == "rmt16_reset128_98304"
        assert mount.memory_mode == "RESET128"
        assert mount.params_sha256 == "aa" * 32
        assert mount.adapter_id == "rmt16_reset128_98304"
        assert len(mount.mount_hash) == 64

    def test_reset128_memory_mode_differs_from_persistent(self):
        reset_mount = _mount()
        persistent = SC.consume_e1_student_contract(
            SC.build_synthetic_student_contract(
                SC.PERSISTENT_STUDENT_CANDIDATE_ID, "test"
            ),
            director_selected_candidate_id=(
                SC.PERSISTENT_STUDENT_CANDIDATE_ID
            ),
            runtime_bundle_hash="c0" * 32,
            ctx="test",
        )
        assert reset_mount.memory_mode != persistent.memory_mode
        assert reset_mount.profile_id != persistent.profile_id
        assert reset_mount.mount_hash != persistent.mount_hash
