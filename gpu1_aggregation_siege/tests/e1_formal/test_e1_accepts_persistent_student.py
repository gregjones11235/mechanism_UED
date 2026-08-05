"""CC2-Student tests: the Persistent Student is an allowed selection.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.
"""
from dicode.teachers.e1_formal import student_contract as SC

_PERSISTENT = SC.PERSISTENT_STUDENT_CANDIDATE_ID


def _mount(**overrides):
    contract = SC.build_synthetic_student_contract(_PERSISTENT, "test")
    kwargs = dict(
        contract=contract,
        director_selected_candidate_id=_PERSISTENT,
        runtime_bundle_hash="c0" * 32,
        ctx="test",
    )
    kwargs.update(overrides)
    return SC.consume_e1_student_contract(**kwargs)


class TestAcceptsPersistentStudent:
    def test_persistent_is_in_the_allowed_set(self):
        assert _PERSISTENT in SC.ALLOWED_STUDENT_CANDIDATE_IDS
        assert len(SC.ALLOWED_STUDENT_CANDIDATE_IDS) == 2

    def test_persistent_mounts_with_its_own_profile(self):
        mount = _mount()
        assert mount.candidate_id == _PERSISTENT
        assert mount.profile_id == "rmt16_persistent_98304"
        assert mount.memory_mode == "PERSISTENT"
        assert mount.params_sha256 == "aa" * 32
        assert mount.adapter_id == "rmt16_persistent_98304"
        assert mount.runtime_bundle_hash == "c0" * 32
        assert len(mount.mount_hash) == 64

    def test_persistent_mount_carries_no_training_capability(self):
        # the shared registry is absent this round => read-only mount
        # is NOT ready; training is NEVER implied
        mount = _mount()
        assert mount.training_ready is False
        assert mount.capability_state == SC.STUDENT_SHARED_REGISTRY_UNBOUND
