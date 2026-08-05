"""CC2-Student tests: read-only mount vs training runtime capability.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

The shared RMT16 adapter proves read-only mount capability
(checkpoint loadable, forward executable, memory progression
verifiable, probe executable) — it can NEVER prove optimizer state /
continuable training / save_full_state / restore_full_state. Training
happens ONLY through the director-injected canonical DiCode runtimes.
"""
from types import SimpleNamespace

from dicode.teachers.e1_formal import student_contract as SC

_PERSISTENT = SC.PERSISTENT_STUDENT_CANDIDATE_ID


def _bundle(candidate_id=_PERSISTENT):
    return SimpleNamespace(
        bundle_hash="c0" * 32,
        student={
            "candidate_id": candidate_id,
            "profile": SC.STUDENT_PROFILE_BY_CANDIDATE[candidate_id],
            "memory_mode": (
                SC.STUDENT_MEMORY_MODE_BY_CANDIDATE[candidate_id]
            ),
            "expected_params_sha256": "aa" * 32,
        },
    )


def _mount(shared=False, training=False):
    return SC.mount_student_from_director_bundle(
        bundle=_bundle(),
        director_selected_candidate_id=None,
        ctx="test",
        shared_registry_bound=shared,
        training_runtime_bound=training,
    )


class TestReadOnlyMountNotTrainingReady:
    def test_shared_registry_unbound_is_honest(self):
        mount = _mount(shared=False, training=False)
        assert mount.read_only_ready is False
        assert mount.training_ready is False
        assert mount.capability_state == SC.STUDENT_SHARED_REGISTRY_UNBOUND

    def test_shared_registry_bound_is_read_only_mount_ready_only(self):
        mount = _mount(shared=True, training=False)
        assert mount.read_only_ready is True
        assert mount.training_ready is False  # NEVER implied
        assert mount.capability_state == SC.STUDENT_READ_ONLY_MOUNT_READY

    def test_training_ready_requires_the_canonical_training_runtime(self):
        mount = _mount(shared=True, training=True)
        assert mount.training_ready is True
        assert mount.capability_state == SC.STUDENT_TRAINING_RUNTIME_READY

    def test_read_only_mount_never_impersonates_training(self):
        # the read-only mount exposes no training surface; training only
        # flows through the director-injected CanonicalDiCodeOneUpdateRuntime
        # + CanonicalDiCodeRunStateCheckpoint (never the adapter)
        mount = _mount(shared=True, training=False)
        assert mount.capability_state == SC.STUDENT_READ_ONLY_MOUNT_READY
        assert mount.capability_state != SC.STUDENT_TRAINING_RUNTIME_READY
