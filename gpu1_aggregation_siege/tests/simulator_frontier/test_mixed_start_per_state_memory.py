# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-11): the mixed-start rollout no longer shares ONE
window-level memory object across episodes.  Every frontier episode gets a
FRESH memory instance (artifact re-read for SAVED_POLICY_MEMORY, fresh
burn-in of the restored bundle's OWN history reference for HISTORY_BURN_IN).
The reachable per-episode helpers are pinned here.
"""

import hashlib

import numpy as np
import pytest

from dicode.simulator_frontier.branch_search_runner import MemoryArtifactRef
from dicode.simulator_frontier.e3_window import (
    E3WindowConfig,
    _fresh_window_memory,
    _verify_window_memory_source,
)
from dicode.simulator_frontier.errors import ProductionBlockedError
from dicode.simulator_frontier.memory_modes import MemoryRestoreMode
from dicode.student_adapters.fake import FakeStudentAdapter, MEMORY_DIM

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _student() -> FakeStudentAdapter:
    return FakeStudentAdapter(candidate_id="FAKE_SEARCH_CONTRACT_ONLY")


def _fresh_memory() -> dict[str, np.ndarray]:
    return {"h": np.zeros((1, MEMORY_DIM), dtype=np.float32)}


def _artifact(tmp_path) -> MemoryArtifactRef:
    path = tmp_path / "policy_memory.npz"
    path.write_bytes(b"SYNTHETIC_MEMORY_ARTIFACT_NOT_SCIENTIFIC_CONTENT")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    student = _student()
    return MemoryArtifactRef(
        path=str(path), sha256=sha,
        memory_spec_hash=student.memory_spec().spec_hash(),
        student_identity_hash=student.identity().identity_hash())


class TestPerStateFreshMemory:
    def test_saved_policy_memory_is_fresh_per_episode(self, tmp_path):
        student = _student()
        artifact = _artifact(tmp_path)
        calls = []

        def loader(_artifact):
            memory = _fresh_memory()
            calls.append(memory)
            return memory

        config = E3WindowConfig(
            memory_mode=MemoryRestoreMode.SAVED_POLICY_MEMORY.value,
            memory_artifact=artifact,
            memory_loader=loader,
        )
        status = _verify_window_memory_source(config, student,
                                              MemoryRestoreMode.SAVED_POLICY_MEMORY)
        assert status == "SAVED_POLICY_MEMORY_VERIFIED"
        m1 = _fresh_window_memory(config, student, None,
                                  MemoryRestoreMode.SAVED_POLICY_MEMORY)
        m2 = _fresh_window_memory(config, student, None,
                                  MemoryRestoreMode.SAVED_POLICY_MEMORY)
        # FRESH instances per episode: distinct objects — never one shared
        # window-level memory object.
        assert m1 is not m2 and calls[0] is not calls[1]

    def test_history_burn_in_uses_the_restored_bundles_own_reference(self):
        student = _student()
        burned = []
        config = E3WindowConfig(
            memory_mode=MemoryRestoreMode.HISTORY_BURN_IN.value,
            history_artifact_ref="FALLBACK_REF",
            burn_in_executor=lambda ref: burned.append(ref) or _fresh_memory(),
        )
        bundle = type("Bundle", (), {"history_reference": "STATE_OWN_HISTORY"})()
        m1 = _fresh_window_memory(config, student, bundle,
                                  MemoryRestoreMode.HISTORY_BURN_IN)
        m2 = _fresh_window_memory(config, student, bundle,
                                  MemoryRestoreMode.HISTORY_BURN_IN)
        assert burned == ["STATE_OWN_HISTORY", "STATE_OWN_HISTORY"]
        assert m1 is not m2

    def test_history_burn_in_without_bundle_reference_blocks(self):
        student = _student()
        config = E3WindowConfig(
            memory_mode=MemoryRestoreMode.HISTORY_BURN_IN.value,
            history_artifact_ref="",
            burn_in_executor=lambda ref: _fresh_memory(),
        )
        with pytest.raises(ProductionBlockedError):
            _fresh_window_memory(config, student, None,
                                 MemoryRestoreMode.HISTORY_BURN_IN)

    def test_zero_memory_is_never_a_production_source(self):
        student = _student()
        config = E3WindowConfig(memory_mode=MemoryRestoreMode.ZERO_MEMORY.value)
        with pytest.raises(ProductionBlockedError):
            _verify_window_memory_source(config, student,
                                         MemoryRestoreMode.ZERO_MEMORY)


class TestArtifactIntegrity:
    def test_artifact_sha256_mismatch_blocks(self, tmp_path):
        student = _student()
        artifact = _artifact(tmp_path)
        # Corrupt the file after the artifact was bound.
        (tmp_path / "policy_memory.npz").write_bytes(b"TAMPERED")
        config = E3WindowConfig(
            memory_mode=MemoryRestoreMode.SAVED_POLICY_MEMORY.value,
            memory_artifact=artifact,
            memory_loader=lambda _artifact: _fresh_memory(),
        )
        with pytest.raises(ProductionBlockedError):
            _verify_window_memory_source(config, student,
                                         MemoryRestoreMode.SAVED_POLICY_MEMORY)

    def test_cross_policy_artifact_blocks(self, tmp_path):
        # An artifact bound to a DIFFERENT adapter identity/spec must block.
        other = FakeStudentAdapter(candidate_id="FAKE_OTHER_CONTRACT_ONLY")
        artifact = _artifact(tmp_path)
        foreign = MemoryArtifactRef(
            path=artifact.path, sha256=artifact.sha256,
            memory_spec_hash=other.memory_spec().spec_hash(),
            student_identity_hash=other.identity().identity_hash())
        config = E3WindowConfig(
            memory_mode=MemoryRestoreMode.SAVED_POLICY_MEMORY.value,
            memory_artifact=foreign,
            memory_loader=lambda _artifact: _fresh_memory(),
        )
        with pytest.raises(ProductionBlockedError):
            _verify_window_memory_source(config, _student(),
                                         MemoryRestoreMode.SAVED_POLICY_MEMORY)
