# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-5): a REFERENCE_POLICY branch may only consume the
Reference policy's OWN memory surface — Student memory is NEVER substituted
into a Reference branch.  The reachable gates are pinned here: a mounted
Reference without its identity/checkpoint binding is refused at construction
time, and a Reference branch without a bound Reference memory surface fails
closed instead of falling back to the Student surface.
"""

import pytest

from dicode.simulator_frontier.branch_search_runner import (
    BranchSearchBlockedError,
    BranchSearchRunConfig,
    BranchSearchRunner,
    SEARCH_SOURCE_REFERENCE_POLICY,
)
from dicode.simulator_frontier.memory_modes import MemoryRestoreRequest
from dicode.student_adapters.fake import FakeStudentAdapter

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _student() -> FakeStudentAdapter:
    return FakeStudentAdapter(candidate_id="FAKE_SEARCH_CONTRACT_ONLY")


def _config() -> BranchSearchRunConfig:
    return BranchSearchRunConfig(
        state_id="s",
        horizon=4,
        requested_n=2,
        memory_mode="SAVED_POLICY_MEMORY",
        memory_request=MemoryRestoreRequest(
            mode="SAVED_POLICY_MEMORY",
            policy_architecture_id="FAKE",
            checkpoint_id="c" * 64),
        success_predicate=lambda _flat: True,
        progress_fn=lambda _flat: 0.5,
    )


class TestReferenceMountBinding:
    def test_reference_without_checkpoint_identity_is_refused(self):
        # A mounted Reference must carry its checkpoint id from construction
        # time — an anonymous Reference can never produce production evidence.
        with pytest.raises(BranchSearchBlockedError):
            BranchSearchRunner(
                student=_student(), student_params=_student()._params,
                step_fn=lambda *a: None, env_params={}, template={},
                observe_fn=lambda s: None,
                capture_student_id="FAKE_CAPTURE_CONTRACT_ONLY",
                search_student_id="FAKE_SEARCH_CONTRACT_ONLY",
                train_student_id="FAKE_TRAIN_CONTRACT_ONLY",
                reference_student=_student(), reference_params={},
                reference_checkpoint_id="",
            )

    def test_reference_with_checkpoint_identity_mounts(self):
        runner = BranchSearchRunner(
            student=_student(), student_params=_student()._params,
            step_fn=lambda *a: None, env_params={}, template={},
            observe_fn=lambda s: None,
            capture_student_id="FAKE_CAPTURE_CONTRACT_ONLY",
            search_student_id="FAKE_SEARCH_CONTRACT_ONLY",
            train_student_id="FAKE_TRAIN_CONTRACT_ONLY",
            reference_student=_student(), reference_params={},
            reference_checkpoint_id="reference-ckpt-001",
        )
        assert runner is not None


class TestReferenceSurfaceIsolation:
    def test_reference_branch_without_reference_surface_blocks(self):
        # No Reference student mounted: a REFERENCE_POLICY memory request must
        # fail closed BEFORE touching any Student surface — Student memory is
        # never substituted into a Reference branch.
        runner = BranchSearchRunner(
            student=_student(), student_params=_student()._params,
            step_fn=lambda *a: None, env_params={}, template={},
            observe_fn=lambda s: None,
            capture_student_id="FAKE_CAPTURE_CONTRACT_ONLY",
            search_student_id="FAKE_SEARCH_CONTRACT_ONLY",
            train_student_id="FAKE_TRAIN_CONTRACT_ONLY",
        )
        with pytest.raises(BranchSearchBlockedError):
            runner._prepare_memory(None, None, _config(),
                                   source=SEARCH_SOURCE_REFERENCE_POLICY)

    def test_reference_branch_without_reference_artifact_blocks(self):
        # Reference mounted but NO reference memory artifact is bound: the
        # Reference surface is unbound, so the branch blocks rather than
        # reading Student memory.
        reference = _student()
        runner = BranchSearchRunner(
            student=_student(), student_params=_student()._params,
            step_fn=lambda *a: None, env_params={}, template={},
            observe_fn=lambda s: None,
            capture_student_id="FAKE_CAPTURE_CONTRACT_ONLY",
            search_student_id="FAKE_SEARCH_CONTRACT_ONLY",
            train_student_id="FAKE_TRAIN_CONTRACT_ONLY",
            reference_student=reference, reference_params={},
            reference_checkpoint_id="reference-ckpt-001",
        )
        with pytest.raises(BranchSearchBlockedError):
            runner._prepare_memory(None, None, _config(),
                                   source=SEARCH_SOURCE_REFERENCE_POLICY)
