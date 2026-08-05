"""CC2 follow-up P0-10/P0-12 tests: ONE Student checkpoint from probe
to update.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
all fixtures are synthetic identity hashes; no real probe or update
runs here.

Covered negative matrix:
* probe checkpoint swap                     -> E1_STUDENT_CHECKPOINT_SWAPPED
* probe identity swap                       -> E1_STUDENT_IDENTITY_SWAPPED
* update input != probe checkpoint          -> E1_UPDATE_STUDENT_MISMATCH
* update output == input                    -> E1_UPDATE_NO_PROGRESS
* re-init input claim (global steps 0)      -> E1_STUDENT_REINIT_CLAIM
* bad types                                 -> E1_STUDENT_CONTINUITY_BAD
"""
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import student_continuity as SC

_STUDENT_IDENTITY = "11" * 32
_STUDENT_CHECKPOINT = "12" * 32
_WINDOW_HASH = "e" * 64
_INPUT_STEPS = 4096


def _binding(**overrides):
    kwargs = dict(
        student_identity_hash=_STUDENT_IDENTITY,
        student_checkpoint_hash=_STUDENT_CHECKPOINT,
        input_global_env_steps=_INPUT_STEPS,
        window_hash=_WINDOW_HASH,
    )
    kwargs.update(overrides)
    return SC.open_student_binding(**kwargs)


def _probe(checkpoint=_STUDENT_CHECKPOINT, identity=_STUDENT_IDENTITY):
    return SimpleNamespace(
        result_id="test-only-probe-1",
        student_checkpoint_hash=checkpoint,
        student_identity_hash=identity,
    )


class TestOpenStudentBinding:
    def test_binds_identity_checkpoint_and_steps(self):
        binding = _binding()
        assert binding.student_identity_hash == _STUDENT_IDENTITY
        assert binding.student_checkpoint_hash == _STUDENT_CHECKPOINT
        assert binding.input_global_env_steps == _INPUT_STEPS
        assert binding.window_hash == _WINDOW_HASH
        assert len(binding.binding_hash) == 64

    def test_reinit_input_claim_refused(self):
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            _binding(input_global_env_steps=0)
        assert excinfo.value.code == SC.E1_STUDENT_REINIT_CLAIM
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            _binding(input_global_env_steps=-3)
        assert excinfo.value.code == SC.E1_STUDENT_REINIT_CLAIM

    def test_bad_hash_refused(self):
        for field in (
            "student_identity_hash",
            "student_checkpoint_hash",
            "window_hash",
        ):
            with pytest.raises(SC.StudentContinuityError) as excinfo:
                _binding(**{field: "short"})
            assert excinfo.value.code == SC.E1_STUDENT_CONTINUITY_BAD


class TestProbeBinding:
    def test_matching_probe_pool_passes(self):
        binding = _binding()
        SC.assert_probe_student_binding(
            binding, (_probe(), _probe()), "test"
        )

    def test_swapped_probe_checkpoint_refused(self):
        binding = _binding()
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            SC.assert_probe_student_binding(
                binding, (_probe(checkpoint="aa" * 32),), "test"
            )
        assert excinfo.value.code == SC.E1_STUDENT_CHECKPOINT_SWAPPED

    def test_swapped_probe_identity_refused(self):
        binding = _binding()
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            SC.assert_probe_student_binding(
                binding, (_probe(identity="bb" * 32),), "test"
            )
        assert excinfo.value.code == SC.E1_STUDENT_IDENTITY_SWAPPED

    def test_empty_pool_refused(self):
        binding = _binding()
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            SC.assert_probe_student_binding(binding, (), "test")
        assert excinfo.value.code == SC.E1_STUDENT_CONTINUITY_BAD

    def test_bad_binding_type_refused(self):
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            SC.assert_probe_student_binding(
                {"binding": "summary"}, (_probe(),), "test"
            )
        assert excinfo.value.code == SC.E1_STUDENT_CONTINUITY_BAD


class TestUpdateInputBinding:
    def test_matching_update_input_binds(self):
        binding = _binding()
        bound = SC.bind_update_input(
            binding,
            update_input_checkpoint_hash=_STUDENT_CHECKPOINT,
            ctx="test",
        )
        assert bound.input_checkpoint_hash == _STUDENT_CHECKPOINT
        assert bound.student_identity_hash == _STUDENT_IDENTITY
        assert bound.input_global_env_steps == _INPUT_STEPS
        assert bound.binding_hash == binding.binding_hash

    def test_different_update_input_refused(self):
        binding = _binding()
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            SC.bind_update_input(
                binding,
                update_input_checkpoint_hash="cc" * 32,
                ctx="test",
            )
        assert excinfo.value.code == SC.E1_UPDATE_STUDENT_MISMATCH

    def test_identical_output_refused(self):
        binding = _binding()
        bound = SC.bind_update_input(
            binding,
            update_input_checkpoint_hash=_STUDENT_CHECKPOINT,
            ctx="test",
        )
        with pytest.raises(SC.StudentContinuityError) as excinfo:
            SC.assert_update_output_differs(
                bound,
                update_output_checkpoint_hash=_STUDENT_CHECKPOINT,
                ctx="test",
            )
        assert excinfo.value.code == SC.E1_UPDATE_NO_PROGRESS

    def test_different_output_passes(self):
        binding = _binding()
        bound = SC.bind_update_input(
            binding,
            update_input_checkpoint_hash=_STUDENT_CHECKPOINT,
            ctx="test",
        )
        SC.assert_update_output_differs(
            bound,
            update_output_checkpoint_hash="dd" * 32,
            ctx="test",
        )
