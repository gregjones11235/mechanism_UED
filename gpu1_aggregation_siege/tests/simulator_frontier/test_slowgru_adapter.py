"""Stage 3 tests (E3): SlowGRU StudentAdapter contract + fail-closed gates.

Hermetic contract tests: the tests here use mock/fixtures and NEVER load the
real 98304 pkl (that is the object-check-only script's job).  Tests cover:

  (a) SlowGRU candidate resolves (profile -> adapter -> handle via mock);
  (b) profile fields correct (obs 8335/action 43/PERSISTENT/all SHAs match);
  (c) wrong checkpoint hash -> fail-closed;
  (d) wrong memory shape -> fail-closed;
  (e) RMT16/SlowGRU cross-mismatch rejection (RMT16 profile + SlowGRU adapter
      -> fail; SLOWGRU profile + RMT16 adapter -> fail).
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from dicode.student_adapters.checkpoint_codec import FORMAT_BAKEOFF_PKL, FORMAT_CC2_PKL
from dicode.student_adapters.protocol import MemoryFieldSpec, StudentAdapter
from dicode.student_adapters.registry import (
    StudentAdapterRegistry,
    StudentProfile,
    default_profile_dir,
    load_student_profile,
)
from dicode.student_adapters.slowgru_adapter import (
    MEMORY_FIELD_KEYS,
    SlowGRUMountError,
    SlowGRUStudentAdapter,
    _flat_to_nested,
    _nested_to_flat,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- profile fixtures ---------------------------------------------------------

def _slowgru_profile_for(params_sha: str, file_sha: str) -> StudentProfile:
    return StudentProfile(
        profile_name="slowgru_persistent_98304",
        candidate_id="SLOWGRU_PERSISTENT_CANONICAL_98304",
        architecture_family="SLOWGRU",
        checkpoint_format=FORMAT_BAKEOFF_PKL,
        global_step=98304,
        total_env_steps=98304,
        params_sha256=params_sha,
        source_commit="57b6925e8834a86742afbb445a83abfd3544a3db",
        observation_shape=(8335,),
        action_count=43,
        memory_mode="PERSISTENT",
        memory_fields={
            "memories": MemoryFieldSpec(shape=(None, 128, 2, 256), dtype="float32"),
            "memories_mask": MemoryFieldSpec(shape=(None, 8, 1, 129), dtype="bool"),
            "memories_mask_idx": MemoryFieldSpec(shape=(None,), dtype="int32"),
            "longstate.h": MemoryFieldSpec(shape=(None, 256), dtype="float32"),
            "longstate.buf": MemoryFieldSpec(shape=(None, 32, 256), dtype="float32"),
            "longstate.count": MemoryFieldSpec(shape=(None,), dtype="int32"),
        },
        notes={
            "carry_mode": "persistent",
            "checkpoint_file_sha256": file_sha,
            "network_src_sha256": "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b",
            "canonical_trainer_src_sha256": "7918333c63bdb6c8917bf423dfb8484942fb46edc6a7c8fa7e36c769cada2545",
            "formal_ranking": "INCONCLUSIVE_FULL_TIE",
            "provisional_rank": 1,
        },
    )


# --- (a) profile loading + candidate resolution --------------------------------

class TestSlowGRUProfileLoading:
    def test_profile_loads_from_yaml(self):
        profile = load_student_profile(
            default_profile_dir() / "slowgru_persistent_98304.yaml")
        assert profile.candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304"
        assert profile.architecture_family == "SLOWGRU"
        assert profile.checkpoint_format == FORMAT_BAKEOFF_PKL
        assert profile.global_step == 98304
        assert profile.total_env_steps == 98304
        assert profile.observation_shape == (8335,)
        assert profile.action_count == 43
        assert profile.memory_mode == "PERSISTENT"
        ident = profile.expected_identity()  # must not raise
        assert ident.params_sha256 == profile.params_sha256
        assert ident.memory_spec_hash == profile.memory_spec().spec_hash()

    def test_profile_fields_correct(self):
        profile = load_student_profile(
            default_profile_dir() / "slowgru_persistent_98304.yaml")
        assert profile.params_sha256 == "99d734b48acfd3499e5b836c7f632a52b1d17a732c3764a24c1935fd82a77ecc"
        assert profile.source_commit == "57b6925e8834a86742afbb445a83abfd3544a3db"
        assert profile.notes["checkpoint_file_sha256"] == (
            "0bc92c9ee28684ba507d6d6d728110000f11d7115126fbaf9137b1f8390a9c47")
        assert profile.notes["network_src_sha256"] == (
            "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b")
        assert profile.notes["canonical_trainer_src_sha256"] == (
            "7918333c63bdb6c8917bf423dfb8484942fb46edc6a7c8fa7e36c769cada2545")
        assert profile.notes["carry_mode"] == "persistent"
        assert profile.notes["formal_ranking"] == "INCONCLUSIVE_FULL_TIE"
        assert profile.notes["formal_winner"] is None
        assert profile.notes["provisional_rank"] == 1
        # Memory fields
        spec = profile.memory_spec()
        assert "memories" in spec.fields
        assert "memories_mask" in spec.fields
        assert "memories_mask_idx" in spec.fields
        assert "longstate.h" in spec.fields
        assert "longstate.buf" in spec.fields
        assert "longstate.count" in spec.fields
        assert spec.mode == "PERSISTENT"

    def test_registry_resolves_slowgru_profile(self):
        profile = load_student_profile(
            default_profile_dir() / "slowgru_persistent_98304.yaml")
        reg = StudentAdapterRegistry()
        reg.register(profile.candidate_id,
                     lambda p: SlowGRUStudentAdapter(
                         p, slowgru_runtime_path="/fake",
                         checkpoint_contract_path="/fake/contract.json",
                         expected_network_src_sha256="b" * 64,
                         expected_trainer_src_sha256="c" * 64))
        adapter = reg.resolve(profile)
        assert isinstance(adapter, StudentAdapter)
        assert adapter.identity().params_sha256 == profile.params_sha256


# --- (b) adapter specs are correct ---------------------------------------------

class TestSlowGRUSpecs:
    def test_specs_are_correct(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        adapter = SlowGRUStudentAdapter(
            profile, slowgru_runtime_path="/fake",
            checkpoint_contract_path="/fake/contract.json",
            expected_network_src_sha256="c" * 64,
            expected_trainer_src_sha256="d" * 64)
        assert adapter.observation_spec().shape == (8335,)
        assert adapter.observation_spec().dtype == "float32"
        assert adapter.action_spec().count == 43
        assert adapter.memory_spec().mode == "PERSISTENT"
        assert adapter.memory_spec().field_names() == tuple(sorted(MEMORY_FIELD_KEYS))
        cs = adapter.checkpoint_spec()
        assert cs.format == FORMAT_BAKEOFF_PKL
        assert cs.params_sha256 == "a" * 64
        assert cs.source_commit == "57b6925e8834a86742afbb445a83abfd3544a3db"
        assert cs.contains_optimizer is False
        assert cs.contains_rng is False
        assert cs.contains_memory is False


# --- (c) constructor rejects wrong family -------------------------------------

class TestSlowGRUConstructorGates:
    def test_rejects_wrong_architecture_family(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        bad = replace(profile, architecture_family="RMT16")
        with pytest.raises(SlowGRUMountError):
            SlowGRUStudentAdapter(bad, slowgru_runtime_path="/fake",
                                  checkpoint_contract_path="/fake/contract.json",
                                  expected_network_src_sha256="c" * 64,
                                  expected_trainer_src_sha256="d" * 64)

    def test_rejects_wrong_checkpoint_format(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        bad = replace(profile, checkpoint_format=FORMAT_CC2_PKL)
        with pytest.raises(SlowGRUMountError):
            SlowGRUStudentAdapter(bad, slowgru_runtime_path="/fake",
                                  checkpoint_contract_path="/fake/contract.json",
                                  expected_network_src_sha256="c" * 64,
                                  expected_trainer_src_sha256="d" * 64)

    def test_rejects_wrong_memory_mode(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        bad = replace(profile, memory_mode="RESET128")
        with pytest.raises(SlowGRUMountError):
            SlowGRUStudentAdapter(bad, slowgru_runtime_path="/fake",
                                  checkpoint_contract_path="/fake/contract.json",
                                  expected_network_src_sha256="c" * 64,
                                  expected_trainer_src_sha256="d" * 64)

    def test_rejects_missing_carry_mode_note(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        bad = replace(profile, notes={"checkpoint_file_sha256": "b" * 64})
        with pytest.raises(SlowGRUMountError):
            SlowGRUStudentAdapter(bad, slowgru_runtime_path="/fake",
                                  checkpoint_contract_path="/fake/contract.json",
                                  expected_network_src_sha256="c" * 64,
                                  expected_trainer_src_sha256="d" * 64)

    def test_rejects_wrong_carry_mode(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        bad = replace(profile, notes={
            "carry_mode": "reset128",
            "checkpoint_file_sha256": "b" * 64,
        })
        with pytest.raises(SlowGRUMountError):
            SlowGRUStudentAdapter(bad, slowgru_runtime_path="/fake",
                                  checkpoint_contract_path="/fake/contract.json",
                                  expected_network_src_sha256="c" * 64,
                                  expected_trainer_src_sha256="d" * 64)


# --- (c) hash mismatch -> fail-closed -----------------------------------------

class TestSlowGRUHashGates:
    def test_identity_mismatch_raises(self, tmp_path):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        adapter = SlowGRUStudentAdapter(
            profile, slowgru_runtime_path="/fake",
            checkpoint_contract_path="/fake/contract.json",
            expected_network_src_sha256="c" * 64,
            expected_trainer_src_sha256="d" * 64)
        foreign = replace(profile.expected_identity(), params_sha256="f" * 64)
        with pytest.raises(SlowGRUMountError):
            adapter.load_full_state("/fake/ckpt.pkl", foreign)

    def test_load_without_runtime_paths_fails_closed(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        adapter = SlowGRUStudentAdapter(profile)  # no runtime paths configured
        with pytest.raises(SlowGRUMountError):
            adapter.load_full_state("/fake/ckpt.pkl", profile.expected_identity())


# --- (d) memory contract validation -------------------------------------------

class TestSlowGRUMemoryContract:
    def test_memory_field_keys(self):
        assert tuple(sorted(MEMORY_FIELD_KEYS)) == tuple(sorted([
            "memories", "memories_mask", "memories_mask_idx",
            "longstate.h", "longstate.buf", "longstate.count",
        ]))

    def test_flat_to_nested_roundtrip(self):
        """Test that flat <-> nested conversion is lossless for public fields."""
        flat = {
            "memories": np.zeros((1, 128, 2, 256), dtype=np.float32),
            "memories_mask": np.zeros((1, 8, 1, 129), dtype=np.bool_),
            "memories_mask_idx": np.zeros((1,), dtype=np.int32) + 129,
            "longstate.h": np.zeros((1, 256), dtype=np.float32),
            "longstate.buf": np.zeros((1, 32, 256), dtype=np.float32),
            "longstate.count": np.zeros((1,), dtype=np.int32),
        }
        nested = _flat_to_nested(flat)
        assert "longstate" in nested
        assert isinstance(nested["longstate"], dict)
        assert nested["longstate"]["h"].shape == (1, 256)
        assert nested["longstate"]["buf"].shape == (1, 32, 256)
        assert nested["longstate"]["count"].shape == (1,)
        # Round-trip
        flat2 = _nested_to_flat(nested)
        for key in MEMORY_FIELD_KEYS:
            assert key in flat2
            assert np.array_equal(flat2[key], flat[key])

    def test_flat_to_nested_preserves_internal_keys(self):
        flat = {
            "memories": np.zeros((1, 128, 2, 256), dtype=np.float32),
            "memories_mask": np.zeros((1, 8, 1, 129), dtype=np.bool_),
            "memories_mask_idx": np.zeros((1,), dtype=np.int32),
            "longstate.h": np.zeros((1, 256), dtype=np.float32),
            "longstate.buf": np.zeros((1, 32, 256), dtype=np.float32),
            "longstate.count": np.zeros((1,), dtype=np.int32),
            "true_done": np.zeros((1,), dtype=np.bool_),
            "step_idx": 5,
        }
        nested = _flat_to_nested(flat)
        assert nested["true_done"] is not None
        assert nested["step_idx"] == 5

    def test_validate_memory_negatives(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        adapter = SlowGRUStudentAdapter(
            profile, slowgru_runtime_path="/fake",
            checkpoint_contract_path="/fake/contract.json",
            expected_network_src_sha256="c" * 64,
            expected_trainer_src_sha256="d" * 64)
        # Not loaded yet
        assert adapter.validate_memory({}, 1)["ok"] is False
        # Missing fields
        bad = {"memories": np.zeros((1, 128, 2, 256), dtype=np.float32)}
        assert adapter.validate_memory(bad, 1)["ok"] is False
        # Wrong shape
        bad_shape = {
            "memories": np.zeros((1, 64, 2, 256), dtype=np.float32),
            "memories_mask": np.zeros((1, 8, 1, 129), dtype=np.bool_),
            "memories_mask_idx": np.zeros((1,), dtype=np.int32),
            "longstate.h": np.zeros((1, 256), dtype=np.float32),
            "longstate.buf": np.zeros((1, 32, 256), dtype=np.float32),
            "longstate.count": np.zeros((1,), dtype=np.int32),
        }
        assert adapter.validate_memory(bad_shape, 1)["ok"] is False
        # Wrong dtype
        bad_dtype = {
            "memories": np.zeros((1, 128, 2, 256), dtype=np.float64),
            "memories_mask": np.zeros((1, 8, 1, 129), dtype=np.bool_),
            "memories_mask_idx": np.zeros((1,), dtype=np.int32),
            "longstate.h": np.zeros((1, 256), dtype=np.float32),
            "longstate.buf": np.zeros((1, 32, 256), dtype=np.float32),
            "longstate.count": np.zeros((1,), dtype=np.int32),
        }
        assert adapter.validate_memory(bad_dtype, 1)["ok"] is False


# --- (e) cross-mismatch: RMT16 profile + SlowGRU adapter -> fail --------------

class TestCrossMismatchRejection:
    def test_rmt16_profile_rejected_by_slowgru_adapter(self):
        rmt16_profile = load_student_profile(
            default_profile_dir() / "rmt16_persistent_98304.yaml")
        with pytest.raises(SlowGRUMountError):
            SlowGRUStudentAdapter(rmt16_profile,
                                  slowgru_runtime_path="/fake",
                                  checkpoint_contract_path="/fake/contract.json",
                                  expected_network_src_sha256="c" * 64,
                                  expected_trainer_src_sha256="d" * 64)

    def test_slowgru_profile_rejected_by_rmt16_adapter(self):
        from dicode.student_adapters.rmt16_adapter import RMT16MountError, RMT16StudentAdapter
        slowgru_profile = load_student_profile(
            default_profile_dir() / "slowgru_persistent_98304.yaml")
        with pytest.raises(RMT16MountError):
            RMT16StudentAdapter(slowgru_profile,
                                driver_source_path="/fake/driver.py",
                                expected_driver_sha256="e" * 64)

    def test_slowgru_adapter_must_be_student_adapter(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        adapter = SlowGRUStudentAdapter(
            profile, slowgru_runtime_path="/fake",
            checkpoint_contract_path="/fake/contract.json",
            expected_network_src_sha256="c" * 64,
            expected_trainer_src_sha256="d" * 64)
        assert isinstance(adapter, StudentAdapter)

    def test_registry_mismatch_rejected(self):
        """Registering SlowGRU adapter for RMT16 profile must fail at resolve."""
        from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter
        slowgru_profile = load_student_profile(
            default_profile_dir() / "slowgru_persistent_98304.yaml")
        reg = StudentAdapterRegistry()
        # Register RMT16 adapter for the SlowGRU candidate_id
        reg.register(slowgru_profile.candidate_id,
                     lambda p: RMT16StudentAdapter(
                         p, driver_source_path="/fake/driver.py",
                         expected_driver_sha256="e" * 64))
        # resolve should still work (it just returns the factory result)
        # but the adapter's constructor should reject the profile
        with pytest.raises(Exception):
            reg.resolve(slowgru_profile)


# --- training surface is PENDING -----------------------------------------------

class TestTrainingSurfacePending:
    def test_save_and_restore_raise_not_implemented(self):
        profile = _slowgru_profile_for("a" * 64, "b" * 64)
        adapter = SlowGRUStudentAdapter(
            profile, slowgru_runtime_path="/fake",
            checkpoint_contract_path="/fake/contract.json",
            expected_network_src_sha256="c" * 64,
            expected_trainer_src_sha256="d" * 64)
        with pytest.raises(NotImplementedError):
            adapter.save_full_state("x.pkl", {}, {})
        with pytest.raises(NotImplementedError):
            adapter.restore_full_state("/fake/ckpt.pkl")