"""BLOCKER-5: RunState architecture memory must serialize/restore REAL values.

The RunState checkpoint MUST carry the real post-session architecture memory
(memories / mem_mask / mem_idx / rmt.* / longstate.* / true_done) — never
shapes, never zeros.  These tests pin the contract for both backends:

  * serialize_memory_state returns the REAL leaf values (a ``values`` map of
    host arrays), NOT a shape-only report.
  * restore_memory_state returns the ORIGINAL values (value-exact round trip,
    dtype + shape exact), never a zero-init fallback.
  * architecture-family mismatch and missing-values on restore fail closed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dicode.training_backend_rmt16 import RMT16TrainingBackend
from dicode.training_backend_slowgru import SlowGRUTrainingBackend

# Real server assets (the SlowGRU backend loads slowgru_runtime in
# init_runner_memory).  Tests are skipped where the runtime is not present.
SLOWGRU_RUNTIME_PATH = "/home/oseasy/student_pool_v1/cc3/slowgru_runtime"
SLOWGRU_CHECKPOINT_CONTRACT_PATH = (
    "/home/oseasy/student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304/"
    "checkpoint_contract.json")
SLOWGRU_CHECKPOINT_PATH = (
    "/home/oseasy/student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304/"
    "ckpt/98304/full_state.pkl")


def _make_rmt16_backend():
    return RMT16TrainingBackend(
        candidate_id="PERSISTENT_RMT16_TEST",
        action_dim=43,
        window_mem=128,
        num_steps=128,
    )


def _make_slowgru_backend():
    if not os.path.isdir(SLOWGRU_RUNTIME_PATH):
        pytest.skip("slowgru_runtime not present on this host")
    return SlowGRUTrainingBackend(
        candidate_id="SLOWGRU_TEST",
        slowgru_runtime_path=SLOWGRU_RUNTIME_PATH,
        checkpoint_contract_path=SLOWGRU_CHECKPOINT_CONTRACT_PATH,
        checkpoint_path=SLOWGRU_CHECKPOINT_PATH,
    )


def _fill_memory(backend, num_envs: int, seed: int = 0):
    """Return a memory dict with non-zero, distinguishable values."""
    mem = backend.init_runner_memory(num_envs)
    rng = np.random.default_rng(seed)
    for key, val in mem.items():
        arr = np.asarray(val)
        if arr.dtype == np.bool_:
            mem[key] = np.asarray(rng.integers(0, 2, size=arr.shape),
                                  dtype=np.bool_)
        elif np.issubdtype(arr.dtype, np.integer):
            mem[key] = np.asarray(rng.integers(1, 50, size=arr.shape),
                                  dtype=arr.dtype)
        else:
            mem[key] = np.asarray(rng.uniform(-3.0, 3.0, size=arr.shape),
                                  dtype=arr.dtype)
    return mem


def test_rmt16_serialize_returns_real_values_not_shapes():
    backend = _make_rmt16_backend()
    mem = _fill_memory(backend, num_envs=3)
    serialized = backend.serialize_memory_state(mem)
    assert serialized["architecture_family"] == "RMT16"
    # BLOCKER-5: the OLD shape-only keys must be gone.
    for shape_key in ("memories_shape", "mem_mask_shape", "mem_idx_shape",
                      "rmt.mem_tokens_shape", "rmt.seg_buf_shape",
                      "rmt.seg_count_shape"):
        assert shape_key not in serialized, \
            f"shape-only key {shape_key!r} still present (BLOCKER-5)"
    assert "values" in serialized
    required = ("memories", "mem_mask", "mem_idx", "rmt.mem_tokens",
                "rmt.seg_buf", "rmt.seg_count")
    assert set(required) <= set(serialized["values"])
    for key in required:
        val = np.asarray(serialized["values"][key])
        assert val.shape == np.asarray(mem[key]).shape
        assert val.size > 0


def test_slowgru_serialize_returns_real_values_not_shapes():
    backend = _make_slowgru_backend()
    mem = _fill_memory(backend, num_envs=3)
    serialized = backend.serialize_memory_state(mem)
    assert serialized["architecture_family"] == "SLOWGRU"
    for shape_key in ("memories_shape", "memories_mask_shape",
                      "memories_mask_idx_shape", "longstate.h_shape",
                      "longstate.buf_shape", "longstate.count_shape"):
        assert shape_key not in serialized, \
            f"shape-only key {shape_key!r} still present (BLOCKER-5)"
    assert "values" in serialized
    for key in mem:
        assert key in serialized["values"], f"missing memory field {key!r}"
        assert np.asarray(serialized["values"][key]).size > 0


def test_rmt16_restore_is_value_exact_not_zero_init():
    backend = _make_rmt16_backend()
    mem = _fill_memory(backend, num_envs=3, seed=7)
    serialized = backend.serialize_memory_state(mem)
    restored = backend.restore_memory_state(serialized)
    for key in mem:
        orig = np.asarray(mem[key])
        got = np.asarray(restored[key])
        assert got.shape == orig.shape, f"{key} shape"
        assert got.dtype == orig.dtype, f"{key} dtype"
        assert np.array_equal(got, orig), f"{key} value mismatch"
    # restore must NOT be the zero-init fallback.
    for key in ("memories", "mem_idx", "rmt.mem_tokens", "rmt.seg_buf"):
        assert np.any(np.asarray(restored[key]) != 0), \
            f"{key} restored as zeros (zero-init fallback is a lie)"


def test_slowgru_restore_is_value_exact_not_zero_init():
    backend = _make_slowgru_backend()
    mem = _fill_memory(backend, num_envs=3, seed=7)
    serialized = backend.serialize_memory_state(mem)
    restored = backend.restore_memory_state(serialized)
    for key in mem:
        orig = np.asarray(mem[key])
        got = np.asarray(restored[key])
        assert got.shape == orig.shape, f"{key} shape"
        assert got.dtype == orig.dtype, f"{key} dtype"
        assert np.array_equal(got, orig), f"{key} value mismatch"
    for key in ("memories", "mem_idx", "longstate.h", "longstate.buf"):
        assert np.any(np.asarray(restored[key]) != 0), \
            f"{key} restored as zeros (zero-init fallback is a lie)"


def test_rmt16_restore_rejects_wrong_architecture_family():
    backend = _make_rmt16_backend()
    serialized = {"architecture_family": "SLOWGRU", "values": {}}
    try:
        backend.restore_memory_state(serialized)
    except RuntimeError as exc:
        assert "architecture_family" in str(exc)
    else:
        raise AssertionError("restore with wrong family must fail closed")


def test_slowgru_restore_rejects_wrong_architecture_family():
    backend = _make_slowgru_backend()
    serialized = {"architecture_family": "RMT16", "values": {}}
    try:
        backend.restore_memory_state(serialized)
    except RuntimeError as exc:
        assert "architecture_family" in str(exc)
    else:
        raise AssertionError("restore with wrong family must fail closed")


def test_slowgru_restore_rejects_missing_values():
    backend = _make_slowgru_backend()
    try:
        backend.restore_memory_state({"architecture_family": "SLOWGRU"})
    except RuntimeError as exc:
        assert "values" in str(exc)
    else:
        raise AssertionError("restore without values must fail closed")


def test_rmt16_restore_rejects_missing_fields():
    backend = _make_rmt16_backend()
    serialized = {"architecture_family": "RMT16",
                  "values": {"memories": np.zeros((2, 128, 2, 256))}}
    try:
        backend.restore_memory_state(serialized)
    except RuntimeError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("restore with partial values must fail closed")
