"""BLOCKER-2/3/4: memory semantics for the RMT16 / SlowGRU training backends.

Pins the contracts the smoke's parameter-lineage evidence depends on:

  * BLOCKER-4: ``reset_runner_memory`` clears ONLY the done envs' memory and
    must NEVER advance ``mem_idx`` for non-done envs (the single per-step
    ``mem_idx`` advance happens exactly once in ``policy_forward_eval``).
  * BLOCKER-2 (strong): RMT16 training REQUIRES the rollout's real per-step
    entering tokens — missing ``rmt_tokens_seq`` / ``rmt_entering_tokens``
    fails closed, never a silent non-RMT fallback.
  * BLOCKER-3: SlowGRU training REQUIRES the rollout's real pre-action
    longstate and done flags — missing ``longstate_prev`` / ``true_done``
    fails closed, never a zero-init recomputation.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from dicode.training_backend_rmt16 import RMT16TrainingBackend
from dicode.training_backend_slowgru import SlowGRUTrainingBackend

SLOWGRU_RUNTIME_PATH = "/home/oseasy/student_pool_v1/cc3/slowgru_runtime"
SLOWGRU_CHECKPOINT_CONTRACT_PATH = (
    "/home/oseasy/student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304/"
    "checkpoint_contract.json")
SLOWGRU_CHECKPOINT_PATH = (
    "/home/oseasy/student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304/"
    "ckpt/98304/full_state.pkl")


def _rmt16():
    return RMT16TrainingBackend(
        candidate_id="PERSISTENT_RMT16_TEST",
        action_dim=43,
        window_mem=128,
        num_steps=128,
    )


def _slowgru():
    if not os.path.isdir(SLOWGRU_RUNTIME_PATH):
        pytest.skip("slowgru_runtime not present on this host")
    return SlowGRUTrainingBackend(
        candidate_id="SLOWGRU_TEST",
        slowgru_runtime_path=SLOWGRU_RUNTIME_PATH,
        checkpoint_contract_path=SLOWGRU_CHECKPOINT_CONTRACT_PATH,
        checkpoint_path=SLOWGRU_CHECKPOINT_PATH,
    )


# ---------------------------------------------------------------------------
# BLOCKER-4: reset_runner_memory clears ONLY done envs, never advances mem_idx
# ---------------------------------------------------------------------------

def test_rmt16_reset_only_clears_done_envs():
    backend = _rmt16()
    memory = backend.init_runner_memory(4)
    # advance all envs away from the initial position
    memory["mem_idx"] = jnp.asarray([100, 55, 7, 128], dtype=jnp.int32)
    memory["mem_mask"] = memory["mem_mask"].at[0].set(
        jnp.ones_like(memory["mem_mask"][0]))
    done = jnp.asarray([True, False, False, True], dtype=jnp.bool_)
    new_memory = backend.reset_runner_memory(memory, done)
    got_idx = np.asarray(new_memory["mem_idx"])
    # done envs (0,3) reset to window_mem=128; non-done envs (1,2) UNCHANGED
    assert got_idx.tolist() == [128, 55, 7, 128], f"mem_idx: {got_idx}"
    # done env 0 mask cleared to zeros
    assert not np.any(np.asarray(new_memory["mem_mask"][0]))
    # non-done env 1 mask carried unchanged
    assert np.array_equal(
        np.asarray(new_memory["mem_mask"][1]), np.asarray(memory["mem_mask"][1]))


def test_rmt16_reset_no_done_never_advances_mem_idx():
    # BLOCKER-4: with done all-False, mem_idx must be untouched — the ONLY
    # per-step advance is inside policy_forward_eval.
    backend = _rmt16()
    memory = backend.init_runner_memory(3)
    memory["mem_idx"] = jnp.asarray([9, 42, 77], dtype=jnp.int32)
    done = jnp.zeros((3,), dtype=jnp.bool_)
    new_memory = backend.reset_runner_memory(memory, done)
    assert np.asarray(new_memory["mem_idx"]).tolist() == [9, 42, 77]


def test_slowgru_reset_only_clears_done_envs():
    backend = _slowgru()
    memory = backend.init_runner_memory(4)
    memory["mem_idx"] = jnp.asarray([120, 3, 64, 1], dtype=jnp.int32)
    memory["longstate.h"] = jnp.ones_like(memory["longstate.h"]) * 5.0
    memory["longstate.count"] = jnp.asarray([17, 2, 9, 31], dtype=jnp.int32)
    done = jnp.asarray([True, False, True, False], dtype=jnp.bool_)
    new_memory = backend.reset_runner_memory(memory, done)
    got_idx = np.asarray(new_memory["mem_idx"]).tolist()
    assert got_idx == [128, 3, 128, 1], f"mem_idx: {got_idx}"
    # done envs 0,2 longstate cleared; non-done envs 1,3 carried
    h = np.asarray(new_memory["longstate.h"])
    assert np.all(h[0] == 0) and np.all(h[2] == 0)
    assert np.all(h[1] == 5.0) and np.all(h[3] == 5.0)
    assert np.asarray(new_memory["longstate.count"]).tolist() == [0, 2, 0, 31]


# ---------------------------------------------------------------------------
# BLOCKER-2 (strong): RMT16 requires real entering tokens, fails closed
# ---------------------------------------------------------------------------

def test_rmt16_policy_forward_train_fails_closed_without_entering_tokens():
    backend = _rmt16()
    memory = {
        "memories": jnp.zeros((2, 128, 2, 256), dtype=jnp.float32),
        "mask": jnp.zeros((2, 8, 1, 129), dtype=jnp.bool_),
        # rmt_tokens_seq intentionally absent
    }
    obs = jnp.zeros((2, 4, 67), dtype=jnp.float32)
    # params are never used: the fail-closed guard raises BEFORE apply.
    try:
        backend.policy_forward_train(params=None, memory=memory, obs=obs)
    except ValueError as exc:
        assert "RMT16_ENTERING_TOKENS_MISSING" in str(exc)
    else:
        raise AssertionError(
            "policy_forward_train without rmt_tokens_seq must fail closed")


def test_rmt16_prepare_training_memory_batch_fails_closed_without_entering():
    backend = _rmt16()
    from types import SimpleNamespace
    traj_batch = SimpleNamespace(
        memories_indices=jnp.zeros((2, 128), dtype=jnp.int32),
        memories_mask=jnp.zeros((2, 128, 8, 1, 129), dtype=jnp.bool_),
        # rmt_entering_tokens intentionally absent
    )
    config = SimpleNamespace(window_grad=16)
    memories_batch = jnp.zeros((2, 128, 128, 2, 256), dtype=jnp.float32)
    try:
        backend.prepare_training_memory_batch(
            traj_batch, memories_batch, config)
    except ValueError as exc:
        assert "RMT16_ENTERING_TOKENS_MISSING" in str(exc)
    else:
        raise AssertionError(
            "prepare_training_memory_batch without entering tokens must fail "
            "closed")


# ---------------------------------------------------------------------------
# BLOCKER-3: SlowGRU requires real longstate + done flags, fails closed
# ---------------------------------------------------------------------------

def test_slowgru_policy_forward_train_fails_closed_without_longstate():
    backend = _slowgru()
    memory = {
        "memories": jnp.zeros((2, 128, 2, 256), dtype=jnp.float32),
        "mask": jnp.zeros((2, 8, 1, 129), dtype=jnp.bool_),
        "true_done": jnp.zeros((2, 16), dtype=jnp.bool_),
        # longstate_prev intentionally absent
    }
    obs = jnp.zeros((2, 16, 67), dtype=jnp.float32)
    # params are never used: the fail-closed guard raises BEFORE apply.
    try:
        backend.policy_forward_train(params=None, memory=memory, obs=obs)
    except ValueError as exc:
        assert "SLOWGRU_LONGSTATE_MISSING" in str(exc)
    else:
        raise AssertionError(
            "policy_forward_train without longstate_prev must fail closed")


def test_slowgru_policy_forward_train_fails_closed_without_true_done():
    backend = _slowgru()
    memory = {
        "memories": jnp.zeros((2, 128, 2, 256), dtype=jnp.float32),
        "mask": jnp.zeros((2, 8, 1, 129), dtype=jnp.bool_),
        # true_done intentionally absent
        "longstate_prev": {
            "h": jnp.zeros((2, 16, 256), dtype=jnp.float32),
            "buf": jnp.zeros((2, 16, 32, 256), dtype=jnp.float32),
            "count": jnp.zeros((2, 16), dtype=jnp.int32),
        },
    }
    obs = jnp.zeros((2, 16, 67), dtype=jnp.float32)
    # params are never used: the fail-closed guard raises BEFORE apply.
    try:
        backend.policy_forward_train(params=None, memory=memory, obs=obs)
    except ValueError as exc:
        assert "SLOWGRU_TRUE_DONE_MISSING" in str(exc)
    else:
        raise AssertionError(
            "policy_forward_train without true_done must fail closed")
