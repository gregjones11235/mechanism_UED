"""SlowGRUTrainingBackend: trains the SLOWGRU_PERSISTENT_CANONICAL_98304 Student.

Wraps ActorCriticSlowGRU (from CC3 slowgru_runtime) with the canonical DiCode PPO
training loop.  The SlowGRU network extends the base GTrXL with a slow-GRU
long-term state that updates every 32 steps via attention pooling.

Memory contract (per env):
  memories         : (128, 2, 256)     — GTrXL fast window memory
  memories_mask    : (8, 1, 129)       — GTrXL attention mask
  memories_mask_idx: ()                — next write position
  longstate.h      : (256,)            — slow GRU hidden state
  longstate.buf    : (32, 256)         — current period's GTrXL hiddens
  longstate.count  : ()                — steps in current period
  true_done        : ()                — reset signal for slow state

Done/reset semantics (per-env):
  - true episode done -> clear fast memory + longstate for that env
  - ordinary step -> carry
  - 128-step segment boundary -> Persistent SlowGRU contract keeps longstate
  - true reset -> longstate cleared per contract (zero-init)

Parameter lineage: checkpoint params MUST equal TrainState initial params.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from .training_backend import StudentTrainingBackend


# SlowGRU constants (frozen)
WINDOW_MEM = 128
NUM_LAYERS = 2
EMBED_SIZE = 256
NUM_HEADS = 8
SLOW_INTERVAL = 32
SLOW_DIM = 256

# Memory field keys
MEMORY_FIELD_KEYS = (
    "memories", "mem_mask", "mem_idx",
    "longstate.h", "longstate.buf", "longstate.count",
    "true_done",
)

# Batch padding for network's x.squeeze() bug
_MIN_BATCH_FOR_NETWORK = 2


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"SlowGRUTrainingBackend: {msg}")


# --- helpers (mirrors ppo_tr.py) ---
def _batch_indices_select(x, y):
    return jax.vmap(lambda a, b: a[b])(x, y)


batch_indices_select = _batch_indices_select
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def _flat_to_longstate(flat: dict) -> dict:
    """Convert flat dotted keys to nested longstate dict."""
    return {
        "h": flat["longstate.h"],
        "buf": flat["longstate.buf"],
        "count": flat["longstate.count"],
    }


def _longstate_to_flat(ls: dict) -> dict:
    """Convert nested longstate to flat dotted keys."""
    return {
        "longstate.h": ls["h"],
        "longstate.buf": ls["buf"],
        "longstate.count": ls["count"],
    }


class SlowGRUTrainingBackend(StudentTrainingBackend):
    """Training backend for the SlowGRU architecture (ActorCriticSlowGRU).

    Constructor args:
        candidate_id: "SLOWGRU_PERSISTENT_CANONICAL_98304"
        slowgru_runtime_path: server path to slowgru_runtime directory
        checkpoint_contract_path: path to checkpoint_contract.json
        checkpoint_path: path to the full_state.pkl checkpoint
        action_dim: number of discrete actions
        carry_mode: "PERSISTENT" or "RESET128"
    """

    def __init__(
        self, *,
        candidate_id: str,
        slowgru_runtime_path: str,
        checkpoint_contract_path: str,
        checkpoint_path: str,
        action_dim: int = 43,
        carry_mode: str = "PERSISTENT",
    ):
        self.architecture_family = "SLOWGRU"
        self.candidate_id = str(candidate_id)
        self._slowgru_runtime_path = str(slowgru_runtime_path)
        self._checkpoint_contract_path = str(checkpoint_contract_path)
        self._checkpoint_path = str(checkpoint_path)
        self._action_dim = int(action_dim)
        self._carry_mode = str(carry_mode)

        self._handle: dict | None = None
        self._network = None
        self._params = None
        self._forward_eval = None
        self._init_longstate = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        _require(bool(self._slowgru_runtime_path),
                 "slowgru_runtime_path not configured")
        _require(os.path.isdir(self._slowgru_runtime_path),
                 f"slowgru_runtime_path not a directory: {self._slowgru_runtime_path!r}")

        # Inject slowgru_runtime into sys.path
        if self._slowgru_runtime_path not in sys.path:
            sys.path.insert(0, self._slowgru_runtime_path)

        import slowgru_runtime as sr

        # Read contract
        import json
        _require(os.path.isfile(self._checkpoint_contract_path),
                 f"checkpoint_contract.json not found: {self._checkpoint_contract_path!r}")
        with open(self._checkpoint_contract_path, "r", encoding="utf-8") as fh:
            contract = json.load(fh)

        # Load candidate via slowgru_runtime
        handle = sr.load_candidate(contract)
        _require(isinstance(handle, dict) and handle.get("params") is not None,
                 "slowgru_runtime.load_candidate returned invalid handle")

        # Seed policy RNG
        sr.seed_policy_rng(handle, 42)

        self._handle = handle
        self._network = handle["network"]
        self._params = handle["params"]
        self._forward_eval = handle["forward_eval"]
        self._init_longstate = handle["init_longstate"]
        self._loaded = True

    # ------------------------------------------------------------------
    # Network + TrainState
    # ------------------------------------------------------------------

    def get_network(self) -> Any:
        self._ensure_loaded()
        return self._network

    def create_train_state_from_checkpoint(
        self, checkpoint_params: Any, tx: Any, rng: jax.Array
    ) -> Any:
        """Create a TrainState from SlowGRU checkpoint params.

        Parameter lineage: checkpoint_params == TrainState.params.
        The checkpoint params are already in the Flax variables dict format
        {"params": ...} as returned by slowgru_runtime.
        """
        self._ensure_loaded()
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint_params)
        return TrainState.create(
            apply_fn=self._network.apply,
            params=params,
            tx=tx,
        )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def init_runner_memory(
        self, num_envs: int
    ) -> Mapping[str, jnp.ndarray]:
        """Initialize SlowGRU memory for num_envs parallel environments."""
        self._ensure_loaded()
        n = int(num_envs)
        return {
            "memories": jnp.zeros((n, WINDOW_MEM, NUM_LAYERS, EMBED_SIZE), dtype=jnp.float32),
            "mem_mask": jnp.zeros((n, NUM_HEADS, 1, WINDOW_MEM + 1), dtype=jnp.bool_),
            "mem_idx": jnp.full((n,), WINDOW_MEM + 1, dtype=jnp.int32),
            "longstate.h": jnp.zeros((n, SLOW_DIM), dtype=jnp.float32),
            "longstate.buf": jnp.zeros((n, SLOW_INTERVAL, EMBED_SIZE), dtype=jnp.float32),
            "longstate.count": jnp.zeros((n,), dtype=jnp.int32),
            "true_done": jnp.zeros((n,), dtype=jnp.bool_),
        }

    def reset_runner_memory(
        self, memory: Mapping[str, jnp.ndarray],
        done: jnp.ndarray
    ) -> Mapping[str, jnp.ndarray]:
        """Reset memory for environments where done is True.

        GTrXL: mem_idx -> window_mem, mem_mask -> zeros
        SlowGRU: longstate -> zeros, true_done preserved as the reset signal
        """
        wm = WINDOW_MEM
        d = done

        new_memory = dict(memory)

        # GTrXL reset (identical to ppo_tr.py _env_step)
        new_memory["mem_idx"] = jnp.where(
            d, wm, jnp.clip(memory["mem_idx"] - 1, 0, wm)
        ).astype(jnp.int32)
        new_memory["mem_mask"] = jnp.where(
            d[:, None, None, None],
            jnp.zeros_like(memory["mem_mask"]),
            memory["mem_mask"],
        )

        # SlowGRU longstate reset on true episode done
        new_memory["longstate.h"] = jnp.where(
            d[:, None], jnp.zeros_like(memory["longstate.h"]), memory["longstate.h"]
        )
        new_memory["longstate.buf"] = jnp.where(
            d[:, None, None], jnp.zeros_like(memory["longstate.buf"]), memory["longstate.buf"]
        )
        new_memory["longstate.count"] = jnp.where(
            d, jnp.zeros_like(memory["longstate.count"]), memory["longstate.count"]
        )

        # true_done carries the reset signal for the next step
        new_memory["true_done"] = d

        return new_memory

    # ------------------------------------------------------------------
    # Policy forward
    # ------------------------------------------------------------------

    def policy_forward_eval(
        self, params: Any, memory: Mapping[str, jnp.ndarray],
        obs: jnp.ndarray
    ) -> tuple:
        """Single-step SlowGRU eval forward.

        Returns (pi, value, memory_out, new_memory).
        """
        self._ensure_loaded()

        memories = memory["memories"]
        mem_mask = memory["mem_mask"]
        mem_idx = memory["mem_idx"]
        longstate = _flat_to_longstate(memory)
        true_done = memory["true_done"]

        done = jnp.zeros((memories.shape[0],), dtype=jnp.bool_)

        # GTrXL mask advance (identical to ppo_tr.py _env_step)
        mem_idx = jnp.where(done, WINDOW_MEM, jnp.clip(mem_idx - 1, 0, WINDOW_MEM)).astype(jnp.int32)
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, WINDOW_MEM + 1)
        ohot = ohot[:, None, None, :].repeat(NUM_HEADS, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)

        # Forward with slow state
        # Pad batch_size=1 to 2 (workaround for x.squeeze() in transformerXL)
        pad = memories.shape[0] < _MIN_BATCH_FOR_NETWORK
        if pad:
            _memories = jnp.concatenate([memories, memories], 0)
            _obs = jnp.concatenate([obs, obs], 0)
            _mask = jnp.concatenate([mem_mask, mem_mask], 0)
            _ls = {k: jnp.concatenate([v, v], 0) for k, v in longstate.items()}
            _reset = jnp.concatenate([true_done, true_done], 0)
        else:
            _memories = memories
            _obs = obs
            _mask = mem_mask
            _ls = longstate
            _reset = true_done

        pi, value, mem_out, ls_new = self._forward_eval(
            params, _memories, _obs, _mask, _ls, _reset)

        if pad:
            pi = jax.tree_util.tree_map(lambda x: x[:1], pi)
            value = value[:1]
            mem_out = mem_out[:1]
            ls_new = {k: v[:1] for k, v in ls_new.items()}

        # Roll memories
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)

        new_memory = {
            "memories": post_memories,
            "mem_mask": mem_mask,
            "mem_idx": mem_idx,
            "longstate.h": ls_new["h"],
            "longstate.buf": ls_new["buf"],
            "longstate.count": ls_new["count"],
            "true_done": true_done,  # carried for next step
        }

        return pi, value, mem_out, new_memory

    def policy_forward_train(
        self, params: Any, memory: Mapping[str, jnp.ndarray],
        obs: jnp.ndarray
    ) -> tuple:
        """Training window forward for SlowGRU.

        Args:
            params: Network parameters.
            memory: Dict with "memories", "mask", "true_done", "longstate_prev".
            obs: (batch, window_grad, obs_dim).

        Returns (pi, value).
        """
        self._ensure_loaded()

        memories_batch = memory["memories"]
        mask = memory["mask"]
        true_done = memory.get("true_done", jnp.zeros((obs.shape[0], obs.shape[1]), dtype=jnp.bool_))
        longstate_prev = memory.get("longstate_prev")
        if longstate_prev is None:
            # Default: zero longstate
            longstate_prev = self._init_longstate(obs.shape[0])

        pi, value = self._network.apply(
            params,  # params is already the Flax variables dict {"params": ...}
            memories_batch, obs, mask, true_done, longstate_prev,
            method=self._network.model_forward_train_longmem,
        )
        return pi, value

    def prepare_training_memory_batch(
        self, traj_batch: Any, memories_batch: Any,
        config: Any
    ) -> Any:
        """Prepare the SlowGRU memory batch for training.

        Returns a dict ready for policy_forward_train.
        """
        # Select memories by indices
        memories_batch_sel = batch_indices_select(
            memories_batch, traj_batch.memories_indices[:, :: config.window_grad]
        )
        memories_batch_sel = batchify(memories_batch_sel)

        # Create mask for window_grad
        memories_mask = traj_batch.memories_mask.reshape(
            (-1, config.window_grad) + traj_batch.memories_mask.shape[2:]
        )
        memories_mask = jnp.swapaxes(memories_mask, 1, 2)
        # Concatenate with zeros
        memories_mask = jnp.concatenate(
            (
                memories_mask,
                jnp.zeros(
                    memories_mask.shape[:-1] + (config.window_grad - 1,),
                    dtype=jnp.bool_,
                ),
            ),
            axis=-1,
        )
        # Roll
        memories_mask = roll_vmap(
            memories_mask, jnp.arange(0, config.window_grad), -1
        )

        result = {
            "memories": memories_batch_sel,
            "mask": memories_mask,
        }

        # true_done per step for slow state update
        if hasattr(traj_batch, "done"):
            done_flat = traj_batch.done.reshape(
                (-1, config.window_grad) + traj_batch.done.shape[2:])
            # squeeze trailing dims
            while done_flat.ndim > 2:
                done_flat = done_flat.squeeze(-1)
            result["true_done"] = done_flat

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_memory_state(
        self, memory: Mapping[str, jnp.ndarray]
    ) -> Mapping[str, Any]:
        """Serialize SlowGRU memory for RunState checkpointing."""
        return {
            "architecture_family": "SLOWGRU",
            "memories_shape": list(np.asarray(memory["memories"]).shape),
            "memories_mask_shape": list(np.asarray(memory["mem_mask"]).shape),
            "memories_mask_idx_shape": list(np.asarray(memory["mem_idx"]).shape),
            "longstate.h_shape": list(np.asarray(memory["longstate.h"]).shape),
            "longstate.buf_shape": list(np.asarray(memory["longstate.buf"]).shape),
            "longstate.count_shape": list(np.asarray(memory["longstate.count"]).shape),
        }

    def restore_memory_state(
        self, serialized: Mapping[str, Any]
    ) -> Mapping[str, jnp.ndarray]:
        """Restore SlowGRU memory from serialized RunState checkpoint."""
        _require(
            serialized.get("architecture_family") == "SLOWGRU",
            "architecture_family mismatch in serialized memory"
        )
        return self.init_runner_memory(
            num_envs=int(serialized["memories_shape"][0])
        )