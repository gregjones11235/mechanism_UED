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
        num_steps: int = 128,
    ):
        self.architecture_family = "SLOWGRU"
        self.candidate_id = str(candidate_id)
        # Canonical student identity used by E3-litesim binding (G1). The
        # checkpoint step is derived from the trailing numeric token of the
        # canonical id (e.g. SLOWGRU_PERSISTENT_CANONICAL_98304 -> 98304).
        self.student_id = self.candidate_id
        try:
            self.checkpoint_step = int(self.candidate_id.rsplit("_", 1)[-1])
        except ValueError:
            self.checkpoint_step = 0
        self._slowgru_runtime_path = str(slowgru_runtime_path)
        self._checkpoint_contract_path = str(checkpoint_contract_path)
        self._checkpoint_path = str(checkpoint_path)
        self._action_dim = int(action_dim)
        self._carry_mode = str(carry_mode)
        self._num_steps = int(num_steps)

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

        # GTrXL reset — BLOCKER-4: reset_runner_memory is responsible ONLY for
        # clearing the done envs' memory (mem_idx -> window_mem, mem_mask ->
        # zeros). It must NOT advance mem_idx for non-done envs — the single
        # per-step mem_idx advance happens exactly once in policy_forward_eval.
        new_memory["mem_idx"] = jnp.where(
            d, wm, memory["mem_idx"]
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
        true_done = memory.get("true_done")
        if true_done is None:
            # BLOCKER-3: never recompute the slow-state PPO loss from a
            # zero-init true_done — the rollout's REAL per-step done flags must
            # be supplied (fail closed).
            raise ValueError(
                "SLOWGRU_TRUE_DONE_MISSING: policy_forward_train received no "
                "true_done — prepare_training_memory_batch must supply the "
                "rollout's real per-step done flags (fail closed)")
        longstate_prev = memory.get("longstate_prev")
        if longstate_prev is None:
            # BLOCKER-3: NEVER recompute PPO loss from zero longstate — the
            # rollout's real hidden state must be supplied. Missing => fail
            # closed (a PASS built on zero-initialized SlowGRU state is a lie).
            raise ValueError(
                "SLOWGRU_LONGSTATE_MISSING: policy_forward_train received no "
                "longstate_prev — the rollout must record real pre-action "
                "longstate (traj_batch.slowgru_longstate) and "
                "prepare_training_memory_batch must supply it (fail closed)")

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

        # BLOCKER-3: the SlowGRU loss must recompute from the rollout's REAL
        # slow state — never zeros.  ``model_forward_train_longmem`` contract:
        #   longstate_prev : (E, dim)  — the PRE-ROLLOUT slow state per env
        #   true_done      : (E, T)    — the PRE-ACTION reset flags the rollout
        #                                used at each of T=num_steps steps
        # ppo_tr's minibatch preserves env-major structure: the trajectory's
        # (num_steps, num_envs, ...) is swapaxes'd to (num_envs, num_steps, ...)
        # and minibatched to (E=num_envs/minibatches, T=num_steps, ...).  So the
        # recorded per-step pre-action longstate is ALREADY (E, T, ...) here,
        # and step 0 of each env is the pre-rollout state.
        ls = getattr(traj_batch, "slowgru_longstate", None)
        if ls is None:
            raise ValueError(
                "SLOWGRU_LONGSTATE_MISSING: prepare_training_memory_batch "
                "found no traj_batch.slowgru_longstate — the rollout must "
                "record real pre-action longstate (fail closed)")
        if ls["h"].ndim < 2 or int(ls["h"].shape[1]) != self._num_steps:
            raise ValueError(
                f"SLOWGRU_LONGSTATE_ORDER: expected (E, T=num_steps={self._num_steps}, "
                f"...) minibatch structure, got h shape {tuple(ls['h'].shape)} "
                f"(fail closed)")
        longstate_prev = {
            key: ls[key][:, 0] for key in ("h", "buf", "count")
        }
        td = ls.get("td")
        if td is None:
            raise ValueError(
                "SLOWGRU_TD_MISSING: prepare_training_memory_batch found no "
                "recorded pre-action true_done (traj_batch.slowgru_longstate "
                "['td']) — the training scan must replay the rollout's real "
                "reset flags (fail closed)")
        result["longstate_prev"] = longstate_prev
        result["true_done"] = td

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_memory_state(
        self, memory: Mapping[str, jnp.ndarray]
    ) -> Mapping[str, Any]:
        """Serialize SlowGRU memory for RunState checkpointing.

        BLOCKER-5: the RunState MUST carry the REAL memory VALUES — never
        shapes, never zeros.  Each leaf is moved to the host (numpy) so the
        checkpoint round-trips value-exact across processes/devices.
        """
        return {
            "architecture_family": "SLOWGRU",
            "carry_mode": self._carry_mode,
            "values": {
                key: np.asarray(memory[key]) for key in MEMORY_FIELD_KEYS
                if key in memory
            },
        }

    def restore_memory_state(
        self, serialized: Mapping[str, Any]
    ) -> Mapping[str, jnp.ndarray]:
        """Restore SlowGRU memory from serialized RunState checkpoint.

        BLOCKER-5: restores the ORIGINAL values — never a zero-init fallback.
        """
        _require(
            serialized.get("architecture_family") == "SLOWGRU",
            "architecture_family mismatch in serialized memory"
        )
        values = serialized.get("values")
        _require(
            isinstance(values, Mapping) and values,
            "serialized SlowGRU memory carries no values (fail closed)"
        )
        _require(
            all(key in values for key in MEMORY_FIELD_KEYS),
            f"serialized SlowGRU memory missing fields: "
            f"{sorted(set(MEMORY_FIELD_KEYS) - set(values))} (fail closed)"
        )
        return {
            key: jnp.asarray(np.asarray(values[key])) for key in MEMORY_FIELD_KEYS
        }