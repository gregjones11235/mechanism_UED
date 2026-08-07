"""RMT16TrainingBackend: trains the PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 Student.

Wraps ActorCriticTransformerRMT16 with the canonical DiCode PPO training loop.
The RMT16 network is a superset of ActorCriticTransformer: identical base
submodule names, plus 16 persistent memory tokens with cross-attention read/update.

Memory contract (per env):
  memories       : (window_mem, num_layers, embed_size)  — GTrXL short memory
  mem_mask       : (num_heads, 1, window_mem + 1)        — GTrXL attention mask
  mem_idx        : scalar                                — next write position
  rmt.mem_tokens : (rmt_num_tokens, embed_size)          — persistent memory tokens
  rmt.seg_buf    : (segment_len, embed_size)             — hidden states in current segment
  rmt.seg_count  : scalar                                — steps in current segment

Parameter lineage: checkpoint params MUST equal TrainState initial params.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from .student_adapters.architectures.rmt16_network import ActorCriticTransformerRMT16
from .student_adapters.architectures.rmt16_memory import RMT16Config
from .student_adapters.architectures.rmt16_anchor import (
    make_apply_eval_rmt,
    make_update_fn,
    rmt_advance_tokens,
    rmt_step_forward,
)
from .training_backend import StudentTrainingBackend


# --- helpers ---

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"RMT16TrainingBackend: {msg}")


def _batch_indices_select(x, y):
    """Select elements from x by indices y along dim=1."""
    return jax.vmap(lambda a, b: a[b])(x, y)


batch_indices_select = _batch_indices_select
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


class RMT16TrainingBackend(StudentTrainingBackend):
    """Training backend for the RMT16 architecture (ActorCriticTransformerRMT16).

    Constructor args:
        candidate_id: "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304" or similar
        action_dim: number of discrete actions
        activation: "relu" or "tanh"
        hidden_layers: actor/critic hidden layer size
        embed_size: encoder/transformer embedding size
        num_heads: attention heads
        qkv_features: QKV feature dimension
        num_layers: transformer layers
        gating: whether GTrXL uses gating
        gating_bias: gating bias
        rmt_num_tokens: number of persistent memory tokens (16)
        window_mem: GTrXL window size
        num_steps: segment length (128, = rollout steps per update)
        carry_mode: "persistent" or "reset128"
    """

    def __init__(
        self, *,
        candidate_id: str,
        action_dim: int,
        activation: str = "relu",
        hidden_layers: int = 256,
        embed_size: int = 256,
        num_heads: int = 8,
        qkv_features: int = 256,
        num_layers: int = 2,
        gating: bool = False,
        gating_bias: float = 0.0,
        rmt_num_tokens: int = 16,
        window_mem: int = 128,
        num_steps: int = 128,
        carry_mode: str = "persistent",
    ):
        self.architecture_family = "RMT16"
        self.candidate_id = str(candidate_id)
        self._action_dim = int(action_dim)
        self._activation = str(activation)
        self._hidden_layers = int(hidden_layers)
        self._embed_size = int(embed_size)
        self._num_heads = int(num_heads)
        self._qkv_features = int(qkv_features)
        self._num_layers = int(num_layers)
        self._gating = bool(gating)
        self._gating_bias = float(gating_bias)
        self._rmt_num_tokens = int(rmt_num_tokens)
        self._window_mem = int(window_mem)
        self._num_steps = int(num_steps)
        self._carry_mode = str(carry_mode)

        self._network = None
        self._apply_eval_rmt = None
        self._rmt_cfg = None

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def get_network(self) -> Any:
        if self._network is None:
            self._build_network()
        return self._network

    def _build_network(self) -> None:
        self._network = ActorCriticTransformerRMT16(
            action_dim=self._action_dim,
            activation=self._activation,
            hidden_layers=self._hidden_layers,
            encoder_size=self._embed_size,
            num_heads=self._num_heads,
            qkv_features=self._qkv_features,
            num_layers=self._num_layers,
            gating=self._gating,
            gating_bias=self._gating_bias,
            rmt_num_tokens=self._rmt_num_tokens,
        )
        self._rmt_cfg = RMT16Config(
            num_tokens=self._rmt_num_tokens,
            segment_len=self._num_steps,
            encoder_size=self._embed_size,
        )

    def create_train_state_from_checkpoint(
        self, checkpoint_params: Any, tx: Any, rng: jax.Array
    ) -> Any:
        """Create a TrainState from RMT16 checkpoint params.

        The apply_fn is bound to the RMT16 network.
        Parameter lineage: checkpoint_params == TrainState.params.
        """
        network = self.get_network()
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint_params)
        return TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def init_runner_memory(
        self, num_envs: int
    ) -> Mapping[str, jnp.ndarray]:
        """Initialize RMT16 memory for num_envs parallel environments.

        Mirrors the CC2 driver reset convention:
          memories/mem_tokens/seg_buf: zeros
          mem_idx: window_mem (= next write position)
          seg_count: 0
        """
        n = int(num_envs)
        wm = self._window_mem
        nl = self._num_layers
        es = self._embed_size
        nh = self._num_heads
        nt = self._rmt_num_tokens
        seg = self._num_steps

        return {
            "memories": jnp.zeros((n, wm, nl, es), dtype=jnp.float32),
            "mem_mask": jnp.zeros((n, nh, 1, wm + 1), dtype=jnp.bool_),
            "mem_idx": jnp.full((n,), wm, dtype=jnp.int32),
            "rmt.mem_tokens": jnp.zeros((n, nt, es), dtype=jnp.float32),
            "rmt.seg_buf": jnp.zeros((n, seg, es), dtype=jnp.float32),
            "rmt.seg_count": jnp.zeros((n,), dtype=jnp.int32),
        }

    def reset_runner_memory(
        self, memory: Mapping[str, jnp.ndarray],
        done: jnp.ndarray
    ) -> Mapping[str, jnp.ndarray]:
        """Reset memory for environments where done is True.

        GTrXL: mem_idx -> window_mem, mem_mask -> zeros
        RMT: mem_tokens/seg_buf/seg_count -> zeros
        """
        wm = self._window_mem

        new_memory = dict(memory)

        # GTrXL reset — BLOCKER-4: reset_runner_memory is responsible ONLY for
        # clearing the done envs' memory (mem_idx -> window_mem, mem_mask ->
        # zeros). It must NOT advance mem_idx for non-done envs — the single
        # per-step mem_idx advance happens exactly once in policy_forward_eval.
        new_memory["mem_idx"] = jnp.where(
            done, wm, memory["mem_idx"]
        ).astype(jnp.int32)
        new_memory["mem_mask"] = jnp.where(
            done[:, None, None, None],
            jnp.zeros_like(memory["mem_mask"]),
            memory["mem_mask"],
        )

        # RMT reset
        new_memory["rmt.mem_tokens"] = jnp.where(
            done[:, None, None], 0.0, memory["rmt.mem_tokens"]
        )
        new_memory["rmt.seg_buf"] = jnp.where(
            done[:, None, None], 0.0, memory["rmt.seg_buf"]
        )
        new_memory["rmt.seg_count"] = jnp.where(
            done, 0, memory["rmt.seg_count"]
        )

        return new_memory

    # ------------------------------------------------------------------
    # Policy forward
    # ------------------------------------------------------------------

    def policy_forward_eval(
        self, params: Any, memory: Mapping[str, jnp.ndarray],
        obs: jnp.ndarray
    ) -> tuple:
        """Single-step RMT16 eval forward.

        Returns (pi, value, memory_out, new_memory).
        """
        network = self.get_network()
        if self._apply_eval_rmt is None:
            self._apply_eval_rmt = make_apply_eval_rmt(network)

        memories = memory["memories"]
        mem_mask = memory["mem_mask"]
        mem_idx = memory["mem_idx"]
        mem_tokens = memory["rmt.mem_tokens"]
        seg_buf = memory["rmt.seg_buf"]
        seg_count = memory["rmt.seg_count"]
        done = jnp.zeros((memories.shape[0],), dtype=jnp.bool_)

        wm = self._window_mem
        nh = self._num_heads

        # GTrXL mask advance
        mem_idx = jnp.where(done, wm, jnp.clip(mem_idx - 1, 0, wm)).astype(jnp.int32)
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, wm + 1)
        ohot = ohot[:, None, None, :].repeat(nh, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)

        # Forward with entering tokens
        logits, value, mem_out, h_t = self._apply_eval_rmt(
            params, memories, obs, mem_mask, mem_tokens,
        )
        pi = jax.nn.softmax(logits)

        from distrax import Categorical
        pi_dist = Categorical(logits=logits)

        # Roll memories
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)

        # Advance RMT state
        update_fn = make_update_fn(network, params)
        old_rmt_st = {"mem_tokens": mem_tokens, "seg_buf": seg_buf, "seg_count": seg_count}
        new_rmt_st = rmt_advance_tokens(old_rmt_st, h_t, done, update_fn, self._rmt_cfg, self._carry_mode)

        new_memory = {
            "memories": post_memories,
            "mem_mask": mem_mask,
            "mem_idx": mem_idx,
            "rmt.mem_tokens": new_rmt_st["mem_tokens"],
            "rmt.seg_buf": new_rmt_st["seg_buf"],
            "rmt.seg_count": new_rmt_st["seg_count"],
        }

        # value is already 1D from the RMT16 network's _heads (squeezed there)
        return pi_dist, value, mem_out, new_memory

    def policy_forward_train(
        self, params: Any, memory: Mapping[str, jnp.ndarray],
        obs: jnp.ndarray
    ) -> tuple:
        """Training window forward for RMT16.

        Args:
            params: Network parameters.
            memory: Dict with "memories", "mask", "rmt_tokens_seq".
            obs: (batch, window_grad, obs_dim).

        Returns (pi, value).
        """
        network = self.get_network()

        memories_batch = memory["memories"]    # (batch, window_grad, window_mem, nl, es) or (batch, window_mem, nl, es)
        mask = memory["mask"]                   # (batch, num_heads, window_grad, window_mem + 1)
        rmt_tokens_seq = memory.get("rmt_tokens_seq")  # (batch, window_grad, num_tokens, es)

        # BLOCKER-2 (strong): RMT16 training MUST be computed from the REAL
        # per-step entering tokens recorded during the rollout
        # (traj_batch.rmt_entering_tokens).  A None rmt_tokens_seq means the
        # rollout-loss PPO loss silently dropped the RMT state — a PASS built on
        # that is a lie.  Fail closed instead of silently falling back to the
        # non-RMT model_forward_train.
        if rmt_tokens_seq is None:
            raise ValueError(
                "RMT16_ENTERING_TOKENS_MISSING: policy_forward_train received "
                "no rmt_tokens_seq — the rollout must record real pre-action "
                "entering tokens (traj_batch.rmt_entering_tokens) and "
                "prepare_training_memory_batch must supply them (fail closed)")
        # shape guard: (batch, window_grad, num_tokens, embed_size)
        if rmt_tokens_seq.ndim != 4 \
                or rmt_tokens_seq.shape[2] != self._rmt_num_tokens:
            raise ValueError(
                f"RMT16_ENTERING_TOKENS_SHAPE: expected "
                f"(batch, window_grad, {self._rmt_num_tokens}, embed_size), got "
                f"{tuple(rmt_tokens_seq.shape)} (fail closed)")

        # network.model_forward_train expects:
        #   memories: (batch, window_mem, nl, es)  -- the WINDOW of memories
        #   obs: (batch, window_grad, obs_dim)       -- the observation window
        #   mask: (batch, num_heads, 1, window_mem + 1) — but we have window_grad dim
        #   rmt_tokens_seq: (batch, window_grad, nt, es)

        pi, value = network.apply(
            {"params": params},
            memories_batch, obs, mask,
            rmt_tokens_seq=rmt_tokens_seq,
            method=network.model_forward_train,
        )
        return pi, value

    def prepare_training_memory_batch(
        self, traj_batch: Any, memories_batch: Any,
        config: Any
    ) -> Any:
        """Prepare the RMT16 memory batch for training.

        ``memories_batch`` is the FULL concatenated memory batch
        (memories_previous + scan_memories, concatenated upstream).

        Returns a dict ready for policy_forward_train.
        """
        # RMT tokens sequence: BLOCKER-2 (strong) — the REAL pre-action entering
        # tokens recorded during the rollout are REQUIRED.  If they are absent,
        # the PPO loss would recompute without the persistent RMT state and the
        # resulting PASS would be a lie — fail closed BEFORE any prep work.
        entering = getattr(traj_batch, "rmt_entering_tokens", None)
        if entering is None:
            raise ValueError(
                "RMT16_ENTERING_TOKENS_MISSING: prepare_training_memory_batch "
                "found no traj_batch.rmt_entering_tokens — the rollout must "
                "record real pre-action entering tokens (fail closed)")
        # Window the entering tokens exactly like obs: collapse every leading
        # dimension into the rollout-entry axis, then split into training
        # windows of ``window_grad`` consecutive steps.
        #   (minibatch, nt, es)        -> (minibatch/window_grad, window_grad, nt, es)
        #   (num_steps, num_envs, nt, es) -> (…/window_grad, window_grad, nt, es)
        entering = jnp.reshape(entering, (-1,) + tuple(entering.shape[-2:]))
        entering = jnp.reshape(
            entering, (-1, config.window_grad, self._rmt_num_tokens,
                       entering.shape[-1]))

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
            "rmt_tokens_seq": entering,
        }

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_memory_state(
        self, memory: Mapping[str, jnp.ndarray]
    ) -> Mapping[str, Any]:
        """Serialize RMT16 memory for RunState checkpointing.

        BLOCKER-5: the RunState MUST carry the REAL memory VALUES — never
        shapes, never zeros.  Each leaf is moved to the host (numpy) so the
        checkpoint round-trips value-exact across processes/devices.
        """
        return {
            "architecture_family": "RMT16",
            "carry_mode": self._carry_mode,
            "values": {
                "memories": np.asarray(memory["memories"]),
                "mem_mask": np.asarray(memory["mem_mask"]),
                "mem_idx": np.asarray(memory["mem_idx"]),
                "rmt.mem_tokens": np.asarray(memory["rmt.mem_tokens"]),
                "rmt.seg_buf": np.asarray(memory["rmt.seg_buf"]),
                "rmt.seg_count": np.asarray(memory["rmt.seg_count"]),
            },
        }

    def restore_memory_state(
        self, serialized: Mapping[str, Any]
    ) -> Mapping[str, jnp.ndarray]:
        """Restore RMT16 memory from serialized RunState checkpoint.

        BLOCKER-5: restores the ORIGINAL values — never a zero-init fallback.
        """
        _require(
            serialized.get("architecture_family") == "RMT16",
            "architecture_family mismatch in serialized memory"
        )
        values = serialized.get("values")
        _require(
            isinstance(values, Mapping) and values,
            "serialized RMT16 memory carries no values (fail closed)"
        )
        required = ("memories", "mem_mask", "mem_idx", "rmt.mem_tokens",
                    "rmt.seg_buf", "rmt.seg_count")
        _require(
            all(key in values for key in required),
            f"serialized RMT16 memory missing fields: "
            f"{sorted(set(required) - set(values))} (fail closed)"
        )
        return {
            key: jnp.asarray(np.asarray(values[key])) for key in required
        }