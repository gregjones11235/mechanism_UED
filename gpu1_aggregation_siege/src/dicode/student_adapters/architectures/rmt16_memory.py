"""LC-RMT16: 16 persistent memory tokens state management.

Architecture (frozen):
  - Original GTrXL 128-step window (UNCHANGED)
  - 16 persistent memory tokens per env, updated via attention every 128-step segment
  - Cross-attention reads from tokens at each step
  - Zero-init residual gate: z_t = h_t + tanh(gate) * rmt_context_t
  - gate=0 => bit-exact with ckpt17500

State per env:
  mem_tokens    : (16, D)    – persistent memory tokens
  seg_buf       : (128, D)   – hidden states accumulated in current segment
  seg_count     : int        – steps in current segment (0..127)

Semantics:
  During each 128-step segment, the actor reads from mem_tokens.
  At segment end (every 128 steps), mem_tokens are updated via
  cross-attention(query=mem_tokens, kv=seg_buf).
  Rollout boundaries do NOT clear tokens; only true done clears.

NOTE: Per-env indexed writes use one-hot masking to avoid GPU scatter bug.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


class RMT16Config:
    def __init__(self, num_tokens: int = 16, segment_len: int = 128,
                 encoder_size: int = 256):
        self.num_tokens = int(num_tokens)
        self.segment_len = int(segment_len)
        self.encoder_size = int(encoder_size)


# ---- one-hot indexed write helper (GPU-safe) ----

def _oh_write_3d(buf, idx, val):
    """buf: (N, M, D), idx: (N,), val: (N, D) → (N, M, D)."""
    oh = jax.nn.one_hot(idx, buf.shape[1])          # (N, M)
    return buf * (1.0 - oh[:, :, None]) + val[:, None, :] * oh[:, :, None]


# ---- public API ----

def rmt16_init(num_envs: int, cfg: RMT16Config):
    return {
        "mem_tokens": jnp.zeros((num_envs, cfg.num_tokens, cfg.encoder_size)),
        "seg_buf":    jnp.zeros((num_envs, cfg.segment_len, cfg.encoder_size)),
        "seg_count":  jnp.zeros((num_envs,), dtype=jnp.int32),
    }


def rmt16_reset_envs(state, done, cfg: RMT16Config):
    d = done
    return {
        "mem_tokens": jnp.where(d[:, None, None], 0.0, state["mem_tokens"]),
        "seg_buf":    jnp.where(d[:, None, None], 0.0, state["seg_buf"]),
        "seg_count":  jnp.where(d, 0, state["seg_count"]),
    }


def rmt16_store_h(state, h_t, cfg: RMT16Config):
    """Store h_t into segment buffer (called every step)."""
    cnt = state["seg_count"]
    new_buf = _oh_write_3d(state["seg_buf"], cnt, h_t)
    return {**state, "seg_buf": new_buf, "seg_count": cnt + 1}


def rmt16_update_tokens(state, update_fn, cfg: RMT16Config):
    """Update mem_tokens from segment buffer (called at segment boundary).

    update_fn: (mem_tokens (N,16,D), seg_buf (N,128,D)) -> new_tokens (N,16,D)
    """
    new_tokens = update_fn(state["mem_tokens"], state["seg_buf"])
    return {
        "mem_tokens": new_tokens,
        "seg_buf":    jnp.zeros_like(state["seg_buf"]),
        "seg_count":  jnp.zeros_like(state["seg_count"]),
    }


def rmt16_step(state, h_t, done, update_fn, cfg: RMT16Config):
    """Advance one env step.

    1. Reset done envs
    2. Store h_t in segment buffer
    3. If segment complete (count == segment_len), update tokens
    """
    state = rmt16_reset_envs(state, done, cfg)
    state = rmt16_store_h(state, h_t, cfg)

    is_boundary = (state["seg_count"] >= cfg.segment_len)  # (N,)

    # Compute updated tokens (always compute, select conditionally)
    updated = rmt16_update_tokens(state, update_fn, cfg)

    # Select: boundary envs get updated tokens, others keep accumulated state
    return {
        "mem_tokens": jnp.where(is_boundary[:, None, None],
                                updated["mem_tokens"], state["mem_tokens"]),
        "seg_buf":    jnp.where(is_boundary[:, None, None],
                                updated["seg_buf"], state["seg_buf"]),
        "seg_count":  jnp.where(is_boundary,
                                updated["seg_count"], state["seg_count"]),
    }
