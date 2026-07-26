"""LC-W512: 512-step raw history state management.

Architecture (frozen):
  - Original GTrXL 128-step window (UNCHANGED)
  - Additional 384-step ring buffer of raw h_t (GTrXL output)
  - 128-step delay line: h_t enters delay, exits into long buffer after 128 steps
  - Cross-attention reads from 384-step buffer with sinusoidal positional encoding
  - Zero-init residual gate: z_t = h_t + tanh(gate) * long_context_t
  - gate=0 => bit-exact with ckpt17500

State per env:
  delay_buf      : (128, D)   – 128-step delay line
  delay_idx      : int        – next write position in delay_buf
  delay_count    : int        – steps accumulated in delay_buf (capped at 128)
  long_buf       : (384, D)   – 384-step ring buffer (older history)
  long_mask      : (384,)     – valid slot mask
  long_idx       : int        – next write position in long_buf

Semantics:
  At step t, long_buf contains h_{t-512}..h_{t-129} (the 384 steps
  BEFORE the GTrXL 128-step window). No overlap with GTrXL memory.
  Rollout boundaries do NOT reset; only true done resets.

NOTE: All per-env indexed writes use one-hot masking instead of
  .at[arange, idx].set() to avoid a GPU scatter bug that drops env 0.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


class W512Config:
    def __init__(self, long_size: int = 384, delay_size: int = 128,
                 encoder_size: int = 256):
        self.long_size = int(long_size)
        self.delay_size = int(delay_size)
        self.encoder_size = int(encoder_size)
        self.total_size = self.long_size + self.delay_size  # 512


# ---- one-hot indexed read/write helpers (GPU-safe) ----

def _oh_read_3d(buf, idx):
    """buf: (N, M, D), idx: (N,) int → (N, D)."""
    oh = jax.nn.one_hot(idx, buf.shape[1])          # (N, M)
    return jnp.einsum("nm,nmd->nd", oh, buf)


def _oh_read_2d(buf, idx):
    """buf: (N, M), idx: (N,) int → (N,)."""
    oh = jax.nn.one_hot(idx, buf.shape[1])          # (N, M)
    return jnp.einsum("nm,nm->n", oh, buf)


def _oh_write_3d(buf, idx, val):
    """buf: (N, M, D), idx: (N,), val: (N, D) → (N, M, D)."""
    oh = jax.nn.one_hot(idx, buf.shape[1])          # (N, M)
    return buf * (1.0 - oh[:, :, None]) + val[:, None, :] * oh[:, :, None]


def _oh_write_2d(buf, idx, val):
    """buf: (N, M), idx: (N,), val: (N,) → (N, M).
    Works for bool and float."""
    oh = jax.nn.one_hot(idx, buf.shape[1], dtype=buf.dtype)  # (N, M)
    return buf * (1 - oh) + val[:, None].astype(buf.dtype) * oh


def _oh_write_bool(buf, idx, val):
    """buf: (N, M) bool, idx: (N,), val: (N,) bool → (N, M) bool."""
    oh = jax.nn.one_hot(idx, buf.shape[1], dtype=jnp.bool_)  # (N, M)
    return jnp.where(oh, val[:, None], buf)


# ---- public API ----

def w512_init(num_envs: int, cfg: W512Config):
    return {
        "delay_buf":   jnp.zeros((num_envs, cfg.delay_size, cfg.encoder_size)),
        "delay_idx":   jnp.zeros((num_envs,), dtype=jnp.int32),
        "delay_count": jnp.zeros((num_envs,), dtype=jnp.int32),
        "long_buf":    jnp.zeros((num_envs, cfg.long_size, cfg.encoder_size)),
        "long_mask":   jnp.zeros((num_envs, cfg.long_size), dtype=jnp.bool_),
        "long_idx":    jnp.zeros((num_envs,), dtype=jnp.int32),
    }


def w512_reset_envs(state, done, cfg: W512Config):
    d = done
    d2 = d[:, None]
    d3 = d[:, None, None]
    return {
        "delay_buf":   jnp.where(d3, 0.0, state["delay_buf"]),
        "delay_idx":   jnp.where(d,  0,   state["delay_idx"]),
        "delay_count": jnp.where(d,  0,   state["delay_count"]),
        "long_buf":    jnp.where(d3, 0.0, state["long_buf"]),
        "long_mask":   jnp.where(d2, False, state["long_mask"]),
        "long_idx":    jnp.where(d,  0,   state["long_idx"]),
    }


def w512_step(state, h_t, done, cfg: W512Config):
    """Advance one env step.

    h_t   : (num_envs, D) – GTrXL output for this step
    done  : (num_envs,) bool

    Flow:
      1. Reset done envs
      2. Read oldest entry from delay_buf (about to be overwritten)
      3. Write h_t into delay_buf
      4. If delay_buf was full (count>=128), push oldest into long_buf
    """
    state = w512_reset_envs(state, done, cfg)

    # Read the entry about to be overwritten (oldest in delay)
    old_h = _oh_read_3d(state["delay_buf"], state["delay_idx"])  # (N, D)

    # Write h_t into delay_buf (one-hot, GPU-safe)
    new_delay = _oh_write_3d(state["delay_buf"], state["delay_idx"], h_t)
    new_didx = (state["delay_idx"] + 1) % cfg.delay_size
    new_dcnt = jnp.minimum(state["delay_count"] + 1, cfg.delay_size)

    # Push old_h into long_buf only when delay was already full
    delay_full = (state["delay_count"] >= cfg.delay_size)  # (N,)

    existing_tok = _oh_read_3d(state["long_buf"], state["long_idx"])
    existing_msk = _oh_read_2d(state["long_mask"].astype(jnp.float32),
                               state["long_idx"])
    write_val = jnp.where(delay_full[:, None], old_h, existing_tok)
    write_msk = jnp.where(delay_full, 1.0, existing_msk)

    new_long  = _oh_write_3d(state["long_buf"], state["long_idx"], write_val)
    new_lmask = _oh_write_2d(state["long_mask"].astype(jnp.float32),
                             state["long_idx"], write_msk).astype(jnp.bool_)
    new_lidx  = jnp.where(delay_full,
                          (state["long_idx"] + 1) % cfg.long_size,
                          state["long_idx"])

    return {
        "delay_buf":   new_delay,
        "delay_idx":   new_didx,
        "delay_count": new_dcnt,
        "long_buf":    new_long,
        "long_mask":   new_lmask,
        "long_idx":    new_lidx,
    }
