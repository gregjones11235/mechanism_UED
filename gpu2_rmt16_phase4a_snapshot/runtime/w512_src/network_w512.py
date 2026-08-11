"""LC-W512: ActorCriticTransformer + 384-step raw history cross-attention.

Architecture (frozen):
  obs → encoder → GTrXL(128-step) → h_t
  h_t → cross-attention(query=h_t, kv=long_buf+posenc) → long_ctx_t
  z_t = h_t + tanh(w512_gate) * long_ctx_t
  z_t → Actor / Value heads

Base submodules keep IDENTICAL names to ActorCriticTransformer so ckpt17500
params load unchanged.  New modules are EXTRA params; w512_gate is
zero-initialised so at init z_t == h_t exactly (bit-exact gate).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from dicode.transformer.transformerXL import Transformer


def _sinusoidal_posenc(seq_len: int, d_model: int):
    """Standard sinusoidal positional encoding, shape (seq_len, d_model)."""
    positions = jnp.arange(seq_len)[:, None]
    dims = jnp.arange(d_model)[None, :]
    angles = positions / jnp.power(10000.0, 2.0 * (dims // 2) / d_model)
    pe = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return pe.astype(jnp.float32)


class ActorCriticTransformerW512(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = False
    gating_bias: float = 0.0
    # W512 additions
    long_size: int = 384

    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        # --- base submodules (IDENTICAL names to ActorCriticTransformer) ---
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias,
        )
        self.actor_ln1 = nn.Dense(self.hidden_layers,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0))
        self.actor_ln2 = nn.Dense(self.hidden_layers,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0))
        self.actor_out = nn.Dense(self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0))
        self.critic_ln1 = nn.Dense(self.hidden_layers,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0))
        self.critic_ln2 = nn.Dense(self.hidden_layers,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0))
        self.critic_out = nn.Dense(1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0))

        # --- W512 modules (NEW params) ---
        # Cross-attention: query=h_t, kv=384-step buffer
        self.w512_cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.encoder_size,
        )
        self.w512_ln = nn.LayerNorm()
        # Zero-init scalar gate → z_t = h_t at init (bit-exact)
        self.w512_gate = self.param("w512_gate",
            nn.initializers.zeros, (1,))
        # Fixed sinusoidal positional encoding for 384 positions
        # Stored as a non-trainable buffer via a constant initializer
        self.w512_posenc = self.param("w512_posenc",
            lambda key, shape: _sinusoidal_posenc(shape[0], shape[1]),
            (self.long_size, self.encoder_size))

    # ----- helpers -----
    def read_long_history(self, h, long_buf, long_mask):
        """Cross-attention read from 384-step buffer.

        h         : (..., D)
        long_buf  : (..., long_size, D)
        long_mask : (..., long_size) bool

        Returns long_ctx : (..., D)
        """
        q = h[..., None, :]                    # (..., 1, D)
        # Add positional encoding to buffer entries
        pe = self.w512_posenc                  # (long_size, D)
        kv = long_buf + pe                     # (..., long_size, D)

        any_valid = long_mask.any(axis=-1, keepdims=True)  # (..., 1)
        safe_mask = jnp.where(
            any_valid[..., None, None],
            long_mask[..., None, None, :],
            jnp.ones_like(long_mask)[..., None, None, :])
        out = self.w512_cross_attn(inputs_q=q, inputs_kv=kv, mask=safe_mask)
        out = out.squeeze(-2)                  # (..., D)
        out = self.w512_ln(out)
        out = jnp.where(any_valid, out, 0.0)
        return out

    def fuse(self, h, long_ctx):
        return h + jnp.tanh(self.w512_gate) * long_ctx

    def _heads(self, z):
        actor_mean = self.activation_fn(self.actor_ln1(z))
        actor_mean = self.activation_fn(self.actor_ln2(actor_mean))
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = self.activation_fn(self.critic_ln1(z))
        critic = self.activation_fn(self.critic_ln2(critic))
        critic = self.critic_out(critic)
        return pi, jnp.squeeze(critic, axis=-1)

    # ----- forward methods -----
    def __call__(self, memories, obs, mask,
                 long_buf=None, long_mask=None):
        return self.model_forward_eval(memories, obs, mask,
                                       long_buf=long_buf, long_mask=long_mask)

    def model_forward_eval(self, memories, obs, mask,
                           long_buf=None, long_mask=None):
        """Single-step eval.  Returns (pi, value, memory_out, h_t)."""
        x, memory_out = self.transformer.forward_eval(memories, obs, mask)
        h_t = x
        if long_buf is not None:
            lc = self.read_long_history(x, long_buf, long_mask)
            x = self.fuse(x, lc)
        pi, value = self._heads(x)
        return pi, value, memory_out, h_t

    def model_forward_train(self, memories, obs, mask,
                            w512_buf=None, w512_mask=None):
        """Window training forward.

        w512_buf  : (batch, long_size, D) – final buffer, shared across window
        w512_mask : (batch, long_size)

        Returns (pi, value).
        """
        x = self.transformer.forward_train(memories, obs, mask)
        # x: (batch, window, D)
        if w512_buf is not None:
            def _read_batch(x_b, buf_b, msk_b):
                # x_b: (window, D), buf_b: (long_size, D), msk_b: (long_size,)
                def _read_step(x_t):
                    return self.read_long_history(x_t, buf_b, msk_b)
                return jax.vmap(_read_step)(x_b)
            lc = jax.vmap(_read_batch, in_axes=(0, 0, 0))(
                x, w512_buf, w512_mask)
            x = x + jnp.tanh(self.w512_gate) * lc
        pi, value = self._heads(x)
        return pi, value
