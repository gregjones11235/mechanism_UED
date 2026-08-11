"""LC-RMT16: ActorCriticTransformer + 16 persistent memory tokens.

Architecture (frozen):
  obs → encoder → GTrXL(128-step) → h_t
  h_t → cross-attention(query=h_t, kv=mem_tokens) → rmt_ctx_t
  z_t = h_t + tanh(rmt_gate) * rmt_ctx_t
  z_t → Actor / Value heads

  Every 128-step segment boundary:
    mem_tokens ← cross-attention(query=mem_tokens, kv=segment_h_buf) + mem_tokens
    (residual update, no fixed-period averaging)

Base submodules keep IDENTICAL names to ActorCriticTransformer so ckpt17500
params load unchanged.  rmt_gate is zero-initialised → bit-exact at init.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from dicode.transformer.transformerXL import Transformer


class ActorCriticTransformerRMT16(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = False
    gating_bias: float = 0.0
    # RMT16 additions
    rmt_num_tokens: int = 16

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

        # --- RMT16 modules (NEW params) ---
        # Read: query=h_t, kv=mem_tokens (at each step)
        self.rmt_read_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.encoder_size,
        )
        self.rmt_read_ln = nn.LayerNorm()
        # Update: query=mem_tokens, kv=segment_h_buf (at segment boundary)
        self.rmt_update_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.encoder_size,
        )
        self.rmt_update_ln = nn.LayerNorm()
        # Zero-init scalar gate → z_t = h_t at init (bit-exact)
        self.rmt_gate = self.param("rmt_gate",
            nn.initializers.zeros, (1,))

    # ----- helpers -----
    def read_rmt(self, h, mem_tokens):
        """Cross-attention read from memory tokens.

        h          : (..., D)
        mem_tokens : (..., num_tokens, D)

        Returns rmt_ctx : (..., D)
        """
        q = h[..., None, :]           # (..., 1, D)
        out = self.rmt_read_attn(inputs_q=q, inputs_kv=mem_tokens)
        out = out.squeeze(-2)         # (..., D)
        out = self.rmt_read_ln(out)
        return out

    def update_rmt_tokens(self, mem_tokens, seg_buf):
        """Update memory tokens from segment hidden states.

        mem_tokens : (batch, num_tokens, D)
        seg_buf    : (batch, segment_len, D)

        Returns updated mem_tokens : (batch, num_tokens, D)
        Residual: new = old + LN(attn(query=old, kv=seg_buf))
        """
        attn_out = self.rmt_update_attn(
            inputs_q=mem_tokens, inputs_kv=seg_buf)
        attn_out = self.rmt_update_ln(attn_out)
        return mem_tokens + attn_out

    def fuse(self, h, rmt_ctx):
        return h + jnp.tanh(self.rmt_gate) * rmt_ctx

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
    def __call__(self, memories, obs, mask, mem_tokens=None):
        return self.model_forward_eval(memories, obs, mask,
                                       mem_tokens=mem_tokens)

    def init_all(self, memories, obs, mask, mem_tokens, seg_buf):
        """Init method that exercises ALL submodules (read + update paths)."""
        pi, value, memory_out, h_t = self.model_forward_eval(
            memories, obs, mask, mem_tokens=mem_tokens)
        new_tokens = self.update_rmt_tokens(mem_tokens, seg_buf)
        return pi, value, memory_out, h_t, new_tokens

    def model_forward_eval(self, memories, obs, mask,
                           mem_tokens=None):
        """Single-step eval.  Returns (pi, value, memory_out, h_t)."""
        x, memory_out = self.transformer.forward_eval(memories, obs, mask)
        h_t = x
        if mem_tokens is not None:
            rc = self.read_rmt(x, mem_tokens)
            x = self.fuse(x, rc)
        pi, value = self._heads(x)
        return pi, value, memory_out, h_t

    def model_forward_train(self, memories, obs, mask,
                            rmt_tokens_seq=None):
        """Window training forward.

        rmt_tokens_seq : (batch, window, num_tokens, D) – mem_tokens per step

        Returns (pi, value).
        """
        x = self.transformer.forward_train(memories, obs, mask)
        if rmt_tokens_seq is not None:
            def _read_one(x_t, tok_t):
                return self.read_rmt(x_t, tok_t)
            rc = jax.vmap(jax.vmap(_read_one,
                                   in_axes=(0, 0)),
                          in_axes=(1, 1), out_axes=1)(x, rmt_tokens_seq)
            x = x + jnp.tanh(self.rmt_gate) * rc
        pi, value = self._heads(x)
        return pi, value
