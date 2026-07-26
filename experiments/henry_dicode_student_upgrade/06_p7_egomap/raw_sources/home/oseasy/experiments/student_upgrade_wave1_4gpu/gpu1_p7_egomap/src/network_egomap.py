"""P7-EGOMAP network: the healthy GTrXL-PPO ActorCriticTransformer with an
added egocentric-map encoder + gated fusion. The base submodules keep IDENTICAL
names/structure so the base ckpt17500 params load into them unchanged; the
egomap encoder + fusion gate are EXTRA params (zero-init gate).

G1 (feature-off bit-exact): when egomap_enabled=False the egomap branch is
skipped entirely and x_fused = x, so pi/value are bit-exact equal to the base.
G6 (transfer): even when enabled, the fusion gate is zero-initialized so at init
x_fused = x exactly -> the network starts behaving exactly like the base.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import distrax

# Reuse the base building blocks unchanged (read-only import; base not modified).
from dicode.network import Transformer  # noqa: E402


class EgoMapCNN(nn.Module):
    """Small CNN encoder: (H, W, C_map) -> (out_dim,)."""
    out_dim: int
    features: tuple = (16, 32)

    @nn.compact
    def __call__(self, x):
        # x: (..., H, W, C). Conv over the last 3 dims; leading dims preserved.
        h = x
        for f in self.features:
            h = nn.Conv(f, kernel_size=(3, 3), padding="SAME",
                        kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        bias_init=nn.initializers.constant(0.0))(h)
            h = nn.relu(h)
        h = jnp.mean(h, axis=(-3, -2))           # global avg pool over H,W
        h = nn.Dense(self.out_dim,
                     kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(h)
        h = nn.relu(h)
        return h


class ActorCriticTransformerEgoMap(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = False
    gating_bias: float = 0.0
    # P7 additions
    egomap_channels: int = 9
    egomap_cnn_features: tuple = (16, 32)

    def setup(self):
        self.activation_fn = nn.relu if self.activation == "relu" else nn.tanh
        # --- base submodules (names identical to ActorCriticTransformer) --- #
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias,
        )
        self.actor_ln1 = nn.Dense(self.hidden_layers, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0))
        self.actor_ln2 = nn.Dense(self.hidden_layers, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0))
        self.actor_out = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.constant(0.0))
        self.critic_ln1 = nn.Dense(self.hidden_layers, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0))
        self.critic_ln2 = nn.Dense(self.hidden_layers, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.constant(0.0))
        # --- P7 egomap branch (extra params) --- #
        self.ego_encoder = EgoMapCNN(out_dim=self.encoder_size,
                                     features=self.egomap_cnn_features)
        # zero-init scalar gate -> x_fused = x at init (and exactly when disabled)
        self.ego_gate = self.param("ego_gate", nn.initializers.zeros, (1,))

    # ----- fusion + heads (shared) ----- #
    def _fuse(self, x, ego_features, enabled):
        if (not enabled) or (ego_features is None):
            return x
        x_ego = self.ego_encoder(ego_features)         # (..., encoder_size)
        return x + self.ego_gate * x_ego

    def _heads(self, x):
        actor_mean = self.activation_fn(self.actor_ln1(x))
        actor_mean = self.activation_fn(self.actor_ln2(actor_mean))
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = self.activation_fn(self.critic_ln1(x))
        critic = self.activation_fn(self.critic_ln2(critic))
        critic = self.critic_out(critic)
        return pi, jnp.squeeze(critic, axis=-1)

    # ----- forward methods mirroring the base, + ego_features ----- #
    def __call__(self, memories, obs, mask, ego_features=None, egomap_enabled=True):
        x, memory_out = self.transformer(memories, obs, mask)
        x = self._fuse(x, ego_features, egomap_enabled)
        pi, value = self._heads(x)
        return pi, value, memory_out

    def model_forward_eval(self, memories, obs, mask, ego_features=None, egomap_enabled=True):
        x, memory_out = self.transformer.forward_eval(memories, obs, mask)
        x = self._fuse(x, ego_features, egomap_enabled)
        pi, value = self._heads(x)
        return pi, value, memory_out

    def model_forward_train(self, memories, obs, mask, ego_features=None, egomap_enabled=True):
        x = self.transformer.forward_train(memories, obs, mask)
        x = self._fuse(x, ego_features, egomap_enabled)
        pi, value = self._heads(x)
        return pi, value
