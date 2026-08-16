"""Slice-only JAX student backend (SLICE/CONTRACT ONLY, not scientific content).

Implements the same surface as dicode.training_backend.StudentTrainingBackend
so the rollout collector and PPOBridge are architecture-generic.  The SlowGRU /
RMT16 production backends plug into the identical code path on the GPU server.
"""
from __future__ import annotations

from typing import Any, Mapping

import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.training import train_state as flax_train_state

from .hashing import hash_pytree


class SliceNet(nn.Module):
    action_dim: int
    mem_dim: int = 16
    hidden: int = 32

    @nn.compact
    def __call__(self, memory: Mapping[str, jnp.ndarray], obs: jnp.ndarray):
        mem = memory["mem"]
        h = nn.relu(nn.Dense(self.hidden)(jnp.concatenate([obs, mem], axis=-1)))
        logits = nn.Dense(self.action_dim)(h)
        value = nn.Dense(1)(h)[..., 0]
        new_mem = jnp.tanh(nn.Dense(self.mem_dim)(jnp.concatenate([h, mem], axis=-1)))
        return distrax.Categorical(logits=logits), value, new_mem


class SliceStudentBackend:
    architecture_family = "slice"

    def __init__(self, obs_dim: int, action_dim: int, mem_dim: int = 16,
                 hidden: int = 32) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.mem_dim = int(mem_dim)
        self.network = SliceNet(action_dim=self.action_dim, mem_dim=mem_dim,
                                hidden=hidden)

    def get_network(self):
        return self.network

    def initial_params(self, rng) -> Any:
        init_mem = {"mem": jnp.zeros((2, self.mem_dim))}
        init_obs = jnp.zeros((2, self.obs_dim))
        return self.network.init(rng, init_mem, init_obs)

    def create_train_state_from_checkpoint(self, checkpoint_params, tx, rng):
        return flax_train_state.TrainState.create(
            apply_fn=self.network.apply, params=checkpoint_params, tx=tx)

    def policy_forward_eval(self, params, memory, obs):
        pi, value, new_mem = self.network.apply(params, memory, obs)
        return pi, value, new_mem, {"mem": new_mem}

    def policy_forward_train(self, params, memory, obs):
        pi, value, _new = self.network.apply(params, memory, obs)
        return pi, value

    def init_runner_memory(self, num_envs: int):
        return {"mem": jnp.zeros((num_envs, self.mem_dim))}

    def reset_runner_memory(self, memory, done):
        return {"mem": jnp.where(done[:, None], 0.0, memory["mem"])}

    def params_hash(self, params) -> str:
        return hash_pytree(params)