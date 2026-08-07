"""StudentTrainingBackend: abstract architecture-specific training operations.

The canonical DiCode PPO core (ppo_tr.py) currently hardcodes ActorCriticTransformer
+ memories/memories_mask/memories_mask_idx + network.model_forward_eval/train.
This module defines the abstract backend interface so the PPO core can call
backend.policy_forward_eval/policy_forward_train/init_memory/reset_memory
instead of always reaching for ActorCriticTransformer.

Concrete backends:
  - RMT16TrainingBackend  (training_backend_rmt16.py)
  - SlowGRUTrainingBackend (training_backend_slowgru.py)
"""

from __future__ import annotations

import abc
from typing import Any, Mapping

import jax
import jax.numpy as jnp


class StudentTrainingBackend(abc.ABC):
    """Abstract backend for training a specific student architecture.

    Each concrete backend provides architecture-specific implementations of
    network creation, policy forward (eval + train), memory management, and
    state serialization.  The canonical DiCode PPO core (GAE, PPO clipping,
    value loss, entropy, optimizer) is NEVER modified — only the
    architecture-specific surface is abstracted.
    """

    # Set by concrete implementations
    architecture_family: str = ""
    candidate_id: str = ""

    # ------------------------------------------------------------------
    # Network + TrainState
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def create_train_state_from_checkpoint(
        self, checkpoint_params: Any, tx: Any, rng: jax.Array
    ) -> Any:
        """Create a Flax TrainState from the loaded checkpoint params.

        The returned TrainState MUST satisfy:
          checkpoint_params_sha256 == trainstate_initial_params_sha256
        (parameter lineage: initial_equals_checkpoint=true).

        Returns a Flax TrainState with params=checkpoint_params, apply_fn bound
        to this backend's network, and the given optimizer.
        """

    @abc.abstractmethod
    def get_network(self) -> Any:
        """Return the Flax network module (nn.Module)."""

    # ------------------------------------------------------------------
    # Policy forward
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def policy_forward_eval(
        self, params: Any, memory: Mapping[str, jnp.ndarray],
        obs: jnp.ndarray
    ) -> tuple:
        """Single-step eval forward for environment rollout.

        Args:
            params: Network parameters (from TrainState).
            memory: Architecture-specific memory dictionary.
            obs: Observation array (num_envs, obs_dim).

        Returns:
            (pi, value, memory_out, new_memory) where:
              pi      : distrax.Categorical distribution
              value   : (num_envs,) value predictions
              memory_out : GTrXL memory output (num_envs, embed_size)
              new_memory : Updated memory dictionary
        """

    @abc.abstractmethod
    def policy_forward_train(
        self, params: Any, memory: Mapping[str, jnp.ndarray],
        obs: jnp.ndarray
    ) -> tuple:
        """Training window forward (no memory output).

        Args:
            params: Network parameters.
            memory: Architecture-specific memory + mask for the training window.
            obs: Observation window (batch, window_grad, obs_dim).

        Returns:
            (pi, value) where:
              pi    : distrax.Categorical distribution
              value : (batch,) value predictions
        """

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def init_runner_memory(
        self, num_envs: int
    ) -> Mapping[str, jnp.ndarray]:
        """Initialize memory for num_envs parallel environments.

        Returns a dict of architecture-specific memory arrays.
        """

    @abc.abstractmethod
    def reset_runner_memory(
        self, memory: Mapping[str, jnp.ndarray],
        done: jnp.ndarray
    ) -> Mapping[str, jnp.ndarray]:
        """Reset memory for environments where done is True.

        Returns updated memory dict with done-env memory cleared.
        """

    @abc.abstractmethod
    def prepare_training_memory_batch(
        self, traj_batch: Any, memories_previous: Any,
        config: Any
    ) -> Any:
        """Prepare the memory batch for training from trajectory data.

        Args:
            traj_batch: Transition batch from rollout.
            memories_previous: Memory state before the rollout.
            config: Training configuration.

        Returns:
            Architecture-specific memory batch ready for policy_forward_train.
        """

    @abc.abstractmethod
    def serialize_memory_state(
        self, memory: Mapping[str, jnp.ndarray]
    ) -> Mapping[str, Any]:
        """Serialize memory for RunState checkpointing.

        Returns a JSON-serializable representation.
        """

    @abc.abstractmethod
    def restore_memory_state(
        self, serialized: Mapping[str, Any]
    ) -> Mapping[str, jnp.ndarray]:
        """Restore memory from a serialized RunState checkpoint."""

    # ------------------------------------------------------------------
    # Architecture identity
    # ------------------------------------------------------------------

    def architecture_identity(self) -> str:
        """Return a compact identity string for evidence."""
        return f"{self.architecture_family}:{self.candidate_id}"