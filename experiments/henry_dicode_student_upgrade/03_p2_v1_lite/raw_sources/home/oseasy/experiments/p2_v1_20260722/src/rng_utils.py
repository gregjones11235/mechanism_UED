"""Checkpointable RNG utilities for P2-v1.

P2-v1 forbids undocumented global np.random action sampling.  Action sampling
uses a local numpy Generator whose bit-generator state is saved and restored
in checkpoints.
"""

from typing import Optional

import numpy as np


def make_action_rng(seed: int = 0) -> np.random.Generator:
    """Create a checkpointable action RNG."""
    return np.random.Generator(np.random.PCG64(seed))


def action_rng_state(action_rng: np.random.Generator) -> dict:
    """Return a picklable state dict for the action RNG."""
    return action_rng.bit_generator.state


def restore_action_rng(state: Optional[dict], seed: int = 0) -> np.random.Generator:
    """Restore an action RNG from a saved state dict.

    If state is None, create a fresh RNG from seed.  This is only valid for
    fresh P2-v1 runs, not for resumed runs.
    """
    rng = make_action_rng(seed)
    if state is not None:
        rng.bit_generator.state = state
    return rng


def sample_actions(
    action_rng: np.random.Generator,
    probs: np.ndarray,
) -> np.ndarray:
    """Sample actions from a [B, A] probability matrix using the local RNG.

    Uses Gumbel-max with the checkpointable Generator.  This is deterministic
    given the saved RNG state and avoids global np.random.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    u = action_rng.random(probs.shape)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    gumbels = -np.log(-np.log(u))
    return np.argmax(np.log(probs) + gumbels, axis=-1).astype(np.int32)
