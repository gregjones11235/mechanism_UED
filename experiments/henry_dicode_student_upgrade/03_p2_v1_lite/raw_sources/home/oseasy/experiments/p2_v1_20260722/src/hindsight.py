"""Hindsight goal relabeling for P2-v1.

P2-v1 requirement (user directive 2026-07-22): hindsight MUST NOT be implemented
by reward shaping alone.  A relabel must SIMULTANEOUSLY:

  1. Re-label the desired goal / task conditioning that the network sees.
     In the Craftax multi-task wrapper the goal/task conditioning is a
     ``embedding_size``-dimensional multi-hot vector APPENDED to the end of the
     symbolic observation (verified: ``CraftaxAugObsTrain.get_obs`` returns
     ``jnp.concatenate([symbolic_obs, task_vector])``).  Relabeling therefore
     replaces the trailing ``embedding_size`` dimensions of every observation
     (and next_observation) with the multi-hot of the newly selected goal.
     This genuinely changes the network's goal-conditioned input.

  2. Recompute the reward under the new goal.  The relabeled reward is the
     goal-conditioned progress/success reward toward the newly selected goal:
     ``r_g[t] = max(ach[t, g] - ach[t-1, g], 0)`` — a +1 the first time goal
     ``g`` is literally achieved.  This depends on the chosen goal, so different
     relabels produce different reward sequences (and therefore different
     returns, value targets, and loss).

Gates (unchanged, MUST NOT be weakened):
  Gate 5 (positive): relabeled goals come ONLY from literally achieved goals.
  Gate 6 (negative): fabricated / unreached hindsight goals are REJECTED.

The module is self-contained — numpy + the local dataclasses only.
"""

from typing import Optional

import numpy as np

from trajectory_replay import ReplaySample, Trajectory

# Number of Craftax achievements.  ``get_achievement_multi_hot`` returns a
# vector of length ``max(Achievement.value) + 1 == 67``.
DEFAULT_EMBEDDING_SIZE = 67


def goal_embedding(goal_index: int, embedding_size: int = DEFAULT_EMBEDDING_SIZE) -> np.ndarray:
    """Multi-hot goal/task-conditioning vector for a single goal index."""
    emb = np.zeros(embedding_size, dtype=np.float32)
    emb[goal_index] = 1.0
    return emb


def apply_goal_conditioning(
    observations: np.ndarray,
    goal_index: int,
    embedding_size: int = DEFAULT_EMBEDDING_SIZE,
) -> np.ndarray:
    """Return observations whose trailing ``embedding_size`` dims are the goal.

    The wrapper appends the task embedding to the END of the observation, so the
    goal-conditioned observation is the raw observation with its trailing
    ``embedding_size`` dimensions replaced by the new goal multi-hot.  The
    original array is not mutated.
    """
    observations = np.asarray(observations)
    if observations.shape[-1] < embedding_size:
        raise ValueError(
            f"Observation last dim {observations.shape[-1]} < embedding_size "
            f"{embedding_size}; cannot apply goal conditioning."
        )
    out = observations.copy().astype(np.float32, copy=False)
    out[..., -embedding_size:] = goal_embedding(goal_index, embedding_size)
    return out


def recompute_reward_for_goal(
    achievements: np.ndarray,
    goal_index: int,
) -> np.ndarray:
    """Goal-conditioned reward: +1 the first time ``goal_index`` is achieved.

    ``r_g[t] = max(ach[t, g] - ach[t-1, g], 0)`` with ach[-1, g] := 0.  This is
    a literal recomputation of reward under the new goal — it is NOT the
    environment reward and it changes with the chosen goal.
    """
    achievements = np.asarray(achievements, dtype=np.float32)
    hit = achievements[:, goal_index]
    prev = np.concatenate([np.zeros_like(hit[:1]), hit[:-1]])
    progress = np.maximum(hit - prev, 0.0)
    return progress.astype(np.float32)


def _select_goal_index(achieved_indices: set, goal_index: Optional[int], goal_name: Optional[str]) -> int:
    """Gate 5/6 enforcement shared by trajectory and sample relabel."""
    if not achieved_indices:
        raise ValueError(
            "Gate 6 (negative): cannot relabel — no achievement was literally "
            "reached (fabricated goal rejected)."
        )
    if goal_index is not None:
        if goal_index not in achieved_indices:
            label = goal_name or f"idx_{goal_index}"
            raise ValueError(
                f"Gate 6 (negative): goal '{label}' (index {goal_index}) was "
                f"NOT literally achieved. Achieved: {sorted(achieved_indices)}. "
                f"Fabricated hindsight goal REJECTED."
            )
        return goal_index
    # First (smallest-index) literally achieved goal.
    return min(achieved_indices)


def relabel_trajectory(
    trajectory: Trajectory,
    goal_index: Optional[int] = None,
    goal_name: Optional[str] = None,
    embedding_size: int = DEFAULT_EMBEDDING_SIZE,
) -> Trajectory:
    """Relabel a trajectory to a literally-achieved goal.

    Applies goal conditioning to observations + next_observations and recomputes
    the reward for the new goal (see module docstring).  The original is not
    mutated.  Raises ValueError on a fabricated goal (Gate 6).
    """
    achieved = trajectory.achieved_goals()
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = _select_goal_index(achieved_indices, goal_index, goal_name)

    new_target = np.zeros_like(trajectory.target_achievements)
    new_target[target_idx] = 1.0

    gc_obs = apply_goal_conditioning(trajectory.observations, target_idx, embedding_size)
    gc_next = (
        apply_goal_conditioning(trajectory.next_observations, target_idx, embedding_size)
        if len(trajectory.next_observations) else np.array([])
    )
    new_rewards = recompute_reward_for_goal(trajectory.achievements, target_idx)

    return Trajectory(
        observations=gc_obs,
        actions=trajectory.actions.copy(),
        rewards=new_rewards,
        dones=trajectory.dones.copy(),
        values=trajectory.values.copy(),
        log_probs=trajectory.log_probs.copy(),
        initial_memory=trajectory.initial_memory.copy(),
        achievements=trajectory.achievements.copy(),
        target_achievements=new_target,
        memory_sequence=(
            trajectory.memory_sequence.copy()
            if len(trajectory.memory_sequence) else np.array([])
        ),
        next_observations=gc_next,
        trajectory_id=trajectory.trajectory_id,
        collected_update_count=trajectory.collected_update_count,
    )


def relabel_sample(
    sample: ReplaySample,
    goal_index: Optional[int] = None,
    goal_name: Optional[str] = None,
    embedding_size: int = DEFAULT_EMBEDDING_SIZE,
) -> ReplaySample:
    """Relabel a ReplaySample to a literally-achieved goal.

    Applies goal conditioning to observations + next_observations and recomputes
    the reward for the new goal.  Behavior log-probs, values, memory, bootstrap
    and policy version (collected_update_count) are carried through unchanged so
    the replay auxiliary update can still compute importance ratios / policy lag.
    """
    achieved = sample.achievements.max(axis=0)
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = _select_goal_index(achieved_indices, goal_index, goal_name)

    new_target = np.zeros_like(sample.target_achievements)
    new_target[target_idx] = 1.0

    gc_obs = apply_goal_conditioning(sample.observations, target_idx, embedding_size)
    gc_next = (
        apply_goal_conditioning(sample.next_observations, target_idx, embedding_size)
        if len(sample.next_observations) else np.array([])
    )
    new_rewards = recompute_reward_for_goal(sample.achievements, target_idx)

    return ReplaySample(
        observations=gc_obs,
        actions=sample.actions,
        rewards=new_rewards,
        dones=sample.dones,
        values=sample.values,
        log_probs=sample.log_probs,
        initial_memory=sample.initial_memory,
        achievements=sample.achievements,
        target_achievements=new_target,
        source_trajectory_id=sample.source_trajectory_id,
        start_step=sample.start_step,
        length=sample.length,
        memory_sequence=sample.memory_sequence,
        next_observations=gc_next,
        next_value=sample.next_value,
        episode_done=sample.episode_done,
        collected_update_count=sample.collected_update_count,
    )


def get_achieved_goal_indices(trajectory: Trajectory) -> list:
    """Sorted list of literally achieved goal indices."""
    achieved = trajectory.achieved_goals()
    return sorted(np.where(achieved > 0)[0].tolist())


def get_achieved_goal_indices_sample(sample: ReplaySample) -> list:
    """Sorted list of literally achieved goal indices in a sample."""
    achieved = sample.achievements.max(axis=0)
    return sorted(np.where(achieved > 0)[0].tolist())
