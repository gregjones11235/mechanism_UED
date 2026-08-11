"""P2-Full-A hindsight goal relabeling (Gate 5/6 unchanged, NOT weakened).

Adapted from P2-v1 hindsight.py for the sparse-anchor replay schema:
  * Trajectory relabel carries memory_anchors + anchor_steps (goal-independent).
  * ReplaySample relabel carries pre_anchor_memory + pre_anchor_step, and applies
    goal conditioning to the loss-region obs/next_obs AND the burn_in_obs (the
    burn-in observations are re-contextualized to the relabeled goal so the
    reconstructed memory reflects the new goal). The anchor memory itself is the
    original-goal memory (standard hindsight approximation); the <=128-step
    burn-in replay with relabeled obs re-contextualizes it.
  * behavior log_probs / policy version are carried through for diagnostics and
    for the V-trace (original-goal) path; the AWR (relabeled) path does NOT use
    them as an importance ratio (see awr.py / Gate G4.4).

Gate 5 (positive): relabeled goals come ONLY from literally achieved goals.
Gate 6 (negative): fabricated / unreached hindsight goals are REJECTED (ValueError).
"""
from typing import Optional

import numpy as np

from replay_buffer import ReplaySample, Trajectory

DEFAULT_EMBEDDING_SIZE = 67


def goal_embedding(goal_index: int, embedding_size: int = DEFAULT_EMBEDDING_SIZE) -> np.ndarray:
    emb = np.zeros(embedding_size, dtype=np.float32)
    emb[goal_index] = 1.0
    return emb


def apply_goal_conditioning(observations, goal_index, embedding_size=DEFAULT_EMBEDDING_SIZE):
    """Replace trailing `embedding_size` dims with the goal multi-hot (no mutation)."""
    observations = np.asarray(observations)
    if observations.size == 0:
        return np.array([])
    if observations.shape[-1] < embedding_size:
        raise ValueError(
            f"Obs last dim {observations.shape[-1]} < embedding_size {embedding_size}.")
    out = observations.copy().astype(np.float32, copy=False)
    out[..., -embedding_size:] = goal_embedding(goal_index, embedding_size)
    return out


def recompute_reward_for_goal(achievements, goal_index):
    """r_g[t] = max(ach[t,g] - ach[t-1,g], 0); ach[-1,g]:=0."""
    achievements = np.asarray(achievements, dtype=np.float32)
    hit = achievements[:, goal_index]
    prev = np.concatenate([np.zeros_like(hit[:1]), hit[:-1]])
    return np.maximum(hit - prev, 0.0).astype(np.float32)


def _select_goal_index(achieved_indices, goal_index, goal_name):
    if not achieved_indices:
        raise ValueError(
            "Gate 6 (negative): cannot relabel — no achievement literally reached "
            "(fabricated goal rejected).")
    if goal_index is not None:
        if goal_index not in achieved_indices:
            label = goal_name or f"idx_{goal_index}"
            raise ValueError(
                f"Gate 6 (negative): goal '{label}' (index {goal_index}) NOT literally "
                f"achieved. Achieved: {sorted(achieved_indices)}. Fabricated goal REJECTED.")
        return goal_index
    return min(achieved_indices)


def relabel_trajectory(trajectory: Trajectory, goal_index=None, goal_name=None,
                       embedding_size=DEFAULT_EMBEDDING_SIZE) -> Trajectory:
    achieved = trajectory.achieved_goals()
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = _select_goal_index(achieved_indices, goal_index, goal_name)
    new_target = np.zeros_like(trajectory.target_achievements)
    new_target[target_idx] = 1.0
    gc_obs = apply_goal_conditioning(trajectory.observations, target_idx, embedding_size)
    gc_next = apply_goal_conditioning(trajectory.next_observations, target_idx, embedding_size)
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
        next_observations=gc_next,
        memory_anchors=(trajectory.memory_anchors.copy()
                        if len(trajectory.memory_anchors) else np.array([])),
        anchor_steps=np.asarray(trajectory.anchor_steps).copy(),
        anchor_masks=(trajectory.anchor_masks.copy()
                      if len(getattr(trajectory, "anchor_masks", np.array([]))) else np.array([])),
        anchor_idxs=np.asarray(getattr(trajectory, "anchor_idxs", np.array([]))).copy(),
        trajectory_id=trajectory.trajectory_id,
        collected_update_count=trajectory.collected_update_count,
    )


def relabel_sample(sample: ReplaySample, goal_index=None, goal_name=None,
                   embedding_size=DEFAULT_EMBEDDING_SIZE) -> ReplaySample:
    """Relabel a ReplaySample to a literally-achieved goal (loss-region achievements)."""
    achieved = sample.achievements.max(axis=0)
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = _select_goal_index(achieved_indices, goal_index, goal_name)
    new_target = np.zeros_like(sample.target_achievements)
    new_target[target_idx] = 1.0
    gc_obs = apply_goal_conditioning(sample.observations, target_idx, embedding_size)
    gc_next = apply_goal_conditioning(sample.next_observations, target_idx, embedding_size)
    gc_burn = apply_goal_conditioning(sample.burn_in_obs, target_idx, embedding_size)
    new_rewards = recompute_reward_for_goal(sample.achievements, target_idx)
    return ReplaySample(
        observations=gc_obs,
        actions=sample.actions,
        rewards=new_rewards,
        dones=sample.dones,
        values=sample.values,
        log_probs=sample.log_probs,
        achievements=sample.achievements,
        target_achievements=new_target,
        pre_anchor_memory=sample.pre_anchor_memory,
        pre_anchor_step=sample.pre_anchor_step,
        burn_in_obs=gc_burn,
        next_observations=gc_next,
        source_trajectory_id=sample.source_trajectory_id,
        start_step=sample.start_step,
        length=sample.length,
        next_value=sample.next_value,
        episode_done=sample.episode_done,
        collected_update_count=sample.collected_update_count,
    )


def get_achieved_goal_indices(trajectory: Trajectory) -> list:
    return sorted(np.where(trajectory.achieved_goals() > 0)[0].tolist())


def get_achieved_goal_indices_sample(sample: ReplaySample) -> list:
    achieved = sample.achievements.max(axis=0)
    return sorted(np.where(achieved > 0)[0].tolist())
