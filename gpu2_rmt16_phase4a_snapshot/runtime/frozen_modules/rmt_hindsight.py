"""RMT16 × P2-Replay — hindsight relabeling for the RMT-extended schema (Phase4A).

Reuses the frozen P2-Full-A hindsight helpers (apply_goal_conditioning,
recompute_reward_for_goal, _select_goal_index — Gate 5/6 NOT weakened). The RMT anchor
fields (mem_tokens/seg_buf/seg_count) are goal-INDEPENDENT (built from h_t) and are
carried through unchanged on relabel, exactly like the GTrXL memory_anchors. The
burn-in observations ARE re-contextualised to the relabeled goal (standard hindsight
approximation; the <=128-step burn-in replay with relabeled obs re-contextualises the
reconstructed memory + tokens). Gate 5 (only literally-achieved goals) / Gate 6
(fabricated goals rejected with ValueError) are inherited unchanged.
"""
import numpy as np

from hindsight import (
    apply_goal_conditioning, recompute_reward_for_goal, _select_goal_index,
    DEFAULT_EMBEDDING_SIZE,
)
from rmt_replay_buffer import RMTTrajectory, RMTReplaySample


def relabel_trajectory_rmt(trajectory: RMTTrajectory, goal_index=None, goal_name=None,
                           embedding_size=DEFAULT_EMBEDDING_SIZE) -> RMTTrajectory:
    achieved = trajectory.achieved_goals()
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = _select_goal_index(achieved_indices, goal_index, goal_name)   # Gate 6
    new_target = np.zeros_like(trajectory.target_achievements)
    new_target[target_idx] = 1.0
    gc_obs = apply_goal_conditioning(trajectory.observations, target_idx, embedding_size)
    gc_next = apply_goal_conditioning(trajectory.next_observations, target_idx, embedding_size)
    new_rewards = recompute_reward_for_goal(trajectory.achievements, target_idx)
    return RMTTrajectory(
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
        memory_anchors=trajectory.memory_anchors.copy(),
        anchor_steps=np.asarray(trajectory.anchor_steps).copy(),
        anchor_masks=trajectory.anchor_masks.copy(),
        anchor_idxs=np.asarray(trajectory.anchor_idxs).copy(),
        trajectory_id=trajectory.trajectory_id,
        collected_update_count=trajectory.collected_update_count,
        rmt_initial_tokens=np.asarray(trajectory.rmt_initial_tokens).copy(),
        rmt_initial_segbuf=np.asarray(trajectory.rmt_initial_segbuf).copy(),
        rmt_initial_segcount=int(trajectory.rmt_initial_segcount),
        rmt_anchor_tokens=np.asarray(trajectory.rmt_anchor_tokens).copy(),
        rmt_anchor_segbuf=np.asarray(trajectory.rmt_anchor_segbuf).copy(),
        rmt_anchor_segcount=np.asarray(trajectory.rmt_anchor_segcount).copy(),
    )


def relabel_sample_rmt(sample: RMTReplaySample, goal_index=None, goal_name=None,
                       embedding_size=DEFAULT_EMBEDDING_SIZE) -> RMTReplaySample:
    achieved = sample.achievements.max(axis=0)
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = _select_goal_index(achieved_indices, goal_index, goal_name)   # Gate 6
    new_target = np.zeros_like(sample.target_achievements)
    new_target[target_idx] = 1.0
    gc_obs = apply_goal_conditioning(sample.observations, target_idx, embedding_size)
    gc_next = apply_goal_conditioning(sample.next_observations, target_idx, embedding_size)
    gc_burn = apply_goal_conditioning(sample.burn_in_obs, target_idx, embedding_size)
    new_rewards = recompute_reward_for_goal(sample.achievements, target_idx)
    return RMTReplaySample(
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
        pre_anchor_rmt_tokens=np.asarray(sample.pre_anchor_rmt_tokens).copy(),
        pre_anchor_rmt_segbuf=np.asarray(sample.pre_anchor_rmt_segbuf).copy(),
        pre_anchor_rmt_segcount=int(sample.pre_anchor_rmt_segcount),
    )
