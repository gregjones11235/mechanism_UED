"""Hindsight goal relabeling: replace the target-conditioning of a trajectory
with achievements that were LITERALLY reached.

D059 Gate 5 (positive): relabeled goals come ONLY from literally achieved goals.
D059 Gate 6 (negative): rejects fabricated / unreached hindsight goals.

The module is self-contained — it only depends on numpy and the local
Trajectory / ReplaySample dataclasses.
"""

from typing import Optional

import numpy as np

from trajectory_replay import ReplaySample, Trajectory


def relabel_trajectory(
    trajectory: Trajectory,
    goal_index: Optional[int] = None,
    goal_name: Optional[str] = None,
) -> Trajectory:
    """Relabel a trajectory's target to a LITERALLY achieved goal.

    Gate 5 enforces: the relabel target MUST be among the achievements
    that were actually reached in this trajectory.  No fabrication.

    Parameters
    ----------
    trajectory : Trajectory
        The stored trajectory to relabel.
    goal_index : int or None
        Specific achievement index to relabel to.  If None, picks the
        first (lowest-index) literally achieved goal, or raises if none.
    goal_name : str or None
        Human-readable label for logging only; not used in logic.

    Returns
    -------
    Trajectory
        A new Trajectory with target_achievements set to the relabel goal.
        The original is not mutated.

    Raises
    ------
    ValueError
        If the requested goal was NOT literally achieved (Gate 6 negative).
    """
    achieved = trajectory.achieved_goals()
    achieved_indices = set(np.where(achieved > 0)[0].tolist())

    if not achieved_indices:
        raise ValueError(
            "Gate 6 (negative): cannot relabel — no achievement was literally "
            "reached in this trajectory (fabricated goal rejected)."
        )

    if goal_index is not None:
        if goal_index not in achieved_indices:
            label = goal_name or f"idx_{goal_index}"
            raise ValueError(
                f"Gate 6 (negative): goal '{label}' (index {goal_index}) was "
                f"NOT literally achieved. Achieved: {sorted(achieved_indices)}. "
                f"Fabricated hindsight goal REJECTED."
            )
        target_idx = goal_index
    else:
        # Pick the first (smallest-index) achieved goal.
        target_idx = min(achieved_indices)

    new_target = np.zeros_like(trajectory.target_achievements)
    new_target[target_idx] = 1.0

    return Trajectory(
        observations=trajectory.observations.copy(),
        actions=trajectory.actions.copy(),
        rewards=trajectory.rewards.copy(),
        dones=trajectory.dones.copy(),
        values=trajectory.values.copy(),
        log_probs=trajectory.log_probs.copy(),
        initial_memory=trajectory.initial_memory.copy(),
        achievements=trajectory.achievements.copy(),
        target_achievements=new_target,
    )


def relabel_sample(
    sample: ReplaySample,
    goal_index: Optional[int] = None,
    goal_name: Optional[str] = None,
) -> ReplaySample:
    """Relabel a ReplaySample's target to a LITERALLY achieved goal.

    Same Gate 5 / Gate 6 enforcement as relabel_trajectory, but operates on
    a ReplaySample (which carries achievements for its sequence slice).
    """
    achieved = sample.achievements.max(axis=0)
    achieved_indices = set(np.where(achieved > 0)[0].tolist())

    if not achieved_indices:
        raise ValueError(
            "Gate 6 (negative): cannot relabel sample — no achievement was "
            "literally reached in this sequence (fabricated goal rejected)."
        )

    if goal_index is not None:
        if goal_index not in achieved_indices:
            label = goal_name or f"idx_{goal_index}"
            raise ValueError(
                f"Gate 6 (negative): goal '{label}' (index {goal_index}) was "
                f"NOT literally achieved in this sample. "
                f"Achieved: {sorted(achieved_indices)}. "
                f"Fabricated hindsight goal REJECTED."
            )
        target_idx = goal_index
    else:
        target_idx = min(achieved_indices)

    new_target = np.zeros_like(sample.target_achievements)
    new_target[target_idx] = 1.0

    return ReplaySample(
        observations=sample.observations,
        actions=sample.actions,
        rewards=sample.rewards,
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
    )


def get_achieved_goal_indices(trajectory: Trajectory) -> list[int]:
    """Return sorted list of literally achieved goal indices."""
    achieved = trajectory.achieved_goals()
    return sorted(np.where(achieved > 0)[0].tolist())


def get_achieved_goal_indices_sample(sample: ReplaySample) -> list[int]:
    """Return sorted list of literally achieved goal indices in a sample."""
    achieved = sample.achievements.max(axis=0)
    return sorted(np.where(achieved > 0)[0].tolist())
