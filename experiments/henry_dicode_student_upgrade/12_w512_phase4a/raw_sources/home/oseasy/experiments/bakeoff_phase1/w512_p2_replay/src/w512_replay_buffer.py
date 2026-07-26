"""W512 × P2 Replay buffer: extends P2-Full-A Trajectory/ReplaySample with W512
long state anchors.

Frozen P2-Full-A hyperparameters: capacity=64, ANCHOR_INTERVAL=128,
MIN_SEQUENCE_LENGTH=129.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Re-export frozen constants from P2-Full-A
ANCHOR_INTERVAL = 128
MIN_SEQUENCE_LENGTH = 129


@dataclass
class W512Trajectory:
    """One complete episode with sparse GTrXL + W512 memory anchors."""
    observations: np.ndarray            # [T, obs_dim]
    actions: np.ndarray                 # [T]
    rewards: np.ndarray                 # [T]
    dones: np.ndarray                   # [T]
    values: np.ndarray                  # [T]
    log_probs: np.ndarray               # [T]
    initial_memory: np.ndarray          # [wm, layers, embed]
    achievements: np.ndarray            # [T, n_ach]
    target_achievements: np.ndarray     # [n_ach]
    next_observations: np.ndarray = field(default_factory=lambda: np.array([]))
    memory_anchors: np.ndarray = field(default_factory=lambda: np.array([]))
    anchor_steps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    anchor_masks: np.ndarray = field(default_factory=lambda: np.array([]))
    anchor_idxs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    trajectory_id: Optional[int] = None
    collected_update_count: int = 0
    # W512 extensions
    init_w512_state: Optional[dict] = None          # W512 state at episode start
    w512_anchor_states: list = field(default_factory=list)  # list of dict per anchor

    @property
    def length(self) -> int:
        return len(self.observations)

    @property
    def n_anchors(self) -> int:
        return 0 if len(self.memory_anchors) == 0 else int(self.memory_anchors.shape[0])

    def achieved_goals(self) -> np.ndarray:
        if len(self.achievements) == 0:
            return np.zeros(self.achievements.shape[1] if self.achievements.ndim > 1 else 0,
                            dtype=np.float32)
        return self.achievements.max(axis=0).astype(np.float32)

    def any_achievement(self) -> bool:
        return bool(self.achieved_goals().any())

    def nearest_anchor_le(self, step: int):
        """(anchor_index, anchor_step, anchor_memory, w512_state) for largest anchor <= step."""
        idx = -1
        best = None
        for i, s in enumerate(self.anchor_steps):
            if int(s) <= step and int(s) > idx:
                idx = int(s)
                best = i
        if best is None:
            raise RuntimeError(f"No anchor <= step {step}")
        w5_st = self.w512_anchor_states[best] if best < len(self.w512_anchor_states) else None
        return best, int(self.anchor_steps[best]), self.memory_anchors[best], w5_st

    def validate_anchors(self):
        expected = list(range(0, max(self.length, 1), ANCHOR_INTERVAL))
        got = [int(s) for s in self.anchor_steps]
        if got != expected:
            raise ValueError(f"Anchor conservation: expected {expected}, got {got}")
        if self.n_anchors != len(expected):
            raise ValueError(f"Anchor count {self.n_anchors} != {len(expected)}")
        if len(self.w512_anchor_states) != len(expected):
            raise ValueError(
                f"W512 anchor state count {len(self.w512_anchor_states)} != {len(expected)}")
        return True


@dataclass
class W512ReplaySample:
    """A sampled loss region with GTrXL + W512 anchor reconstruction data."""
    observations: np.ndarray        # [L, obs_dim]
    actions: np.ndarray             # [L]
    rewards: np.ndarray             # [L]
    dones: np.ndarray               # [L]
    values: np.ndarray              # [L]
    log_probs: np.ndarray           # [L]
    achievements: np.ndarray        # [L, n_ach]
    target_achievements: np.ndarray  # [n_ach]
    pre_anchor_memory: np.ndarray   # [wm, layers, embed]
    pre_anchor_step: int
    burn_in_obs: np.ndarray         # [gap, obs_dim]
    next_observations: np.ndarray = field(default_factory=lambda: np.array([]))
    source_trajectory_id: int = -1
    start_step: int = 0
    length: int = 0
    next_value: float = 0.0
    episode_done: bool = False
    collected_update_count: int = 0
    # W512 extension: anchor W512 long state
    pre_anchor_w512_state: Optional[dict] = None

    @property
    def burn_in_length(self) -> int:
        return 0 if len(self.burn_in_obs) == 0 else int(self.burn_in_obs.shape[0])


class W512ReplayBuffer:
    """Fixed-capacity ring of W512Trajectories. Frozen: capacity=64."""

    def __init__(self, capacity: int = 64, seed: int = 42):
        self.capacity = capacity
        self._buffer: list = []
        self._next_id: int = 0
        self._rng = np.random.RandomState(seed)
        self.trajectories_collected = 0
        self.trajectories_inserted = 0
        self.replay_samples_drawn = 0
        self.total_anchors_stored = 0

    def insert(self, trajectory: W512Trajectory) -> int:
        if trajectory.length < 1:
            raise ValueError("Rejected: empty trajectory.")
        if not trajectory.dones[-1]:
            raise ValueError("Gate 4: trajectory does not end with done=True.")
        trajectory.validate_anchors()
        traj_id = self._next_id
        self._next_id += 1
        trajectory.trajectory_id = traj_id
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(trajectory)
        self.trajectories_inserted += 1
        self.total_anchors_stored += trajectory.n_anchors
        return traj_id

    def can_sample(self) -> bool:
        return any(t.length >= MIN_SEQUENCE_LENGTH for t in self._buffer)

    def sample(self, sequence_length=None) -> W512ReplaySample:
        if not self._buffer:
            raise RuntimeError("Replay buffer is empty.")
        eligible = [(i, t) for i, t in enumerate(self._buffer)
                    if t.length >= MIN_SEQUENCE_LENGTH]
        if not eligible:
            raise RuntimeError("Gate 4: no trajectory >= 129 steps.")
        _, traj = eligible[self._rng.randint(len(eligible))]
        max_len = traj.length
        if sequence_length is None:
            sequence_length = self._rng.randint(MIN_SEQUENCE_LENGTH, max_len + 1)
        if sequence_length > max_len:
            sequence_length = max_len
        max_start = max_len - sequence_length
        start_step = self._rng.randint(0, max_start + 1) if max_start > 0 else 0
        end_step = start_step + sequence_length

        a_idx, a_step, a_mem, a_w512 = traj.nearest_anchor_le(start_step)
        burn_in_obs = traj.observations[a_step:start_step].copy()

        next_obs = np.array([])
        if len(traj.next_observations) >= end_step:
            next_obs = traj.next_observations[start_step:end_step].copy()
        slice_done = bool(traj.dones[end_step - 1])
        if slice_done:
            next_value = 0.0
        elif end_step < traj.length:
            next_value = float(traj.values[end_step])
        else:
            raise RuntimeError("Nonterminal sample at trajectory boundary.")

        self.replay_samples_drawn += 1

        return W512ReplaySample(
            observations=traj.observations[start_step:end_step].copy(),
            actions=traj.actions[start_step:end_step].copy(),
            rewards=traj.rewards[start_step:end_step].copy(),
            dones=traj.dones[start_step:end_step].copy(),
            values=traj.values[start_step:end_step].copy(),
            log_probs=traj.log_probs[start_step:end_step].copy(),
            achievements=traj.achievements[start_step:end_step].copy(),
            target_achievements=traj.target_achievements.copy(),
            pre_anchor_memory=np.asarray(a_mem).copy(),
            pre_anchor_step=int(a_step),
            burn_in_obs=burn_in_obs,
            next_observations=next_obs,
            source_trajectory_id=traj.trajectory_id,
            start_step=int(start_step),
            length=int(sequence_length),
            next_value=next_value,
            episode_done=slice_done,
            collected_update_count=int(traj.collected_update_count),
            pre_anchor_w512_state=a_w512,
        )

    def __len__(self):
        return len(self._buffer)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "buffer": self._buffer,
            "next_id": self._next_id,
            "rng_state": self._rng.get_state(),
            "trajectories_collected": self.trajectories_collected,
            "trajectories_inserted": self.trajectories_inserted,
            "replay_samples_drawn": self.replay_samples_drawn,
            "total_anchors_stored": self.total_anchors_stored,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "W512ReplayBuffer":
        buf = cls(capacity=state["capacity"])
        buf._buffer = state["buffer"]
        buf._next_id = state["next_id"]
        buf._rng.set_state(state["rng_state"])
        buf.trajectories_collected = state["trajectories_collected"]
        buf.trajectories_inserted = state["trajectories_inserted"]
        buf.replay_samples_drawn = state["replay_samples_drawn"]
        buf.total_anchors_stored = state["total_anchors_stored"]
        return buf


def relabel_sample_w512(sample: W512ReplaySample, goal_index=None,
                        goal_name=None, embedding_size=67):
    """Hindsight relabel for W512ReplaySample: applies goal conditioning
    and preserves W512 anchor state."""
    import hindsight as H
    # Apply standard hindsight relabeling
    achieved = sample.achievements.max(axis=0)
    achieved_indices = set(np.where(achieved > 0)[0].tolist())
    target_idx = H._select_goal_index(achieved_indices, goal_index, goal_name)
    new_target = np.zeros_like(sample.target_achievements)
    new_target[target_idx] = 1.0
    gc_obs = H.apply_goal_conditioning(sample.observations, target_idx,
                                        embedding_size)
    gc_next = H.apply_goal_conditioning(sample.next_observations, target_idx,
                                         embedding_size)
    gc_burn = H.apply_goal_conditioning(sample.burn_in_obs, target_idx,
                                         embedding_size)
    new_rewards = H.recompute_reward_for_goal(sample.achievements, target_idx)
    return W512ReplaySample(
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
        pre_anchor_w512_state=sample.pre_anchor_w512_state,
    )
