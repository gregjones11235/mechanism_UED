"""P2-Full-A replay buffer: complete-episode storage with SPARSE memory anchors.

Frozen design §2 / §11 (p2_full_frozen_design.md v2):
  * NO per-step memory_sequence storage (the disk killer; ~256 KB/step).
  * Store a pre-action GTrXL memory ANCHOR every 128 steps (+ episode start).
  * Replay reconstructs memory at the loss-window start by replaying the current
    network forward from the NEAREST anchor <= start (<=128 steps), NOT from a
    zero memory (zero-memory mid-episode burn-in is forbidden / incorrect).
  * Anchors enter trajectory conservation, checkpoint and exact-resume tests.

The Trajectory stores the full anchor set (for conservation). A ReplaySample
carries the single anchor nearest-before-start plus the burn-in observations
needed to reconstruct memory at the loss-window start; the loss region itself is
forwarded with current params (V-trace-consistent re-burn-in).
"""
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ANCHOR_INTERVAL = 128
MIN_SEQUENCE_LENGTH = 129  # strictly > 128


def anchor_steps_for_length(length: int) -> list:
    """Anchor step indices for an episode of `length` steps: 0,128,256,...<length."""
    return list(range(0, max(length, 1), ANCHOR_INTERVAL))


@dataclass
class Trajectory:
    """One complete episode (T can exceed 128) with sparse memory anchors."""
    observations: np.ndarray            # [T, obs_dim]
    actions: np.ndarray                 # [T]
    rewards: np.ndarray                 # [T]
    dones: np.ndarray                   # [T] (True at final step)
    values: np.ndarray                  # [T]
    log_probs: np.ndarray               # [T] behavior mu (collection-time policy)
    initial_memory: np.ndarray          # [window_mem, n_layers, embed] pre-action mem at step 0
    achievements: np.ndarray            # [T, n_ach]
    target_achievements: np.ndarray     # [n_ach]
    next_observations: np.ndarray = field(default_factory=lambda: np.array([]))  # [T, obs_dim]
    memory_anchors: np.ndarray = field(default_factory=lambda: np.array([]))     # [N, wm, layers, embed]
    anchor_steps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # [N]
    # Optional pre-step forward state per anchor for BIT-EXACT reconstruction:
    # anchor_masks[k] = mem_mask entering the anchor iteration [num_heads,1,wm+1]
    # anchor_idxs[k]  = mem_idx entering the anchor iteration (scalar int)
    # These let replay reproduce the exact rollout mask evolution from the anchor.
    anchor_masks: np.ndarray = field(default_factory=lambda: np.array([]))       # [N, heads,1,wm+1]
    anchor_idxs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))   # [N]
    trajectory_id: Optional[int] = None
    collected_update_count: int = 0

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

    def anchor_at(self, step: int) -> Optional[np.ndarray]:
        """Pre-action memory anchor at exact `step`, or None."""
        for i, s in enumerate(self.anchor_steps):
            if int(s) == step:
                return self.memory_anchors[i]
        return None

    def nearest_anchor_le(self, step: int):
        """(anchor_step, anchor_memory) for the largest anchor step <= `step`."""
        idx = -1
        best = None
        for i, s in enumerate(self.anchor_steps):
            if int(s) <= step and int(s) > idx:
                idx = int(s)
                best = i
        if best is None:
            raise RuntimeError(f"No anchor <= step {step}; episode-start anchor missing.")
        return int(self.anchor_steps[best]), self.memory_anchors[best]

    def validate_anchors(self):
        """Conservation: episode-start anchor present, count == ceil(L/128), steps correct."""
        expected = anchor_steps_for_length(self.length)
        got = [int(s) for s in self.anchor_steps]
        if got != expected:
            raise ValueError(
                f"Anchor conservation failed: expected steps {expected}, got {got}.")
        if self.n_anchors != len(expected):
            raise ValueError(
                f"Anchor count {self.n_anchors} != expected {len(expected)}.")
        if len(self.anchor_steps) == 0 or int(self.anchor_steps[0]) != 0:
            raise ValueError("Episode-start anchor (step 0) missing.")
        return True


@dataclass
class ReplaySample:
    """A sampled loss region plus the anchor data to reconstruct its start memory."""
    observations: np.ndarray        # [L, obs_dim]  loss region [start, end)
    actions: np.ndarray             # [L]
    rewards: np.ndarray             # [L]
    dones: np.ndarray               # [L]
    values: np.ndarray              # [L]
    log_probs: np.ndarray           # [L] behavior mu (diagnostic; V-trace uses it, AWR does not)
    achievements: np.ndarray        # [L, n_ach]
    target_achievements: np.ndarray  # [n_ach]
    # anchor reconstruction data
    pre_anchor_memory: np.ndarray   # [wm, layers, embed] memory at nearest anchor <= start
    pre_anchor_step: int            # absolute step of that anchor
    burn_in_obs: np.ndarray         # [gap, obs_dim] obs[pre_anchor_step : start], gap<=128
    next_observations: np.ndarray = field(default_factory=lambda: np.array([]))  # [L, obs_dim]
    source_trajectory_id: int = -1
    start_step: int = 0
    length: int = 0
    next_value: float = 0.0          # collected V at state after last step (diagnostic/fallback)
    episode_done: bool = False
    collected_update_count: int = 0

    @property
    def burn_in_length(self) -> int:
        return 0 if len(self.burn_in_obs) == 0 else int(self.burn_in_obs.shape[0])


@dataclass
class ReplayCounters:
    trajectories_collected: int = 0
    trajectories_inserted: int = 0
    replay_samples_drawn: int = 0
    total_sequence_length: int = 0
    relabelled_samples: int = 0
    gradient_updates: int = 0
    total_anchors_stored: int = 0
    target_achievement_coverage: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "trajectories_collected": self.trajectories_collected,
            "trajectories_inserted": self.trajectories_inserted,
            "replay_samples_drawn": self.replay_samples_drawn,
            "total_sequence_length": self.total_sequence_length,
            "relabelled_samples": self.relabelled_samples,
            "gradient_updates": self.gradient_updates,
            "total_anchors_stored": self.total_anchors_stored,
            "target_achievement_coverage": dict(self.target_achievement_coverage),
        }


class ReplayBuffer:
    """Fixed-capacity ring of complete episodes with sparse memory anchors."""

    def __init__(self, capacity: int = 64, seed: int = 42):
        self.capacity = capacity
        self._buffer: list = []
        self._next_id: int = 0
        self._rng = np.random.RandomState(seed)
        self.counters = ReplayCounters()
        self._achievement_coverage: dict = defaultdict(int)

    def insert(self, trajectory: Trajectory) -> int:
        if trajectory.length < 1:
            raise ValueError("Rejected: empty trajectory.")
        if not trajectory.dones[-1]:
            raise ValueError(
                "Gate 4 (negative): trajectory does not end with done=True; "
                "truncated-only fragments REJECTED.")
        # anchor conservation (episode-start anchor + correct count/steps)
        trajectory.validate_anchors()
        traj_id = self._next_id
        self._next_id += 1
        trajectory.trajectory_id = traj_id
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(trajectory)
        self.counters.trajectories_inserted += 1
        self.counters.total_anchors_stored += trajectory.n_anchors
        for idx in np.where(trajectory.achieved_goals() > 0)[0]:
            self._achievement_coverage[int(idx)] += 1
        return traj_id

    def can_sample(self) -> bool:
        return any(t.length >= MIN_SEQUENCE_LENGTH for t in self._buffer)

    def _get_by_id(self, traj_id):
        for t in self._buffer:
            if t.trajectory_id == traj_id:
                return t
        return None

    def sample(self, sequence_length=None, trajectory_id=None, start_step=None) -> ReplaySample:
        if not self._buffer:
            raise RuntimeError("Replay buffer is empty.")
        eligible = [(i, t) for i, t in enumerate(self._buffer) if t.length >= MIN_SEQUENCE_LENGTH]
        if not eligible:
            raise RuntimeError(
                "Gate 4 (negative): no trajectory exceeds 128 steps; truncated-only REJECTED.")
        if trajectory_id is None:
            _, traj = eligible[self._rng.randint(len(eligible))]
        else:
            traj = self._get_by_id(trajectory_id)
            if traj is None or traj.length < MIN_SEQUENCE_LENGTH:
                raise RuntimeError(f"Trajectory {trajectory_id} not found / too short.")
        trajectory_id = traj.trajectory_id
        max_len = traj.length
        if sequence_length is None:
            sequence_length = self._rng.randint(MIN_SEQUENCE_LENGTH, max_len + 1)
        if sequence_length < MIN_SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length {sequence_length} <= 128 REJECTED.")
        if sequence_length > max_len:
            raise ValueError(f"sequence_length {sequence_length} > trajectory length {max_len}.")
        max_start = max_len - sequence_length
        if start_step is None:
            start_step = self._rng.randint(0, max_start + 1) if max_start > 0 else 0
        if start_step < 0 or start_step > max_start:
            raise ValueError(f"Invalid start_step={start_step}.")
        end_step = start_step + sequence_length

        # nearest anchor <= start_step  (<=128 behind)
        a_step, a_mem = traj.nearest_anchor_le(start_step)
        burn_in_obs = traj.observations[a_step:start_step].copy()  # gap<=128

        next_obs = np.array([])
        if len(traj.next_observations) >= end_step:
            next_obs = traj.next_observations[start_step:end_step].copy()
        slice_done = bool(traj.dones[end_step - 1])
        if slice_done:
            next_value = 0.0
        elif end_step < traj.length:
            next_value = float(traj.values[end_step])
        else:
            raise RuntimeError("Nonterminal sample ends at trajectory boundary without next value.")

        self.counters.replay_samples_drawn += 1
        self.counters.total_sequence_length += sequence_length

        return ReplaySample(
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
            source_trajectory_id=trajectory_id,
            start_step=int(start_step),
            length=int(sequence_length),
            next_value=next_value,
            episode_done=slice_done,
            collected_update_count=int(getattr(traj, "collected_update_count", 0)),
        )

    def sample_batch(self, n: int, **kw) -> list:
        return [self.sample(**kw) for _ in range(n)]

    def __len__(self):
        return len(self._buffer)

    @property
    def longest_trajectory_length(self) -> int:
        return max((t.length for t in self._buffer), default=0)

    @property
    def achievement_coverage(self) -> dict:
        return dict(self._achievement_coverage)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "buffer": self._buffer,
            "next_id": self._next_id,
            "counters": self.counters,
            "achievement_coverage": dict(self._achievement_coverage),
            "rng_state": self._rng.get_state(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ReplayBuffer":
        buf = cls(capacity=state["capacity"])
        buf._buffer = state["buffer"]
        buf._next_id = state["next_id"]
        buf.counters = state["counters"]
        buf._achievement_coverage = defaultdict(int, state.get("achievement_coverage", {}))
        buf._rng.set_state(state["rng_state"])
        return buf

    def hash_digest(self) -> str:
        h = hashlib.sha256()
        h.update(str(len(self._buffer)).encode())
        for t in self._buffer:
            h.update(str(t.length).encode())
            h.update(t.observations.tobytes())
            h.update(np.asarray(t.anchor_steps).tobytes())
            if t.n_anchors:
                h.update(np.asarray(t.memory_anchors).tobytes())
        h.update(str(self.counters.snapshot()).encode())
        return h.hexdigest()[:16]
