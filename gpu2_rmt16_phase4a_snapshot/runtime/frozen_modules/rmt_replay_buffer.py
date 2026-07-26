"""RMT16 × P2-Replay — replay buffer schema extension (Phase4A).

EXTENDS (does not modify) the frozen P2-Full-A `replay_buffer.py` dataclasses with the
RMT16 persistent-memory anchor fields. The P2 anchor carries ONLY the GTrXL short memory
(memories/mask/idx). RMT16 adds 16 persistent memory tokens (+ segment buffer + count),
so every anchor must ALSO snapshot them — otherwise the reconstructed pre-action state
cannot be bit-exact (directive §五: anchor = GTrXL short memory + 16 RMT tokens +
seg_buf + seg_count + episode position + done/mask + RNG + behavior policy info).

All P2-Full-A frozen coefficients / conservation rules are inherited UNCHANGED
(capacity=64, ANCHOR_INTERVAL=128, MIN_SEQUENCE_LENGTH=129, deterministic sampling,
no success/quality/TD-error prioritisation). This module adds NO new replay coefficient.

RMT anchor fields are goal-independent (built from h_t), so on hindsight relabel they are
carried through unchanged exactly like the GTrXL memory_anchors (standard hindsight
approximation; the <=128-step burn-in with relabeled obs re-contextualises them).
"""
from dataclasses import dataclass, field

import numpy as np

from replay_buffer import (
    ReplayBuffer, ReplaySample, Trajectory,
    ANCHOR_INTERVAL, MIN_SEQUENCE_LENGTH, anchor_steps_for_length,
)


@dataclass
class RMTTrajectory(Trajectory):
    """A complete episode with sparse GTrXL memory anchors AND RMT16 token anchors.

    RMT additions (aligned 1:1 with the GTrXL anchors; same anchor_steps):
      rmt_initial_tokens   : [num_tokens, D]   tokens at episode step 0 (entering state)
      rmt_initial_segbuf   : [segment_len, D]  seg_buf at episode step 0
      rmt_initial_segcount : int               seg_count at episode step 0
      rmt_anchor_tokens    : [N, num_tokens, D] mem_tokens entering each anchor step
      rmt_anchor_segbuf    : [N, segment_len, D] seg_buf entering each anchor step
      rmt_anchor_segcount  : [N]               seg_count entering each anchor step
    """
    rmt_initial_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    rmt_initial_segbuf: np.ndarray = field(default_factory=lambda: np.array([]))
    rmt_initial_segcount: int = 0
    rmt_anchor_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    rmt_anchor_segbuf: np.ndarray = field(default_factory=lambda: np.array([]))
    rmt_anchor_segcount: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def validate_anchors(self):
        """P2 conservation PLUS RMT anchor count/steps == GTrXL anchor count/steps."""
        super().validate_anchors()
        expected = anchor_steps_for_length(self.length)
        n_exp = len(expected)
        if self.rmt_anchor_tokens.size == 0:
            raise ValueError("RMT anchor tokens missing (RMT conservation failed).")
        if int(self.rmt_anchor_tokens.shape[0]) != n_exp:
            raise ValueError(
                f"RMT anchor count {self.rmt_anchor_tokens.shape[0]} != expected {n_exp}.")
        got_steps = [int(s) for s in self.rmt_anchor_segcount_steps()]
        if got_steps != expected:
            raise ValueError(
                f"RMT anchor steps {got_steps} != expected {expected}.")
        if int(self.rmt_anchor_segbuf.shape[0]) != n_exp:
            raise ValueError("RMT seg_buf anchor count mismatch.")
        if int(np.asarray(self.rmt_anchor_segcount).shape[0]) != n_exp:
            raise ValueError("RMT seg_count anchor count mismatch.")
        return True

    def rmt_anchor_segcount_steps(self):
        """RMT anchors share anchor_steps with the GTrXL anchors (1:1 aligned)."""
        return list(self.anchor_steps)

    def nearest_rmt_anchor_le(self, step: int):
        """(anchor_step, tokens, segbuf, segcount) for largest anchor step <= step."""
        a_step, _ = self.nearest_anchor_le(step)  # reuse base index search (raises if none)
        for i, s in enumerate(self.anchor_steps):
            if int(s) == a_step:
                return (a_step,
                        np.asarray(self.rmt_anchor_tokens[i]).copy(),
                        np.asarray(self.rmt_anchor_segbuf[i]).copy(),
                        int(np.asarray(self.rmt_anchor_segcount)[i]))
        raise RuntimeError(f"RMT anchor at step {a_step} not found.")


@dataclass
class RMTReplaySample(ReplaySample):
    """A sampled loss region + GTrXL anchor data + RMT16 anchor data for reconstruction."""
    pre_anchor_rmt_tokens: np.ndarray = field(default_factory=lambda: np.array([]))   # [num_tokens, D]
    pre_anchor_rmt_segbuf: np.ndarray = field(default_factory=lambda: np.array([]))   # [segment_len, D]
    pre_anchor_rmt_segcount: int = 0


class RMTReplayBuffer(ReplayBuffer):
    """Fixed-capacity ring of complete RMT episodes (inherits P2 frozen sampling).

    Sampling rules are inherited VERBATIM from P2-Full-A (deterministic RandomState,
    complete episodes only, length > 128, contiguous L_seq window, nearest-anchor-le
    burn-in). NO success/quality/TD-error prioritisation is added (directive §七).
    """

    def insert(self, trajectory: "RMTTrajectory") -> int:
        if not isinstance(trajectory, RMTTrajectory):
            raise TypeError("RMTReplayBuffer requires RMTTrajectory.")
        return super().insert(trajectory)   # base validates anchors (incl. RMT via override)

    def sample(self, sequence_length=None, trajectory_id=None, start_step=None) -> RMTReplaySample:
        """Same as P2 sample but returns an RMTReplaySample with RMT anchor data attached."""
        if not self._buffer:
            raise RuntimeError("Replay buffer is empty.")
        eligible = [(i, t) for i, t in enumerate(self._buffer) if t.length >= MIN_SEQUENCE_LENGTH]
        if not eligible:
            raise RuntimeError("No trajectory exceeds 128 steps; truncated-only REJECTED.")
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

        a_step, a_mem = traj.nearest_anchor_le(start_step)
        burn_in_obs = traj.observations[a_step:start_step].copy()
        # RMT anchor at the SAME nearest step
        _, rmt_tok, rmt_segbuf, rmt_segcount = traj.nearest_rmt_anchor_le(start_step)
        # GTrXL anchor mask/idx (bit-exact reconstruction of the rollout mask evolution)
        a_mask, a_idx = np.array([]), 0
        if len(getattr(traj, "anchor_masks", np.array([]))) and len(traj.anchor_masks):
            for i, s in enumerate(traj.anchor_steps):
                if int(s) == a_step:
                    a_mask = np.asarray(traj.anchor_masks[i]).copy()
                    a_idx = int(np.asarray(traj.anchor_idxs)[i])
                    break

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

        return RMTReplaySample(
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
            pre_anchor_rmt_tokens=rmt_tok,
            pre_anchor_rmt_segbuf=rmt_segbuf,
            pre_anchor_rmt_segcount=int(rmt_segcount),
        )

    def sample_batch(self, n: int, **kw) -> list:
        return [self.sample(**kw) for _ in range(n)]
