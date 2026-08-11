"""W512 × P2-Replay — replay buffer schema extension (CC2 corrected §二).

EXTENDS (does not modify) the frozen P2-Full-A `replay_buffer.py` dataclasses with the W512
raw-history anchor fields, EXACTLY mirroring `rmt_replay_buffer.py`. The P2 anchor carries ONLY
the GTrXL short memory (memories/mask/idx). W512 adds a 512-step raw-history state — a 128-step
delay line (delay_buf/delay_idx/delay_count) feeding a 384-step ring (long_buf/long_mask/long_idx)
of GTrXL h_t, plus the episode-local segment counter (seg_step) that drives RESET128 clearing — so
every anchor must ALSO snapshot that whole state. Otherwise the reconstructed pre-action state
cannot be bit-exact (the same conservation principle as the RMT16 anchor, directive §五).

All P2-Full-A frozen coefficients / conservation rules are inherited UNCHANGED (capacity=64,
ANCHOR_INTERVAL=128, MIN_SEQUENCE_LENGTH=129, deterministic eligible-only sampling, no
success/quality/TD-error prioritisation). This module adds NO new replay coefficient.

W512 anchor fields are goal-independent (built from h_t), so on any hypothetical hindsight
relabel they would carry through unchanged exactly like the GTrXL memory_anchors. (The canonical
W512 run is replay_mode=original_vtrace / hindsight=false, so no relabel ever touches them; the
fields are simply carried, matching the RMT schema.)

The policy-version RANGE provenance block + validators + EligibleSampleBatch are imported
VERBATIM from rmt_replay_buffer (they are network-agnostic pure functions / dataclasses; that
module imports only numpy + replay_buffer, no JAX, so importing it here is frozen-safe).
"""
from dataclasses import dataclass, field

import numpy as np

from replay_buffer import (
    ReplayBuffer, ReplaySample, Trajectory,
    ANCHOR_INTERVAL, MIN_SEQUENCE_LENGTH, anchor_steps_for_length,
)
# Network-agnostic provenance helpers / eligible-batch result, reused unchanged.
from rmt_replay_buffer import (
    EligibleSampleBatch,
    validate_policy_version_range_fields,
    validate_sample_policy_version_range,
)


@dataclass
class W512Trajectory(Trajectory):
    """A complete episode with sparse GTrXL memory anchors AND W512 raw-history anchors.

    W512 additions (aligned 1:1 with the GTrXL anchors; same anchor_steps). Each anchor snapshots
    the ENTERING w512_memory state (before that step's RESET128 clear), so reconstruction re-applies
    the identical transition from the anchor (bit-exact by construction):
      w512_initial_*    : the w512 state entering episode step 0 (the fresh state)
      w512_anchor_delay_buf   : [N, delay_size, D]   delay line entering each anchor step
      w512_anchor_delay_idx   : [N]                  delay write index
      w512_anchor_delay_count : [N]                  delay fill count
      w512_anchor_long_buf    : [N, long_size, D]    384 ring of h_t
      w512_anchor_long_mask   : [N, long_size]       valid-slot mask (bool)
      w512_anchor_long_idx    : [N]                  ring write index
      w512_anchor_seg_step    : [N]                  episode-local segment counter (drives reset128)
    """
    # ---- entering state at episode step 0 (fresh) ----
    w512_initial_delay_buf: np.ndarray = field(default_factory=lambda: np.array([]))
    w512_initial_delay_idx: int = 0
    w512_initial_delay_count: int = 0
    w512_initial_long_buf: np.ndarray = field(default_factory=lambda: np.array([]))
    w512_initial_long_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    w512_initial_long_idx: int = 0
    w512_initial_seg_step: int = 0
    # ---- per-anchor entering state (1:1 with anchor_steps) ----
    w512_anchor_delay_buf: np.ndarray = field(default_factory=lambda: np.array([]))
    w512_anchor_delay_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    w512_anchor_delay_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    w512_anchor_long_buf: np.ndarray = field(default_factory=lambda: np.array([]))
    w512_anchor_long_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    w512_anchor_long_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    w512_anchor_seg_step: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # ---- split provenance (ADDITIVE, default 0; identical to the RMT schema) ----
    outer_update_index: int = 0
    policy_version_start: int = 0
    policy_version_end: int = 0
    policy_version_span: int = 0
    policy_version_at_collection: int = 0   # DEPRECATED alias of policy_version_start

    def validate_anchors(self):
        """P2 conservation PLUS W512 anchor count == GTrXL anchor count (1:1 aligned)."""
        super().validate_anchors()
        expected = anchor_steps_for_length(self.length)
        n_exp = len(expected)
        if self.w512_anchor_long_buf.size == 0:
            raise ValueError("W512 anchor long_buf missing (W512 conservation failed).")
        if int(self.w512_anchor_long_buf.shape[0]) != n_exp:
            raise ValueError(
                f"W512 anchor count {self.w512_anchor_long_buf.shape[0]} != expected {n_exp}.")
        for nm, arr in (("delay_buf", self.w512_anchor_delay_buf),
                        ("delay_idx", self.w512_anchor_delay_idx),
                        ("delay_count", self.w512_anchor_delay_count),
                        ("long_mask", self.w512_anchor_long_mask),
                        ("long_idx", self.w512_anchor_long_idx),
                        ("seg_step", self.w512_anchor_seg_step)):
            if int(np.asarray(arr).shape[0]) != n_exp:
                raise ValueError(f"W512 anchor {nm} count mismatch "
                                 f"({int(np.asarray(arr).shape[0])} != {n_exp}).")
        # policy-version range invariants (base insert validates anchors -> rejects bad range)
        self.validate_policy_version_range()
        return True

    def validate_policy_version_range(self):
        """Validate the episode policy-version RANGE block (pure; never modifies a field)."""
        return validate_policy_version_range_fields(
            self.policy_version_start, self.policy_version_end,
            self.policy_version_span, self.policy_version_at_collection)

    def nearest_w512_anchor_le(self, step: int):
        """(anchor_step, w512_state_dict) for the largest anchor step <= step.

        w512_state_dict has keys delay_buf/delay_idx/delay_count/long_buf/long_mask/long_idx/
        seg_step (numpy, ENTERING state at that anchor)."""
        a_step, _ = self.nearest_anchor_le(step)   # reuse base index search (raises if none)
        for i, s in enumerate(self.anchor_steps):
            if int(s) == a_step:
                st = {
                    "delay_buf":   np.asarray(self.w512_anchor_delay_buf[i]).copy(),
                    "delay_idx":   int(np.asarray(self.w512_anchor_delay_idx)[i]),
                    "delay_count": int(np.asarray(self.w512_anchor_delay_count)[i]),
                    "long_buf":    np.asarray(self.w512_anchor_long_buf[i]).copy(),
                    "long_mask":   np.asarray(self.w512_anchor_long_mask[i]).copy(),
                    "long_idx":    int(np.asarray(self.w512_anchor_long_idx)[i]),
                    "seg_step":    int(np.asarray(self.w512_anchor_seg_step)[i]),
                }
                return a_step, st
        raise RuntimeError(f"W512 anchor at step {a_step} not found.")


@dataclass
class W512ReplaySample(ReplaySample):
    """A sampled loss region + GTrXL anchor data + W512 anchor data for reconstruction."""
    pre_anchor_w512_delay_buf: np.ndarray = field(default_factory=lambda: np.array([]))   # [delay_size, D]
    pre_anchor_w512_delay_idx: int = 0
    pre_anchor_w512_delay_count: int = 0
    pre_anchor_w512_long_buf: np.ndarray = field(default_factory=lambda: np.array([]))    # [long_size, D]
    pre_anchor_w512_long_mask: np.ndarray = field(default_factory=lambda: np.array([]))   # [long_size]
    pre_anchor_w512_long_idx: int = 0
    pre_anchor_w512_seg_step: int = 0
    # source-episode policy-version RANGE (propagated verbatim by sample(); default 0)
    policy_version_start: int = 0
    policy_version_end: int = 0
    policy_version_span: int = 0
    policy_version_at_collection: int = 0


class W512ReplayBuffer(ReplayBuffer):
    """Fixed-capacity ring of complete W512 episodes (inherits P2 frozen sampling).

    Sampling rules are inherited VERBATIM from P2-Full-A (deterministic RandomState, complete
    episodes only, length > 128, contiguous L_seq window, nearest-anchor-le burn-in). NO
    success/quality/TD-error prioritisation is added (directive §七). sample_eligible is the
    deterministic eligible-only sampler (identical control flow to RMTReplayBuffer.sample_eligible).
    """

    def insert(self, trajectory: "W512Trajectory") -> int:
        if not isinstance(trajectory, W512Trajectory):
            raise TypeError("W512ReplayBuffer requires W512Trajectory.")
        return super().insert(trajectory)   # base validates anchors (incl. W512 via override)

    def sample(self, sequence_length=None, trajectory_id=None, start_step=None) -> W512ReplaySample:
        """Same as P2 sample but returns a W512ReplaySample with W512 anchor data attached."""
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
        # W512 anchor state at the SAME nearest step
        _, w512_st = traj.nearest_w512_anchor_le(start_step)
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

        sample = W512ReplaySample(
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
            policy_version_start=int(getattr(traj, "policy_version_start", 0)),
            policy_version_end=int(getattr(traj, "policy_version_end", 0)),
            policy_version_span=int(getattr(traj, "policy_version_span", 0)),
            policy_version_at_collection=int(getattr(traj, "policy_version_start", 0)),
            pre_anchor_w512_delay_buf=w512_st["delay_buf"],
            pre_anchor_w512_delay_idx=int(w512_st["delay_idx"]),
            pre_anchor_w512_delay_count=int(w512_st["delay_count"]),
            pre_anchor_w512_long_buf=w512_st["long_buf"],
            pre_anchor_w512_long_mask=w512_st["long_mask"],
            pre_anchor_w512_long_idx=int(w512_st["long_idx"]),
            pre_anchor_w512_seg_step=int(w512_st["seg_step"]),
        )
        # the constructed sample's policy-version range must satisfy the invariants (read-only)
        validate_sample_policy_version_range(sample)
        return sample

    def sample_batch(self, n: int, **kw) -> list:
        return [self.sample(**kw) for _ in range(n)]

    def sample_eligible(self, sequence_length, rng, batch_size) -> EligibleSampleBatch:
        """Eligible-ONLY, fixed-size, deterministic replay sampling (identical control flow to
        RMTReplayBuffer.sample_eligible — directive §七). Pre-filters to trajectories with
        length >= sequence_length; never selects a too-short trajectory; empty eligible set ->
        explicit status="NOT_READY" (no redraw / no retry / no exception); OK -> exactly
        batch_size samples; all randomness from the supplied rng (bit-identical given same buffer
        + rng state)."""
        sequence_length = int(sequence_length)
        batch_size = int(batch_size)
        if sequence_length < MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"sample_eligible: sequence_length {sequence_length} < {MIN_SEQUENCE_LENGTH} REJECTED.")
        if batch_size < 1:
            raise ValueError(f"sample_eligible: batch_size {batch_size} < 1 REJECTED.")

        eligible = [(i, t) for i, t in enumerate(self._buffer) if t.length >= sequence_length]
        eligible_count = len(eligible)
        if eligible_count == 0:
            return EligibleSampleBatch(status="NOT_READY", samples=[], sample_ids=[],
                                       start_offsets=[], sequence_lengths=[],
                                       eligible_count=0, batch_size=batch_size)

        samples, sample_ids, start_offsets, sequence_lengths = [], [], [], []
        for _ in range(batch_size):
            _, traj = eligible[rng.randint(eligible_count)]
            max_start = traj.length - sequence_length
            start_step = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
            s = self.sample(trajectory_id=traj.trajectory_id, start_step=start_step,
                            sequence_length=sequence_length)
            samples.append(s)
            sample_ids.append(int(traj.trajectory_id))
            start_offsets.append(int(start_step))
            sequence_lengths.append(int(sequence_length))

        return EligibleSampleBatch(status="OK", samples=samples, sample_ids=sample_ids,
                                   start_offsets=start_offsets,
                                   sequence_lengths=sequence_lengths,
                                   eligible_count=eligible_count, batch_size=batch_size)
