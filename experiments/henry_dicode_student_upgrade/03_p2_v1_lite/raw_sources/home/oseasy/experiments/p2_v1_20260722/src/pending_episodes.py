"""Cross-rollout persistent per-env episode buffers (P2-v1 方案B).

Background
----------
The P2-v1 rollout collector (``p2_v1_core.collect_rollout``) originally rebuilt
its per-env episode buffer from scratch on every call.  A Trajectory was emitted
only when an episode hit ``done=True`` *within* a single rollout, so a captured
trajectory's length was bounded by ``rollout_steps``.  With
``ROLLOUT_STEPS=128`` and ``TrajectoryReplayBuffer.MIN_SEQUENCE_LENGTH=129``
(strictly > 128), ``can_sample()`` was therefore ALWAYS False and the replay
auxiliary update (and the hindsight relabel gated on it) was structurally
unreachable at any number of env steps (see reports/p2_v1_level2_threshold_audit.md).

方案B (user decision 2026-07-22) fixes this by persisting each env slot's
in-progress episode buffer ACROSS rollouts:

  * rollout ends, episode NOT done  -> buffer retained, next rollout appends;
  * episode truly done              -> full Trajectory formed, inserted into
                                       replay, slot reset with a NEW episode id;
  * a rollout boundary is NEVER written as a done, never loses the pre-boundary
    prefix, and never re-appends a boundary transition;
  * ``initial_memory`` is the memory before the episode's FIRST action (set once
    when the episode starts, then carried across rollouts) -- never a mid-rollout
    boundary memory;
  * each env slot buffers independently (no cross-slot splicing);
  * the pending buffers are checkpointable so an interrupted run resumes exactly.

Because an episode can now span several rollouts, its captured length can exceed
``rollout_steps`` and reach ``>= MIN_SEQUENCE_LENGTH`` -- making replay/hindsight
genuinely reachable.

This module does NOT change the on-policy PPO main path: the RolloutBatch is
built from the per-step accumulators exactly as before; the pending buffers only
feed the complete-episode replay channel.  With replay/hindsight disabled the
training is bit-identical to the previous collector (see test 9 / original PPO
equivalence).
"""

from typing import List, Optional


def _fresh_slot_dict() -> dict:
    """A fresh, empty per-slot transition accumulator.

    The keys mirror the historical ``ep[e]`` dict in ``collect_rollout`` so the
    Trajectory construction stays byte-for-byte identical:

      obs/act/rew/don/val/lp/next_obs/mem_pre/mask_pre/ach : per-step lists
      init_mem : memory BEFORE the episode's first action (set once)
    """
    return {
        "obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
        "next_obs": [], "mem_pre": [], "mask_pre": [], "init_mem": None,
        "ach": [],
    }


class PendingEpisodeBuffers:
    """Persistent per-env-slot episode buffers carried across rollouts.

    Attributes
    ----------
    num_envs : int
        Number of parallel env slots (fixed for the run).
    slots : list[dict]
        One ``_fresh_slot_dict()`` per env slot, holding the transitions of the
        episode currently being collected on that slot.
    episode_id : list[int]
        Stable monotonic episode id per slot.  A new id is assigned every time a
        slot starts a fresh episode (initial reset or post-done auto-reset).
    policy_version : list[int]
        The ``collected_update_count`` at which the slot's current episode began
        being collected (provenance for the off-policy lag accounting).
    next_episode_id : int
        Counter for the next fresh episode id (checkpointed for exact resume).
    """

    def __init__(
        self,
        num_envs: int,
        first_episode_id: int = 0,
        first_policy_version: int = 0,
    ):
        self.num_envs = int(num_envs)
        self.next_episode_id = int(first_episode_id)
        self.slots: List[dict] = []
        self.episode_id: List[int] = []
        self.policy_version: List[int] = []
        for _ in range(self.num_envs):
            self.slots.append(_fresh_slot_dict())
            self.episode_id.append(self.next_episode_id)
            self.policy_version.append(int(first_policy_version))
            self.next_episode_id += 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_slot(self, e: int, policy_version: int) -> int:
        """Reset slot ``e`` to a fresh buffer with a NEW episode id.

        Called after an episode on slot ``e`` completes (done=True) so the
        auto-reset observation starts a brand-new episode (auto-reset isolation:
        the terminal transition stays in the finished episode, the reset
        observation begins the next).  Returns the new episode id.
        """
        self.slots[e] = _fresh_slot_dict()
        self.episode_id[e] = self.next_episode_id
        self.policy_version[e] = int(policy_version)
        self.next_episode_id += 1
        return self.episode_id[e]

    # ------------------------------------------------------------------
    # Accounting (used by the no-dup/no-loss invariant test)
    # ------------------------------------------------------------------

    def slot_lengths(self) -> List[int]:
        """Current accumulated length of each slot's pending episode."""
        return [len(s["obs"]) for s in self.slots]

    def total_pending_transitions(self) -> int:
        """Total transitions held across all pending (incomplete) episodes."""
        return sum(len(s["obs"]) for s in self.slots)

    def is_empty(self) -> bool:
        return self.total_pending_transitions() == 0

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serializable snapshot for the full checkpoint.

        The per-slot dicts contain Python lists of numpy arrays / scalars, which
        pickle faithfully (bit-exact) so a restored buffer reproduces the same
        completed trajectories as an uninterrupted run.
        """
        return {
            "num_envs": self.num_envs,
            "next_episode_id": self.next_episode_id,
            "slots": self.slots,
            "episode_id": list(self.episode_id),
            "policy_version": list(self.policy_version),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PendingEpisodeBuffers":
        obj = cls.__new__(cls)
        obj.num_envs = int(state["num_envs"])
        obj.next_episode_id = int(state["next_episode_id"])
        obj.slots = state["slots"]
        obj.episode_id = list(state["episode_id"])
        obj.policy_version = list(state["policy_version"])
        return obj

    def __repr__(self) -> str:
        return (f"PendingEpisodeBuffers(num_envs={self.num_envs}, "
                f"pending={self.total_pending_transitions()}, "
                f"lengths={self.slot_lengths()}, "
                f"next_episode_id={self.next_episode_id})")
