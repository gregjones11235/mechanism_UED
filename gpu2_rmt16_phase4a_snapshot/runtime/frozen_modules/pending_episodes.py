"""Cross-rollout persistent per-env episode buffers (P2-Full-A, sparse-anchor schema).

Adapted from P2-v1 方案B. Each env slot persists its in-progress episode ACROSS
rollouts so a completed episode can exceed rollout_steps (and thus reach
MIN_SEQUENCE_LENGTH=129 for replay). Differences from P2-v1 for the frozen §2 schema:

  * NO per-step memory_sequence storage (the ~256 KB/step disk killer is removed).
  * Instead each slot accumulates SPARSE memory ANCHORS: the pre-action entering state
    (memory, mask, idx) snapshotted at episode steps 0,128,256,... Replay reconstructs
    any loss-window-start memory from the nearest anchor (<=128 burn-in steps).
  * initial_memory is still the memory before the episode's first action (set once).
  * The terminal transition stays in the finished episode; auto-reset begins a new
    episode (new id) on the next step. Rollout boundaries are never written as done.
  * Buffers + anchors are checkpointable for exact resume.
"""
from typing import List, Optional


def _fresh_slot_dict() -> dict:
    return {
        "obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
        "next_obs": [], "ach": [],
        "init_mem": None,                 # memory before the episode's first action
        # sparse anchors (entering state at episode steps 0,128,256,...)
        "anchor_mem": [],                 # list of [wm, layers, embed]
        "anchor_mask": [],                # list of [heads, 1, wm+1]
        "anchor_idx": [],                 # list of int
        "anchor_step": [],                # list of int (episode-local step)
    }


class PendingEpisodeBuffers:
    """Persistent per-env-slot episode buffers carried across rollouts."""

    def __init__(self, num_envs: int, first_episode_id: int = 0,
                 first_policy_version: int = 0):
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

    def reset_slot(self, e: int, policy_version: int) -> int:
        """Reset slot e to a fresh buffer with a NEW episode id (auto-reset isolation)."""
        self.slots[e] = _fresh_slot_dict()
        self.episode_id[e] = self.next_episode_id
        self.policy_version[e] = int(policy_version)
        self.next_episode_id += 1
        return self.episode_id[e]

    # ---- anchor accounting -------------------------------------------
    def add_anchor(self, e: int, step: int, memory, mask, idx) -> None:
        s = self.slots[e]
        s["anchor_mem"].append(memory)
        s["anchor_mask"].append(mask)
        s["anchor_idx"].append(int(idx))
        s["anchor_step"].append(int(step))

    def slot_anchor_steps(self, e: int) -> List[int]:
        return list(self.slots[e]["anchor_step"])

    # ---- transition accounting ---------------------------------------
    def slot_lengths(self) -> List[int]:
        return [len(s["obs"]) for s in self.slots]

    def total_pending_transitions(self) -> int:
        return sum(len(s["obs"]) for s in self.slots)

    def total_pending_anchors(self) -> int:
        return sum(len(s["anchor_mem"]) for s in self.slots)

    def is_empty(self) -> bool:
        return self.total_pending_transitions() == 0

    # ---- checkpoint serialization ------------------------------------
    def state_dict(self) -> dict:
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
                f"anchors={self.total_pending_anchors()}, "
                f"lengths={self.slot_lengths()}, "
                f"next_episode_id={self.next_episode_id})")
