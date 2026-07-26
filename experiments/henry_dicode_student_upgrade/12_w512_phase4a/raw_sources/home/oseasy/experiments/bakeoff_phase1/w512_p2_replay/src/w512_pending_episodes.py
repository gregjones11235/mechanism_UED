"""W512 × P2 pending episode buffers: extends P2-Full-A PendingEpisodeBuffers
with W512 long state anchor storage.
"""
from typing import List


def _fresh_slot_dict_w512() -> dict:
    return {
        "obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
        "next_obs": [], "ach": [],
        "init_mem": None,
        "init_w512_state": None,
        "anchor_mem": [],
        "anchor_mask": [],
        "anchor_idx": [],
        "anchor_step": [],
        "w512_anchor_state": [],       # list of dict per anchor
    }


class W512PendingEpisodeBuffers:
    """Persistent per-env-slot episode buffers with W512 long state anchors."""

    def __init__(self, num_envs: int, first_policy_version: int = 0):
        self.num_envs = int(num_envs)
        self.next_episode_id = 0
        self.slots: List[dict] = []
        self.episode_id: List[int] = []
        self.policy_version: List[int] = []
        for _ in range(self.num_envs):
            self.slots.append(_fresh_slot_dict_w512())
            self.episode_id.append(self.next_episode_id)
            self.policy_version.append(int(first_policy_version))
            self.next_episode_id += 1

    def reset_slot_w512(self, e: int, policy_version: int) -> int:
        self.slots[e] = _fresh_slot_dict_w512()
        self.episode_id[e] = self.next_episode_id
        self.policy_version[e] = int(policy_version)
        self.next_episode_id += 1
        return self.episode_id[e]

    def add_anchor(self, e: int, step: int, memory, mask, idx) -> None:
        s = self.slots[e]
        s["anchor_mem"].append(memory)
        s["anchor_mask"].append(mask)
        s["anchor_idx"].append(int(idx))
        s["anchor_step"].append(int(step))

    def total_pending_transitions(self) -> int:
        return sum(len(s["obs"]) for s in self.slots)

    def total_pending_anchors(self) -> int:
        return sum(len(s["anchor_mem"]) for s in self.slots)

    def state_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "next_episode_id": self.next_episode_id,
            "slots": self.slots,
            "episode_id": list(self.episode_id),
            "policy_version": list(self.policy_version),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "W512PendingEpisodeBuffers":
        obj = cls.__new__(cls)
        obj.num_envs = int(state["num_envs"])
        obj.next_episode_id = int(state["next_episode_id"])
        obj.slots = state["slots"]
        obj.episode_id = list(state["episode_id"])
        obj.policy_version = list(state["policy_version"])
        return obj
