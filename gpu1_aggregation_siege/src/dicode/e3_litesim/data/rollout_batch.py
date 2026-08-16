"""OnPolicyRolloutBatch: the Transition-compatible batch produced by the
Lightweight Simulator Data Engine and consumed by PPOBridge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OnPolicyRolloutBatch:
    obs: Any               # (T, B, O)
    actions: Any           # (T, B)
    rewards: Any           # (T, B)
    dones: Any             # (T, B)
    old_logp: Any          # (T, B)
    old_value: Any         # (T, B)
    entering_memory: dict  # (B, ...) architecture-correct entering state
    bootstrap_value: Any   # (B,)
    start_state_ids: list
    frontier_family: str
    rollout_length: int
    horizon: int
    policy_hash: str
    student_version: str
    terminal_reason: list
    trace: list = field(default_factory=list)   # batched EnvState snapshots
    memory_trace: list = field(default_factory=list)  # per-step entering memory
    metrics: dict = field(default_factory=dict)

    @property
    def num_transitions(self) -> int:
        return int(self.obs.shape[0] * self.obs.shape[1])

    def to_dict(self) -> dict:
        return {
            "start_state_ids": list(self.start_state_ids),
            "frontier_family": self.frontier_family,
            "rollout_length": self.rollout_length,
            "horizon": self.horizon,
            "num_transitions": self.num_transitions,
            "policy_hash": self.policy_hash,
            "student_version": self.student_version,
            "terminal_reason": list(self.terminal_reason),
        }