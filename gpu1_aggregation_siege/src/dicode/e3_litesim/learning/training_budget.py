"""Training budget tracking: every simulator transition is accounted."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingBudget:
    max_simulator_transitions: int = 10 ** 9
    max_ppo_updates: int = 10 ** 9
    used_transitions: int = 0
    used_ppo_updates: int = 0

    def spend(self, transitions: int = 0, ppo_updates: int = 0) -> None:
        self.used_transitions += int(transitions)
        self.used_ppo_updates += int(ppo_updates)
        if self.used_transitions > self.max_simulator_transitions:
            raise RuntimeError("BUDGET_EXCEEDED: simulator transitions")
        if self.used_ppo_updates > self.max_ppo_updates:
            raise RuntimeError("BUDGET_EXCEEDED: ppo updates")

    def remaining(self) -> dict:
        return {
            "transitions": self.max_simulator_transitions - self.used_transitions,
            "ppo_updates": self.max_ppo_updates - self.used_ppo_updates,
        }