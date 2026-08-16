"""Strict transition / budget accounting (G7)."""
from __future__ import annotations

import time
from typing import Dict

from ..runtime.hashing import hash_payload

CATEGORIES = ("probe", "diagnosis", "training", "anchor", "original")


class TransitionAccounting:
    def __init__(self) -> None:
        self.counters: Dict[str, int] = {c: 0 for c in CATEGORIES}
        self.ppo_updates = 0
        self.llm_calls = 0
        self.llm_tokens = 0
        self.wall_clock_sec = 0.0
        self._t0 = time.time()

    def record(self, category: str, transitions: int) -> None:
        if category not in self.counters:
            raise KeyError(f"unknown accounting category {category}")
        self.counters[category] += int(transitions)

    def record_ppo(self, updates: int) -> None:
        self.ppo_updates += int(updates)

    @property
    def total(self) -> int:
        return sum(self.counters.values())

    def finalize(self, *, student_version: str = "") -> dict:
        self.wall_clock_sec = time.time() - self._t0
        body = {
            **self.counters,
            "total_simulator_transitions": self.total,
            "ppo_updates": self.ppo_updates,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "wall_clock_sec": round(self.wall_clock_sec, 3),
            "env_steps_per_sec": (self.total / self.wall_clock_sec
                                  if self.wall_clock_sec > 0 else 0.0),
            "student_version": student_version,
        }
        body["accounting_hash"] = hash_payload(
            {k: v for k, v in body.items() if k != "accounting_hash"})
        return body