"""DeterministicScheduler: config-driven distribution weights (no LLM)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from ..runtime.hashing import hash_payload


@dataclass
class SchedulerConfig:
    repair: float = 0.50
    expansion: float = 0.20
    high_lp: float = 0.10
    anchors: float = 0.10
    original: float = 0.10


class DeterministicScheduler:
    def __init__(self, config: SchedulerConfig = SchedulerConfig()) -> None:
        self.config = config

    def build_distribution(self, frontier, evidence=None,
                           learning_progress=None, forgetting: float = 0.0):
        w = {"repair": self.config.repair, "expansion": self.config.expansion,
             "high_lp": self.config.high_lp, "anchors": self.config.anchors,
             "original": self.config.original}
        if evidence is not None and not getattr(evidence, "unknown", True):
            w["repair"] += 0.10
            w["expansion"] = max(0.05, w["expansion"] - 0.10)
        if forgetting and forgetting > 0.1:
            w["anchors"] += 0.05
            w["high_lp"] = max(0.05, w["high_lp"] - 0.05)
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        body = {"frontier": getattr(frontier, "spec_hash", ""),
                "evidence": getattr(evidence, "evidence_hash", None),
                "learning_progress": learning_progress,
                "forgetting": forgetting, "config": asdict(self.config)}
        return {"weights": w, "scheduler_hash": hash_payload(body)}