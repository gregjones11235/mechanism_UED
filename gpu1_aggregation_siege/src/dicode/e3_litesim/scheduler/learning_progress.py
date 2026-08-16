"""Learning progress / regret / forgetting trackers (deterministic EMAs)."""
from __future__ import annotations

from typing import Dict


class LearningProgressTracker:
    def __init__(self, ema: float = 0.5) -> None:
        self.ema = ema
        self._last: Dict[str, float] = {}
        self._ema: Dict[str, float] = {}
        self._best: Dict[str, float] = {}

    def update(self, family: str, success_rate: float) -> dict:
        prev = self._last.get(family)
        lp = (success_rate - prev) if prev is not None else 0.0
        self._last[family] = success_rate
        self._ema[family] = (self.ema * success_rate +
                             (1 - self.ema) * self._ema.get(family, success_rate))
        self._best[family] = max(self._best.get(family, 0.0), success_rate)
        return {"family": family, "learning_progress": lp,
                "regret_proxy": 1.0 - success_rate,
                "forgetting": self._best[family] - success_rate}

    def forgetting_of(self, family: str, current: float) -> float:
        return max(0.0, self._best.get(family, 0.0) - current)