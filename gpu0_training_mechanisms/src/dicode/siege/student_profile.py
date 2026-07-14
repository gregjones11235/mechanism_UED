"""StudentProfileLog — tracks skill mastery on held-out evaluation.

Tier definitions (reporting only, never passed to LLM/selector):
  Tier 0: Never attempted or SR < 0.1
  Tier 1: 0.1 <= SR < 0.5 (emerging)
  Tier 2: 0.5 <= SR < 0.8 (developing)
  Tier 3: 0.8 <= SR < 0.95 (proficient)
  Tier 4: SR >= 0.95 (mastered)

Tier labels are NEVER passed to: LLM, modeler, generator, selector, auction, Copeland.
Only raw held-out SR values and binary mastery flags may be consumed.
"""

import json, os, time
from collections import defaultdict
from typing import Optional


class StudentProfileLog:
    """Tracks per-achievement held-out evaluation success rates."""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path
        self.achievements: dict[str, dict] = {}  # ach_name -> {sr, history, tier, sessions_since_last}
        self.session_count = 0
        self._loaded = False
        if state_path and os.path.exists(state_path):
            self.load(state_path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.achievements = data.get("achievements", {})
        self.session_count = data.get("session_count", 0)
        self._loaded = True

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.state_path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "achievements": self.achievements,
                "session_count": self.session_count,
            }, f, indent=2)

    def update(self, held_out_metrics: dict) -> None:
        """Update profile with new held-out evaluation results.

        Args:
            held_out_metrics: Dict mapping achievement_name -> success_rate (0-1).
        """
        self.session_count += 1
        for ach_name, sr in held_out_metrics.items():
            sr = float(sr)
            if ach_name not in self.achievements:
                self.achievements[ach_name] = {"history": [], "tier": 0}
            entry = self.achievements[ach_name]
            entry["history"].append({"session": self.session_count, "sr": sr})
            # Keep last 50 sessions
            if len(entry["history"]) > 50:
                entry["history"] = entry["history"][-50:]
            entry["sr"] = sr
            entry["tier"] = self._compute_tier(sr)
            entry["sessions_since_evaluated"] = 0
        # Increment sessions_since for unevaluated achievements
        for ach_name in self.achievements:
            if ach_name not in held_out_metrics:
                self.achievements[ach_name]["sessions_since_evaluated"] = \
                    self.achievements[ach_name].get("sessions_since_evaluated", 0) + 1

    @staticmethod
    def _compute_tier(sr: float) -> int:
        if sr >= 0.95: return 4
        if sr >= 0.8:  return 3
        if sr >= 0.5:  return 2
        if sr >= 0.1:  return 1
        return 0

    def get_tier(self, ach_name: str) -> int:
        return self.achievements.get(ach_name, {}).get("tier", 0)

    def get_sr(self, ach_name: str) -> float:
        return self.achievements.get(ach_name, {}).get("sr", 0.0)

    def is_mastered(self, ach_name: str) -> bool:
        """Binary: mastered = tier >= 4 (SR >= 0.95 on held-out)."""
        return self.get_tier(ach_name) >= 4

    def is_proficient(self, ach_name: str) -> bool:
        """Binary: proficient = tier >= 3 (SR >= 0.8 on held-out)."""
        return self.get_tier(ach_name) >= 3

    def get_forgetting_risk(self, ach_name: str, threshold: float = 0.05) -> bool:
        """Check if achievement shows evidence of forgetting.

        Forgetting = SR declined by more than threshold from best observed.
        """
        entry = self.achievements.get(ach_name, {})
        history = entry.get("history", [])
        if len(history) < 2:
            return False
        best = max(h["sr"] for h in history)
        current = history[-1]["sr"]
        return (best - current) > threshold

    @property
    def tier3_plus_count(self) -> int:
        return sum(1 for a in self.achievements.values() if a.get("tier", 0) >= 3)

    @property
    def tier4_count(self) -> int:
        return sum(1 for a in self.achievements.values() if a.get("tier", 0) >= 4)

    @property
    def summary(self) -> dict:
        tiers = defaultdict(int)
        for a in self.achievements.values():
            tiers[a.get("tier", 0)] += 1
        return {
            "session_count": self.session_count,
            "total_achievements": len(self.achievements),
            "tier_distribution": dict(tiers),
            "tier3_plus": self.tier3_plus_count,
            "tier4": self.tier4_count,
        }
