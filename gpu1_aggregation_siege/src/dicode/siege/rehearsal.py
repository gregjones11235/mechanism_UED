"""Forgetting-triggered rehearsal — prevents catastrophic forgetting.

Controlled by empirical held-out evidence (SR decline), not heuristic timers.
"""
import json, os, time
from typing import Optional


class ForgettingRehearsal:
    """Triggers rehearsal when held-out SR declines for mastered skills."""

    def __init__(self, forgetting_threshold: float = 0.05, state_path: Optional[str] = None):
        self.forgetting_threshold = forgetting_threshold
        self.state_path = state_path
        self.rehearsal_log: list[dict] = []
        self.active_rehearsals: set[str] = set()
        if state_path and os.path.exists(state_path):
            self.load(state_path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.forgetting_threshold = data.get("forgetting_threshold", 0.05)
        self.rehearsal_log = data.get("rehearsal_log", [])
        self.active_rehearsals = set(data.get("active_rehearsals", []))

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.state_path
        if not p: return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "forgetting_threshold": self.forgetting_threshold,
                "rehearsal_log": self.rehearsal_log,
                "active_rehearsals": list(self.active_rehearsals),
            }, f, indent=2)

    def detect_forgetting(self, student_profile) -> list[str]:
        """Detect skills showing empirical forgetting evidence.

        Returns list of achievement names needing rehearsal.
        """
        at_risk = []
        for ach_name in student_profile.achievements:
            if student_profile.get_tier(ach_name) >= 3:  # Only track proficient+
                if student_profile.get_forgetting_risk(ach_name, self.forgetting_threshold):
                    at_risk.append(ach_name)
        return at_risk

    def update(self, student_profile, session: int) -> dict:
        """Update rehearsal state based on current student profile."""
        at_risk = self.detect_forgetting(student_profile)
        previously_active = self.active_rehearsals.copy()
        self.active_rehearsals = set(at_risk)

        newly_at_risk = self.active_rehearsals - previously_active
        recovered = previously_active - self.active_rehearsals

        entry = {
            "session": session,
            "timestamp": time.time(),
            "at_risk_count": len(at_risk),
            "newly_at_risk": list(newly_at_risk),
            "recovered": list(recovered),
            "active": list(self.active_rehearsals),
        }
        self.rehearsal_log.append(entry)
        return entry

    @property
    def rehearsal_active(self) -> bool:
        return len(self.active_rehearsals) > 0

    @property
    def rehearsal_count(self) -> int:
        return len(self.active_rehearsals)

    @property
    def summary(self) -> dict:
        return {
            "active_rehearsals": self.rehearsal_count,
            "total_triggers": len(self.rehearsal_log),
            "forgetting_threshold": self.forgetting_threshold,
        }
