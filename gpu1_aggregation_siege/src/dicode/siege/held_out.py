"""Held-out evidence collection for unbiased skill mastery measurement.

Evaluates student checkpoints on held-out Craftax tasks that are never
used for training. Produces per-achievement success rates.
"""

import json, os, time
from typing import Optional


class HeldOutEvaluator:
    """Collects held-out evaluation evidence.

    In production, this would run the student on held-out Craftax tasks
    using the existing evaluation infrastructure. For the isolated port,
    it provides the interface and state management.
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path
        self.evaluations: list[dict] = []  # [{session, global_step, metrics}]
        self.held_out_tasks: list[str] = []
        if state_path and os.path.exists(state_path):
            self.load(state_path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.evaluations = data.get("evaluations", [])
        self.held_out_tasks = data.get("held_out_tasks", [])

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.state_path
        if not p: return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "evaluations": self.evaluations,
                "held_out_tasks": self.held_out_tasks,
            }, f, indent=2)

    def set_held_out_tasks(self, task_ids: list[str]) -> None:
        self.held_out_tasks = list(task_ids)

    def record_evaluation(self, session: int, global_step: int,
                          per_achievement_sr: dict) -> None:
        """Record held-out evaluation results.

        Args:
            session: Curriculum session index.
            global_step: Global environment step count.
            per_achievement_sr: Dict mapping achievement_name -> success_rate.
        """
        self.evaluations.append({
            "session": session,
            "global_step": global_step,
            "timestamp": time.time(),
            "metrics": dict(per_achievement_sr),
        })

    def get_latest_metrics(self) -> dict:
        if not self.evaluations:
            return {}
        return dict(self.evaluations[-1]["metrics"])

    @property
    def evaluation_count(self) -> int:
        return len(self.evaluations)

    @property
    def summary(self) -> dict:
        return {
            "evaluations": self.evaluation_count,
            "held_out_tasks": len(self.held_out_tasks),
            "latest_session": self.evaluations[-1]["session"] if self.evaluations else None,
        }
