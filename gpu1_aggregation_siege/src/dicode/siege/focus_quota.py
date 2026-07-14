"""Focus quota — ensures SIEGE chain tasks are not starved by competitive selection.

The selector may allocate only within valid candidates. The focus quota
sets a minimum number of chain-relevant tasks that MUST be included.
"""
import json, os
from typing import Optional


class FocusQuota:
    """Enforces minimum chain-task allocation in curriculum selection."""

    def __init__(self, min_chain_tasks: int = 2, state_path: Optional[str] = None):
        self.min_chain_tasks = min_chain_tasks
        self.state_path = state_path
        self.quota_history: list[dict] = []
        if state_path and os.path.exists(state_path):
            self.load(state_path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.min_chain_tasks = data.get("min_chain_tasks", 2)
        self.quota_history = data.get("quota_history", [])

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.state_path
        if not p: return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "min_chain_tasks": self.min_chain_tasks,
                "quota_history": self.quota_history,
            }, f, indent=2)

    def check(self, selected_ids: list[str], chain_task_ids: list[str],
              session: int) -> dict:
        """Check if the selection satisfies the focus quota.

        Args:
            selected_ids: Tasks selected by the aggregation mechanism.
            chain_task_ids: Tasks relevant to active SIEGE chains.
            session: Current session index.

        Returns:
            Dict with 'satisfied' (bool), 'chain_count' (int), 'deficit' (int).
        """
        chain_set = set(chain_task_ids)
        selected_chain = [t for t in selected_ids if t in chain_set]
        count = len(selected_chain)
        deficit = max(0, self.min_chain_tasks - count)

        result = {
            "session": session,
            "satisfied": deficit == 0,
            "chain_count": count,
            "min_required": self.min_chain_tasks,
            "deficit": deficit,
            "total_selected": len(selected_ids),
        }
        self.quota_history.append(result)
        return result

    def enforce(self, selected_ids: list[str], chain_task_ids: list[str],
                candidate_pool: list[str], session: int) -> list[str]:
        """Enforce focus quota by replacing non-chain tasks with chain tasks.

        Does NOT modify if quota is already satisfied.
        """
        check = self.check(selected_ids, chain_task_ids, session)
        if check["satisfied"]:
            return list(selected_ids)

        chain_set = set(chain_task_ids)
        selected_set = set(selected_ids)
        # Replace non-chain tasks with chain tasks
        non_chain = [t for t in selected_ids if t not in chain_set]
        available_chain = [t for t in chain_task_ids if t not in selected_set]

        result = list(selected_ids)
        for i in range(min(check["deficit"], len(available_chain), len(non_chain))):
            result.remove(non_chain[i])
            result.append(available_chain[i])

        return result

    @property
    def summary(self) -> dict:
        violations = sum(1 for h in self.quota_history if not h["satisfied"])
        return {
            "min_chain_tasks": self.min_chain_tasks,
            "total_checks": len(self.quota_history),
            "violations": violations,
        }
