"""Cost tracking for LLM curriculum judgments.

Tracks estimated costs per provider, role, and task.
Never hard-codes permanent pricing — reads from provider config.
"""

import json
import os
import time
from collections import defaultdict
from typing import Optional


class LLMCostTracker:
    """Tracks and caps costs for LLM API calls."""

    def __init__(
        self,
        max_total_cost: float = 1.0,
        max_api_calls: int = 60,
        log_path: Optional[str] = None,
    ):
        self.max_total_cost = max_total_cost
        self.max_api_calls = max_api_calls
        self.log_path = log_path

        self.total_cost = 0.0
        self.total_calls = 0
        self.by_provider = defaultdict(lambda: {"calls": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0})
        self.by_role = defaultdict(lambda: {"calls": 0, "cost": 0.0})
        self.calls = []  # Individual call records

    def can_call(self, estimated_tokens: int = 500) -> bool:
        """Check if another API call is within budget."""
        # Rough cost estimate for a typical judgment call
        est_cost = estimated_tokens / 1000.0 * 0.0006  # assume worst-case pricing
        if self.total_calls >= self.max_api_calls:
            return False
        if self.total_cost + est_cost > self.max_total_cost:
            return False
        return True

    def record_call(self, call_result: dict) -> None:
        """Record a completed API call."""
        provider = call_result.get("provider", "unknown")
        role = call_result.get("role", "unknown")
        cost = call_result.get("estimated_cost", 0.0)
        input_tokens = call_result.get("input_tokens_est", 0)
        output_tokens = call_result.get("output_tokens_est", 0)

        self.total_cost += cost
        self.total_calls += 1
        self.by_provider[provider]["calls"] += 1
        self.by_provider[provider]["cost"] += cost
        self.by_provider[provider]["input_tokens"] += input_tokens
        self.by_provider[provider]["output_tokens"] += output_tokens
        self.by_role[role]["calls"] += 1
        self.by_role[role]["cost"] += cost

        record = {
            "timestamp": time.time(),
            "provider": provider,
            "role": role,
            "task_id": call_result.get("task_id", "unknown"),
            "success": call_result.get("success", False),
            "input_tokens_est": input_tokens,
            "output_tokens_est": output_tokens,
            "estimated_cost": cost,
        }
        self.calls.append(record)

        if self.log_path:
            self._write_log(record)

    def _write_log(self, record: dict) -> None:
        """Write a call record to the cost log."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def get_summary(self) -> dict:
        """Get a summary of all costs."""
        return {
            "total_cost": round(self.total_cost, 8),
            "total_calls": self.total_calls,
            "max_allowed_calls": self.max_api_calls,
            "max_allowed_cost": self.max_total_cost,
            "budget_remaining": round(max(0, self.max_total_cost - self.total_cost), 8),
            "calls_remaining": max(0, self.max_api_calls - self.total_calls),
            "by_provider": dict(self.by_provider),
            "by_role": dict(self.by_role),
            "cost_per_successful_call": (
                round(self.total_cost / max(1, sum(1 for c in self.calls if c.get("success"))), 8)
            ),
        }
