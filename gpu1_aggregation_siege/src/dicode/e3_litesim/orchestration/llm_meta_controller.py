"""LLMMetaController: interface only, OUT of the high-frequency loop.

Trigger conditions (v1 contract): UNKNOWN frontier, same frontier stagnates
K rounds, new unsupported capability family, periodic high-level review.
The ordinary Probe->Frontier->Data->PPO->Reprobe loop never calls it and
runs fully with E3_NO_LLM=true.
"""
from __future__ import annotations

import os
from typing import Sequence


class LLMMetaController:
    def __init__(self, stagnation_k: int = 3) -> None:
        self.stagnation_k = stagnation_k

    def should_trigger(self, history: Sequence[dict]) -> tuple:
        if not history:
            return False, "no_history"
        last = history[-1]
        if last.get("frontier_status") == "UNKNOWN":
            return True, "unknown_frontier"
        recent = [h.get("frontier_tier") for h in history[-self.stagnation_k:]]
        if (len(recent) >= self.stagnation_k and len(set(recent)) == 1
                and last.get("frontier_status") in ("UNSTABLE", "FAILED")):
            return True, "frontier_stagnation"
        known = {h.get("skill_family") for h in history[:-1]}
        if known and last.get("skill_family") not in known:
            return True, "new_capability_family"
        return False, "routine"

    def invoke(self, payload: dict) -> dict:
        if os.environ.get("E3_NO_LLM", "true").lower() == "true":
            raise RuntimeError(
                "E3_NO_LLM=true: LLM is interface-only in v1 and must not be "
                "called by the hot loop")
        raise NotImplementedError("LLMMetaController.invoke: v1 interface only")