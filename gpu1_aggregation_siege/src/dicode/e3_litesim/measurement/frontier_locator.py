"""Deterministic Frontier Locator: probe results -> FrontierSpec (no LLM)."""
from __future__ import annotations

from typing import Optional

from ..data.frontier_spec import FrontierSpec, finalize
from .capability_probe import CapabilityMeasurementResult
from .tier_registry import TierRegistry

_FAIL_STATUSES = ("FRONTIER", "UNSTABLE", "FAILED")


def locate_frontier(measurement: CapabilityMeasurementResult,
                    registry: TierRegistry) -> FrontierSpec:
    results = sorted(measurement.tier_results,
                     key=lambda r: registry.get(r.tier_id).order)
    mastered = [r for r in results if r.status == "MASTERED"]
    highest_pass: Optional[str] = mastered[-1].tier_id if mastered else None
    failing = [r for r in results if r.status in _FAIL_STATUSES]
    if failing:
        target = failing[0]
    else:
        unknown = [r for r in results if r.status == "UNKNOWN"]
        target = unknown[0] if unknown else results[-1]
    tier = registry.get(target.tier_id)
    horizon = max(16, tier.horizon // 2)
    return finalize(FrontierSpec(
        skill_family=target.skill_family, tier=target.tier_id,
        probe_id=target.probe_id, mastered_before=highest_pass,
        failing_here=target.tier_id, status=target.status,
        rollout_horizon=horizon,
        success_predicate=tier.success_predicate_name,
        progress_metric="success_rate", priority=1,
        allowed_variations=("frozen", "prefix_variant")))