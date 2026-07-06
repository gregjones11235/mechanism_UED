"""Skill Graph Scheduler (minimal core).

Given the student's per-achievement success rates (from
``process_evaluation_metrics``, keys like ``skill_collect_wood`` valued 0..100),
locate the current *learnable frontier* — the shallowest not-yet-mastered depth
tier — via the existing ``reachable_ceiling``, and package the unmastered
achievements in that tier as the generation target.

This file is deliberately thin: it reuses existing repo machinery and only wires
it together + formats the target for the generation prompt.

Reused (not reimplemented):
  - auction.craftax_achievements: reachable_ceiling, tier_mastery, DEPTH_TIERS,
    MASTERY_THRESHOLD_DEFAULT
  - dicode.dreaming.auction_integration: profile_to_target_gap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from auction.craftax_achievements import (
    DEPTH_TIERS,
    MASTERY_THRESHOLD_DEFAULT,
    reachable_ceiling,
    tier_mastery,
)
from dicode.dreaming.auction_integration import profile_to_target_gap


@dataclass
class SchedulerTarget:
    """What the scheduler decides the student should practice next."""
    tier: int                          # frontier depth tier (1..4) to target
    target_achievements: list[str]     # unmastered achievements in that tier, hardest first
    tier_mastery: dict[int, float]     # mean mastery per tier (0..1)
    frontier_mastery: float            # mean mastery of the frontier tier (0..1)
    gap_type: str                      # "advance" (frontier not yet mastered) | "consolidate"


def pick_target(
    evaluation_metrics: Optional[Mapping[str, float]],
    *,
    threshold: float = MASTERY_THRESHOLD_DEFAULT,
    max_target_achievements: int = 6,
) -> SchedulerTarget:
    """Locate the learnable frontier tier and the achievements to practice there.

    Args:
        evaluation_metrics: dict from ``process_evaluation_metrics``,
            e.g. ``{"skill_collect_wood": 99.89, "skill_collect_iron": 0.0, ...}``
            (values are success rates in 0..100). ``None``/empty -> defaults to tier 1.
        threshold: a tier counts as "mastered" at >= this mean SR (fraction). Default 0.60.
        max_target_achievements: cap on how many frontier achievements to return.

    Returns:
        A SchedulerTarget describing the frontier tier + the specific unmastered
        achievements to focus generation on.
    """
    target_gap = profile_to_target_gap(dict(evaluation_metrics) if evaluation_metrics else {})
    mastery = tier_mastery(target_gap)
    frontier = reachable_ceiling(target_gap, threshold=threshold)

    # Rank the frontier tier's achievements hardest-first (largest gap = lowest SR).
    tier_names = DEPTH_TIERS[frontier]
    ranked = sorted(tier_names, key=lambda a: target_gap.get(a, 1.0), reverse=True)
    unmastered = [a for a in ranked if target_gap.get(a, 1.0) > (1.0 - threshold)]
    target_achievements = (unmastered or ranked)[:max_target_achievements]

    frontier_mastery = mastery.get(frontier, 0.0)
    # By construction reachable_ceiling returns the shallowest tier with mastery < threshold,
    # so frontier_mastery < threshold ("advance") — except when ALL tiers are mastered, where it
    # returns the deepest tier (mastery >= threshold) and we keep practicing it ("consolidate").
    gap_type = "advance" if frontier_mastery < threshold else "consolidate"

    return SchedulerTarget(
        tier=frontier,
        target_achievements=target_achievements,
        tier_mastery=mastery,
        frontier_mastery=frontier_mastery,
        gap_type=gap_type,
    )


def format_target_for_prompt(target: SchedulerTarget) -> str:
    """Render the target as a constraint string to inject into the env-generation prompt."""
    skills = ", ".join(target.target_achievements) if target.target_achievements else "(none)"
    return (
        f"Focus the generated level on the student's current learning frontier: depth tier "
        f"{target.tier}. The student has NOT yet mastered these achievements at this tier: "
        f"{skills}. Design a level that requires and teaches these specific skills, building on "
        f"the already-mastered shallower tiers. Do not jump to deeper tiers than tier {target.tier}."
    )
