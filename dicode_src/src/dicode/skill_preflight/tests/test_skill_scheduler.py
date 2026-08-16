"""Offline tests for the Skill Graph Scheduler.

No RL / no model needed — feed synthetic per-achievement success rates and assert
the scheduler locates the correct learnable frontier tier.

Run on the pod:
    cd /workspace/mechanism_UED/dicode_src
    uv run pytest src/dicode/skill_preflight/tests/test_skill_scheduler.py -v
"""
from __future__ import annotations

import pytest

from auction.craftax_achievements import DEPTH_TIERS
from dicode.skill_preflight.skill_scheduler import (
    pick_target,
    format_target_for_prompt,
)


def make_metrics(mastered_tiers, mastered_sr: float = 95.0, unmastered_sr: float = 5.0) -> dict:
    """Build eval-style metrics ({'skill_<name>': SR_0_100}) where the given tiers are
    mastered (high SR) and all others are not (low SR). Emits ALL achievements, mirroring
    the real Craftax held-out eval which reports every achievement (incl. 0.0)."""
    metrics: dict[str, float] = {}
    for tier, names in DEPTH_TIERS.items():
        sr = mastered_sr if tier in mastered_tiers else unmastered_sr
        for a in names:
            metrics[f"skill_{a}"] = sr
    return metrics


@pytest.mark.parametrize(
    "mastered_tiers, expected_frontier",
    [
        (set(), 1),          # nothing mastered -> stay at tier 1
        ({1}, 2),            # tier 1 mastered -> frontier tier 2
        ({1, 2}, 3),         # tiers 1-2 mastered -> frontier tier 3
        ({1, 2, 3}, 4),      # tiers 1-3 mastered -> frontier tier 4
        ({1, 2, 3, 4}, 4),   # all mastered -> deepest tier (consolidate)
    ],
)
def test_frontier_tier(mastered_tiers, expected_frontier):
    target = pick_target(make_metrics(mastered_tiers))
    assert target.tier == expected_frontier


def test_all_mastered_is_consolidate():
    target = pick_target(make_metrics({1, 2, 3, 4}))
    assert target.tier == 4
    assert target.gap_type == "consolidate"


def test_frontier_is_advance():
    target = pick_target(make_metrics({1}))  # frontier = tier 2, unmastered
    assert target.gap_type == "advance"
    assert target.frontier_mastery < 0.60


def test_target_achievements_in_frontier_tier():
    target = pick_target(make_metrics({1}))  # frontier = tier 2
    assert len(target.target_achievements) > 0
    assert all(a in DEPTH_TIERS[2] for a in target.target_achievements)


def test_target_achievements_hardest_first():
    # give tier-2 a mix: one easy (high SR), rest hard (low SR); easy one should rank last
    metrics = make_metrics({1})
    easy = next(iter(DEPTH_TIERS[2]))
    metrics[f"skill_{easy}"] = 99.0  # nearly mastered -> should be filtered / ranked last
    target = pick_target(metrics)
    assert easy not in target.target_achievements or target.target_achievements[-1] == easy


def test_empty_metrics_defaults_to_tier1():
    assert pick_target(None).tier == 1
    assert pick_target({}).tier == 1


def test_format_target_for_prompt_mentions_tier_and_skills():
    target = pick_target(make_metrics({1}))
    text = format_target_for_prompt(target)
    assert f"tier {target.tier}" in text
    assert target.target_achievements[0] in text


def test_max_target_achievements_cap():
    target = pick_target(make_metrics(set()), max_target_achievements=3)
    assert len(target.target_achievements) <= 3
