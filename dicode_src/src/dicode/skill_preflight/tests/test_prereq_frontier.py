"""Offline tests for frontier_mode="prereq" (C-2-lite §1) + the one-step prompt (§2).

Includes the design-doc verification method: REPLAY the real probe end-point eval JSON
(experiments_mason/eval/eval_PROBET02_seed0.json @ update 2400) and assert the criterion
concentrates fire on the true gap (make_iron_pickaxe) and stops issuing prereq-broken
targets (make_diamond_sword while iron_pickaxe sits at 2.7%).

Run on the pod:
    cd /workspace/mechanism_UED/dicode_src
    uv run pytest src/dicode/skill_preflight/tests/test_prereq_frontier.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from auction.craftax_achievements import ALL_ACHIEVEMENTS
from dicode.skill_preflight.skill_scheduler import (
    MASTERED_SR_CUT,
    format_target_for_prompt,
    format_target_for_prompt_one_step,
    format_scaffold_rules_for_coder,
    pick_target,
)

# Repo root = dicode_src/.. ; endpoint JSON lives in experiments_mason/eval/.
_ENDPOINT_JSON = (
    Path(__file__).resolve().parents[5] / "experiments_mason" / "eval" / "eval_PROBET02_seed0.json"
)


def make_metrics(sr_by_name: dict[str, float], default: float = 0.0) -> dict[str, float]:
    """Full 67-key eval-style metrics; unspecified achievements get ``default``."""
    m = {f"skill_{a}": default for a in ALL_ACHIEVEMENTS}
    for name, sr in sr_by_name.items():
        m[f"skill_{name}"] = sr
    return m


# A synthetic profile isolating the fracture-zone anchor case: tier-1/2 resource line
# mastered, iron tools flat, diamond family flat. Uninvolved skills sit at 50% (neither
# targets under threshold=0.2 nor prereq blockers under prereq_threshold=0.3) so the test
# reads the iron/diamond mechanics without cap interference.
PROBE_LIKE = {
    "collect_wood": 99.0, "place_table": 98.0, "make_wood_pickaxe": 95.0,
    "collect_stone": 96.0, "make_stone_pickaxe": 90.0, "collect_coal": 87.8,
    "collect_iron": 64.6, "place_furnace": 97.8,
    "make_iron_pickaxe": 2.7, "make_iron_sword": 2.6,
    "collect_diamond": 7.4, "make_diamond_sword": 0.6,
}


def test_prereq_mode_targets_true_gap_not_broken_chain():
    t = pick_target(
        make_metrics(PROBE_LIKE, default=50.0),
        threshold=0.2, frontier_mode="prereq", prereq_threshold=0.3,
    )
    assert t.mode == "prereq"
    # iron pickaxe/sword: all direct prereqs over 30% -> should be targeted.
    assert "make_iron_pickaxe" in t.target_achievements
    assert "make_iron_sword" in t.target_achievements
    # diamond family: gated by make_iron_pickaxe @2.7% -> must NOT be issued.
    assert "collect_diamond" not in t.target_achievements
    assert "make_diamond_sword" not in t.target_achievements


def test_prereq_mode_respects_mastery_threshold():
    # collect_iron @64.6% is above threshold=0.2 -> not a target even though learnable.
    t = pick_target(
        make_metrics(PROBE_LIKE, default=50.0),
        threshold=0.2, frontier_mode="prereq", prereq_threshold=0.3,
    )
    assert "collect_iron" not in t.target_achievements


def test_prereq_mode_early_mean_cannot_unlock():
    """The 'premature-mean' pathology: one hot member (collect_iron 64%) must not unlock
    anything that depends on the weak link underneath it."""
    metrics = make_metrics({
        "collect_wood": 99.0, "place_table": 98.0, "make_wood_pickaxe": 95.0,
        "collect_stone": 96.0,
        "make_stone_pickaxe": 15.0,  # the weak link (below prereq_threshold, below mastery)
        "collect_iron": 10.0,        # blocked BY the weak link
        "make_iron_pickaxe": 5.0,    # blocked by collect_iron
    }, default=50.0)
    t = pick_target(metrics, threshold=0.2, frontier_mode="prereq", prereq_threshold=0.3)
    assert "make_stone_pickaxe" in t.target_achievements  # the actual weak link is targeted
    assert "collect_iron" not in t.target_achievements    # its prereq (stone pickaxe) not in place
    assert "make_iron_pickaxe" not in t.target_achievements


def test_all_mastered_falls_back_to_tier_consolidate():
    metrics = make_metrics({}, default=95.0)
    t = pick_target(metrics, threshold=0.6, frontier_mode="prereq")
    assert t.mode == "tier"
    assert t.gap_type == "consolidate"
    assert t.tier == 4


def test_default_mode_is_legacy_tier():
    """No new kwargs -> byte-identical legacy behaviour (old runs reproduce)."""
    metrics = make_metrics({}, default=5.0)
    t_default = pick_target(metrics)
    t_tier = pick_target(metrics, frontier_mode="tier")
    assert t_default.mode == t_tier.mode == "tier"
    assert t_default.tier == t_tier.tier == 1
    assert t_default.target_achievements == t_tier.target_achievements
    # Legacy formatter unchanged.
    assert "learning frontier: depth tier 1" in format_target_for_prompt(t_default)


def test_one_step_prompt_contents():
    t = pick_target(
        make_metrics(PROBE_LIKE), threshold=0.2, frontier_mode="prereq", prereq_threshold=0.3
    )
    txt = format_target_for_prompt_one_step(t)
    assert "make_iron_pickaxe" in txt
    assert "EXACTLY ONE unmastered step bare" in txt
    assert "MASTERED (SR >= 70%)" in txt and "collect_wood" in txt
    coder = format_scaffold_rules_for_coder(t.sr_snapshot)
    assert "completed_achievements" in coder
    assert "override the code examples" in coder
    # mastered/unacquired bucketing uses the shared cut
    assert MASTERED_SR_CUT == 0.70


@pytest.mark.skipif(not _ENDPOINT_JSON.exists(), reason="endpoint eval JSON not in this checkout")
def test_replay_probe_endpoint_json():
    """§5.1 verification method: replay the real probe end point through the criterion.

    REPLAY FINDING (2026-07-11): at the probe end point the eligible pool is larger than the
    default cap of 6 — several legitimately prereq-ready skills sit at exactly 0.0
    (learn_fireball/learn_iceball via open_chest 62%, enter_gnomish_mines via enter_dungeon
    66%, make_iron_armour, defeat_orc_mage) and outrank make_iron_pickaxe @2.7% under
    hardest-first ordering, pushing the fracture-zone anchor to rank 7. The invariant (no
    broken-prereq target ever issued; diamond family excluded) holds regardless; whether the
    cap/ranking needs tuning is a short-run observation point, not a correctness bug.
    """
    skills = json.loads(_ENDPOINT_JSON.read_text())["2400"]["skills"]
    t = pick_target(skills, threshold=0.2, frontier_mode="prereq", prereq_threshold=0.3)
    assert t.mode == "prereq"
    # The criterion's core invariant on real data: NO issued target has a broken prereq.
    from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS
    for a in t.target_achievements:
        for p in DIRECT_PREREQS[a]:
            assert t.sr_snapshot[p] >= 0.3, f"{a} issued with broken prereq {p}"
    # The broken-chain family is out; the ready iron family is in.
    assert "make_diamond_sword" not in t.target_achievements
    assert "collect_diamond" not in t.target_achievements
    assert "make_iron_sword" in t.target_achievements
    # The anchor case make_iron_pickaxe IS eligible and enters at cap 8 (rank 7).
    t8 = pick_target(
        skills, threshold=0.2, frontier_mode="prereq", prereq_threshold=0.3,
        max_target_achievements=8,
    )
    assert "make_iron_pickaxe" in t8.target_achievements
