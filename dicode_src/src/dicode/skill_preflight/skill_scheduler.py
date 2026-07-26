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

C-2-lite §1 (2026-07-11): a second frontier criterion, ``frontier_mode="prereq"``. The tier-mean
criterion lets early-maturing members (collect_iron 64%) inflate a tier's mean while the true
bottleneck (make_iron_pickaxe 2.7%) lies flat, so diamond-family tasks get issued before their
prerequisites exist. Prereq mode replaces "tier mean over threshold" with a PER-NODE readiness
test on the direct-prerequisite graph (prereq_graph.DIRECT_PREREQS): a skill is targetable iff
every direct prerequisite is individually over ``prereq_threshold`` AND the skill itself is
unmastered. Default mode stays "tier" — old runs reproduce bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from auction.craftax_achievements import (
    ACHIEVEMENT_DEPTH,
    ALL_ACHIEVEMENTS,
    DEPTH_TIERS,
    MASTERY_THRESHOLD_DEFAULT,
    reachable_ceiling,
    tier_mastery,
)
from dicode.dreaming.auction_integration import profile_to_target_gap
from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS


@dataclass
class SchedulerTarget:
    """What the scheduler decides the student should practice next."""
    tier: int                          # frontier depth tier (1..4) to target
    target_achievements: list[str]     # unmastered achievements in that tier, hardest first
    tier_mastery: dict[int, float]     # mean mastery per tier (0..1)
    frontier_mastery: float            # mean mastery of the frontier tier (0..1)
    gap_type: str                      # "advance" (frontier not yet mastered) | "consolidate"
    mode: str = "tier"                 # criterion that produced this target ("tier" | "prereq")
    sr_snapshot: dict[str, float] = field(default_factory=dict)  # per-achievement SR 0..1


def _sr_map(evaluation_metrics: Optional[Mapping[str, float]]) -> dict[str, float]:
    """Per-achievement SR in [0,1] over ALL 67 achievements; unmeasured -> 0.0.

    Missing-is-unmastered matches the conservative philosophy of tier_mastery(default): an
    achievement the eval never reported must NOT read as ready. The real held-out eval emits
    all 67 keys, so in production nothing is actually missing.
    """
    target_gap = profile_to_target_gap(dict(evaluation_metrics) if evaluation_metrics else {})
    return {a: 1.0 - target_gap.get(a, 1.0) for a in ALL_ACHIEVEMENTS}


def pick_target(
    evaluation_metrics: Optional[Mapping[str, float]],
    *,
    threshold: float = MASTERY_THRESHOLD_DEFAULT,
    max_target_achievements: int = 6,
    frontier_mode: str = "tier",
    prereq_threshold: float = 0.3,
) -> SchedulerTarget:
    """Locate the learnable frontier and the achievements to practice there.

    Args:
        evaluation_metrics: dict from ``process_evaluation_metrics``,
            e.g. ``{"skill_collect_wood": 99.89, "skill_collect_iron": 0.0, ...}``
            (values are success rates in 0..100). ``None``/empty -> defaults to tier 1.
        threshold: mastery threshold (fraction). In "tier" mode: a tier counts as mastered at
            >= this MEAN SR. In "prereq" mode: a skill counts as mastered (drops out of the
            target pool) at >= this INDIVIDUAL SR. Default 0.60.
        max_target_achievements: cap on how many frontier achievements to return.
        frontier_mode: "tier" (legacy, tier-mean criterion — default, old runs unchanged) or
            "prereq" (C-2-lite §1: per-node direct-prerequisite readiness).
        prereq_threshold: prereq mode only — a direct prerequisite counts as "in place" at
            >= this individual SR. Default 0.3.

    Returns:
        A SchedulerTarget describing the frontier + the specific achievements to focus
        generation on. ``sr_snapshot`` carries the full per-achievement SR map (0..1) for
        downstream consumers (one-step prompt, scaffold gate).
    """
    target_gap = profile_to_target_gap(dict(evaluation_metrics) if evaluation_metrics else {})
    mastery = tier_mastery(target_gap)
    sr = _sr_map(evaluation_metrics)

    if frontier_mode == "prereq":
        # A skill is targetable iff (a) it is itself unmastered and (b) EVERY direct
        # prerequisite is individually over prereq_threshold. This is the fix for the
        # tier-mean blindspot: no more issuing diamond_sword while make_iron_pickaxe
        # sits at 2.7% under a 60%+ tier mean.
        eligible = [
            a for a in ALL_ACHIEVEMENTS
            if sr[a] < threshold
            and all(sr[p] >= prereq_threshold for p in DIRECT_PREREQS[a])
        ]
        if eligible:
            # Hardest-first (lowest SR), deeper tier breaks ties, then name for determinism.
            eligible.sort(key=lambda a: (sr[a], -ACHIEVEMENT_DEPTH[a], a))
            target_achievements = eligible[:max_target_achievements]
            frontier = max(ACHIEVEMENT_DEPTH[a] for a in target_achievements)
            return SchedulerTarget(
                tier=frontier,
                target_achievements=target_achievements,
                tier_mastery=mastery,
                frontier_mastery=mastery.get(frontier, 0.0),
                gap_type="advance",
                mode="prereq",
                sr_snapshot=sr,
            )
        # Nothing eligible — in practice only when EVERYTHING is mastered above threshold
        # (zero-prereq skills are always eligible while unmastered, so a fresh run still
        # yields the tier-1 basics). Fall through to the tier criterion, which returns the
        # deepest tier as a "consolidate" target, so the scheduler always emits something.

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
        mode="tier",
        sr_snapshot=sr,
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


# --- C-2-lite §2: one-step frontier scaffolding (prompt layer) -----------------------------
#
# Replaces the semantics of "SCAFFOLD every missing prerequisite and list it as Completed"
# (the instruction the leak audit traced 100% of scaffolding to) with a mastery-driven rule:
# exactly ONE unmastered step may be exposed bare; deeper UNMASTERED prerequisites may be
# scaffolded; MASTERED prefixes must be performed in-episode and never pre-marked. The text
# rides the [Curriculum focus] injection, so it only ever reaches the LLM when the scheduler
# flag is on — baseline prompts are untouched.

MASTERED_SR_CUT = 0.70   # a skill counts as "mastered" for scaffold decisions at >= this SR
LEARNING_SR_CUT = 0.30   # below this it counts as "not acquired" (scaffoldable if not focus)


def _snapshot_lines(sr: Mapping[str, float]) -> tuple[str, str, str]:
    """(mastered, in-progress, unacquired) comma-lists from a 0..1 SR snapshot."""
    mastered = sorted(a for a, v in sr.items() if v >= MASTERED_SR_CUT)
    learning = sorted(
        (a for a, v in sr.items() if LEARNING_SR_CUT <= v < MASTERED_SR_CUT),
        key=lambda a: -sr[a],
    )
    unacquired = sorted(a for a, v in sr.items() if v < LEARNING_SR_CUT)
    fmt = lambda names: ", ".join(names) if names else "(none)"
    learning_fmt = (
        ", ".join(f"{a} ({sr[a] * 100:.0f}%)" for a in learning) if learning else "(none)"
    )
    return fmt(mastered), learning_fmt, fmt(unacquired)


def format_target_for_prompt_one_step(
    target: SchedulerTarget, *, mastered_exemption: bool = False, r3_v2: bool = False
) -> str:
    """One-step variant of the [Curriculum focus] block (design/docstring stage).

    Injects the mastery snapshot explicitly (the profile string upstream omits 0% skills,
    which is exactly the set the scaffold rule needs to reason about) and states the
    one-bare-step scaffolding contract.
    """
    skills = ", ".join(target.target_achievements) if target.target_achievements else "(none)"
    mastered, learning, unacquired = _snapshot_lines(target.sr_snapshot)
    return (
        f"Train these frontier skills (each has all direct prerequisites in place but is "
        f"itself unmastered): {skills}.\n"
        f"\n"
        f"Mastery snapshot (held-out evaluation):\n"
        f"- MASTERED (SR >= 70%): {mastered}\n"
        f"- IN PROGRESS (30-70%): {learning}\n"
        f"- NOT ACQUIRED (< 30%): {unacquired}\n"
        f"\n"
        f"SCAFFOLDING CONTRACT (one bare step):\n"
        f"1. The task must expose EXACTLY ONE unmastered step bare: the focus skill and its "
        + ("immediate prerequisites that the agent has NOT yet mastered "
           if mastered_exemption else "immediate prerequisites ")
        + f"must be performed by the agent during the episode, from the "
        f"resources the world provides — never granted in the starting inventory and never "
        f"listed as completed/prerequisite achievements."
        + (" Prerequisites the agent already MASTERS (>= 70%) MAY be provided or skipped "
           "(e.g. starting floor, inventory) so the episode's practice budget concentrates "
           "on the unmastered step." if mastered_exemption else "")
        + f"\n"
        f"2. You MAY scaffold prerequisites that are deeper in the chain ONLY if the agent has "
        f"NOT acquired them (< 30%) and they are not the training focus.\n"
        + ("3. Do NOT PRE-MARK skills the agent already MASTERS (>= 70%) as completed "
           "achievements: the agent performs those itself. PROVIDING a mastered prerequisite "
           "in the starting inventory is permitted by rule 1 above - e.g. if make_stone_pickaxe "
           "is mastered and the focus skill is collect_iron, start the agent with a stone "
           "pickaxe rather than making it re-craft one.\n"
           if r3_v2 else
           "3. Do NOT pre-mark or provision skills the agent already MASTERS (>= 70%): the agent "
           "performs those itself. Granting them adds nothing and corrupts the task's meaning.\n")
        + f"4. In the docstring, the Prerequisites section may list ONLY items permitted by rule "
        f"2. If that leaves it empty, write 'Prerequisites: none'."
    )


def format_scaffold_rules_for_coder(
    sr: Mapping[str, float], *, mastered_exemption: bool = False, r3_v2: bool = False
) -> str:
    """Scaffolding constraint block appended to the CODE-generation user prompt.

    The coder stage sees few-shot code examples that all demonstrate the leak pattern
    (set_player_inventory + completed_achievements pre-marking + near-spawn mobs), so the
    rule must explicitly outrank both the examples and any docstring text that conflicts.
    """
    mastered, _, unacquired = _snapshot_lines(sr)
    return (
        "\n\n## SCAFFOLDING CONSTRAINTS (override the code examples and any conflicting "
        "docstring text)\n"
        f"- Skills the agent already MASTERS: {mastered}\n"
        f"- Skills the agent has NOT acquired: {unacquired}\n"
        "1. `self.completed_achievements` may contain ONLY achievements from the NOT-acquired "
        "list that are NOT in `self.relevant_achievements`. NEVER pre-mark a mastered "
        "achievement and NEVER pre-mark a relevant achievement.\n"
        "2. Do NOT grant items via `set_player_inventory` that substitute a relevant "
        "achievement or its immediate prerequisite (e.g. no pre-made iron pickaxe when the "
        "task trains make_iron_pickaxe or collect_diamond)."
        + (" EXCEPTION: prerequisites in the MASTERS list above may be provided or skipped "
           "— e.g. a starting floor that skips a mastered descent is encouraged."
           + (" A mastered TOOL TIER may likewise be granted directly: if make_stone_pickaxe "
              "is mastered and the task trains collect_iron, giving pickaxe level 2 in "
              "`set_player_inventory` is CORRECT, not a violation." if r3_v2 else "")
           if mastered_exemption else "") + "\n"
        "3. If the code examples above pre-mark achievements or hand out inventory more "
        "liberally, IGNORE that pattern — these constraints take precedence."
    )
