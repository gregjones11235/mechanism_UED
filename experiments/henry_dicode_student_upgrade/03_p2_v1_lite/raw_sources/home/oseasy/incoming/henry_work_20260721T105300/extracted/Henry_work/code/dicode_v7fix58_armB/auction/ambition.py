"""AmbitionGain — the bid term targeting the student's CURRENT gap on the target env.

After moving "depth" into the Coverage weights (depth-tier-derived, §coverage.py), AmbitionGain
carries only the *dynamic target gap*: for the achievements a proposal teaches, how far is the
student still from mastering them ON THE TARGET env? This makes Coverage (static value of *which*
achievements) and AmbitionGain (dynamic *how badly the student needs them now*) orthogonal.

target_gap[a] in [0,1]: 1 - (student's success rate on achievement a on the target env).
Source at integration time: wandb evaluation/skill_<a> (§2.1), i.e. global_agent_profile.
A proposal teaching achievements the student already aces (gap≈0) gets low AmbitionGain even if
those achievements are deep; one teaching achievements the student fails (gap≈1) gets high.

MODULARITY: AmbitionGain is a per-proposal independent score (a function of that proposal's own
achievements and the fixed target-gap vector), so it is MODULAR — adding it to the bid keeps the
overall bid submodular (same argument as endorsement.py). It does NOT depend on the selected set.
"""

from __future__ import annotations

from collections.abc import Mapping

from .craftax_achievements import (
    ALL_ACHIEVEMENTS,
    depth_of,
    tier_overreach_factor,
)
from .proposal import Proposal


def _validate_gap(target_gap: Mapping[str, float]) -> None:
    bad = set(target_gap) - ALL_ACHIEVEMENTS
    if bad:
        raise ValueError(f"target_gap references unknown achievements: {sorted(bad)}")
    for a, g in target_gap.items():
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"target_gap[{a}] must be in [0,1], got {g}")


def ambition_gain(
    proposal: Proposal,
    target_gap: Mapping[str, float],
    *,
    use_depth_weight: bool = True,
    reachable_ceiling: int | None = None,
    overreach_decay: float = 0.3,
) -> float:
    """AmbitionGain(proposal) = sum over its achievements of target_gap[a] * depth_weight[a] * reach.

    Args:
        proposal: the candidate.
        target_gap: {achievement: 1 - student_SR_on_target} in [0,1]. Missing achievement => gap 0
            (treated as already mastered / not a target concern).
        use_depth_weight: if True, weight each gap by the achievement's depth tier so that a gap
            on a deep late-game achievement counts more than the same gap on an early one. This is
            the "dependency-chain depth x target gap" of §2.3. If False, depth is ignored (pure gap).
        reachable_ceiling: the deepest tier the student can currently be pushed toward
            (craftax_achievements.reachable_ceiling). If given, gap on achievements STRICTLY DEEPER
            than the ceiling is soft-discounted by ``overreach_decay ** (tier - ceiling)`` — the
            ability-gate that stops ambitious out-bidding feasible on tiers the student can't reach
            (v1_experiment.md §10.9). None => no gate (legacy behaviour, all tiers full weight).
        overreach_decay: geometric decay per tier beyond the ceiling (0.3 => 1 over -> 0.30). Only
            used when reachable_ceiling is not None. In [0,1); 0 would be a hard zero-out.

    Returns:
        A non-negative scalar. Still per-proposal independent (ceiling is a fixed student-state
        scalar, not a function of the selected set) => MODULAR, so the assembled bid stays
        submodular and the (1-1/e) greedy guarantee is preserved.
    """
    _validate_gap(target_gap)
    total = 0.0
    for a in proposal.achievements:
        gap = target_gap.get(a, 0.0)
        w = float(depth_of(a)) if use_depth_weight else 1.0
        reach = (
            tier_overreach_factor(depth_of(a), reachable_ceiling, decay=overreach_decay)
            if reachable_ceiling is not None
            else 1.0
        )
        total += gap * w * reach
    return total
