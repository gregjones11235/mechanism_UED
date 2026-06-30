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

from .craftax_achievements import ALL_ACHIEVEMENTS, depth_of
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
) -> float:
    """AmbitionGain(proposal) = sum over its achievements of target_gap[a] * depth_weight[a].

    Args:
        proposal: the candidate.
        target_gap: {achievement: 1 - student_SR_on_target} in [0,1]. Missing achievement => gap 0
            (treated as already mastered / not a target concern).
        use_depth_weight: if True, weight each gap by the achievement's depth tier so that a gap
            on a deep late-game achievement counts more than the same gap on an early one. This is
            the "dependency-chain depth x target gap" of §2.3. If False, depth is ignored (pure gap).

    Returns:
        A non-negative scalar. Per-proposal independent => modular.
    """
    _validate_gap(target_gap)
    total = 0.0
    for a in proposal.achievements:
        gap = target_gap.get(a, 0.0)
        w = float(depth_of(a)) if use_depth_weight else 1.0
        total += gap * w
    return total
