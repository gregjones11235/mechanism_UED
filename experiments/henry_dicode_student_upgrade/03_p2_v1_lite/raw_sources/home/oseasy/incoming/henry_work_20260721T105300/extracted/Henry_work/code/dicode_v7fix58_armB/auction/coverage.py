"""Weighted Coverage — the core set function the auction maximizes (v1_experiment.md §7.5).

Coverage(S) = total non-negative weight of the achievements covered by the proposals in S.

    Coverage(S) = sum_{a in (union of S's achievements)} weight[a]

This is a *weighted coverage function*. For ANY non-negative weights it is **monotone and
submodular** (the standard set-cover result). Submodularity = diminishing returns:

    gain(x | S) >= gain(x | T)   whenever  S subseteq T          [the defining inequality]

where gain(x | S) = Coverage(S ∪ {x}) - Coverage(S) = weight of x's achievements not yet
covered by S. Submodularity + monotonicity is exactly the precondition for the
Nemhauser (1 - 1/e) greedy guarantee that backs v1's top-k selection.

Design decisions (see discussion 2026-06-30):
- WEIGHTED (not raw count): the objective is non-uniform — late-game achievements (tier 4,
  where DiCode's selling point lives) matter far more than tier-1 ones baselines already ace.
  Weighting makes greedy prefer scarce/deep achievements. Weighted coverage stays submodular
  for non-negative weights, so the (1-1/e) guarantee is NOT lost.
- weights are an INJECTED parameter (default = depth-tier-derived). v2 passes Walrasian dynamic
  shadow prices as the weights through this same function → zero rework, v1/v2 share Coverage.
- "depth" lives here as the weight; AmbitionGain (separate file) then carries only the
  *dynamic target gap*, so the two bid terms are orthogonal (static value vs dynamic gap).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .craftax_achievements import ALL_ACHIEVEMENTS, depth_of
from .proposal import Proposal

# Default per-achievement weights derived from depth tier: deeper == scarcer == more valuable.
# tier 1..4 -> weight. Chosen so tier-4 (late-game, baseline-collapses) dominates the objective
# but no tier is zero (every achievement is worth *something*). Tunable; v2 overrides entirely.
_TIER_WEIGHT: Mapping[int, float] = {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0}

DEFAULT_WEIGHTS: dict[str, float] = {a: _TIER_WEIGHT[depth_of(a)] for a in ALL_ACHIEVEMENTS}


def _validate_weights(weights: Mapping[str, float]) -> None:
    bad = set(weights) - ALL_ACHIEVEMENTS
    if bad:
        raise ValueError(f"weights reference unknown achievements: {sorted(bad)}")
    neg = [a for a, w in weights.items() if w < 0.0]
    if neg:
        # Non-negativity is what guarantees submodularity. Refuse to silently break the theorem.
        raise ValueError(f"weights must be non-negative (else Coverage is not submodular): {neg}")


def covered_set(proposals: Iterable[Proposal]) -> frozenset[str]:
    """Union of achievements covered by a set of proposals."""
    out: set[str] = set()
    for p in proposals:
        out |= p.achievements
    return frozenset(out)


def coverage(
    proposals: Iterable[Proposal],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    *,
    already_covered: frozenset[str] = frozenset(),
) -> float:
    """Coverage(S) = total weight of achievements covered by S but not in ``already_covered``.

    ``already_covered`` lets the archive's existing coverage be discounted (we only credit
    *new* coverage beyond what the archive already has), per §2.3 "complementary coverage".
    """
    _validate_weights(weights)
    new = covered_set(proposals) - already_covered
    return float(sum(weights.get(a, 0.0) for a in new))


def coverage_gain(
    candidate: Proposal,
    selected: Iterable[Proposal],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    *,
    already_covered: frozenset[str] = frozenset(),
) -> float:
    """Marginal gain of adding ``candidate`` to ``selected``: weight of its newly-covered achievements.

    gain(x | S) = Coverage(S ∪ {x}) - Coverage(S). This is what greedy top-k maximizes each step.
    By submodularity this is non-increasing as ``selected`` grows.
    """
    _validate_weights(weights)
    base = covered_set(selected) | already_covered
    new = candidate.achievements - base
    return float(sum(weights.get(a, 0.0) for a in new))
