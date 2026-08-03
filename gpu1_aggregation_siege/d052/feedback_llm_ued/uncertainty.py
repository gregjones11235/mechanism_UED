"""Episode-count-derived uncertainty (board input layer, C4).

Probe statistics computed from few episodes are NOISY; the six-role board
must see a confidence-interval half-width with every rate, never a bare point
estimate. The half-width is the normal approximation
``z * sqrt(p (1 - p) / n)`` — conservative at the small n this loop uses,
deterministic, and fail-closed on illegal inputs (no silent n=0 division).
"""
from __future__ import annotations

from dataclasses import dataclass

#: two-sided 95% normal quantile (round constant; no scipy dependency)
Z_95 = 1.96


def ci_halfwidth(p: float, n: int, z: float = Z_95) -> float:
    """Half-width of the CI for a rate ``p`` estimated from ``n`` episodes."""
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"ILLEGAL_EPISODE_COUNT: {n!r} — an uncertainty "
                         "estimate requires at least one episode")
    if not isinstance(p, (int, float)) or isinstance(p, bool) or \
            not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"ILLEGAL_RATE: {p!r} must lie in [0, 1]")
    if z <= 0:
        raise ValueError(f"ILLEGAL_Z: {z!r}")
    p = float(p)
    return z * (p * (1.0 - p) / n) ** 0.5


@dataclass(frozen=True)
class UncertainRate:
    """A rate estimate with its episode count and CI half-width."""

    estimate: float
    episodes: int
    ci: float

    @property
    def lo(self) -> float:
        return max(0.0, self.estimate - self.ci)

    @property
    def hi(self) -> float:
        return min(1.0, self.estimate + self.ci)

    def overlaps(self, other: "UncertainRate") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi


def rate_with_ci(successes: int, episodes: int,
                 z: float = Z_95) -> UncertainRate:
    if not isinstance(successes, int) or successes < 0 or successes > episodes:
        raise ValueError(
            f"ILLEGAL_SUCCESS_COUNT: {successes!r} of {episodes!r}")
    p = successes / episodes
    return UncertainRate(estimate=p, episodes=episodes,
                         ci=ci_halfwidth(p, episodes, z))


def episodes_from_transitions(transitions: int, rollout_length: int) -> int:
    """Floor-derived episode count (lower bound for non-aligned rollouts).

    The symbolic runner's transitions are exact multiples of the rollout
    length; a real executor's are not, so this floor division is documented
    as a CONSERVATIVE lower bound — fewer assumed episodes means wider CIs,
    never overconfidence.
    """
    if transitions < 0 or rollout_length <= 0:
        raise ValueError(
            f"ILLEGAL_TRANSITIONS: transitions={transitions!r} "
            f"rollout_length={rollout_length!r}")
    return transitions // rollout_length
