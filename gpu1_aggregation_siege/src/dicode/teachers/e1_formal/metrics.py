"""Stage 7: probe-based metrics — regret, behavioral gap, learnability (G2).

Gate G2 implementation. Every number produced here derives ONLY from
THIS candidate's own real dual probes (one Student probe, one Reference
probe) with Wilson confidence intervals:

* ``regret          = max(0, ref_p - stu_p)``  (point estimates);
* ``behavioral_gap  = |ref_p - stu_p|``;
* ``classify_learnability`` -> one of LEARNABLE / SATURATED /
  BOTH_UNREACHABLE / INSUFFICIENT_EVIDENCE, decided from the
  conservative gap confidence bound ``gap_lo = ref_lo - stu_hi``.

The archive learnability prior (LP) is carried ONLY as the inert side
field ``learnability_prior_lp`` on the verdict. It is never read by the
selector, never enters any ranking, and can never substitute for real
probe evidence. There is NO fixed learnability substitute value of any
kind in this module (the former draft's constant default has been
deleted and is grep-audited by tests).

Missing probes never degrade to a guessed number: the verdict state is
``LEARNABILITY_UNAVAILABLE`` and the selection side fails closed with
``SELECTION_BLOCKED_NO_REAL_EVIDENCE`` (see ``selector.py``).

All thresholds come from the frozen teacher configuration with NO
defaults (``LEARNABILITY_THRESHOLD_MISSING`` otherwise). Pure standard
library; fail-closed with greppable codes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .schemas import E1SchemaError, assert_selector_admissible

#: learnability verdict states (G2)
LEARNABLE = "LEARNABLE"
SATURATED = "SATURATED"
BOTH_UNREACHABLE = "BOTH_UNREACHABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
#: no real probe exists for this candidate (never a guessed number)
LEARNABILITY_UNAVAILABLE = "LEARNABILITY_UNAVAILABLE"

#: probe sides
SIDE_STUDENT = "student"
SIDE_REFERENCE = "reference"
_VALID_SIDES = frozenset({SIDE_STUDENT, SIDE_REFERENCE})

# fail-closed codes
METRICS_BAD_TYPE = "METRICS_BAD_TYPE"
METRICS_MISSING_FIELD = "METRICS_MISSING_FIELD"
METRICS_UNKNOWN_FIELD = "METRICS_UNKNOWN_FIELD"
METRICS_OUT_OF_RANGE = "METRICS_OUT_OF_RANGE"
METRICS_CANDIDATE_MISMATCH = "METRICS_CANDIDATE_MISMATCH"
LEARNABILITY_THRESHOLD_MISSING = "LEARNABILITY_THRESHOLD_MISSING"
CI_LEVEL_UNSUPPORTED = "CI_LEVEL_UNSUPPORTED"

#: Normal quantiles for the supported two-sided CI levels. These are
#: mathematical constants of the Wilson interval (NOT tunable thresholds
#: and NOT defaulted policy values): the ci_level itself must always be
#: supplied by the frozen configuration.
_Z_BY_CI_LEVEL = {
    0.90: 1.6448536269514722,
    0.95: 1.9599639845400540,
    0.99: 2.5758293035489004,
}


class MetricsError(E1SchemaError):
    """Fail-closed metrics violation; ``code`` is greppable."""


def z_for_ci_level(ci_level: Any, ctx: str) -> float:
    """Normal quantile for a supported CI level (fail-closed lookup)."""
    if isinstance(ci_level, bool) or not isinstance(ci_level, float):
        raise MetricsError(
            METRICS_BAD_TYPE,
            f"{ctx}: ci_level must be a float, got {type(ci_level).__name__}",
        )
    for level, z in _Z_BY_CI_LEVEL.items():
        if ci_level == level:
            return z
    raise MetricsError(
        CI_LEVEL_UNSUPPORTED,
        f"{ctx}: ci_level {ci_level!r} is unsupported "
        f"(supported: {sorted(_Z_BY_CI_LEVEL)})",
    )


def wilson_interval(
    successes: int, episodes: int, z: float
) -> Tuple[float, float]:
    """Wilson score interval ``(lo, hi)`` for a binomial success rate.

    Deterministic pure-python implementation; ``episodes`` must be >= 1.
    """
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise MetricsError(
            METRICS_BAD_TYPE, "wilson: successes must be int",
        )
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise MetricsError(
            METRICS_BAD_TYPE, "wilson: episodes must be int",
        )
    if episodes < 1:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"wilson: episodes must be >= 1, got {episodes}",
        )
    if successes < 0 or successes > episodes:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"wilson: successes {successes} outside [0, {episodes}]",
        )
    if not math.isfinite(z) or z <= 0:
        raise MetricsError(
            METRICS_OUT_OF_RANGE, f"wilson: z must be positive finite, got {z}"
        )
    n = float(episodes)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# ---------------------------------------------------------------------------
# Probe results (real dual-probe consumption, fail-closed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    """One real probe of THIS candidate by one side (Student/Reference)."""

    candidate_id: str
    side: str  # SIDE_STUDENT or SIDE_REFERENCE
    successes: int
    episodes: int
    provenance: str  # selector-admissible (stamped CANDIDATE_EVALUATION)

    @property
    def success_rate(self) -> float:
        if self.episodes < 1:
            raise MetricsError(
                METRICS_OUT_OF_RANGE,
                f"probe {self.candidate_id}/{self.side}: episodes < 1",
            )
        return self.successes / self.episodes


_PROBE_FIELDS = frozenset(
    {"candidate_id", "side", "successes", "episodes", "provenance"}
)


def consume_probe_result(mapping: Any, ctx: str) -> ProbeResult:
    """Parse one real probe record fail-closed (no defaults, no guesses)."""
    if not isinstance(mapping, Mapping):
        raise MetricsError(
            METRICS_BAD_TYPE,
            f"{ctx}: probe must be a mapping, got {type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k not in _PROBE_FIELDS)
    if unknown:
        raise MetricsError(
            METRICS_UNKNOWN_FIELD, f"{ctx}: unknown probe field(s) {unknown}"
        )
    for name in ("candidate_id", "side", "provenance"):
        if name not in mapping:
            raise MetricsError(
                METRICS_MISSING_FIELD, f"{ctx}: probe missing {name!r}"
            )
    for name in ("successes", "episodes"):
        if name not in mapping:
            raise MetricsError(
                METRICS_MISSING_FIELD, f"{ctx}: probe missing {name!r}"
            )
    candidate_id = mapping["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise MetricsError(
            METRICS_BAD_TYPE, f"{ctx}: probe candidate_id must be non-empty str"
        )
    side = mapping["side"]
    if not isinstance(side, str) or side not in _VALID_SIDES:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: probe side {side!r} not in {sorted(_VALID_SIDES)}",
        )
    counts = {}
    for name in ("successes", "episodes"):
        value = mapping[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise MetricsError(
                METRICS_BAD_TYPE,
                f"{ctx}: probe {name!r} must be int, got "
                f"{type(value).__name__}",
            )
        if value < 0:
            raise MetricsError(
                METRICS_OUT_OF_RANGE,
                f"{ctx}: probe {name!r} must be >= 0, got {value}",
            )
        counts[name] = value
    if counts["successes"] > counts["episodes"]:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: probe successes {counts['successes']} exceed episodes "
            f"{counts['episodes']}",
        )
    provenance = assert_selector_admissible(mapping["provenance"], ctx)
    return ProbeResult(
        candidate_id=candidate_id.strip(),
        side=side,
        successes=counts["successes"],
        episodes=counts["episodes"],
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Learnability thresholds (frozen config only — NO defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LearnabilityThresholds:
    """G2 thresholds; every value must come from the frozen config."""

    tau_saturated: float    # Student CI lower bound at/above => SATURATED
    tau_reachable: float    # Reference evidence of reachability for LEARNABLE
    tau_unreachable: float  # Reference CI upper bound below => BOTH_UNREACHABLE
    delta_min: float        # minimum meaningful conservative gap for LEARNABLE
    min_episodes: int       # per-side episode floor below => no verdict
    ci_level: float         # Wilson CI level (supported set only)


_THRESHOLD_FLOAT_FIELDS = (
    "tau_saturated",
    "tau_reachable",
    "tau_unreachable",
    "delta_min",
)


def consume_learnability_thresholds(
    mapping: Any, ctx: str
) -> LearnabilityThresholds:
    """Parse the threshold block fail-closed; NO field has a default."""
    if not isinstance(mapping, Mapping):
        raise MetricsError(
            LEARNABILITY_THRESHOLD_MISSING,
            f"{ctx}: learnability threshold block must be a mapping, got "
            f"{type(mapping).__name__} (thresholds come from the frozen "
            "config only; nothing is defaulted)",
        )
    known = frozenset(_THRESHOLD_FLOAT_FIELDS) | {
        "min_episodes",
        "ci_level",
    }
    unknown = sorted(k for k in mapping if k not in known)
    if unknown:
        raise MetricsError(
            METRICS_UNKNOWN_FIELD,
            f"{ctx}: unknown threshold field(s) {unknown}",
        )
    values = {}
    for name in _THRESHOLD_FLOAT_FIELDS:
        if name not in mapping or mapping[name] is None:
            raise MetricsError(
                LEARNABILITY_THRESHOLD_MISSING,
                f"{ctx}: threshold {name!r} missing (no default exists)",
            )
        value = mapping[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricsError(
                METRICS_BAD_TYPE,
                f"{ctx}: threshold {name!r} must be a number",
            )
        value = float(value)
        if not math.isfinite(value):
            raise MetricsError(
                METRICS_OUT_OF_RANGE,
                f"{ctx}: threshold {name!r} must be finite",
            )
        values[name] = value
    for name in ("tau_saturated", "tau_reachable", "tau_unreachable"):
        if not (0.0 <= values[name] <= 1.0):
            raise MetricsError(
                METRICS_OUT_OF_RANGE,
                f"{ctx}: threshold {name!r} must be in [0, 1], got "
                f"{values[name]}",
            )
    if values["delta_min"] <= 0.0:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: delta_min must be > 0, got {values['delta_min']}",
        )
    if not (
        values["tau_unreachable"]
        < values["tau_reachable"]
        <= values["tau_saturated"]
    ):
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: thresholds must satisfy tau_unreachable < "
            f"tau_reachable <= tau_saturated, got {values['tau_unreachable']}"
            f" / {values['tau_reachable']} / {values['tau_saturated']}",
        )
    if "min_episodes" not in mapping or mapping["min_episodes"] is None:
        raise MetricsError(
            LEARNABILITY_THRESHOLD_MISSING,
            f"{ctx}: threshold 'min_episodes' missing (no default exists)",
        )
    min_episodes = mapping["min_episodes"]
    if isinstance(min_episodes, bool) or not isinstance(min_episodes, int):
        raise MetricsError(
            METRICS_BAD_TYPE, f"{ctx}: min_episodes must be int"
        )
    if min_episodes < 1:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: min_episodes must be >= 1, got {min_episodes}",
        )
    if "ci_level" not in mapping or mapping["ci_level"] is None:
        raise MetricsError(
            LEARNABILITY_THRESHOLD_MISSING,
            f"{ctx}: threshold 'ci_level' missing (no default exists)",
        )
    ci_level = mapping["ci_level"]
    if isinstance(ci_level, bool) or not isinstance(ci_level, (int, float)):
        raise MetricsError(METRICS_BAD_TYPE, f"{ctx}: ci_level must be a number")
    ci_level = float(ci_level)
    z_for_ci_level(ci_level, ctx)  # fail-closed support check
    return LearnabilityThresholds(
        tau_saturated=values["tau_saturated"],
        tau_reachable=values["tau_reachable"],
        tau_unreachable=values["tau_unreachable"],
        delta_min=values["delta_min"],
        min_episodes=min_episodes,
        ci_level=ci_level,
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LearnabilityVerdict:
    """G2 verdict for ONE candidate, derived from its own real probes.

    ``learnability_prior_lp`` is an inert archive prior recorded for
    audit ONLY — the selector never reads it and it never substitutes
    for probe evidence.
    """

    candidate_id: str
    state: str
    note: str
    student_success_rate: Optional[float] = None
    student_ci: Optional[Tuple[float, float]] = None
    reference_success_rate: Optional[float] = None
    reference_ci: Optional[Tuple[float, float]] = None
    gap: Optional[float] = None            # ref_p - stu_p (point estimate)
    gap_ci_lower: Optional[float] = None   # conservative: ref_lo - stu_hi
    regret: Optional[float] = None         # max(0, ref_p - stu_p)
    behavioral_gap: Optional[float] = None  # |ref_p - stu_p|
    episodes_student: Optional[int] = None
    episodes_reference: Optional[int] = None
    learnability_prior_lp: Optional[float] = None


def _validate_prior_lp(value: Any, ctx: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricsError(
            METRICS_BAD_TYPE,
            f"{ctx}: learnability_prior_lp must be a number or absent",
        )
    value = float(value)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: learnability_prior_lp must be in [0, 1], got {value}",
        )
    return value


def build_learnability_verdict(
    *,
    candidate_id: str,
    student_probe: Optional[ProbeResult],
    reference_probe: Optional[ProbeResult],
    thresholds: LearnabilityThresholds,
    learnability_prior_lp: Optional[float] = None,
    ctx: str = "learnability",
) -> LearnabilityVerdict:
    """Classify ONE candidate from its own real dual probes (G2).

    Missing probe(s) yield ``LEARNABILITY_UNAVAILABLE`` — never a
    guessed number. The conservative gap bound is
    ``gap_lo = ref_lo - stu_hi``.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise MetricsError(
            METRICS_BAD_TYPE, f"{ctx}: candidate_id must be non-empty str"
        )
    if not isinstance(thresholds, LearnabilityThresholds):
        raise MetricsError(
            METRICS_BAD_TYPE,
            f"{ctx}: thresholds must be LearnabilityThresholds",
        )
    prior_lp = _validate_prior_lp(learnability_prior_lp, ctx)
    candidate_id = candidate_id.strip()

    if student_probe is None or reference_probe is None:
        return LearnabilityVerdict(
            candidate_id=candidate_id,
            state=LEARNABILITY_UNAVAILABLE,
            note="no real dual probe for this candidate; no substitute "
            "value exists; selection must fail closed",
            learnability_prior_lp=prior_lp,
        )
    if not isinstance(student_probe, ProbeResult) or not isinstance(
        reference_probe, ProbeResult
    ):
        raise MetricsError(
            METRICS_BAD_TYPE, f"{ctx}: probes must be ProbeResult instances"
        )
    if student_probe.side != SIDE_STUDENT:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: student probe has side {student_probe.side!r}",
        )
    if reference_probe.side != SIDE_REFERENCE:
        raise MetricsError(
            METRICS_OUT_OF_RANGE,
            f"{ctx}: reference probe has side {reference_probe.side!r}",
        )
    for probe in (student_probe, reference_probe):
        if probe.candidate_id != candidate_id:
            raise MetricsError(
                METRICS_CANDIDATE_MISMATCH,
                f"{ctx}: probe candidate_id {probe.candidate_id!r} != "
                f"{candidate_id!r}",
            )

    z = z_for_ci_level(thresholds.ci_level, ctx)
    if (
        student_probe.episodes < thresholds.min_episodes
        or reference_probe.episodes < thresholds.min_episodes
    ):
        return LearnabilityVerdict(
            candidate_id=candidate_id,
            state=INSUFFICIENT_EVIDENCE,
            note=f"probe episodes below min_episodes="
            f"{thresholds.min_episodes}",
            student_success_rate=student_probe.success_rate,
            reference_success_rate=reference_probe.success_rate,
            episodes_student=student_probe.episodes,
            episodes_reference=reference_probe.episodes,
            learnability_prior_lp=prior_lp,
        )

    stu_lo, stu_hi = wilson_interval(
        student_probe.successes, student_probe.episodes, z
    )
    ref_lo, ref_hi = wilson_interval(
        reference_probe.successes, reference_probe.episodes, z
    )
    stu_p = student_probe.success_rate
    ref_p = reference_probe.success_rate
    gap = ref_p - stu_p
    gap_lo = ref_lo - stu_hi  # conservative lower bound of the gap
    base = dict(
        student_success_rate=stu_p,
        student_ci=(stu_lo, stu_hi),
        reference_success_rate=ref_p,
        reference_ci=(ref_lo, ref_hi),
        gap=gap,
        gap_ci_lower=gap_lo,
        regret=max(0.0, ref_p - stu_p),
        behavioral_gap=abs(ref_p - stu_p),
        episodes_student=student_probe.episodes,
        episodes_reference=reference_probe.episodes,
        learnability_prior_lp=prior_lp,
    )

    if ref_hi < thresholds.tau_unreachable:
        state, note = BOTH_UNREACHABLE, (
            f"reference CI upper bound {ref_hi:.6f} < tau_unreachable="
            f"{thresholds.tau_unreachable}; neither side can reach the task"
        )
    elif stu_lo >= thresholds.tau_saturated:
        state, note = SATURATED, (
            f"student CI lower bound {stu_lo:.6f} >= tau_saturated="
            f"{thresholds.tau_saturated}; no learning headroom"
        )
    elif (
        gap_lo >= thresholds.delta_min
        and ref_hi >= thresholds.tau_reachable
    ):
        state, note = LEARNABLE, (
            f"conservative gap lower bound {gap_lo:.6f} >= delta_min="
            f"{thresholds.delta_min} and reference reachable "
            f"(ref CI upper {ref_hi:.6f} >= tau_reachable="
            f"{thresholds.tau_reachable})"
        )
    else:
        state, note = INSUFFICIENT_EVIDENCE, (
            f"verdict indeterminate: gap_lo={gap_lo:.6f} < delta_min="
            f"{thresholds.delta_min} or reference not shown reachable "
            f"(ref CI upper {ref_hi:.6f} < tau_reachable="
            f"{thresholds.tau_reachable}); no verdict is fabricated"
        )
    return LearnabilityVerdict(
        candidate_id=candidate_id, state=state, note=note, **base
    )
