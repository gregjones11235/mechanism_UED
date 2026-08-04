"""Round-3 P0-5: criterion-wise Soft Copeland selector (formal path).

The FORMAL E1 selection path: eight behavioral criteria are each
normalized independently (``rank_percentile_v1``), the pairwise
Copeland contest happens INSIDE each criterion, and the per-criterion
Copeland scores are combined by an audited weight vector. The
"average all scores first, then one Copeland" shape of the E1-S
ablation path (``selector.py``) is STRUCTURALLY ABSENT here — no
aggregated-strength function exists in this module.

Pipeline::

    CriterionSignals (8 criteria, all required, has_real_probe=True)
      -> rank_percentile_v1 per criterion (in-package replica of the
         d052 normalization semantics; parity is test-side only, the
         runtime never imports d052)
      -> pairwise Copeland INSIDE each criterion (sorted ids; +1 / 0 /
         +0.5 ties)
      -> weighted aggregation (Fraction weights, sum exactly 1)
      -> rank (composite DESC, candidate_id ASC)
      -> greedy take k under the per-family cap (NO backfill)

Criterion direction (pinned): all criteria are HIGHER-IS-BETTER except
``simulator_cost`` (cheaper is better), whose percentile is inverted
(``1 - p``) before the Copeland contest. Regret criteria are
higher-is-better BY DESIGN of this branch: larger Student-vs-Reference
regret means more untrained behavior and therefore higher curriculum
priority.

Fail-closed evidence rule: empty pool, ANY ``has_real_probe=False``, or
ANY missing criterion value fails with
``SELECTION_BLOCKED_NO_REAL_EVIDENCE`` — archive priors, heuristics or
LLM role scores never substitute for real probe evidence. LLM role
scores never enter this path at all (E1-S ablation only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .canonical import canonical_sha256
from .schemas import E1Code, assert_selector_admissible
from .selector import (
    CRITIC_HARD_VETO,
    CRITIC_SCORE_ONLY,
    CRITIC_SOFT_PENALTY,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    SelectionOutcome,
    SelectorError,
)

CRITERION_SELECTOR_NAME = "CRITERION_WISE_COPELAND"
NORMALIZATION_NAME = "rank_percentile_v1"

#: the eight behavioral criteria — ALL required for every candidate
CRITERIA = (
    "front_regret",
    "global_regret",
    "behavioral_gap",
    "learnability",
    "learning_progress",
    "diversity",
    "global_retention",
    "simulator_cost",
)

#: criteria where LOWER raw values are better (percentile inverted)
LOWER_IS_BETTER = ("simulator_cost",)

# fail-closed codes (greppable)
CSEL_BAD_TYPE = "CSEL_BAD_TYPE"
CSEL_MISSING_FIELD = "CSEL_MISSING_FIELD"
CSEL_UNKNOWN_FIELD = "CSEL_UNKNOWN_FIELD"
CSEL_OUT_OF_RANGE = "CSEL_OUT_OF_RANGE"
CSEL_WEIGHT_MISMATCH = "CSEL_WEIGHT_MISMATCH"
CSEL_FAMILY_CAP_BAD_TYPE = "CSEL_FAMILY_CAP_BAD_TYPE"

_VALID_POLICIES = frozenset(
    {CRITIC_HARD_VETO, CRITIC_SOFT_PENALTY, CRITIC_SCORE_ONLY}
)


@dataclass(frozen=True)
class CriterionSignals:
    """One candidate's criterion evidence (real-probe-backed only).

    ``values`` carries the criterion -> raw value pairs the caller can
    actually evidence; ``select_criterion_batch`` fails closed unless
    EVERY candidate carries ALL eight ``CRITERIA``.
    """

    candidate_id: str
    family_id: str
    values: Tuple[Tuple[str, float], ...]
    provenance: str
    has_real_probe: bool

    def values_dict(self) -> Dict[str, float]:
        return dict(self.values)


_SIGNAL_FIELDS = frozenset(
    {"candidate_id", "family_id", "values", "provenance", "has_real_probe"}
)


def _require_finite_number(name: str, value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectorError(
            CSEL_BAD_TYPE, f"{ctx}: {name} must be a number, got {value!r}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise SelectorError(
            CSEL_BAD_TYPE, f"{ctx}: {name} must be finite, got {value!r}"
        )
    return value


def rank_percentile_v1(values: Sequence[float]) -> List[float]:
    """In-package replica of the d052 rank-percentile normalization.

    ``normalized(c) = (# values strictly less than c) / (n - 1)``;
    a singleton normalizes to 0.5; ties share their percentile.
    Deterministic and order-independent. The runtime NEVER imports
    d052; equivalence with the canonical implementation is enforced
    test-side only.
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    return [
        sum(1 for other in values if other < value) / (n - 1)
        for value in values
    ]


def consume_criterion_signal(mapping: Any, ctx: str) -> CriterionSignals:
    """Parse one candidate's criterion signals fail-closed."""
    if not isinstance(mapping, Mapping):
        raise SelectorError(
            CSEL_BAD_TYPE,
            f"{ctx}: signals must be a mapping, got {type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k not in _SIGNAL_FIELDS)
    if unknown:
        raise SelectorError(
            CSEL_UNKNOWN_FIELD, f"{ctx}: unknown signal field(s) {unknown}"
        )
    for field in ("candidate_id", "family_id"):
        if field not in mapping:
            raise SelectorError(
                CSEL_MISSING_FIELD, f"{ctx}: missing {field!r}"
            )
        value = mapping[field]
        if not isinstance(value, str) or not value.strip():
            raise SelectorError(
                CSEL_BAD_TYPE, f"{ctx}: {field} must be a non-empty str"
            )
    if "values" not in mapping:
        raise SelectorError(CSEL_MISSING_FIELD, f"{ctx}: missing 'values'")
    raw_values = mapping["values"]
    if not isinstance(raw_values, Mapping):
        raise SelectorError(
            CSEL_BAD_TYPE, f"{ctx}: values must be a mapping"
        )
    if len(raw_values) == 0:
        raise SelectorError(
            CSEL_MISSING_FIELD, f"{ctx}: values must not be empty"
        )
    values: List[Tuple[str, float]] = []
    for criterion, value in raw_values.items():
        if criterion not in CRITERIA:
            raise SelectorError(
                CSEL_UNKNOWN_FIELD,
                f"{ctx}: unknown criterion {criterion!r} (expected the "
                f"eight CRITERIA {list(CRITERIA)})",
            )
        values.append(
            (criterion, _require_finite_number(f"values[{criterion}]", value, ctx))
        )
    if "provenance" not in mapping:
        raise SelectorError(CSEL_MISSING_FIELD, f"{ctx}: missing 'provenance'")
    provenance = assert_selector_admissible(mapping["provenance"], ctx)
    if "has_real_probe" not in mapping:
        raise SelectorError(
            CSEL_MISSING_FIELD, f"{ctx}: missing 'has_real_probe'"
        )
    if not isinstance(mapping["has_real_probe"], bool):
        raise SelectorError(
            CSEL_BAD_TYPE, f"{ctx}: has_real_probe must be bool"
        )
    return CriterionSignals(
        candidate_id=mapping["candidate_id"].strip(),
        family_id=mapping["family_id"].strip(),
        values=tuple(values),
        provenance=provenance,
        has_real_probe=mapping["has_real_probe"],
    )


def consume_criterion_signals(
    mappings: Any, ctx: str
) -> Tuple[CriterionSignals, ...]:
    """Parse all criterion signals; candidate ids must be unique."""
    if not isinstance(mappings, (list, tuple)):
        raise SelectorError(
            CSEL_BAD_TYPE,
            f"{ctx}: signals must be a sequence of mappings, got "
            f"{type(mappings).__name__}",
        )
    signals = tuple(
        consume_criterion_signal(raw, f"{ctx}[{i}]")
        for i, raw in enumerate(mappings)
    )
    ids = [sig.candidate_id for sig in signals]
    if len(set(ids)) != len(ids):
        raise SelectorError(
            CSEL_BAD_TYPE, f"{ctx}: duplicate candidate_id in {ids}"
        )
    return signals


def normalize_per_criterion(
    signals: Sequence[CriterionSignals],
) -> Dict[str, Dict[str, float]]:
    """rank_percentile_v1 per criterion (LOWER_IS_BETTER inverted)."""
    normalized: Dict[str, Dict[str, float]] = {}
    for criterion in CRITERIA:
        holders = sorted(
            (sig for sig in signals if criterion in sig.values_dict()),
            key=lambda sig: sig.candidate_id,
        )
        raw = [sig.values_dict()[criterion] for sig in holders]
        percentiles = rank_percentile_v1(raw)
        if criterion in LOWER_IS_BETTER:
            percentiles = [1.0 - p for p in percentiles]
        normalized[criterion] = {
            sig.candidate_id: p for sig, p in zip(holders, percentiles)
        }
    return normalized


def copeland_per_criterion(
    normalized: Mapping[str, Mapping[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Pairwise Copeland INSIDE each criterion (sorted ids, tie 0.5).

    No cross-criterion averaged strength exists anywhere in this
    module — the contest is per criterion by construction.
    """
    per_criterion: Dict[str, Dict[str, float]] = {}
    for criterion in CRITERIA:
        scores = normalized.get(criterion)
        if scores is None:
            continue
        ids = sorted(scores)
        copeland = {cid: 0.0 for cid in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if scores[a] > scores[b]:
                    copeland[a] += 1.0
                elif scores[a] < scores[b]:
                    copeland[b] += 1.0
                else:
                    copeland[a] += 0.5
                    copeland[b] += 0.5
        per_criterion[criterion] = copeland
    return per_criterion


def resolve_weights(
    weights: Optional[Mapping[str, Any]]
) -> Tuple[Dict[str, Fraction], str]:
    """Resolve the aggregation weights fail-closed.

    Absent => the supervisor-sanctioned equal weights (1/8 each),
    recorded as ``weights_source="default_equal"``. Present => must
    cover ALL eight criteria exactly, every weight a positive
    ``Fraction`` (or int), and the sum EXACTLY 1 — else
    ``CSEL_WEIGHT_MISMATCH``. Floats are rejected (exact rational
    arithmetic only).
    """
    if weights is None:
        equal = {criterion: Fraction(1, 8) for criterion in CRITERIA}
        return equal, "default_equal"
    if not isinstance(weights, Mapping):
        raise SelectorError(
            CSEL_WEIGHT_MISMATCH,
            "criterion weights must be a mapping covering all eight "
            f"CRITERIA, got {type(weights).__name__}",
        )
    if set(weights) != set(CRITERIA):
        raise SelectorError(
            CSEL_WEIGHT_MISMATCH,
            f"criterion weights must cover exactly {list(CRITERIA)}, got "
            f"{sorted(weights)}",
        )
    resolved: Dict[str, Fraction] = {}
    for criterion in CRITERIA:
        value = weights[criterion]
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise SelectorError(
                CSEL_WEIGHT_MISMATCH,
                f"criterion weight {criterion!r} must be a Fraction (or "
                f"int), got {value!r} — exact rational arithmetic only",
            )
        fraction = Fraction(value)
        if fraction <= 0:
            raise SelectorError(
                CSEL_WEIGHT_MISMATCH,
                f"criterion weight {criterion!r} must be positive, got "
                f"{fraction}",
            )
        resolved[criterion] = fraction
    if sum(resolved.values()) != Fraction(1):
        raise SelectorError(
            CSEL_WEIGHT_MISMATCH,
            f"criterion weights must sum to exactly 1, got "
            f"{sum(resolved.values())}",
        )
    return resolved, "configured"


def aggregate_copeland(
    per_criterion: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, Fraction],
) -> Dict[str, float]:
    """Weighted sum of the per-criterion Copeland scores.

    Summed in the fixed ``CRITERIA`` order for determinism; candidates
    missing from a criterion contribute 0 for that criterion (selection
    blocks earlier when evidence is incomplete, so this is defensive).
    """
    candidates = sorted(
        {cid for scores in per_criterion.values() for cid in scores}
    )
    aggregate: Dict[str, float] = {}
    for cid in candidates:
        total = Fraction(0)
        for criterion in CRITERIA:
            scores = per_criterion.get(criterion)
            if scores is None or cid not in scores:
                continue
            total += weights[criterion] * Fraction(scores[cid]).limit_denominator(
                10 ** 12
            )
        aggregate[cid] = float(total)
    return aggregate


def derive_criterion_values(
    *,
    probe_result: Optional[Mapping[str, Any]] = None,
    retention_probe: Optional[float] = None,
    axis_count: Optional[int] = None,
    pool_axis_max: Optional[int] = None,
    episodes: Optional[float] = None,
) -> Dict[str, float]:
    """Derive criterion values from REAL inputs only — never fabricate.

    The formal source is the shared ``CandidateProbeResult`` (CC4);
    this round no such object exists, so every criterion whose real
    input is absent is OMITTED (and criterion-wise selection fails
    closed with ``SELECTION_BLOCKED_NO_REAL_EVIDENCE``). Derivation:

    * front_regret / global_regret / behavioral_gap / learnability /
      learning_progress — the corresponding numeric field of the shared
      probe result (omitted when the probe result or field is absent);
    * diversity — ``axis_count / pool_axis_max`` clipped to [0, 1]
      (both counts required);
    * global_retention — the ``retention_probe`` value in [0, 1]. G3:
      retention has NO real source this round (disabled, no
      substitute); this input exists only for a future labelled real
      probe record;
    * simulator_cost — ``episodes`` (raw cost; the selector inverts
      the direction for LOWER_IS_BETTER).
    """
    values: Dict[str, float] = {}
    if probe_result is not None:
        if not isinstance(probe_result, Mapping):
            raise SelectorError(
                CSEL_BAD_TYPE,
                "probe_result must be a mapping (the shared "
                f"CandidateProbeResult), got {type(probe_result).__name__}",
            )
        for criterion in (
            "front_regret",
            "global_regret",
            "behavioral_gap",
            "learnability",
            "learning_progress",
        ):
            if criterion in probe_result:
                values[criterion] = _require_finite_number(
                    f"probe_result[{criterion}]", probe_result[criterion],
                    "derive_criterion_values",
                )
    if axis_count is not None and pool_axis_max is not None:
        axis_count_f = _require_finite_number(
            "axis_count", axis_count, "derive_criterion_values"
        )
        pool_axis_max_f = _require_finite_number(
            "pool_axis_max", pool_axis_max, "derive_criterion_values"
        )
        if axis_count_f < 0 or pool_axis_max_f <= 0:
            raise SelectorError(
                CSEL_OUT_OF_RANGE,
                "diversity derivation requires axis_count >= 0 and "
                "pool_axis_max > 0",
            )
        values["diversity"] = min(1.0, axis_count_f / pool_axis_max_f)
    if retention_probe is not None:
        retention = _require_finite_number(
            "retention_probe", retention_probe, "derive_criterion_values"
        )
        if retention < 0.0 or retention > 1.0:
            raise SelectorError(
                CSEL_OUT_OF_RANGE,
                f"retention_probe outside [0, 1]: {retention}",
            )
        values["global_retention"] = retention
    if episodes is not None:
        cost = _require_finite_number(
            "episodes", episodes, "derive_criterion_values"
        )
        if cost < 0:
            raise SelectorError(
                CSEL_OUT_OF_RANGE, f"episodes must be >= 0, got {cost}"
            )
        values["simulator_cost"] = cost
    return values


def select_criterion_batch(
    signals: Sequence[CriterionSignals],
    *,
    k: int,
    seed: int,
    critic_policy: str,
    family_cap: int,
    weights: Optional[Mapping[str, Any]] = None,
) -> SelectionOutcome:
    """Criterion-wise Soft Copeland top-k under the per-family cap.

    Fail-closed evidence rule FIRST: empty pool, any
    ``has_real_probe=False`` or any missing criterion value raises
    ``SELECTION_BLOCKED_NO_REAL_EVIDENCE``. Then: per-criterion
    normalization + Copeland, weighted aggregation, rank (composite
    DESC, candidate_id ASC), greedy take under ``family_cap`` —
    shortfall is honest ``STATUS_INSUFFICIENT`` (NO backfill, NO
    k-reduction, NO re-LLM).
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SelectorError(
            CSEL_OUT_OF_RANGE, f"k must be an int >= 1, got {k!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SelectorError(
            CSEL_BAD_TYPE, f"seed must be an int, got {seed!r}"
        )
    if critic_policy not in _VALID_POLICIES:
        raise SelectorError(
            CSEL_OUT_OF_RANGE,
            f"critic_policy {critic_policy!r} not in {sorted(_VALID_POLICIES)}",
        )
    if isinstance(family_cap, bool) or not isinstance(family_cap, int):
        raise SelectorError(
            CSEL_FAMILY_CAP_BAD_TYPE,
            f"family_cap must be an int, got {family_cap!r}",
        )
    if family_cap < 1:
        raise SelectorError(
            CSEL_OUT_OF_RANGE,
            f"family_cap must be >= 1, got {family_cap}",
        )

    # ---- fail-closed evidence gate (BEFORE any scoring) ----------------
    if len(signals) == 0:
        raise SelectorError(
            E1Code.SELECTION_BLOCKED_NO_REAL_EVIDENCE,
            "no criterion signals at all; criterion-wise selection is "
            "blocked (no real evidence, no substitute)",
        )
    missing_probe = sorted(
        sig.candidate_id for sig in signals if not sig.has_real_probe
    )
    if missing_probe:
        raise SelectorError(
            E1Code.SELECTION_BLOCKED_NO_REAL_EVIDENCE,
            f"candidates without a real dual probe: {missing_probe}; "
            "criterion-wise selection is blocked — archive priors, "
            "heuristics or LLM role scores never substitute for real "
            "probe evidence",
        )
    incomplete = sorted(
        sig.candidate_id
        for sig in signals
        if set(sig.values_dict()) != set(CRITERIA)
    )
    if incomplete:
        raise SelectorError(
            E1Code.SELECTION_BLOCKED_NO_REAL_EVIDENCE,
            f"candidates missing criterion value(s): {incomplete}; all "
            f"eight CRITERIA {list(CRITERIA)} are required — missing "
            "real evidence blocks selection, never fabricated",
        )

    resolved_weights, weights_source = resolve_weights(weights)
    normalized = normalize_per_criterion(signals)
    per_criterion = copeland_per_criterion(normalized)
    composite = aggregate_copeland(per_criterion, resolved_weights)

    ranked = sorted(composite.items(), key=lambda t: (-float(t[1]), t[0]))
    family_counts: Dict[str, int] = {}
    selected: List[str] = []
    for cid, _score in ranked:
        if len(selected) >= k:
            break
        family = next(
            sig.family_id for sig in signals if sig.candidate_id == cid
        )
        if family_counts.get(family, 0) >= family_cap:
            continue  # per-family cap honored; NO backfill
        family_counts[family] = family_counts.get(family, 0) + 1
        selected.append(cid)

    if len(selected) >= k:
        status = STATUS_OK
        note = ""
    else:
        status = STATUS_INSUFFICIENT
        note = (
            f"only {len(selected)} candidates selected for k={k} under "
            f"family_cap={family_cap} and critic_policy={critic_policy}; "
            "NO backfill / NO k-reduction / NO re-LLM"
        )
    selection_hash = canonical_sha256(
        {
            "selector": CRITERION_SELECTOR_NAME,
            "normalization": NORMALIZATION_NAME,
            "weights": [
                [criterion, str(resolved_weights[criterion])]
                for criterion in CRITERIA
            ],
            "weights_source": weights_source,
            "family_cap": family_cap,
            "k": k,
            "seed": seed,
            "critic_policy": critic_policy,
            "selected_ids": sorted(selected),
        }
    )
    return SelectionOutcome(
        selector=CRITERION_SELECTOR_NAME,
        critic_policy=critic_policy,
        k_requested=k,
        seed=seed,
        candidate_count_in=len(signals),
        eligible_count=len(signals),
        selected_ids=tuple(selected),
        rejected_by_critic=(),  # criterion signals carry no critic field
        status=status,
        selection_hash=selection_hash,
        shortfall_note=note,
    )
