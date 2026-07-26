"""rank_percentile_v1 — per-role score normalization.

Spec (task §角色分数归一化):
  * per-role INDEPENDENT (each role column normalized on its own distribution);
  * output strictly bounded to [0, 1];
  * raw scores PRESERVED alongside the normalized value;
  * DETERMINISTIC ties (same raw -> same normalized & rank, independent of input
    order);
  * NO reverse-weighting from outcomes (normalization depends only on the raw
    signal column, never on downstream training results).

Definition (competition-rank percentile):
    normalized(c) = (# candidates with raw strictly less than c) / (n - 1)   n > 1
                  = 0.5                                                     n == 1
  * min raw -> 0.0; a unique max raw -> 1.0; tied candidates share one value.
  * tie_group = dense index (0-based) of the raw value among sorted distinct raws.
  * rank = # strictly-less (0-based competition rank).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from d052.schemas.roles import NormalizedEntry, NormalizedRoleScores, ScoringRole

NORMALIZATION = "rank_percentile_v1"


class NormalizationError(Exception):
    EMPTY_COLUMN = "EMPTY_COLUMN"
    NON_FINITE = "NON_FINITE"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def rank_percentile_v1(role: ScoringRole,
                       scores: Sequence[Tuple[str, float]]) -> NormalizedRoleScores:
    """Normalize one role's (candidate_id, raw) column to [0,1].

    Deterministic: depends only on the multiset of raw values, not input order.
    """
    import math

    items = list(scores)
    if not items:
        raise NormalizationError(
            NormalizationError.EMPTY_COLUMN,
            f"role={role.value}: cannot normalize an empty score column")
    for cid, raw in items:
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise NormalizationError(
                NormalizationError.NON_FINITE,
                f"role={role.value} candidate={cid}: non-finite raw {raw!r}")

    n = len(items)
    raws = [float(r) for _, r in items]
    distinct = sorted(set(raws))
    # dense tie-group index per raw value
    dense_index = {v: i for i, v in enumerate(distinct)}
    # count strictly-less per raw value (competition rank), precomputed
    less_count: Dict[float, int] = {}
    for v in distinct:
        less_count[v] = sum(1 for x in raws if x < v)

    entries: List[NormalizedEntry] = []
    for cid, raw in items:
        raw = float(raw)
        if n == 1:
            normalized = 0.5
        else:
            normalized = less_count[raw] / (n - 1)
        # guard the closed interval against float round-off
        normalized = min(1.0, max(0.0, normalized))
        entries.append(NormalizedEntry(
            candidate_id=cid, raw=raw, normalized=normalized,
            rank=less_count[raw], tie_group=dense_index[raw]))

    return NormalizedRoleScores(role=role, normalization=NORMALIZATION,
                                entries=entries)


def normalize_role_matrix(
    matrix: Mapping[ScoringRole, Sequence[Tuple[str, float]]]
) -> Dict[ScoringRole, NormalizedRoleScores]:
    """Normalize each role column INDEPENDENTLY. Roles do not share a scale."""
    return {role: rank_percentile_v1(role, cols) for role, cols in matrix.items()}


def normalized_map(col: NormalizedRoleScores) -> Dict[str, float]:
    """candidate_id -> normalized score (convenience for selectors)."""
    return {e.candidate_id: e.normalized for e in col.entries}
