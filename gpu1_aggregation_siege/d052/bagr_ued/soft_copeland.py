"""Criterion-wise Soft Copeland over environments (task section 12).

THE ONLY place the scoring dimensions are combined — and even here the
combination is explicit, versioned, and auditable. Upstream, front_regret /
global_regret / behavioral_gap remain THREE SEPARATE quantities; this module
receives all of them plus learning_progress, learnability, diversity,
global_retention, critic_penalty (the "at least" input set of section 12) and
alpha_front (front-vs-global regret weighting).

CC3 fix2 (task §1-§3) — CRITERION-WISE rewrite. The v1 implementation had two
defects this version removes:

  1. alpha_front was multiplied into the RAW front/global regret BEFORE the
     min-max normalization, so any uniform positive alpha is canceled out by
     the normalization and CANNOT change the FRONT/GLOBAL preference — the
     alpha split was cosmetically visible but numerically inert on the
     ranking;
  2. a single weighted-sum ``strength`` was computed and the pairwise
     comparison was then run over that scalar — i.e. a weighted-sum ranking,
     NOT a criterion-wise Soft Copeland.

This version:

  * normalizes each RAW dimension INDEPENDENTLY with NO alpha before
    normalization (degenerate constant dimension -> 0.5, recorded in the
    normalization provenance as constant=true);
  * compares candidates CRITERION BY CRITERION with a numerically stable soft
    preference  p_k(i>j) = sigmoid(sign_k * (s_i_k - s_j_k) / temperature_k)
    (critic_penalty is lower-is-better -> sign_k = -1; all others +1);
  * applies dimension weights AT THE PAIRWISE LEVEL — front_weight =
    alpha_front, global_weight = 1 - alpha_front (strictly positive because
    the schema refuses alpha_front == 1), the rest from the versioned weight
    table — with NO pre-aggregated strength;
  * copeland_i = mean over j != i of the weighted pairwise preference
    (equivalently 0.5 + mean pairwise_margin; the raw margin sum is recorded
    per entry);
  * records the FULL audit trail: per-dimension normalization provenance
    (min / max / constant / normalized values), the resolved weights, the
    temperatures, the candidate-id-aligned pairwise preference AND pairwise
    margin matrices, the final Copeland scores, and the deterministic
    tie-break rule — all bound into ranking_hash for bit-identical replay.

Final environment VALUE still requires REAL rollout validation; these scores
are dry-run mock-evidence based and labeled so in the controller certificate.
"""
from __future__ import annotations

import math
from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.schemas.common import CanonicalModel, validate_finite

#: CC3 fix2 (§1): new weights contract. front/global weights are NOT in this
#: table — they are resolved AT THE PAIRWISE LEVEL from alpha_front
#: (front_weight = alpha_front, global_weight = 1 - alpha_front).
WEIGHTS_VERSION = "bagr_ued.soft_copeland.weights.v2"
TEMPERATURES_VERSION = "bagr_ued.soft_copeland.temperatures.v1"

#: the eight RAW dimensions normalized independently (no alpha before
#: normalization). Order is the canonical dimension order of the matrices.
RAW_DIMENSIONS = (
    "front_regret",
    "global_regret",
    "behavioral_gap",
    "learning_progress",
    "learnability",
    "diversity",
    "global_retention",
    "critic_penalty",
)

#: configurable dimension weights applied AT THE PAIRWISE LEVEL (versioned;
#: change => new version string). critic_penalty is a full-weight criterion
#: (lower is better — handled by the sign, not by the weight).
DIMENSION_WEIGHTS: Dict[str, float] = {
    "behavioral_gap": 1.0,
    "learning_progress": 0.8,
    "learnability": 0.8,
    "diversity": 0.6,
    "global_retention": 0.6,
    "critic_penalty": 1.0,
}

#: per-dimension soft-comparison temperatures (versioned, auditable). A
#: uniform 1.0 on min-max normalized [0,1] differences keeps every criterion
#: contribution strictly inside (0, 1) — a constant dimension degrades to the
#: neutral preference 0.5 instead of dominating.
DIMENSION_TEMPERATURES: Dict[str, float] = {k: 1.0 for k in RAW_DIMENSIONS}

#: lower-is-better dimensions: the pairwise sign is flipped so a LOWER raw
#: value yields p(i>j) > 0.5.
_LOWER_IS_BETTER = frozenset({"critic_penalty"})

TIE_EPS = 1e-12

#: deterministic tie-break rule (recorded in the ranking audit trail)
TIE_BREAK_RULE = "copeland_score_desc__environment_id_asc"


class EnvironmentScoreBundle(CanonicalModel):
    """The >=8 signals Soft Copeland must receive, all separate fields."""

    environment_id: str = Field(min_length=1)
    front_regret: float = Field(ge=0.0)
    global_regret: float = Field(ge=0.0)
    behavioral_gap: float = Field(ge=0.0)
    learning_progress: float = Field(ge=0.0, le=1.0)
    learnability: float = Field(ge=0.0, le=1.0)
    diversity: float = Field(ge=0.0, le=1.0)
    global_retention: float = Field(ge=0.0, le=1.0)
    critic_penalty: float = Field(ge=0.0, le=1.0)
    #: CC1 audit fix1 (§8): alpha_front is structurally < 1.0 (strict). The
    #: global pairwise weight (1 - alpha_front) must ALWAYS be strictly
    #: positive — the schema refuses alpha_front == 1.0 outright.
    alpha_front: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _finite_all(self) -> "EnvironmentScoreBundle":
        for name in ("front_regret", "global_regret", "behavioral_gap",
                     "learning_progress", "learnability", "diversity",
                     "global_retention", "critic_penalty", "alpha_front"):
            validate_finite(getattr(self, name), name)
        return self


class CopelandEntry(CanonicalModel):
    environment_id: str
    copeland_score: float = Field(ge=0.0, le=1.0)
    #: raw sum over j != i of pairwise_margin(i, j); copeland_score equals
    #: 0.5 + copeland_margin_sum / (n - 1) — both forms are recorded
    copeland_margin_sum: float = 0.0
    rank: int = Field(ge=1)
    #: per-dimension NORMALIZED values (no alpha premultiplied) + the raw
    #: critic penalty + the weights used — the full criterion audit trail
    components: Dict[str, float] = Field(default_factory=dict)


class CopelandRanking(CanonicalModel):
    weights_version: str = WEIGHTS_VERSION
    temperatures_version: str = TEMPERATURES_VERSION
    #: the run-level alpha actually used (uniform across the batch)
    alpha_front_used: float = Field(default=0.0, ge=0.0, lt=1.0)
    #: resolved pairwise-level weights INCLUDING front=alpha / global=1-alpha
    dimension_weights: Dict[str, float] = Field(default_factory=dict)
    dimension_temperatures: Dict[str, float] = Field(default_factory=dict)
    #: candidate ids in canonical (sorted) order — the matrices are aligned
    #: to this order row-for-row and column-for-column
    environment_order: List[str] = Field(default_factory=list)
    #: per raw dimension: {min, max, constant, normalized[]} — provenance of
    #: the independent normalization (NO alpha applied before this step)
    normalization_provenance: Dict[str, dict] = Field(default_factory=dict)
    #: pairwise_preference[i][j] = P(candidate_i > candidate_j) in (0, 1);
    #: diagonal is the neutral 0.5
    pairwise_preference: List[List[float]] = Field(default_factory=list)
    #: pairwise_margin[i][j] = pairwise_preference[i][j] - 0.5
    pairwise_margin: List[List[float]] = Field(default_factory=list)
    tie_break_rule: str = TIE_BREAK_RULE
    entries: List[CopelandEntry] = Field(default_factory=list)
    ranking_hash: str = ""


def _sigmoid(x: float) -> float:
    """Numerically stable logistic — equivalent to 1/(1+exp(-x)) for all x."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _minmax(values: List[float]) -> tuple:
    """Independent min-max normalization of one RAW dimension.

    Returns (normalized, provenance). A degenerate constant dimension maps to
    0.5 for every candidate (neutral pairwise contribution) and is recorded
    constant=true — nothing is silently dropped.
    """
    lo, hi = min(values), max(values)
    constant = (hi - lo) <= TIE_EPS
    norm = [0.5] * len(values) if constant else \
        [(v - lo) / (hi - lo) for v in values]
    prov = dict(min=round(lo, 12), max=round(hi, 12), constant=constant,
                normalized=[round(v, 9) for v in norm])
    return norm, prov


def soft_copeland_rank(bundles: List[EnvironmentScoreBundle]) -> CopelandRanking:
    """Criterion-wise Soft Copeland (CC3 fix2 §1-§3). See module docstring."""
    if len(set(b.environment_id for b in bundles)) != len(bundles):
        raise ValueError("DUPLICATE_ENVIRONMENT_SCORES")
    if not bundles:
        raise ValueError("EMPTY_ENVIRONMENT_SCORES")

    # fail-closed on any non-finite input (defense in depth beyond the schema)
    for b in bundles:
        for dim in RAW_DIMENSIONS:
            v = getattr(b, dim)
            if not math.isfinite(v):
                raise ValueError(
                    f"NON_FINITE_SCORE_INPUT: {b.environment_id}:{dim}={v!r}")
    alphas = {b.alpha_front for b in bundles}
    if len(alphas) != 1:
        raise ValueError(
            f"ALPHA_FRONT_NOT_UNIFORM: alpha_front is a run-level weighting, "
            f"got {sorted(alphas)} across the batch")
    alpha_front = next(iter(alphas))

    ordered = sorted(bundles, key=lambda b: b.environment_id)
    n = len(ordered)
    env_order = [b.environment_id for b in ordered]

    # 1. independent normalization of each RAW dimension — NO alpha before
    #    normalization (fix2 §1.1; this is exactly what v1 got wrong)
    raw_cols = {dim: [getattr(b, dim) for b in ordered]
                for dim in RAW_DIMENSIONS}
    norm: Dict[str, List[float]] = {}
    norm_prov: Dict[str, dict] = {}
    for dim in RAW_DIMENSIONS:
        norm[dim], norm_prov[dim] = _minmax(raw_cols[dim])

    # 2. pairwise-level weights: front = alpha, global = 1 - alpha (strictly
    #    positive — schema refuses alpha == 1), rest from the versioned table
    weights: Dict[str, float] = {
        "front_regret": alpha_front,
        "global_regret": 1.0 - alpha_front,
    }
    weights.update(DIMENSION_WEIGHTS)
    total_w = sum(weights[d] for d in RAW_DIMENSIONS)

    # 3. criterion-wise soft pairwise preference (NO pre-aggregated strength)
    pref = [[0.5] * n for _ in range(n)]
    margin = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            acc = 0.0
            for dim in RAW_DIMENSIONS:
                sign = -1.0 if dim in _LOWER_IS_BETTER else 1.0
                t = DIMENSION_TEMPERATURES[dim]
                diff = sign * (norm[dim][i] - norm[dim][j]) / t
                acc += weights[dim] * _sigmoid(diff)
            p = acc / total_w
            pref[i][j] = round(p, 9)
            margin[i][j] = round(p - 0.5, 9)

    # 4. copeland_i = mean_{j != i} pairwise_preference(i, j)
    #              = 0.5 + mean pairwise margin   (equivalent; both recorded)
    copeland: List[float] = []
    margin_sum: List[float] = []
    for i in range(n):
        if n == 1:
            copeland.append(0.5)
            margin_sum.append(0.0)
            continue
        s = sum(pref[i][j] for j in range(n) if j != i) / (n - 1)
        copeland.append(s)
        margin_sum.append(sum(margin[i][j] for j in range(n) if j != i))

    rows = sorted(range(n),
                  key=lambda i: (-copeland[i], ordered[i].environment_id))
    entries = []
    for rank, i in enumerate(rows, start=1):
        comp = {dim: round(norm[dim][i], 9) for dim in RAW_DIMENSIONS}
        comp["critic_penalty_raw"] = ordered[i].critic_penalty
        comp["alpha_front_used"] = alpha_front
        comp["pairwise_weight_front_regret"] = weights["front_regret"]
        comp["pairwise_weight_global_regret"] = weights["global_regret"]
        entries.append(CopelandEntry(
            environment_id=ordered[i].environment_id,
            copeland_score=round(copeland[i], 9),
            copeland_margin_sum=round(margin_sum[i], 9),
            rank=rank,
            components=comp))

    ranking = CopelandRanking(
        alpha_front_used=alpha_front,
        dimension_weights=weights,
        dimension_temperatures=dict(DIMENSION_TEMPERATURES),
        environment_order=env_order,
        normalization_provenance=norm_prov,
        pairwise_preference=pref,
        pairwise_margin=margin,
        entries=entries)
    object.__setattr__(ranking, "ranking_hash", canonical_sha256(
        {"weights_version": WEIGHTS_VERSION,
         "temperatures_version": TEMPERATURES_VERSION,
         "alpha_front_used": alpha_front,
         "dimension_weights": weights,
         "dimension_temperatures": DIMENSION_TEMPERATURES,
         "environment_order": env_order,
         "normalization_provenance": norm_prov,
         "pairwise_preference": pref,
         "pairwise_margin": margin,
         "tie_break_rule": TIE_BREAK_RULE,
         "entries": [e.model_dump() for e in entries]}))
    return ranking
