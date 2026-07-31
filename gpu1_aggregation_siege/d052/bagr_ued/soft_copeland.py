"""Soft Copeland over environments (task section 12).

THE ONLY place the scoring dimensions are combined — and even here the
combination is explicit, versioned, and auditable. Upstream, front_regret /
global_regret / behavioral_gap remain THREE SEPARATE quantities; this module
receives all of them plus learning_progress, learnability, diversity,
global_retention, critic_penalty (the "at least" input set of section 12) and
alpha_front (front-vs-global regret weighting).

Semantics (documented lineage of d052.selectors.copeland):
  * each dimension is min-max normalized over the batch (degenerate constant
    dimension -> 0.5), so nothing is silently dropped;
  * front component = alpha_front * front_regret, global component =
    (1 - alpha_front) * global_regret — the alpha split is VISIBLE, not buried;
  * strength = sum(weight_i * normalized_i) - critic_penalty;
  * full pairwise Copeland: win +1 / tie +0.5 (|d| <= 1e-12) / loss 0,
    normalized by (n - 1);
  * deterministic tie-break (copeland_score DESC, strength DESC, id ASC) ->
    bit-identical replay -> ranking_hash.

Final environment VALUE still requires REAL rollout validation; these scores
are dry-run mock-evidence based and labeled so in the controller certificate.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.schemas.common import CanonicalModel, validate_finite

WEIGHTS_VERSION = "bagr_ued.soft_copeland.weights.v1"

#: documented dimension weights (versioned; change => new version string)
DIMENSION_WEIGHTS: Dict[str, float] = {
    "alpha_front_regret": 1.0,
    "one_minus_alpha_global_regret": 1.0,
    "behavioral_gap": 1.0,
    "learning_progress": 0.8,
    "learnability": 0.8,
    "diversity": 0.6,
    "global_retention": 0.6,
}

TIE_EPS = 1e-12


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
    #: global component weight (1 - alpha_front) must ALWAYS be strictly
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
    strength: float
    copeland_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    components: Dict[str, float] = Field(default_factory=dict)


class CopelandRanking(CanonicalModel):
    weights_version: str = WEIGHTS_VERSION
    entries: List[CopelandEntry] = Field(default_factory=list)
    ranking_hash: str = ""


def _minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi - lo <= TIE_EPS:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def soft_copeland_rank(bundles: List[EnvironmentScoreBundle]) -> CopelandRanking:
    if len(set(b.environment_id for b in bundles)) != len(bundles):
        raise ValueError("DUPLICATE_ENVIRONMENT_SCORES")
    ordered = sorted(bundles, key=lambda b: b.environment_id)
    n = len(ordered)

    # dimension columns (the visible alpha split lives here)
    cols = {
        "alpha_front_regret": [b.alpha_front * b.front_regret for b in ordered],
        "one_minus_alpha_global_regret":
            [(1.0 - b.alpha_front) * b.global_regret for b in ordered],
        "behavioral_gap": [b.behavioral_gap for b in ordered],
        "learning_progress": [b.learning_progress for b in ordered],
        "learnability": [b.learnability for b in ordered],
        "diversity": [b.diversity for b in ordered],
        "global_retention": [b.global_retention for b in ordered],
    }
    norm = {k: _minmax(v) for k, v in cols.items()}

    strengths: List[float] = []
    components: List[Dict[str, float]] = []
    for i, b in enumerate(ordered):
        comp = {k: round(norm[k][i], 9) for k in cols}
        s = sum(DIMENSION_WEIGHTS[k] * norm[k][i] for k in cols) - b.critic_penalty
        strengths.append(s)
        comp["critic_penalty_subtracted"] = b.critic_penalty
        comp["strength"] = round(s, 9)
        components.append(comp)

    # full pairwise Copeland
    copeland = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = strengths[i] - strengths[j]
            if d > TIE_EPS:
                copeland[i] += 1.0
            elif abs(d) <= TIE_EPS:
                copeland[i] += 0.5
    if n > 1:
        copeland = [c / (n - 1) for c in copeland]

    rows = sorted(
        range(n),
        key=lambda i: (-copeland[i], -strengths[i], ordered[i].environment_id))
    entries = [CopelandEntry(
        environment_id=ordered[i].environment_id,
        strength=round(strengths[i], 9),
        copeland_score=round(copeland[i], 9),
        rank=rank,
        components=components[i]) for rank, i in enumerate(rows, start=1)]
    ranking = CopelandRanking(entries=entries)
    object.__setattr__(ranking, "ranking_hash", canonical_sha256(
        {"weights_version": WEIGHTS_VERSION,
         "dimension_weights": DIMENSION_WEIGHTS,
         "entries": [e.model_dump() for e in entries]}))
    return ranking
