"""C12: multi-criterion Stage-2 selection — shared Soft Copeland, no
hand-written scalar.

This module REPLACES the former hand-written weighted-sum ``_full_score``:
the Stage-2 full-probe metrics are mapped into the eight RAW criteria the
shared ``d052.bagr_ued.soft_copeland`` module consumes (the criteria are
STORED SEPARATELY on every stage-2 audit record — never collapsed into a
single hand-written scalar at this layer), and the family-diverse greedy
pick of the 12 dynamic slots is driven by the Copeland scores that shared
module produces. Soft Copeland is therefore NOT forked: this direction
consumes the common canonical implementation (weights/temperatures versions,
pairwise matrices, ``ranking_hash``) unchanged.

Criterion mapping (symbolic-probe degenerate limits, documented — never
silently dropped):

====================  =======================================================
front_regret          ``ProbeMetrics.regret`` (Reference-vs-Student success gap)
global_regret         ``ProbeMetrics.regret`` — the symbolic probe has NO
                      separate global regret channel, so the alpha split
                      re-weights ONE signal (documented degenerate limit)
behavioral_gap        ``max(0, reference_behavior_activation
                      - student_behavior_activation)``
learning_progress     ``student_front_progress``
learnability          ``learnability``
diversity             pool-relative family rarity:
                      ``1 - (family_count(fam) - 1) / max(1, pool_size - 1)``
global_retention      ``global_retention``
critic_penalty        simulator cost: ``simulator_transitions /
                      max_transitions`` — every Stage-2 probe consumes the
                      IDENTICAL episode budget, so this dimension is constant
                      across the pool and degrades to the neutral 0.5 inside
                      ``soft_copeland_rank`` (recorded ``constant=true`` in
                      its normalization provenance; documented limit)
====================  =======================================================

ENGINEERING_SCAFFOLD: mock probe metrics; fully deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

from d052.bagr_ued.soft_copeland import (
    RAW_DIMENSIONS,
    CopelandRanking,
    EnvironmentScoreBundle,
    soft_copeland_rank,
)
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import ProbeMetrics


def stage2_criteria(m: ProbeMetrics, *, family: str,
                    family_counts: Mapping[str, int], pool_size: int,
                    max_transitions: int) -> Dict[str, float]:
    """Map one full-probe metric set to the eight RAW Soft-Copeland criteria.

    Every value is clamped into the ``EnvironmentScoreBundle`` schema ranges;
    the mapping is a pure function of the probe metrics + pool composition,
    so a replayed run reproduces it bit-for-bit.
    """
    if pool_size <= 0:
        raise ValueError(f"ILLEGAL_STAGE2_POOL_SIZE: {pool_size}")
    if max_transitions <= 0:
        raise ValueError(f"ILLEGAL_STAGE2_MAX_TRANSITIONS: {max_transitions}")
    fam_n = int(family_counts.get(family, 0))
    if fam_n <= 0:
        raise ValueError(
            f"STAGE2_FAMILY_COUNT_MISSING: {family!r} not counted in the "
            "stage-2 pool")
    diversity = 1.0 - (fam_n - 1) / float(max(1, pool_size - 1))
    return dict(
        front_regret=min(1.0, max(0.0, m.regret)),
        # degenerate limit: see module docstring (one signal, alpha split)
        global_regret=min(1.0, max(0.0, m.regret)),
        behavioral_gap=min(1.0, max(0.0, m.reference_behavior_activation
                                    - m.student_behavior_activation)),
        learning_progress=min(1.0, max(0.0, m.student_front_progress)),
        learnability=min(1.0, max(0.0, m.learnability)),
        diversity=min(1.0, max(0.0, diversity)),
        global_retention=min(1.0, max(0.0, m.global_retention)),
        critic_penalty=min(1.0, max(0.0, m.simulator_transitions
                                    / float(max_transitions))))


def copeland_stage2_selection(
        pool: List[Tuple[object, ProbeMetrics]], *, keep: int,
        max_transitions: int,
        alpha_front: float = C.ALPHA_FRONT_STAGE2,
        family_penalty: float = C.STAGE2_FAMILY_PENALTY
) -> Tuple[List[Tuple[object, ProbeMetrics]], Dict[str, dict],
           CopelandRanking]:
    """Rank the full-probed pool with the SHARED Soft Copeland and greedily
    pick ``keep`` family-diverse candidates.

    Returns ``(picked, audit, ranking)``: ``picked`` preserves the greedy
    selection order; ``audit`` maps candidate_id -> {criteria (the eight
    separate RAW values), copeland_score, copeland_rank}; ``ranking`` is the
    full hash-bound audit trail of ``soft_copeland_rank``.

    No hand-written composite score participates in the cut: the ONLY scalar
    driving the pick is the shared module's Copeland score, with the
    documented per-family diversity penalty on the greedy pass.
    """
    if not pool:
        raise ValueError("EMPTY_STAGE2_POOL")
    if keep <= 0:
        raise ValueError(f"ILLEGAL_STAGE2_KEEP: {keep}")
    for cand, m in pool:
        if not isinstance(m, ProbeMetrics):
            raise ValueError(
                f"ILLEGAL_STAGE2_POOL_METRICS: {cand.candidate_id!r}")

    family_counts: Dict[str, int] = {}
    for cand, _m in pool:
        family_counts[cand.environment_family] = \
            family_counts.get(cand.environment_family, 0) + 1

    criteria: Dict[str, Dict[str, float]] = {}
    bundles: List[EnvironmentScoreBundle] = []
    for cand, m in pool:
        crit = stage2_criteria(m, family=cand.environment_family,
                               family_counts=family_counts,
                               pool_size=len(pool),
                               max_transitions=max_transitions)
        criteria[cand.candidate_id] = crit
        bundles.append(EnvironmentScoreBundle(
            environment_id=cand.candidate_id,
            front_regret=crit["front_regret"],
            global_regret=crit["global_regret"],
            behavioral_gap=crit["behavioral_gap"],
            learning_progress=crit["learning_progress"],
            learnability=crit["learnability"],
            diversity=crit["diversity"],
            global_retention=crit["global_retention"],
            critic_penalty=crit["critic_penalty"],
            alpha_front=alpha_front))

    ranking = soft_copeland_rank(bundles)
    score_by_id = {e.environment_id: e.copeland_score
                   for e in ranking.entries}
    rank_by_id = {e.environment_id: e.rank for e in ranking.entries}
    audit = {cid: dict(criteria=dict(criteria[cid]),
                       copeland_score=score_by_id[cid],
                       copeland_rank=rank_by_id[cid])
             for cid in criteria}

    # greedy family-diverse pick over the Copeland scores (deterministic:
    # score desc, then candidate_id asc; per-family penalty on each repeat)
    remaining = sorted(pool, key=lambda t: (-score_by_id[t[0].candidate_id],
                                            t[0].candidate_id))
    picked: List[Tuple[object, ProbeMetrics]] = []
    picked_family_counts: Dict[str, int] = {}
    while len(picked) < keep and remaining:
        best_i = None
        best_eff = None
        for i, (cand, _m) in enumerate(remaining):
            eff = (score_by_id[cand.candidate_id]
                   - family_penalty
                   * picked_family_counts.get(cand.environment_family, 0))
            if best_eff is None or eff > best_eff or \
                    (eff == best_eff and cand.candidate_id
                     < remaining[best_i][0].candidate_id):
                best_eff = eff
                best_i = i
        cand, m = remaining.pop(best_i)
        picked_family_counts[cand.environment_family] = \
            picked_family_counts.get(cand.environment_family, 0) + 1
        picked.append((cand, m))
    return picked, audit, ranking


#: the canonical eight criterion names (re-export for tests / audits)
CRITERION_NAMES = tuple(RAW_DIMENSIONS)
