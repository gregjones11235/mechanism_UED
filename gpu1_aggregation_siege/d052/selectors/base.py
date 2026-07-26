"""Unified selector machinery: shared signals model, deterministic ranking,
critic-policy eligibility, and the no-backfill selection funnel that EVERY
selector family routes through.

Design decisions (documented, NO_SILENT_ASSUMPTION):
  * Pure-python, numpy-free -> maximal determinism, no heavy deps.
  * Determinism tie-break is ALWAYS (composite DESC, candidate_id ASC); identical
    inputs (regardless of candidate order) -> identical selected_ids + selection_hash
    (bit-identical replay).
  * The legacy aggregation.py (blob 92a7e8b6) and auction.py (blob ec351728) cannot
    be imported at runtime here (four-package `import dicode` collision + jax/craftax
    deps absent). These canonical selectors RE-IMPLEMENT the documented semantics
    self-contained; exact numeric parity with legacy is NOT claimed (Phase-1 found
    the archived selector input contract is disjoint from the candidate bundle).

Adapter decision recorded for the reuse map: REUSE_WITH_ADAPTER realized as a
clean self-contained re-implementation against canonical schemas.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

from pydantic import Field, model_validator

from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.selector import (
    CriticPolicy,
    SelectionStatus,
    SelectorConfig,
    SelectionResult,
    compute_selection_hash,
)

BaseFn = Callable[["CandidateSignals"], float]


class CandidateSignals(CanonicalModel):
    """One candidate's normalized role signals + critic verdict + cost."""

    candidate_id: str = Field(min_length=1)
    #: role value (tutor/critic/explorer/...) -> normalized score in [0,1]
    role_scores: Dict[str, float] = Field(default_factory=dict)
    critic_reject: bool = False
    #: normalized critic penalty in [0,1] (consumed by soft_penalty only)
    critic_penalty: float = 0.0
    #: positive finite per-candidate cost (consumed by budgeted selectors only)
    cost: float = 1.0
    #: deterministic modeler-alignment bonus in [0,1] (consumed by S2 only)
    modeler_bonus: float = 0.0

    @model_validator(mode="after")
    def _bounds(self) -> "CandidateSignals":
        for k, v in self.role_scores.items():
            fv = float(v)
            if not math.isfinite(fv) or not (0.0 <= fv <= 1.0):
                raise ValueError(f"role_scores[{k}] out of [0,1]: {v}")
        if not (0.0 <= self.critic_penalty <= 1.0):
            raise ValueError(f"critic_penalty out of [0,1]: {self.critic_penalty}")
        if not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError(f"cost must be positive finite, got {self.cost}")
        if not (0.0 <= self.modeler_bonus <= 1.0):
            raise ValueError(f"modeler_bonus out of [0,1]: {self.modeler_bonus}")
        return self


class SelectorSignals(CanonicalModel):
    """All signals for one selection run, bound to ONE shared frozen pool."""

    pool_hash: str
    candidates: List[CandidateSignals]

    @model_validator(mode="after")
    def _unique(self) -> "SelectorSignals":
        validate_sha256_hex(self.pool_hash, "pool_hash")
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("DUPLICATE_CANDIDATE_SIGNALS")
        return self

    def by_id(self) -> Dict[str, CandidateSignals]:
        return {c.candidate_id: c for c in self.candidates}


class SelectorError(Exception):
    POOL_MISMATCH = "POOL_MISMATCH"
    SIGNAL_POOL_MISMATCH = "SIGNAL_POOL_MISMATCH"
    UNKNOWN_SELECTOR = "UNKNOWN_SELECTOR"
    MISSING_BUDGET = "MISSING_BUDGET"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def mean_role_scores(sig: CandidateSignals, roles: Sequence[str],
                     default: float = 0.0) -> float:
    """Deterministic mean of a candidate's normalized scores over ``roles``.

    Missing role scores count as ``default`` (0.0). Order-independent.
    """
    if not roles:
        return default
    vals = [float(sig.role_scores.get(r, default)) for r in roles]
    return sum(vals) / len(vals)


def eligible_ids(signals: SelectorSignals, policy: CriticPolicy) -> List[str]:
    """Candidates eligible under the critic policy (hard_veto excludes rejected)."""
    if policy is CriticPolicy.HARD_VETO:
        return [c.candidate_id for c in signals.candidates if not c.critic_reject]
    return [c.candidate_id for c in signals.candidates]  # soft_penalty / score_only


def composite_score(sig: CandidateSignals, policy: CriticPolicy,
                    base: float) -> float:
    """Apply critic policy to a family-specific base score, deterministically."""
    if policy is CriticPolicy.SOFT_PENALTY:
        return base - sig.critic_penalty
    return base  # hard_veto (already filtered) / score_only (critic ignored)


def rank_by(signals: SelectorSignals, config: SelectorConfig,
            base_fn: BaseFn) -> Tuple[List[Tuple[str, float]], List[str]]:
    """Rank eligible candidates by (composite DESC, candidate_id ASC).

    Returns (ordered [(candidate_id, composite)], rejected_by_critic sorted).
    ``base_fn`` supplies the family-specific pre-critic composite.
    """
    rejected = sorted(c.candidate_id for c in signals.candidates if c.critic_reject)
    by = signals.by_id()
    ranked: List[Tuple[str, float]] = []
    for cid in eligible_ids(signals, config.critic_policy):
        sig = by[cid]
        ranked.append((cid, composite_score(sig, config.critic_policy,
                                            float(base_fn(sig)))))
    ranked.sort(key=lambda t: (-float(t[1]), t[0]))
    return ranked, rejected


def _finalize(config: SelectorConfig, signals: SelectorSignals,
              eligible_count: int, rejected: List[str],
              chosen_ordered: Sequence[str], shortfall_reason: str) -> SelectionResult:
    """Build the audit-grade result; NO backfill / k-reduction / re-LLM."""
    if len(chosen_ordered) >= config.k:
        selected = list(chosen_ordered[:config.k])
        status = SelectionStatus.OK
        note = ""
    else:
        selected = list(chosen_ordered)
        status = SelectionStatus.INSUFFICIENT_ELIGIBLE_CANDIDATES
        note = shortfall_reason
    selection_hash = compute_selection_hash(
        config.selector.value, config.critic_policy.value, config.k, config.seed,
        selected)
    return SelectionResult(
        selector=config.selector,
        critic_policy=config.critic_policy,
        k_requested=config.k,
        seed=config.seed,
        candidate_count_in=len(signals.candidates),
        eligible_count=eligible_count,
        selected_ids=selected,
        rejected_by_critic=rejected,
        selection_status=status,
        selection_hash=selection_hash,
        shortfall_note=note,
    )


def select_unbudgeted(config: SelectorConfig, signals: SelectorSignals,
                      base_fn: BaseFn) -> SelectionResult:
    """Top-k over the full eligible ranking (shortfall only from eligibility)."""
    ranked, rejected = rank_by(signals, config, base_fn)
    chosen = [cid for cid, _ in ranked]
    reason = (f"only {len(ranked)} eligible candidates for k={config.k} under "
              f"critic_policy={config.critic_policy.value}; selected all "
              f"{len(chosen)}; NO backfill / NO k-reduction / NO re-LLM")
    return _finalize(config, signals, len(ranked), rejected, chosen, reason)


def select_budgeted(config: SelectorConfig, signals: SelectorSignals,
                    base_fn: BaseFn) -> SelectionResult:
    """Greedy highest-composite-first under a cumulative-cost cap.

    Walks the deterministic ranking; admits a candidate iff cumulative cost stays
    <= budget; stops once k are admitted. Shortfall (eligibility OR budget) ->
    INSUFFICIENT with an honest note; never backfills.
    """
    if config.budget is None:
        raise SelectorError(
            SelectorError.MISSING_BUDGET,
            f"selector {config.selector.value} requires a budget")
    ranked, rejected = rank_by(signals, config, base_fn)
    by = signals.by_id()
    chosen: List[str] = []
    cumulative = 0.0
    for cid, _ in ranked:
        if len(chosen) >= config.k:
            break
        cost = by[cid].cost
        if cumulative + cost <= config.budget:
            chosen.append(cid)
            cumulative += cost
    reason = (f"budget={config.budget} admitted only {len(chosen)} of k={config.k} "
              f"under cumulative cost (eligible={len(ranked)}, "
              f"critic_policy={config.critic_policy.value}); NO backfill / "
              f"NO k-reduction / NO re-LLM")
    return _finalize(config, signals, len(ranked), rejected, chosen, reason)
