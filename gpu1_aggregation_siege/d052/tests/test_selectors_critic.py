"""GATE 8 — critic-policy consumption (hard_veto / soft_penalty / score_only).

hard_veto (default): critic_reject=True excludes; a shortfall yields
INSUFFICIENT_ELIGIBLE_CANDIDATES with NO backfill / NO k-reduction / NO re-LLM.
soft_penalty: critic signal subtracts from the composite (can reorder), candidate
stays eligible. score_only: critic recorded but never affects eligibility/rank.
"""
import pytest

from d052.generation import build_pool
from d052.schemas.selector import (
    CriticPolicy,
    SelectionStatus,
    SelectorConfig,
    SelectorType,
)
from d052.selectors import CandidateSignals, SelectorSignals, select

_TP = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
       "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}


def _pool(ids):
    return build_pool(
        "p", [{"task_id": i, "task_params": dict(_TP),
               "target_achievements": ["collect_wood"]} for i in ids])


def _signals(pool, **over):
    cands = []
    for i in pool.candidates:
        kw = dict(over.get(i.task_id, {}))
        cands.append(CandidateSignals(candidate_id=i.task_id, **kw))
    return SelectorSignals(pool_hash=pool.pool_hash, candidates=cands)


# --- hard_veto --------------------------------------------------------------

def test_hard_veto_excludes_rejected():
    pool = _pool(["a", "b", "c"])
    sig = _signals(pool, c={"critic_reject": True},
                   a={"role_scores": {"tutor": 0.9}},
                   b={"role_scores": {"tutor": 0.5}})
    cfg = SelectorConfig(selector=SelectorType.AUCTION_RAW, k=2, seed=1)
    res = select(cfg, pool, sig)
    assert res.critic_policy is CriticPolicy.HARD_VETO
    assert res.selected_ids == ["a", "b"]
    assert res.rejected_by_critic == ["c"]
    assert res.eligible_count == 2
    assert res.selection_status is SelectionStatus.OK


def test_hard_veto_shortfall_is_insufficient_no_backfill():
    pool = _pool(["a", "b", "c"])
    sig = _signals(pool, c={"critic_reject": True})
    cfg = SelectorConfig(selector=SelectorType.AUCTION_RAW, k=3, seed=1)
    res = select(cfg, pool, sig)
    assert res.selection_status is SelectionStatus.INSUFFICIENT_ELIGIBLE_CANDIDATES
    assert len(res.selected_ids) == 2          # all eligible, < k
    assert res.eligible_count == 2
    assert res.shortfall_note                  # honest reason recorded
    assert "NO backfill" in res.shortfall_note
    assert res.rejected_by_critic == ["c"]


def test_hard_veto_all_rejected_yields_empty_insufficient():
    pool = _pool(["a", "b"])
    sig = _signals(pool, a={"critic_reject": True}, b={"critic_reject": True})
    cfg = SelectorConfig(selector=SelectorType.SOFT_COPELAND, k=1, seed=1)
    res = select(cfg, pool, sig)
    assert res.selection_status is SelectionStatus.INSUFFICIENT_ELIGIBLE_CANDIDATES
    assert res.selected_ids == []
    assert res.eligible_count == 0


# --- soft_penalty -----------------------------------------------------------

def test_soft_penalty_reorders_but_keeps_eligible():
    pool = _pool(["a", "b"])
    # a has the higher base bid but a heavy critic penalty -> b wins
    sig = _signals(pool,
                   a={"role_scores": {"tutor": 0.9}, "critic_penalty": 0.8},
                   b={"role_scores": {"tutor": 0.5}, "critic_penalty": 0.0})
    cfg = SelectorConfig(selector=SelectorType.AUCTION_RAW, k=1, seed=1,
                         critic_policy=CriticPolicy.SOFT_PENALTY)
    res = select(cfg, pool, sig)
    assert res.selected_ids == ["b"]
    assert res.eligible_count == 2          # a still eligible (not vetoed)
    assert res.selection_status is SelectionStatus.OK


def test_soft_penalty_does_not_apply_when_zero():
    pool = _pool(["a", "b"])
    sig = _signals(pool, a={"role_scores": {"tutor": 0.9}},
                   b={"role_scores": {"tutor": 0.5}})
    cfg = SelectorConfig(selector=SelectorType.AUCTION_RAW, k=1, seed=1,
                         critic_policy=CriticPolicy.SOFT_PENALTY)
    res = select(cfg, pool, sig)
    assert res.selected_ids == ["a"]


# --- score_only -------------------------------------------------------------

def test_score_only_ignores_critic_for_rank_and_eligibility():
    pool = _pool(["a", "b"])
    # a is critic-rejected AND penalized, but score_only ignores both -> a wins
    sig = _signals(pool,
                   a={"role_scores": {"tutor": 0.9}, "critic_reject": True,
                      "critic_penalty": 0.9},
                   b={"role_scores": {"tutor": 0.5}})
    cfg = SelectorConfig(selector=SelectorType.AUCTION_RAW, k=1, seed=1,
                         critic_policy=CriticPolicy.SCORE_ONLY)
    res = select(cfg, pool, sig)
    assert res.selected_ids == ["a"]
    assert res.eligible_count == 2
    assert res.rejected_by_critic == ["a"]   # verdict still recorded for audit


def test_score_only_matches_no_critic_ordering():
    pool = _pool(["a", "b", "c"])
    base = {"a": {"role_scores": {"tutor": 0.9}},
            "b": {"role_scores": {"tutor": 0.6}},
            "c": {"role_scores": {"tutor": 0.3}}}
    penalized = {"a": {"role_scores": {"tutor": 0.9}, "critic_penalty": 0.9},
                 "b": {"role_scores": {"tutor": 0.6}, "critic_penalty": 0.5},
                 "c": {"role_scores": {"tutor": 0.3}, "critic_reject": True}}
    cfg_plain = SelectorConfig(selector=SelectorType.SOFT_COPELAND, k=3, seed=1)
    cfg_score = SelectorConfig(selector=SelectorType.SOFT_COPELAND, k=3, seed=1,
                               critic_policy=CriticPolicy.SCORE_ONLY)
    plain = select(cfg_plain, pool, _signals(pool, **base))
    scored = select(cfg_score, pool, _signals(pool, **penalized))
    assert plain.selected_ids == scored.selected_ids   # critic had zero effect
