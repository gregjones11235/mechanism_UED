"""GATE 9 — deterministic bit-identical replay + the unified selector interface.

Every selector type runs through ONE entry point (select) over ONE shared frozen
pool, and identical inputs (regardless of candidate order) reproduce identical
selected_ids + selection_hash. Also covers the shared-pool invariants, the
budgeted cost-cap, and the S0 content-order floor.
"""
import pytest

from d052.generation import build_pool
from d052.schemas.selector import (
    SelectionStatus,
    SelectorConfig,
    SelectorType,
)
from d052.selectors import (
    CandidateSignals,
    SelectorError,
    SelectorSignals,
    copeland_scores,
    select,
)

_TP = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
       "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}
SCORING = ["tutor", "critic", "explorer"]


def _pool(ids):
    return build_pool(
        "p", [{"task_id": i, "task_params": dict(_TP),
               "target_achievements": ["collect_wood"]} for i in ids])


def _sig(pool, spec):
    """spec: candidate_id -> CandidateSignals kwargs."""
    return SelectorSignals(
        pool_hash=pool.pool_hash,
        candidates=[CandidateSignals(candidate_id=c.task_id,
                                     **spec.get(c.task_id, {}))
                    for c in pool.candidates])


def _rich_spec():
    # distinct, varied signals so every family has a non-trivial ranking
    return {
        "a": {"role_scores": {"tutor": 0.9, "explorer": 0.2, "critic": 0.5},
              "critic_penalty": 0.1, "cost": 1.0, "modeler_bonus": 0.3},
        "b": {"role_scores": {"tutor": 0.4, "explorer": 0.8, "critic": 0.6},
              "critic_penalty": 0.2, "cost": 1.0, "modeler_bonus": 0.1},
        "c": {"role_scores": {"tutor": 0.6, "explorer": 0.5, "critic": 0.4},
              "critic_penalty": 0.0, "cost": 1.0, "modeler_bonus": 0.2},
        "d": {"role_scores": {"tutor": 0.1, "explorer": 0.9, "critic": 0.7},
              "critic_penalty": 0.3, "cost": 1.0, "modeler_bonus": 0.0},
        "e": {"role_scores": {"tutor": 0.7, "explorer": 0.3, "critic": 0.2},
              "critic_penalty": 0.1, "cost": 1.0, "modeler_bonus": 0.4},
    }


def _all_configs():
    return [
        SelectorConfig(selector=SelectorType.S0_CANONICAL_BASELINE, k=2, seed=7),
        SelectorConfig(selector=SelectorType.S1_THREE_ROLE, k=2, seed=7,
                       roles=SCORING),
        SelectorConfig(selector=SelectorType.S2_FOUR_ROLE_MODELER, k=2, seed=7,
                       roles=SCORING),
        SelectorConfig(selector=SelectorType.SOFT_COPELAND, k=2, seed=7),
        SelectorConfig(selector=SelectorType.BUDGETED_SOFT_COPELAND, k=2, seed=7,
                       budget=2.5),
        SelectorConfig(selector=SelectorType.AUCTION_RAW, k=2, seed=7),
        SelectorConfig(selector=SelectorType.AUCTION_BUDGETED, k=2, seed=7,
                       budget=2.5),
    ]


# --- unified interface: all 7 selectors run through select() ---------------

def test_all_seven_selectors_produce_valid_results():
    pool = _pool(["a", "b", "c", "d", "e"])
    spec = _rich_spec()
    seen = set()
    for cfg in _all_configs():
        res = select(cfg, pool, _sig(pool, spec))
        assert res.selector is cfg.selector
        assert len(res.selected_ids) == 2
        assert res.selection_status is SelectionStatus.OK
        assert res.candidate_count_in == 5
        seen.add(cfg.selector)
    assert len(seen) == 7


# --- determinism: bit-identical replay under candidate-order shuffle -------

@pytest.mark.parametrize("cfg", _all_configs(),
                         ids=lambda c: c.selector.value)
def test_bit_identical_replay_under_shuffle(cfg):
    pool = _pool(["a", "b", "c", "d", "e"])
    spec = _rich_spec()
    forward = _sig(pool, spec)
    reversed_sig = SelectorSignals(
        pool_hash=pool.pool_hash,
        candidates=list(reversed(forward.candidates)))
    r1 = select(cfg, pool, forward)
    r2 = select(cfg, pool, reversed_sig)
    assert r1.model_dump() == r2.model_dump()   # bit-for-bit identical


def test_selection_hash_binds_seed_and_k():
    pool = _pool(["a", "b", "c"])
    spec = {"a": {"role_scores": {"tutor": 0.9}},
            "b": {"role_scores": {"tutor": 0.5}},
            "c": {"role_scores": {"tutor": 0.1}}}
    base = dict(selector=SelectorType.AUCTION_RAW, k=2)
    h_seed7 = select(SelectorConfig(seed=7, **base), pool,
                     _sig(pool, spec)).selection_hash
    h_seed8 = select(SelectorConfig(seed=8, **base), pool,
                     _sig(pool, spec)).selection_hash
    h_k1 = select(SelectorConfig(seed=7, selector=SelectorType.AUCTION_RAW,
                                 k=1), pool, _sig(pool, spec)).selection_hash
    assert h_seed7 != h_seed8     # seed is part of the hash
    assert h_seed7 != h_k1        # k is part of the hash


# --- shared-frozen-pool invariants -----------------------------------------

def test_pool_hash_mismatch_hard_fails():
    pool1 = _pool(["a", "b"])
    pool2 = build_pool(
        "q", [{"task_id": "x", "task_params": dict(_TP),
               "target_achievements": ["eat_cow"]}])
    sig = _sig(pool1, {})
    cfg = SelectorConfig(selector=SelectorType.S0_CANONICAL_BASELINE, k=1, seed=1)
    with pytest.raises(SelectorError) as ei:
        select(cfg, pool2, sig)
    assert ei.value.code == SelectorError.POOL_MISMATCH


def test_signal_pool_coverage_must_be_exact():
    pool = _pool(["a", "b", "c"])
    partial = SelectorSignals(
        pool_hash=pool.pool_hash,
        candidates=[CandidateSignals(candidate_id="a")])
    cfg = SelectorConfig(selector=SelectorType.S0_CANONICAL_BASELINE, k=1, seed=1)
    with pytest.raises(SelectorError) as ei:
        select(cfg, pool, partial)
    assert ei.value.code == SelectorError.SIGNAL_POOL_MISMATCH


# --- S0 content-order floor -------------------------------------------------

def test_s0_is_lexicographic_content_order():
    pool = _pool(["c", "a", "b"])
    cfg = SelectorConfig(selector=SelectorType.S0_CANONICAL_BASELINE, k=2, seed=1)
    res = select(cfg, pool, _sig(pool, {}))
    assert res.selected_ids == ["a", "b"]   # candidate_id ascending, no LLM


# --- Copeland pairwise semantics -------------------------------------------

def test_copeland_pairwise_scores():
    pool = _pool(["a", "b", "c"])
    spec = {"a": {"role_scores": {"tutor": 0.9}},
            "b": {"role_scores": {"tutor": 0.5}},
            "c": {"role_scores": {"tutor": 0.1}}}
    scores = copeland_scores(_sig(pool, spec))
    assert scores == {"a": 2.0, "b": 1.0, "c": 0.0}


def test_copeland_tie_splits_half():
    pool = _pool(["a", "b"])
    spec = {"a": {"role_scores": {"tutor": 0.5}},
            "b": {"role_scores": {"tutor": 0.5}}}
    scores = copeland_scores(_sig(pool, spec))
    assert scores == {"a": 0.5, "b": 0.5}


# --- budgeted cost-cap ------------------------------------------------------

def test_budgeted_admits_greedy_highest_first_under_cap():
    pool = _pool(["a", "b", "c"])
    # a has the top bid but is too expensive -> skipped; b,c admitted
    spec = {"a": {"role_scores": {"tutor": 0.9}, "cost": 5.0},
            "b": {"role_scores": {"tutor": 0.8}, "cost": 1.0},
            "c": {"role_scores": {"tutor": 0.7}, "cost": 1.0}}
    cfg = SelectorConfig(selector=SelectorType.AUCTION_BUDGETED, k=2, seed=1,
                         budget=2.0)
    res = select(cfg, pool, _sig(pool, spec))
    assert res.selected_ids == ["b", "c"]
    assert res.selection_status is SelectionStatus.OK


def test_budget_shortfall_is_insufficient_with_note():
    pool = _pool(["a", "b", "c"])
    spec = {i: {"role_scores": {"tutor": 0.5}, "cost": 1.0}
            for i in ("a", "b", "c")}
    cfg = SelectorConfig(selector=SelectorType.AUCTION_BUDGETED, k=3, seed=1,
                         budget=1.5)   # room for exactly 1
    res = select(cfg, pool, _sig(pool, spec))
    assert res.selection_status is SelectionStatus.INSUFFICIENT_ELIGIBLE_CANDIDATES
    assert len(res.selected_ids) == 1
    assert res.eligible_count == 3          # all eligible; budget (not critic) bound
    assert "budget" in res.shortfall_note
