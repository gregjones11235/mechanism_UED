"""GATE 2 (Phase 2.5) — judgment-cache read + deterministic bit-identical replay.

The same (pool, judgments, selector config, seed) reproduces a bit-identical
selection_hash regardless of judgment insertion order or signal order, and the
cache_hash is order-insensitive. Changing the seed changes the hash (seed binding).
"""
import pytest

from d052.counterfactual.judgment_cache import JudgmentCache
from d052.counterfactual.pipeline import (
    build_phase25_judgments,
    build_phase25_pool,
)
from d052.counterfactual.prompts import role_judgment_prompt_hash
from d052.roles.protocol import RoleName
from d052.schemas.roles import ScoringRole
from d052.schemas.selector import SelectorConfig, SelectorType
from d052.selectors import select

_ROLE_NAMES = [RoleName.TUTOR, RoleName.CRITIC, RoleName.EXPLORER]
_JPH = role_judgment_prompt_hash(_ROLE_NAMES)


def _config(seed=7):
    return SelectorConfig(selector=SelectorType.S1_THREE_ROLE, k=8, seed=seed,
                          roles=[ScoringRole.TUTOR, ScoringRole.CRITIC,
                                 ScoringRole.EXPLORER])


def _cache(pool, judgments):
    c = JudgmentCache(pool.pool_hash, _JPH)
    c.put_many(judgments)
    return c


def test_cache_hash_is_insertion_order_insensitive():
    pool = build_phase25_pool()
    js = build_phase25_judgments(pool)
    forward = _cache(pool, js)
    reversed_cache = _cache(pool, list(reversed(js)))
    assert forward.cache_hash() == reversed_cache.cache_hash()


def test_bit_identical_selection_hash_under_judgment_order_shuffle():
    pool = build_phase25_pool()
    js = build_phase25_judgments(pool)
    cfg = _config(seed=7)
    r1 = select(cfg, pool, _cache(pool, js).build_signals(pool, cfg))
    r2 = select(cfg, pool, _cache(pool, list(reversed(js))).build_signals(pool, cfg))
    assert r1.model_dump() == r2.model_dump()      # bit-for-bit identical
    assert r1.selection_hash == r2.selection_hash


def test_bit_identical_selection_hash_under_signal_order_shuffle():
    pool = build_phase25_pool()
    js = build_phase25_judgments(pool)
    cfg = _config(seed=7)
    sig = _cache(pool, js).build_signals(pool, cfg)
    from d052.selectors.base import SelectorSignals
    shuffled = SelectorSignals(pool_hash=pool.pool_hash,
                               candidates=list(reversed(sig.candidates)))
    r1 = select(cfg, pool, sig)
    r2 = select(cfg, pool, shuffled)
    assert r1.selection_hash == r2.selection_hash
    assert r1.selected_ids == r2.selected_ids


def test_selection_hash_binds_seed():
    pool = build_phase25_pool()
    js = build_phase25_judgments(pool)
    cache = _cache(pool, js)
    h7 = select(_config(seed=7), pool,
                cache.build_signals(pool, _config(seed=7))).selection_hash
    h8 = select(_config(seed=8), pool,
                cache.build_signals(pool, _config(seed=8))).selection_hash
    assert h7 != h8


def test_replay_is_exact_three_times():
    pool = build_phase25_pool()
    js = build_phase25_judgments(pool)
    cfg = _config(seed=7)
    hashes = set()
    for _ in range(3):
        cache = _cache(pool, js)             # rebuilt from scratch each time
        hashes.add(select(cfg, pool, cache.build_signals(pool, cfg)).selection_hash)
    assert len(hashes) == 1                  # one unique selection_hash across replays
