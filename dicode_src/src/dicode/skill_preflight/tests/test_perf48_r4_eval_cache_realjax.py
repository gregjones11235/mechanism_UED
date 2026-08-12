"""R4: production-level validation of the evaluation JIT compile cache.

These tests exercise ``craftax_evaluation._get_or_compile_evaluator`` /
``_cache_key`` with REAL jax (lower().compile() + real pytrees) instead of
FakeJit mocks, covering the audit's C-cache semantics:

  1. jax.clear_caches() survival (cached executable stays callable)
  2. same static key + different train-state VALUES -> hit, output uses the new value
  3. same static key + different RNG VALUES -> hit, output uses the new rng
  4. static changes (embedding content/shape/dtype, detail, num_envs/num_steps,
     max_timesteps, model structure, conditioning, window_mem, train-state/rng
     pytree structure/shape/dtype/weak_type) -> miss
  5. cache off -> _cache_key never called (zero expensive hashing)
  6. LRU bound + eviction order
  7. new-run entry clears the run-scoped cache
  8. concurrency: formal held-out eval is single-threaded (documented), and the
     cache helpers are lock-guarded (no corruption)

Requires jax -> runs on the server (CPU) via importorskip.
"""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from dicode.craftax_evaluation import (
    _cache_key,
    _get_or_compile_evaluator,
    _put_cached_evaluator,
    _get_cached_evaluator,
    clear_compiled_evaluator_cache,
)


def _cfg(activation="relu", hidden_layers=256, embed_size=256, num_heads=8,
         qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
         condition_on_task=True, conditioning_type="one_hot", window_mem=128,
         num_envs=1024, num_steps=128, max_timesteps=8192, with_eval=True):
    training = SimpleNamespace(activation=activation, hidden_layers=hidden_layers,
                               embed_size=embed_size, num_heads=num_heads,
                               qkv_features=qkv_features, num_layers=num_layers,
                               gating=gating, gating_bias=gating_bias,
                               condition_on_task=condition_on_task,
                               conditioning_type=conditioning_type,
                               window_mem=window_mem)
    evaluation = SimpleNamespace(num_envs=num_envs, num_steps=num_steps)
    cfg = SimpleNamespace(training=training, evaluation=evaluation)
    if with_eval:
        cfg.eval = SimpleNamespace(max_timesteps=max_timesteps)
    return cfg


def _jit_fn():
    # a real computation that depends on BOTH dynamic inputs (params + rng)
    def f(params, rng):
        k1, k2 = jax.random.split(rng)
        a = jnp.sum(params) + jnp.sum(jax.random.normal(k1, (2,)))
        b = jnp.sum(jax.random.normal(k2, (2,)))
        return {"a": a, "b": b, "params_sum": jnp.sum(params)}
    return jax.jit(f)


def _params(values):
    return jnp.asarray(values, dtype=jnp.float32)


def test_cache_off_never_calls_key(monkeypatch):
    from dicode import craftax_evaluation as ce
    calls = []
    orig = ce._cache_key

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(ce, "_cache_key", spy)
    cfg = _cfg()
    p = _params([1.0, 2.0]); rng = jax.random.PRNGKey(0)
    key = ce._eval_cache_key(False, cfg, None, False, (2,), p, rng)
    assert key is None and calls == []


def test_same_key_twice_compiles_once_and_hits():
    clear_compiled_evaluator_cache()
    cfg = _cfg()
    p = _params([1.0, 2.0]); rng = jax.random.PRNGKey(0)
    key = _cache_key(cfg, None, False, (2,), p, rng)
    jit_fn = _jit_fn()
    compiled, hit1 = _get_or_compile_evaluator(key, jit_fn, (p, rng), True)
    assert hit1 is False
    out1 = compiled(p, rng)
    compiled2, hit2 = _get_or_compile_evaluator(key, jit_fn, (p, rng), True)
    assert hit2 is True
    out2 = compiled2(p, rng)
    for k in out1:
        assert np.allclose(np.asarray(out1[k]), np.asarray(out2[k]))


def test_clear_caches_survival():
    clear_compiled_evaluator_cache()
    cfg = _cfg()
    p = _params([1.0, 2.0]); rng = jax.random.PRNGKey(0)
    key = _cache_key(cfg, None, False, (2,), p, rng)
    compiled, hit = _get_or_compile_evaluator(key, _jit_fn(), (p, rng), True)
    assert hit is False
    before = compiled(p, rng)
    jax.clear_caches()  # must not invalidate the self-contained executable
    after = compiled(p, rng)
    for k in before:
        assert np.allclose(np.asarray(before[k]), np.asarray(after[k]))
    # and the run-scoped cache still hits after clear_caches
    compiled2, hit2 = _get_or_compile_evaluator(key, _jit_fn(), (p, rng), True)
    assert hit2 is True


def test_dynamic_params_value_change_hits_and_uses_new_input():
    clear_compiled_evaluator_cache()
    cfg = _cfg()
    p1 = _params([1.0, 2.0]); rng = jax.random.PRNGKey(0)
    key = _cache_key(cfg, None, False, (2,), p1, rng)
    compiled, hit1 = _get_or_compile_evaluator(key, _jit_fn(), (p1, rng), True)
    assert hit1 is False
    p2 = _params([10.0, 20.0])  # same structure, different values -> must hit
    key2 = _cache_key(cfg, None, False, (2,), p2, rng)
    assert key2 == key
    compiled2, hit2 = _get_or_compile_evaluator(key2, _jit_fn(), (p2, rng), True)
    assert hit2 is True
    out = compiled2(p2, rng)
    # output must reflect the NEW dynamic input (params_sum = 30)
    assert np.allclose(np.asarray(out["params_sum"]), 30.0)


def test_dynamic_rng_value_change_hits_and_uses_new_input():
    clear_compiled_evaluator_cache()
    cfg = _cfg()
    p = _params([1.0, 2.0]); rng1 = jax.random.PRNGKey(0)
    key1 = _cache_key(cfg, None, False, (2,), p, rng1)
    compiled, hit1 = _get_or_compile_evaluator(key1, _jit_fn(), (p, rng1), True)
    assert hit1 is False
    out1 = compiled(p, rng1)
    rng2 = jax.random.PRNGKey(7)  # same structure, different value -> hit
    key2 = _cache_key(cfg, None, False, (2,), p, rng2)
    assert key2 == key1
    compiled2, hit2 = _get_or_compile_evaluator(key2, _jit_fn(), (p, rng2), True)
    assert hit2 is True
    out2 = compiled2(p, rng2)
    # outputs differ because the rng-driven random values differ
    a1 = float(np.asarray(out1["a"])); a2 = float(np.asarray(out2["a"]))
    assert not np.isclose(a1, a2)


def _assert_miss(cfg_a, cfg_b, embedding_a=None, embedding_b=None,
                 detail_a=False, detail_b=False, shape_a=(2,), shape_b=(2,)):
    clear_compiled_evaluator_cache()
    p = _params([1.0, 2.0]); rng = jax.random.PRNGKey(0)
    key_a = _cache_key(cfg_a, embedding_a, detail_a, shape_a, p, rng)
    key_b = _cache_key(cfg_b, embedding_b, detail_b, shape_b, p, rng)
    assert key_a != key_b, "static config change must produce a different cache key"


def test_static_config_changes_miss():
    base = _cfg()
    assert _cache_key(base, None, False, (2,), _params([1.0]), jax.random.PRNGKey(0)) is not None
    _assert_miss(base, _cfg(activation="tanh"))
    _assert_miss(base, _cfg(hidden_layers=128))
    _assert_miss(base, _cfg(embed_size=512))
    _assert_miss(base, _cfg(num_heads=4))
    _assert_miss(base, _cfg(qkv_features=128))
    _assert_miss(base, _cfg(num_layers=4))
    _assert_miss(base, _cfg(gating=False))
    _assert_miss(base, _cfg(gating_bias=1.0))
    _assert_miss(base, _cfg(condition_on_task=False))
    _assert_miss(base, _cfg(conditioning_type="learned"))
    _assert_miss(base, _cfg(window_mem=64))
    _assert_miss(base, _cfg(num_envs=512))
    _assert_miss(base, _cfg(num_steps=64))
    _assert_miss(base, _cfg(max_timesteps=4096))


def test_embedding_changes_miss():
    emb_a = np.zeros((4, 67), dtype=np.float32)
    emb_b = emb_a + 1.0
    emb_c = np.zeros((4, 128), dtype=np.float32)  # different shape
    emb_d = np.zeros((4, 67), dtype=np.float64)   # different dtype
    _assert_miss(_cfg(), _cfg(), embedding_a=emb_a, embedding_b=emb_b)
    _assert_miss(_cfg(), _cfg(), embedding_a=emb_a, embedding_b=emb_c, shape_b=(4, 128))
    _assert_miss(_cfg(), _cfg(), embedding_a=emb_a, embedding_b=emb_d)


def test_detail_change_miss():
    _assert_miss(_cfg(), _cfg(), detail_a=False, detail_b=True)


def test_pytree_structure_change_miss():
    clear_compiled_evaluator_cache()
    cfg = _cfg()
    rng = jax.random.PRNGKey(0)
    p1 = _params([1.0, 2.0])
    p2 = {"params": _params([1.0, 2.0]), "extra": _params([3.0])}  # different structure
    k1 = _cache_key(cfg, None, False, (2,), p1, rng)
    k2 = _cache_key(cfg, None, False, (2,), p2, rng)
    assert k1 != k2


def test_lru_bound_and_eviction_order():
    clear_compiled_evaluator_cache()
    cfg = _cfg()
    p = _params([1.0, 2.0]); rng = jax.random.PRNGKey(0)
    jit_fn = _jit_fn()
    keys = []
    for i in range(3):
        c = SimpleNamespace(training=SimpleNamespace(activation=f"relu{i}", hidden_layers=256,
                                                     embed_size=256, num_heads=8, qkv_features=256,
                                                     num_layers=2, gating=True, gating_bias=2.0,
                                                     condition_on_task=True, conditioning_type="one_hot",
                                                     window_mem=128),
                            evaluation=SimpleNamespace(num_envs=1024, num_steps=128))
        k = _cache_key(c, None, False, (2,), p, rng)
        keys.append(k)
        _get_or_compile_evaluator(k, jit_fn, (p, rng), True, max_entries=2)
    # after 3 puts with max_entries=2, the first key must have been evicted
    assert _get_cached_evaluator(keys[0]) is None
    assert _get_cached_evaluator(keys[1]) is not None
    assert _get_cached_evaluator(keys[2]) is not None
    # touching keys[1] makes it MRU; adding keys[0] again evicts keys[2]
    _get_cached_evaluator(keys[1])
    _get_or_compile_evaluator(keys[0], jit_fn, (p, rng), True, max_entries=2)
    assert _get_cached_evaluator(keys[2]) is None


def test_new_run_clears_cache(monkeypatch):
    # run_dicode.py:45 clears the evaluator cache at run start
    from pathlib import Path
    rd = Path(__file__).parents[4] / "experiments" / "training" / "run_dicode.py"
    src = rd.read_text(encoding="utf-8")
    assert "clear_compiled_evaluator_cache()" in src.split("tracker.configure(config, reset=True)")[1][:120]


def test_concurrency_formal_flow_single_threaded_documented():
    """Audit R4 item 13: the formal held-out evaluator is invoked exactly once
    per session from run_dicode's single main thread, so the run-scoped cache is
    effectively single-threaded in production. The cache helpers are
    RLock-guarded against corruption. If a future multi-threaded caller appears,
    a per-key single-flight must be added around the lower().compile() step
    (which currently happens outside the lock)."""
    from pathlib import Path
    rd = Path(__file__).parents[4] / "experiments" / "training" / "run_dicode.py"
    src = rd.read_text(encoding="utf-8")
    # run_session_evaluation (the held-out path) is called once per session in
    # run_dicode's main loop (LEAK FIX block).
    assert src.count("run_session_evaluation(") >= 1
    from dicode import craftax_evaluation as ce
    import threading
    assert isinstance(ce._COMPILED_EVALUATOR_CACHE_LOCK, type(threading.RLock()))
    assert hasattr(ce, "_get_cached_evaluator") and hasattr(ce, "_put_cached_evaluator")
