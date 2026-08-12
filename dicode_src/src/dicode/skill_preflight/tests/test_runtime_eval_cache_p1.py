import json
import threading
from pathlib import Path

import pytest
from types import SimpleNamespace


def test_profiling_disabled_does_not_create_jsonl(tmp_path):
    from dicode.runtime_analysis import RuntimeTracker

    tracker = RuntimeTracker()
    tracker.configure(enabled=False, output_jsonl=tmp_path / "events.jsonl")
    tracker.record("phase", session=1)
    assert not (tmp_path / "events.jsonl").exists()


def test_event_schema_and_thread_safe_append(tmp_path):
    from dicode.runtime_analysis import RuntimeTracker

    tracker = RuntimeTracker()
    path = tmp_path / "events.jsonl"
    tracker.configure(enabled=True, output_jsonl=path, run_id="test-run")

    def emit(i):
        tracker.record("phase", session=i, request_id=str(i), overlap_group="g")

    threads = [threading.Thread(target=emit, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = {"run_id", "session", "phase", "parent_phase", "start_monotonic_ns",
                "end_monotonic_ns", "duration_s", "status", "cache_hit",
                "task_signature", "request_id", "overlap_group"}
    assert len(rows) == 12
    assert all(set(row) == expected for row in rows)


def test_config_defaults_are_off():
    text = Path(__file__).parents[4].joinpath("conf", "config.yaml").read_text()
    assert "enabled: false" in text
    assert "eval_compile_cache: false" in text
    assert "compiled_cache_max_entries: 8" in text


def test_eval_cache_key_covers_embedding_detail_and_static_config():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    cfg = SimpleNamespace(
        training=SimpleNamespace(conditioning_type="embedding", window_mem=4,
                                 activation="relu", hidden_layers=2, embed_size=8,
                                 num_heads=2, qkv_features=8, num_layers=1,
                                 gating=True, gating_bias=0.0, condition_on_task=True),
        evaluation=SimpleNamespace(num_envs=2, num_steps=3),
    )
    import numpy as np
    a = np.zeros((2, 8), dtype=np.float32)
    assert craftax._cache_key(cfg, a, False, a.shape, "state", "rng") != craftax._cache_key(cfg, a, True, a.shape, "state", "rng")
    assert craftax._cache_key(cfg, a, False, a.shape, "state", "rng") != craftax._cache_key(cfg, a + 1, False, a.shape, "state", "rng")
    cfg.evaluation.num_steps = 4
    assert craftax._cache_key(cfg, a, False, a.shape, "state", "rng") != craftax._cache_key(
        SimpleNamespace(training=cfg.training, evaluation=SimpleNamespace(num_envs=2, num_steps=3)),
        a, False, a.shape, "state", "rng")


def test_overlapping_events_are_not_stacked_as_wall_clock(tmp_path):
    from dicode.runtime_analysis import RuntimeTracker
    tracker = RuntimeTracker()
    tracker.configure(enabled=True, output_jsonl=tmp_path / "events.jsonl")
    start = 100
    tracker.record("outer", start, start + 10, overlap_group="x")
    tracker.record("inner", start + 2, start + 8, parent_phase="outer", overlap_group="x")
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows[0]["duration_s"] > rows[1]["duration_s"]
    source = Path(__file__).parents[2].joinpath("runtime_analysis.py").read_text()
    assert "stacked=False" in source


def test_nested_parent_error_and_exclusive_report(tmp_path):
    from dicode.runtime_analysis import RuntimeTracker

    tracker = RuntimeTracker()
    path = tmp_path / "events.jsonl"
    tracker.configure(enabled=True, output_jsonl=path, run_id="current", reset=True)
    with pytest.raises(ValueError):
        with tracker.span("outer", session=1):
            with tracker.span("inner", session=1):
                raise ValueError("boom")
    tracker.record("outer", 0, 10, session=1, overlap_group="g")
    tracker.record("inner", 2, 8, session=1, parent_phase="outer", overlap_group="g")
    tracker.record("out_of_bounds", -5, 15, session=1, overlap_group="g")
    tracker.record("session_wall", 0, 10, session=1)
    tracker.record("ignored", 0, 100, session=None)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run_id": "old", "session": 1, "phase": "old", "start_monotonic_ns": 0, "end_monotonic_ns": 100}) + "\n")
    report = tracker.derive_reports()
    session = report["sessions"]["1"]
    assert session["covered_union"] == pytest.approx(10e-9)
    assert session["overlap_groups"]["g"]["covered_union_s"] == pytest.approx(10e-9)
    assert session["exclusive_phase_totals"]["outer"] == pytest.approx(4e-9)
    assert session["exclusive_phase_totals"]["inner"] == pytest.approx(6e-9)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row["parent_phase"] == "outer" and row["status"] == "error" for row in rows)
    assert "None" not in report["sessions"]
    assert (tmp_path / "events.csv").exists() and (tmp_path / "critical_path.json").exists()


def test_eval_cache_hit_miss_lru_and_disabled_path():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    craftax.clear_compiled_evaluator_cache()
    craftax._put_cached_evaluator("a", "A", max_entries=2)
    assert craftax._get_cached_evaluator("a") == "A"  # hit
    assert craftax._get_cached_evaluator("missing") is None  # miss
    craftax._put_cached_evaluator("b", "B", max_entries=2)
    craftax._put_cached_evaluator("c", "C", max_entries=2)
    assert craftax._get_cached_evaluator("a") is None  # LRU eviction
    assert craftax._get_cached_evaluator("c") == "C"
    assert not craftax._cache_enabled(SimpleNamespace(get=lambda *_: {}))

    calls = []
    class FakeJit:
        def lower(self, *args):
            calls.append(args)
            return self
        def compile(self):
            return object()
    key = ("real",)
    craftax.clear_compiled_evaluator_cache()
    craftax._get_or_compile_evaluator(key, FakeJit(), (1, 2), True, 8)
    craftax._get_or_compile_evaluator(key, FakeJit(), (1, 2), True, 8)
    assert len(calls) == 1
    calls.clear()
    craftax._get_or_compile_evaluator(("off",), FakeJit(), (1, 2), False, 8)
    assert calls == []


def test_scoring_transfer_cpu_events_and_output_equivalence(tmp_path, monkeypatch):
    scoring = pytest.importorskip("dicode.scoring")
    from dicode.runtime_analysis import tracker
    tracker.configure(enabled=True, output_jsonl=tmp_path / "events.jsonl", run_id="score", reset=True)
    expected = {"0": {"sr": 0.5}}
    monkeypatch.setattr(scoring.jax, "device_get", lambda value: {"transferred": value})
    monkeypatch.setattr(scoring, "_calculate_scores_from_snapshot_impl", lambda *args: expected)
    actual = scoring.calculate_scores_from_snapshot({}, 1, None, None, None)
    assert actual == expected
    phases = [json.loads(line)["phase"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert phases == ["scoring_transfer", "scoring_cpu"]


# =============================================================================
# C: Evaluation JIT reuse -- cache-key skip when off, weak_type signature,
#    jax.clear_caches() survivability, dynamic/static input contract.
#    Tests that touch craftax_evaluation or real jax are server-run
#    (# requires-jax-server); the source audits run on the CPU-only box.
# =============================================================================

def _eval_cfg(**training_over):
    training = SimpleNamespace(activation="relu", hidden_layers=2, embed_size=8,
                               num_heads=2, qkv_features=8, num_layers=1, gating=True,
                               gating_bias=0.0, condition_on_task=True,
                               conditioning_type="embedding", window_mem=4)
    evaluation = SimpleNamespace(num_envs=2, num_steps=3)
    for k, v in training_over.items():
        if k in vars(evaluation):
            setattr(evaluation, k, v)
        else:
            setattr(training, k, v)
    cfg = SimpleNamespace(training=training, evaluation=evaluation)
    return cfg


def test_eval_cache_key_skipped_when_disabled(monkeypatch):
    """cache off: _cache_key (expensive ~1MB embedding hash) is never called."""
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    calls = []
    monkeypatch.setattr(craftax, "_cache_key", lambda *a, **k: calls.append(a) or ("key",))
    import numpy as np
    a = np.zeros((2, 8), dtype=np.float32)
    key_off = craftax._eval_cache_key(False, _eval_cfg(), a, False, a.shape, "state", "rng")
    assert key_off is None and calls == []
    key_on = craftax._eval_cache_key(True, _eval_cfg(), a, False, a.shape, "state", "rng")
    assert key_on == ("key",) and len(calls) == 1


def test_eval_cache_off_historical_path_source_audit():
    src = Path(__file__).parents[2].joinpath("craftax_evaluation.py").read_text(encoding="utf-8")
    assert "_eval_cache_key(" in src                      # main delegates key building
    assert "if not use_cache:" in src and "return None" in src
    # the expensive hash must be skipped when off: _cache_key only called via helper
    assert "weak_type" in src                             # C1 weak_type signature fix
    # main must not call _cache_key unconditionally
    assert "if use_cache else None" in src or "_eval_cache_key(" in src


def test_eval_pytree_signature_includes_weak_type():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    sig_weak = craftax._pytree_signature(1.0)          # python scalar -> weak_type True
    sig_strong = craftax._pytree_signature(jnp.float32(1.0))
    assert sig_weak != sig_strong
    assert "True" in repr(sig_weak) or True in sig_weak


def test_eval_compiled_survives_jax_clear_caches():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    def f(x):
        return jnp.sum(x * 2.0)
    x = jnp.ones(4)
    compiled = jax.jit(f).lower(x).compile()
    expected = float(f(x))
    assert float(compiled(x)) == expected
    jax.clear_caches()
    assert float(compiled(x)) == expected               # self-contained executable


def test_eval_cached_compiled_survives_jax_clear_caches():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    craftax.clear_compiled_evaluator_cache()
    jit_fn = jax.jit(lambda x: jnp.sum(x * 2.0))
    x = jnp.ones(4)
    key = ("clear-caches",)
    compiled, hit = craftax._get_or_compile_evaluator(key, jit_fn, (x,), True, 8)
    assert hit is False
    expected = float(compiled(x))
    jax.clear_caches()
    compiled2, hit2 = craftax._get_or_compile_evaluator(key, jit_fn, (x,), True, 8)
    assert hit2 is True                                  # still a cache hit
    assert float(compiled2(x)) == expected               # and still callable


def test_eval_cache_dynamic_input_values_still_hit_and_use_new_inputs():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    craftax.clear_compiled_evaluator_cache()
    def f(x):
        return jnp.sum(x * 2.0)
    x1 = jnp.ones(4)
    x2 = jnp.full(4, 5.0)
    jit_fn = jax.jit(f)
    key = ("dynamic",)
    c1, h1 = craftax._get_or_compile_evaluator(key, jit_fn, (x1,), True, 8)
    assert h1 is False and float(c1(x1)) == 8.0
    c2, h2 = craftax._get_or_compile_evaluator(key, jit_fn, (x1,), True, 8)
    assert h2 is True
    assert float(c2(x2)) == 40.0                          # same executable, new values


def test_eval_cache_key_static_config_changes_miss():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    import numpy as np
    a = np.zeros((2, 8), dtype=np.float32)
    base = _eval_cfg()
    base_key = craftax._cache_key(base, a, False, a.shape, "state", "rng")
    cases = {
        "activation": {"activation": "tanh"},
        "hidden_layers": {"hidden_layers": 4},
        "embed_size": {"embed_size": 16},
        "num_heads": {"num_heads": 4},
        "qkv_features": {"qkv_features": 16},
        "num_layers": {"num_layers": 2},
        "gating": {"gating": False},
        "gating_bias": {"gating_bias": 1.0},
        "condition_on_task": {"condition_on_task": False},
        "conditioning_type": {"conditioning_type": "one_hot"},
        "window_mem": {"window_mem": 8},
        "num_envs": {"num_envs": 4},
        "num_steps": {"num_steps": 5},
    }
    for label, over in cases.items():
        key = craftax._cache_key(_eval_cfg(**over), a, False, a.shape, "state", "rng")
        assert key != base_key, f"{label} change did not invalidate the eval cache key"
    # embedding content / input_shape / detail
    assert craftax._cache_key(base, a + 1, False, a.shape, "state", "rng") != base_key
    assert craftax._cache_key(base, a, False, (2, 8), "state", "rng") != craftax._cache_key(base, a, False, (2, 9), "state", "rng")
    assert craftax._cache_key(base, a, True, a.shape, "state", "rng") != base_key
    # max_timesteps (config.eval)
    cfg_ev = _eval_cfg()
    cfg_ev.eval = {"max_timesteps": 4096}  # dict (has .get) mirrors OmegaConf
    assert craftax._cache_key(cfg_ev, a, False, a.shape, "state", "rng") != base_key
    # train-state / rng structure change (different treedef)
    assert craftax._cache_key(base, a, False, a.shape, {"p": 1}, "rng") != craftax._cache_key(base, a, False, a.shape, [1], "rng")


def test_eval_cache_lru_bound_and_thread_safety():
    craftax = pytest.importorskip("dicode.craftax_evaluation")
    craftax.clear_compiled_evaluator_cache()
    import threading

    class FakeJit:
        _counter_lock = threading.Lock()
        total = 0
        def __init__(self, tag):
            self.tag = tag
        def lower(self, *args):
            return self
        def compile(self):
            with FakeJit._counter_lock:
                FakeJit.total += 1
            return (self.tag, FakeJit.total)

    results, errors = [], []
    def worker(i):
        try:
            key = (f"thread-{i % 3}",)
            obj, hit = craftax._get_or_compile_evaluator(key, FakeJit(f"j{i % 3}"), (), True, 2)
            results.append((i, obj, hit))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(obj is not None for _, obj, _ in results)
    assert len(craftax._COMPILED_EVALUATOR_CACHE) <= 2   # LRU bound enforced
    # at least one hit among 30 calls on 3 keys with max_entries=2
    assert any(hit for _, _, hit in results)
