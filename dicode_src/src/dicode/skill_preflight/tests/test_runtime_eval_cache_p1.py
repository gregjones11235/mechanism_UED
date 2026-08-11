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
