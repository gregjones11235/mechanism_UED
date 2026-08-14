"""CPU/fake-runtime tests for the fused-preflight benchmark tools."""
from __future__ import annotations

import importlib.util
import ast
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PERF = Path(__file__).parents[4] / "experiments" / "performance"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


BASE_CONFIG = {
    "dicode_manager": {"score_function": "learnability", "max_updates_per_session": 100},
    "training": {
        "num_envs": 1024,
        "num_steps": 128,
        "total_timesteps": 2_000_000_000,
        "conditioning_type": "one_hot",
        "condition_on_task": True,
        "compact_scoring_payload": False,
    },
    "performance": {
        "preflight_reuse_loaded_tasks": False,
        "eval_compile_cache": False,
        "learnability_fused_preflight_summary": False,
        "validation_cache": False,
        "compact_preflight_payload": False,
        "train_compile_cache": False,
        "embedding_cache": False,
    },
    "runtime_profiling": {"enabled": False},
}


def test_b4_single_exact_overlay_diff_and_fixed_flags():
    config = load("perf48_fastpath_config")
    off = config.build_overlay_dict(BASE_CONFIG, comparison="B4_SINGLE", arm="B4_OFF")
    on = config.build_overlay_dict(BASE_CONFIG, comparison="B4_SINGLE", arm="B4_ON")
    gate = config.verify_overlay_pair(off, on, comparison="B4_SINGLE")
    assert gate["diff_paths"] == ["performance.learnability_fused_preflight_summary"]
    for arm in (off, on):
        assert arm["performance"]["preflight_reuse_loaded_tasks"] is True
        assert arm["performance"]["eval_compile_cache"] is True
        assert arm["performance"]["validation_cache"] is False
        assert arm["performance"]["compact_preflight_payload"] is False


def test_final_combo_exact_overlay_diff_and_no_old_caches():
    config = load("perf48_fastpath_config")
    baseline = config.build_overlay_dict(
        BASE_CONFIG, comparison="FINAL_COMBO", arm="BASELINE"
    )
    fast = config.build_overlay_dict(
        BASE_CONFIG, comparison="FINAL_COMBO", arm="FAST_COMBO"
    )
    gate = config.verify_overlay_pair(baseline, fast, comparison="FINAL_COMBO")
    assert set(gate["diff_paths"]) == {
        "performance.preflight_reuse_loaded_tasks",
        "performance.eval_compile_cache",
        "performance.learnability_fused_preflight_summary",
        "performance.validation_cache",
    }
    assert all(baseline["performance"][key] is False for key in config.SWITCHES)
    assert all(fast["performance"][key] is True for key in config.SWITCHES)
    for arm in (baseline, fast):
        assert all(arm["performance"][key] is False for key in config.FORCED_FALSE)


def test_overlay_gate_rejects_extra_or_missing_diff_and_nonlearnability():
    config = load("perf48_fastpath_config")
    off = config.build_overlay_dict(BASE_CONFIG, comparison="B4_SINGLE", arm="B4_OFF")
    on = config.build_overlay_dict(BASE_CONFIG, comparison="B4_SINGLE", arm="B4_ON")
    bad = json.loads(json.dumps(on))
    bad["performance"]["train_compile_cache"] = True
    with pytest.raises(ValueError):
        config.verify_overlay_pair(off, bad, comparison="B4_SINGLE")
    same = json.loads(json.dumps(off))
    with pytest.raises(ValueError):
        config.verify_overlay_pair(off, same, comparison="B4_SINGLE")
    nonlearnability = json.loads(json.dumps(BASE_CONFIG))
    nonlearnability["dicode_manager"]["score_function"] = "pvl"
    with pytest.raises(ValueError):
        config.build_overlay_dict(
            nonlearnability, comparison="B4_SINGLE", arm="B4_ON"
        )


def test_benchmark_config_pair_is_exact_and_manifest_bound(tmp_path):
    import yaml

    config = load("perf48_fastpath_config")
    benchmark = load("perf48_fastpath_benchmark")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    left = tmp_path / "b4_off.yaml"
    right = tmp_path / "b4_on.yaml"
    config.write_overlay_yaml(
        base, comparison="B4_SINGLE", arm="B4_OFF", out_path=left
    )
    config.write_overlay_yaml(
        base, comparison="B4_SINGLE", arm="B4_ON", out_path=right
    )
    manifest = {"source_config": {"config": {
        "left": {"path": str(left), "sha256": benchmark._sha256_file(left)},
        "right": {"path": str(right), "sha256": benchmark._sha256_file(right)},
    }}}
    args = SimpleNamespace(
        comparison="B4_SINGLE", config_left=str(left), config_right=str(right)
    )
    assert benchmark.verify_config_pair(args, manifest)["valid"]
    manifest["source_config"]["config"]["right"]["sha256"] = "bad"
    with pytest.raises(RuntimeError):
        benchmark.verify_config_pair(args, manifest)


class _Tracker:
    enabled = True

    def __init__(self):
        self.phases = []

    @contextmanager
    def span(self, phase):
        self.phases.append(phase)
        yield


def test_harness_fake_runtime_fused_summary_uses_pure_helper_not_legacy_scorer(monkeypatch):
    harness = load("perf48_fastpath_harness")
    monkeypatch.setattr(
        harness,
        "_config_get",
        lambda cfg, key, default=None: (
            cfg.get(key, default) if "." not in key
            else cfg.get(key.split(".")[0], {}).get(key.split(".")[1], default)
        ),
    )
    config = json.loads(json.dumps(BASE_CONFIG))
    config["performance"]["learnability_fused_preflight_summary"] = True
    calls = {"legacy": 0, "contract": 0}

    def convert(finished, successes, count):
        assert count == 2
        return {
            "0": {"sr": 0.5, "priority_score": 0.25},
            "1": {"sr": -1.0, "priority_score": 0.0},
        }

    runtime = {
        "require_learnability_fused_contract": lambda score: calls.__setitem__(
            "contract", calls["contract"] + 1
        ),
        "device_get": lambda value: value,
        "learnability_scores_from_counts": convert,
        "calculate_scores_from_snapshot": lambda *args: calls.__setitem__(
            "legacy", calls["legacy"] + 1
        ),
    }
    tracker = _Tracker()
    scores, mode, payload = harness._score_preflight_result(
        {
            "learnability_summary": {
                "finished_counts": np.array([4, 0], dtype=np.int32),
                "success_counts": np.array([2, 0], dtype=np.int32),
            }
        },
        config,
        2,
        runtime,
        tracker,
    )
    assert mode == "fused" and payload == 16
    assert scores["0"]["sr"] == 0.5
    assert calls == {"legacy": 0, "contract": 1}
    assert tracker.phases == ["scoring_transfer", "scoring_cpu"]


def test_harness_fake_runtime_legacy_uses_original_scoring(monkeypatch):
    harness = load("perf48_fastpath_harness")
    monkeypatch.setattr(
        harness,
        "_config_get",
        lambda cfg, key, default=None: cfg.get(key, default),
    )
    config = {
        "performance": {"learnability_fused_preflight_summary": False},
        "dicode_manager.score_function": "learnability",
    }
    expected = {"0": {"sr": 0.25, "priority_score": 0.1875}}
    calls = []
    runtime = {
        "calculate_scores_from_snapshot": lambda *args: calls.append(args) or expected,
    }
    raw = {
        "scoring_window_data": object(),
        "task_achievement_mask": np.zeros((1, 2), dtype=bool),
        "task_completed_mask": np.zeros((1, 2), dtype=bool),
    }
    scores, mode, payload = harness._score_preflight_result(
        raw, config, 1, runtime, _Tracker()
    )
    assert scores == expected and mode == "legacy" and payload == -1
    assert len(calls) == 1


def _result_doc(benchmark, *, comparison="B4_SINGLE", arm="B4_OFF", wall=100.0,
                preflight=100.0, peak=100.0, free=8000.0):
    semantic = {field: "same" for field in benchmark.SEMANTIC_FIELDS}
    performance = {field: 0.0 for field in benchmark.PERF_FIELDS}
    performance.update({
        "session_wall_s": wall,
        "preflight_wall_s": preflight,
        "gpu_peak_memory_mib": peak,
        "gpu_min_free_mib": free,
    })
    flags = benchmark._cfg.EXPECTED_FLAGS[arm]
    return {
        **semantic,
        **performance,
        "classification": benchmark.CLASSIFICATION,
        "comparison": comparison,
        "manifest_sha256": "manifest",
        "source_commit": "source",
        "gpu_uuid": "GPU-1",
        "stage": "early",
        "repeat": 0,
        "arm": arm,
        "llm_api_calls": 0,
        "checkpoint_loadable": True,
        "compact_scoring_payload": False,
        "validation_cache_enabled": flags["validation_cache"],
        "validation_cache_exercised": False,
        "validation_cache_speedup_claimed": False,
        "preflight_summary_mode": (
            "fused" if flags["learnability_fused_preflight_summary"] else "legacy"
        ),
        "runtime_source_evidence": {"verified": True},
        "profiling": {
            "enabled": True,
            "event_count": 1,
            "events_csv_sha256": "events",
            "critical_path_sha256": "critical",
        },
        "env_evidence": {"jax_version": "0.6.2"},
        "preflight_task_reload_occurred": not flags["preflight_reuse_loaded_tasks"],
        "preflight_task_reload_explicit_absent": flags["preflight_reuse_loaded_tasks"],
        "eval_compile_span_count": 1 if flags["eval_compile_cache"] else 0,
        "eval_cache_hit_count": 1 if flags["eval_compile_cache"] else 0,
        "eval_first_cache_miss": flags["eval_compile_cache"],
        **{marker: False for marker in benchmark.RUNTIME_MARKERS},
    }


def _pair(benchmark, comparison, left_wall, right_wall, left_preflight, right_preflight):
    left_arm, right_arm = benchmark.COMPARISONS[comparison]
    return {
        "stage": "early",
        "repeat": 0,
        "left": _result_doc(
            benchmark, comparison=comparison, arm=left_arm,
            wall=left_wall, preflight=left_preflight,
        ),
        "right": _result_doc(
            benchmark, comparison=comparison, arm=right_arm,
            wall=right_wall, preflight=right_preflight,
        ),
    }


def test_semantic_mismatch_and_runtime_marker_are_rejected():
    benchmark = load("perf48_fastpath_benchmark")
    pair = _pair(benchmark, "B4_SINGLE", 100, 99, 100, 85)
    assert benchmark.compare_semantics(pair["left"], pair["right"]) == "SEMANTIC_PASS"
    pair["right"][benchmark.SEMANTIC_FIELDS[0]] = "different"
    assert benchmark.compare_semantics(pair["left"], pair["right"]) == "REJECTED_SEMANTIC_MISMATCH"
    pair = _pair(benchmark, "B4_SINGLE", 100, 99, 100, 85)
    pair["right"]["oom"] = True
    assert benchmark.compare_semantics(pair["left"], pair["right"]) == "REJECTED_RUNTIME_FAILURE"


def test_b4_and_final_thresholds_and_validation_nonclaim():
    benchmark = load("perf48_fastpath_benchmark")
    b4 = [_pair(benchmark, "B4_SINGLE", 100, 100, 100, 89) for _ in range(6)]
    verdict = benchmark.aggregate_pairs(b4, comparison="B4_SINGLE", required_pairs=6)
    assert verdict["conclusion"] == "B4_PASS"
    assert verdict["validation_cache_exercised"] is False
    b4[0]["right"]["session_wall_s"] = 102
    assert benchmark.aggregate_pairs(
        b4, comparison="B4_SINGLE", required_pairs=6
    )["conclusion"] == "NO_SPEEDUP"

    final = [_pair(benchmark, "FINAL_COMBO", 100, 89, 100, 79) for _ in range(6)]
    assert benchmark.aggregate_pairs(
        final, comparison="FINAL_COMBO", required_pairs=6
    )["conclusion"] == "FINAL_COMBO_PASS"
    insufficient = [_pair(benchmark, "FINAL_COMBO", 100, 95, 100, 85) for _ in range(6)]
    assert benchmark.aggregate_pairs(
        insufficient, comparison="FINAL_COMBO", required_pairs=6
    )["conclusion"] == "NO_SPEEDUP"


def test_mechanism_gate_requires_validation_unexercised():
    benchmark = load("perf48_fastpath_benchmark")
    pair = _pair(benchmark, "FINAL_COMBO", 100, 89, 100, 79)
    assert benchmark.verify_mechanisms(
        pair["left"], pair["right"], comparison="FINAL_COMBO"
    )["ok"]
    pair["right"]["validation_cache_exercised"] = True
    assert not benchmark.verify_mechanisms(
        pair["left"], pair["right"], comparison="FINAL_COMBO"
    )["ok"]


def test_pair_filter_alternation_and_early_stoploss():
    benchmark = load("perf48_fastpath_benchmark")
    assert benchmark.parse_pair_filter() == [
        ("early", 0), ("early", 1), ("mid", 0),
        ("mid", 1), ("late", 0), ("late", 1),
    ]
    assert benchmark.parse_pair_filter(stage="mid") == [("mid", 0), ("mid", 1)]
    assert benchmark.parse_pair_filter(repeat=1) == [
        ("early", 1), ("mid", 1), ("late", 1)
    ]
    assert benchmark.parse_pair_filter(only_pairs="early:0,late:1") == [
        ("early", 0), ("late", 1)
    ]
    with pytest.raises(ValueError):
        benchmark.parse_pair_filter(only_pairs="bad:0")
    slow = _pair(benchmark, "B4_SINGLE", 100, 102, 100, 101)
    assert benchmark.should_stop_after_early(slow, comparison="B4_SINGLE")


def test_result_validation_and_resume_do_not_launch(monkeypatch, tmp_path):
    benchmark = load("perf48_fastpath_benchmark")
    document = _result_doc(benchmark, arm="B4_OFF")
    result_dir = tmp_path / "arm"
    result_dir.mkdir()
    (result_dir / "RESULT.json").write_text(json.dumps(document), encoding="utf-8")
    args = SimpleNamespace(
        manifest_sha256="manifest",
        comparison="B4_SINGLE",
        gpu_uuid="GPU-1",
        source_commit="source",
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must resume")),
    )
    resumed = benchmark.run_fastpath_arm(args, "early", 0, "B4_OFF", result_dir)
    assert resumed["arm"] == "B4_OFF"


def test_fatal_marker_gpu_env_and_deploy_source_binding(tmp_path):
    benchmark = load("perf48_fastpath_benchmark")
    log = tmp_path / "stderr"
    log.write_text("Out of memory", encoding="utf-8")
    assert benchmark.fatal_in([log])
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    args = SimpleNamespace(gpu_uuid="GPU-X", deterministic_xla=True)
    env = benchmark._arm_env(args, str(source), tmp_path / "out")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-X"
    assert env["PYTHONPATH"] == str(source / "src")
    assert env["XLA_FLAGS"].count(benchmark._pair.DETERMINISTIC_FLAG) == 1

    deploy = load("perf48_fastpath_deploy")
    assert "src/dicode/skill_preflight/learnability_summary.py" in deploy.SOURCE_FILES
    assert "experiments/training/run_dicode.py" in deploy.SOURCE_FILES


def test_fastpath_deploy_reuses_frozen_stage_materials_and_binds_new_sources(tmp_path):
    import networkx as nx
    import yaml

    deploy = load("perf48_fastpath_deploy")
    frozen = tmp_path / "frozen"
    source = tmp_path / "source"
    out = tmp_path / "deploy"
    frozen.mkdir()
    stages = []
    for index, name in enumerate(("early", "mid", "late"), 1):
        stage_dir = frozen / "stages" / name
        checkpoint = stage_dir / "checkpoint" / str(index)
        checkpoint.mkdir(parents=True)
        (checkpoint / "_CHECKPOINT_METADATA").write_text("ok", encoding="utf-8")
        graph = nx.DiGraph()
        graph.add_node(
            "task_1", code="class Env:\n    pass\n", description=name,
            status="desc_generated", type="generated",
        )
        nx.write_graphml(graph, stage_dir / "task_graph.graphml")
        np.save(stage_dir / "conditioning.npy", np.zeros((2, 67), dtype=np.float32))
        stages.append({
            "name": name,
            "task_ids": ["task_1"],
            "global_step": index,
            "initial_env_steps": index * 1024 * 128,
            "archive_reconstruction_limit": "all",
        })
    (frozen / "manifest.json").write_text(
        json.dumps({"stages": stages}), encoding="utf-8"
    )
    for relative in deploy.SOURCE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    evidence = deploy.build_deploy(
        frozen_run=frozen, base_config=base, source=source, out=out
    )
    assert evidence["validation_cache_exercised"] is False
    assert set(evidence["overlay_gates"]) == {"B4_SINGLE", "FINAL_COMBO"}
    assert (out / "manifest.json").is_file()
    assert set(Path(path).name for path in evidence["configs"].values()) == {
        "b4_off.yaml", "b4_on.yaml", "baseline.yaml", "fast_combo.yaml"
    }
    loaded = deploy._manifest.load_manifest(out / "manifest.json")
    source_entries = loaded["source_config"]["source"]
    bound_paths = {Path(entry["path"]).resolve() for entry in source_entries.values()}
    assert (source / "src/dicode/skill_preflight/learnability_summary.py").resolve() in bound_paths
    assert (source / "experiments/training/run_dicode.py").resolve() in bound_paths


def test_harness_arm_contract_all_four_arms():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    harness = load("perf48_fastpath_harness")
    config = load("perf48_fastpath_config")
    for comparison, arms in config.COMPARISONS.items():
        for arm in arms:
            overlay = config.build_overlay_dict(
                BASE_CONFIG, comparison=comparison, arm=arm
            )
            cfg = OmegaConf.create(overlay)
            harness._config_contract(cfg)
            harness._arm_contract(cfg, comparison, arm)
    bad = config.build_overlay_dict(
        BASE_CONFIG, comparison="B4_SINGLE", arm="B4_ON"
    )
    bad["dicode_manager"]["score_function"] = "pvl"
    with pytest.raises(RuntimeError):
        harness._config_contract(OmegaConf.create(bad))


def test_preflight_constructs_real_runtime_exactly_once():
    """Prevent duplicate imports/source binding in harness preflight mode."""
    source = (PERF / "perf48_fastpath_harness.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    preflight = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_preflight"
    )
    calls = [
        node for node in ast.walk(preflight)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_real_runtime"
    ]
    assert len(calls) == 1
