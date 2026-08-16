"""CPU-only contract tests for the dual-pipeline research tools."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest


PERF = Path(__file__).parents[4] / "experiments" / "performance"
SOURCE_ROOT = PERF.parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _common(component: str) -> dict:
    benchmark = load("perf48_dual_pipeline_benchmark")
    sources = (
        benchmark.A_RUNTIME_SOURCES
        if component == "A"
        else benchmark.B_RUNTIME_SOURCES
    )
    gpu = {
        "A": "GPU-8df11537-ab79-722d-606f-411966196c4c",
        "B": "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",
    }[component]
    return {
        "classification": "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE",
        "not_semantic_mainline": True,
        "component": component,
        "manifest_sha256": "manifest",
        "source_commit": "source",
        "gpu_uuid": gpu,
        "stage": "early",
        "repeat": 0,
        "llm_api_calls": 0,
        "validation_cache_speedup_included": False,
        "validation_replay_scope": "not_executed_not_timed",
        "task_ids": ["one", "two"],
        "task_assignment_sha256": "assignment",
        "task_code_hashes": ["code-one", "code-two"],
        "input_rng_sha256": "input-rng",
        "frozen_rng": [1, 2],
        "checkpoint_input_path": "/frozen/checkpoint/100",
        "checkpoint_input_sha256": "checkpoint",
        "conditioning_path": "/frozen/conditioning.npy",
        "conditioning_file_sha256": "conditioning-file",
        "embedding_hash": "embedding",
        "conditioning_type": "one_hot",
        "conditioning_shape": [2, 67],
        "conditioning_dtype": "float32",
        "config_evidence": {"path": "/frozen/fast.yaml", "sha256": "config"},
        "compile_cache_dir": f"/isolated/{component}",
        "barrier": {"enabled": False, "mode": "control_direct"},
        "component_started_monotonic_ns": 1_000_000_000,
        "component_ended_monotonic_ns": 501_000_000_000,
        "component_wall_s": 500.0,
        "runtime_source_evidence": {
            "verified": True,
            "paths": {name: f"/source/{relative}" for name, relative in sources.items()},
            "hashes": {name: "source-hash" for name in sources},
            "expected_relatives": dict(sources),
        },
        "env_evidence": {"jax_version": "0.test", "gpu": component},
        "runtime_failure": False,
        "fatal_error": False,
        "oom": False,
        "xid": False,
        "checkpoint_error": False,
        "gpu_violation": False,
    }


def _result(component: str, cache_suffix: str = "control") -> dict:
    harness = load("perf48_dual_pipeline_harness")
    result = _common(component)
    result["compile_cache_dir"] = f"/isolated/{component}/{cache_suffix}"
    if component == "A":
        result.update(
            component_scope="fused_preflight_only",
            candidate_task_load_ids=["one", "two"],
            candidate_task_load_sha256="task-load-order",
            preflight_rng_sha256="preflight-rng",
            params_sha256_before="params-before",
            optimizer_sha256_before="optimizer-before",
            score_projection=[
                {"task_id": "one", "sr": 0.25, "priority_score": 0.1875},
                {"task_id": "two", "sr": 0.75, "priority_score": 0.1875},
            ],
            scoring_fingerprint="scores",
            accepted_ids=["two"],
            rejected_ids=["one"],
            archive_before_sha256="archive-before",
            archive_after_sha256="archive-after",
            preflight_env_steps=40 * 1024 * 128,
            preflight_summary_mode="fused",
            preflight_return_payload_bytes=16,
        )
    else:
        result.update(
            params_sha256_before="params-before",
            params_sha256_after="params-after",
            optimizer_sha256_before="optimizer-before",
            optimizer_sha256_after="optimizer-after",
            checkpoint_reloaded_params_sha256="params-after",
            checkpoint_reloaded_optimizer_sha256="optimizer-after",
            checkpoint_loadable=True,
            checkpoint_output_path=f"/isolated/{cache_suffix}/checkpoint/200",
            train_rng_sha256="train-rng",
            outer_rng_after_sha256="outer-rng",
            input_global_update_step=100,
            global_update_step=200,
            global_env_steps=26_214_400,
            updates=100,
            env_steps=13_107_200,
            scoring_fingerprint="training-scores",
            component_started_monotonic_ns=1_000_000_000,
            component_ended_monotonic_ns=801_000_000_000,
            component_wall_s=800.0,
        )
    return harness._hashed_document(result)


def _strict_manifest(tmp_path: Path):
    manifest_module = load("perf48_combo_manifest")
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.py"
    source.write_text("EVIDENCE = True\n", encoding="utf-8")
    config = tmp_path / "fast.yaml"
    config.write_text("performance:\n  validation_cache: true\n", encoding="utf-8")
    stages = {}
    for index, stage in enumerate(("early", "mid", "late"), 1):
        graph_path = tmp_path / f"{stage}.graphml"
        graph = nx.DiGraph()
        graph.add_node("candidate", code=f"class Env:\n    pass\n# {stage}\n")
        nx.write_graphml(graph, graph_path)
        checkpoint = tmp_path / f"checkpoint_{index}"
        checkpoint.mkdir()
        (checkpoint / "_CHECKPOINT_METADATA").write_text("ok", encoding="utf-8")
        conditioning = tmp_path / f"{stage}.npy"
        np.save(conditioning, np.zeros((2, 67), dtype=np.float32))
        stages[stage] = {
            "graph": str(graph_path),
            "checkpoint": str(checkpoint),
            "task_ids": ["candidate"],
            "global_step": index,
            "initial_env_steps": index * 1024 * 128,
            "archive_reconstruction_limit": "all",
            "conditioning_path": str(conditioning),
        }
    spec = {
        "base_dir": str(tmp_path),
        "budget": {
            "timesteps": 2_000_000_000,
            "num_envs": 1024,
            "num_steps": 128,
            "updates": 100,
        },
        "conditioning_type": "one_hot",
        "source": {"dummy": str(source)},
        "config": {"fast_combo": str(config)},
        "stages": stages,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_module.write_manifest(
        manifest_module.build_combo_manifest(spec), manifest_path
    )
    return manifest_path, config, source


def test_atomic_self_hash_and_tamper(tmp_path):
    harness = load("perf48_dual_pipeline_harness")
    path = tmp_path / "result.json"
    result = harness.atomic_json(
        path, {"classification": harness.CLASSIFICATION, "llm_api_calls": 0}
    )
    assert harness.load_hashed_json(path) == result
    document = json.loads(path.read_text(encoding="utf-8"))
    document["llm_api_calls"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        harness.load_hashed_json(path)


@pytest.mark.parametrize("component", ("A", "B"))
def test_component_result_validation_and_semantic_mismatch(component):
    benchmark = load("perf48_dual_pipeline_benchmark")
    control = _result(component, "control")
    concurrent = _result(component, "concurrent")
    validated = benchmark.validate_component_result(
        control,
        component=component,
        manifest_sha256="manifest",
        source_commit="source",
        gpu_uuid=control["gpu_uuid"],
        stage="early",
        repeat=0,
    )
    assert validated == control
    assert benchmark.compare_component_semantics(control, concurrent, component)["ok"]
    changed = dict(concurrent)
    field = "archive_after_sha256" if component == "A" else "params_sha256_after"
    changed[field] = "changed"
    changed = load("perf48_dual_pipeline_harness")._hashed_document(changed)
    comparison = benchmark.compare_component_semantics(control, changed, component)
    assert comparison["ok"] is False
    assert field in comparison["differences"]


def test_component_a_scope_and_validation_cache_claim_fail_closed():
    benchmark = load("perf48_dual_pipeline_benchmark")
    result = _result("A")
    result["component_scope"] = "worker_main_validation_replay"
    result = load("perf48_dual_pipeline_harness")._hashed_document(result)
    with pytest.raises(RuntimeError, match="scope mismatch"):
        benchmark.validate_component_result(
            result,
            component="A",
            manifest_sha256="manifest",
            source_commit="source",
            gpu_uuid=result["gpu_uuid"],
            stage="early",
            repeat=0,
        )
    assert "validation_cache_enabled" not in result
    assert "validation_cache_exercised" not in result


def test_concurrent_component_requires_valid_barrier_receipt():
    benchmark = load("perf48_dual_pipeline_benchmark")
    harness = load("perf48_dual_pipeline_harness")
    result = _result("A")
    result["barrier"] = {
        "enabled": True,
        "mode": "ready_go",
        "barrier_id": "barrier",
        "ready_sha256": "ready",
        "ready_monotonic_ns": 100,
        "go_sha256": "go",
        "go_monotonic_ns": 200,
        "go_observed_monotonic_ns": 300,
    }
    result = harness._hashed_document(result)
    assert benchmark.validate_component_result(
        result,
        component="A",
        manifest_sha256="manifest",
        source_commit="source",
        gpu_uuid=result["gpu_uuid"],
        stage="early",
        repeat=0,
        expected_barrier=True,
    )
    result["barrier"]["go_sha256"] = ""
    result = harness._hashed_document(result)
    with pytest.raises(RuntimeError, match="receipt incomplete"):
        benchmark.validate_component_result(
            result,
            component="A",
            manifest_sha256="manifest",
            source_commit="source",
            gpu_uuid=result["gpu_uuid"],
            stage="early",
            repeat=0,
            expected_barrier=True,
        )


def test_shared_frozen_inputs_ignore_gpu_and_component_but_not_rng():
    benchmark = load("perf48_dual_pipeline_benchmark")
    a, b = _result("A"), _result("B")
    assert benchmark.compare_shared_inputs(a, b)["ok"] is True
    b["input_rng_sha256"] = "changed"
    assert benchmark.compare_shared_inputs(a, b)["differences"] == {
        "input_rng_sha256": {"A": "input-rng", "B": "changed"}
    }


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"runtime_ok": False}, "REJECTED_RUNTIME_FAILURE"),
        ({"shared_ok": False}, "REJECTED_SHARED_INPUT_MISMATCH"),
        ({"semantic_a_ok": False}, "REJECTED_SEMANTIC_COMPONENT_A"),
        ({"semantic_b_ok": False}, "REJECTED_SEMANTIC_COMPONENT_B"),
        ({"start_skew_s": 2.000001}, "REJECTED_START_SKEW"),
        ({"slowdown_a": 0.100001}, "REJECTED_COMPONENT_A_SLOWDOWN"),
        ({"slowdown_b": 0.100001}, "REJECTED_COMPONENT_B_SLOWDOWN"),
        ({"hidden_wall_s": 399.999}, "NO_HIDDEN_WALL"),
        ({}, "PASS"),
    ],
)
def test_exact_alternate_conclusions(kwargs, expected):
    benchmark = load("perf48_dual_pipeline_benchmark")
    values = {
        "runtime_ok": True,
        "shared_ok": True,
        "semantic_a_ok": True,
        "semantic_b_ok": True,
        "start_skew_s": 2.0,
        "slowdown_a": 0.10,
        "slowdown_b": 0.10,
        "hidden_wall_s": 400.0,
    }
    values.update(kwargs)
    assert benchmark.judge(**values) == expected
    assert expected in benchmark.CONCLUSIONS


@pytest.mark.parametrize("target", ("manifest", "source", "config", "archive"))
def test_strict_frozen_input_tamper_fails_closed(tmp_path, monkeypatch, target):
    harness = load("perf48_dual_pipeline_harness")
    manifest_path, config, source = _strict_manifest(tmp_path / target)
    monkeypatch.setattr(harness._fast, "_load_config", lambda path: {})
    monkeypatch.setattr(harness._fast, "_config_contract", lambda cfg: None)
    monkeypatch.setattr(harness._fast, "_arm_contract", lambda *args: None)
    monkeypatch.setattr(harness._train, "_config_contract", lambda cfg: None)
    monkeypatch.setattr(harness._train, "_arm_values", lambda cfg: (False, "learnability"))
    harness.load_inputs(manifest_path, config)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target == "manifest":
        document["classification"] = "tampered"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
    elif target == "source":
        source.write_text("EVIDENCE = False\n", encoding="utf-8")
    elif target == "config":
        config.write_text("performance:\n  validation_cache: false\n", encoding="utf-8")
    else:
        Path(document["stages"][0]["graph"]["path"]).write_text(
            "tampered", encoding="utf-8"
        )
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        harness.load_inputs(manifest_path, config)


def test_reexported_training_runtime_binds_to_ppo_definition_and_tamper(tmp_path):
    harness = load("perf48_dual_pipeline_harness")
    module_path = tmp_path / "src" / "dicode" / "ppo_tr.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "def run_training_session():\n    return 1\n", encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("dual_runtime_source", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    manifest = {
        "source_config": {
            "source": {
                "src/dicode/ppo_tr.py": {
                    "path": str(module_path.resolve()),
                    "sha256": harness._file_sha256(module_path),
                }
            }
        }
    }
    evidence = harness._bind_sources(
        {"run_training_session": module.run_training_session},
        manifest,
        {"run_training_session": "src/dicode/ppo_tr.py"},
    )
    assert evidence["verified"]
    assert evidence["expected_relatives"]["run_training_session"] == (
        "src/dicode/ppo_tr.py"
    )
    module_path.write_text(
        "def run_training_session():\n    return 2\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source binding mismatch"):
        harness._bind_sources(
            {"run_training_session": module.run_training_session},
            manifest,
            {"run_training_session": "src/dicode/ppo_tr.py"},
        )


def test_real_training_reexport_source_is_ppo_tr():
    from dicode.training import (
        _calculate_task_distribution,
        _create_achievement_masks,
        run_training_session,
    )

    assert Path(inspect.getsourcefile(run_training_session)).resolve() == (
        SOURCE_ROOT / "src/dicode/ppo_tr.py"
    ).resolve()
    assert Path(inspect.getsourcefile(_calculate_task_distribution)).resolve() == (
        SOURCE_ROOT / "src/dicode/training.py"
    ).resolve()
    assert Path(inspect.getsourcefile(_create_achievement_masks)).resolve() == (
        SOURCE_ROOT / "src/dicode/training.py"
    ).resolve()
    harness = load("perf48_dual_pipeline_harness")
    benchmark = load("perf48_dual_pipeline_benchmark")
    deploy = load("perf48_dual_pipeline_deploy")
    harness_source = Path(harness.__file__).read_text(encoding="utf-8")
    assert '"run_training_session": "src/dicode/ppo_tr.py"' in harness_source
    assert benchmark.B_RUNTIME_SOURCES["run_training_session"] == (
        "src/dicode/ppo_tr.py"
    )
    assert {"src/dicode/ppo_tr.py", "src/dicode/training.py"}.issubset(
        deploy.ALL_SOURCE_FILES
    )


def _derived_dual_manifest(tmp_path: Path):
    combo = load("perf48_combo_manifest")
    deploy = load("perf48_dual_pipeline_deploy")
    parent_path, config, _ = _strict_manifest(tmp_path / "frozen")
    source_root = tmp_path / "source"
    entries = {}
    for relative in deploy.ALL_SOURCE_FILES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# bound {relative}\n", encoding="utf-8")
        if relative in deploy.BASE_SOURCE_FILES:
            entries[relative] = {
                "path": str(path.resolve()),
                "sha256": deploy._file_sha256(path),
            }
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["source_config"]["source"] = entries
    combo.write_manifest(parent, parent_path)
    commit = "6f0625dec8cb1fa6bcc7c3ad1912a5e298b279ec"
    out = tmp_path / "dual_deploy"
    evidence = deploy.build_deploy(
        fastpath_manifest=parent_path,
        source=source_root,
        source_commit=commit,
        out=out,
    )
    return out / "manifest.json", config, source_root, parent_path, evidence, commit


def test_dual_deploy_derives_self_hashed_source_complete_manifest(tmp_path):
    harness = load("perf48_dual_pipeline_harness")
    deploy = load("perf48_dual_pipeline_deploy")
    manifest_path, _, source, parent, evidence, commit = _derived_dual_manifest(
        tmp_path
    )
    manifest = deploy._manifest.load_manifest(manifest_path)
    metadata = harness.verify_dual_manifest(manifest, commit)
    assert set(manifest["source_config"]["source"]) == set(deploy.ALL_SOURCE_FILES)
    training = manifest["source_config"]["source"]["src/dicode/training.py"]
    assert Path(training["path"]).resolve() == (
        source / "src/dicode/training.py"
    ).resolve()
    assert training["sha256"] == deploy._file_sha256(training["path"])
    assert metadata["parent_manifest"]["path"] == str(parent.resolve())
    assert evidence["manifest_sha256"] == manifest["manifest_sha256"]
    loaded_evidence = deploy.load_deploy_evidence(
        manifest_path.parent / "dual_pipeline_deploy_evidence.json"
    )
    assert loaded_evidence == evidence


@pytest.mark.parametrize("target", ("manifest", "training", "parent", "evidence"))
def test_dual_deploy_tamper_fails_closed(tmp_path, target):
    harness = load("perf48_dual_pipeline_harness")
    deploy = load("perf48_dual_pipeline_deploy")
    manifest_path, _, source, parent, _, commit = _derived_dual_manifest(tmp_path)
    evidence_path = manifest_path.parent / "dual_pipeline_deploy_evidence.json"
    if target == "manifest":
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        document["dual_pipeline"]["source_commit"] = "0" * 40
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest_sha256"):
            deploy._manifest.load_manifest(manifest_path)
    elif target == "training":
        (source / "src/dicode/training.py").write_text(
            "# tampered\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="source/config"):
            deploy._manifest.load_manifest(manifest_path)
    elif target == "parent":
        parent.write_text(parent.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        manifest = deploy._manifest.load_manifest(manifest_path)
        with pytest.raises(RuntimeError, match="parent manifest binding"):
            harness.verify_dual_manifest(manifest, commit)
    else:
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
        document["llm_api_calls"] = 1
        evidence_path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="evidence hash"):
            deploy.load_deploy_evidence(evidence_path)


def _execution(
    component: str,
    wall: float,
    start: int,
    peak: int = 1000,
    component_wall: float | None = None,
):
    result = _result(component, "execution")
    component_wall = wall if component_wall is None else component_wall
    result["component_started_monotonic_ns"] = start
    result["component_ended_monotonic_ns"] = start + int(component_wall * 1e9)
    result["component_wall_s"] = component_wall
    result = load("perf48_dual_pipeline_harness")._hashed_document(result)
    return {
        "component": component,
        "pid": 100 if component == "A" else 200,
        "argv": [component],
        "started_monotonic_ns": start,
        "ended_monotonic_ns": start + int(wall * 1e9),
        "process_wall_s": wall,
        "returncode": 0,
        "out": f"/{component}",
        "compile_cache_dir": f"/{component}/cache",
        "gpu_peak_memory_mib": peak,
        "gpu_min_free_mib": 12_000,
        "fatal_marker": None,
        "monitor_interval_s": 2.0,
        "result": result,
    }


def test_orchestrator_orders_controls_then_concurrent_and_passes(tmp_path, monkeypatch):
    benchmark = load("perf48_dual_pipeline_benchmark")
    calls = []
    monkeypatch.setattr(
        benchmark,
        "_static_gate",
        lambda args: setattr(args, "manifest_sha256", "manifest") or {"verified": True},
    )

    def launch(args, root, label, components):
        calls.append((label, components))
        if label == "control_A":
            return {"A": _execution("A", 600.0, 0, component_wall=500.0)}
        if label == "control_B":
            return {"B": _execution("B", 900.0, 0, component_wall=800.0)}
        return {
            "A": _execution("A", 700.0, int(1e9), component_wall=520.0),
            "B": _execution("B", 1000.0, int(1.5e9), component_wall=840.0),
        }

    monkeypatch.setattr(benchmark, "_launch_group", launch)
    args = SimpleNamespace(
        root=str(tmp_path / "run"),
        manifest="manifest.json",
        config="fast.yaml",
        harness="harness.py",
        python="python",
        source="source",
        source_commit="source",
        stage="early",
        repeat=0,
        gpu_a_index=2,
        gpu_a_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
        gpu_b_index=3,
        gpu_b_uuid="GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",
        barrier_timeout_s=120.0,
    )
    result = benchmark.run_benchmark(args)
    assert calls == [
        ("control_A", ("A",)),
        ("control_B", ("B",)),
        ("concurrent", ("A", "B")),
    ]
    assert result["conclusion"] == "PASS"
    assert result["timing"]["start_skew_s"] == 0.5
    assert result["timing"]["hidden_wall_s"] == pytest.approx(459.5)
    assert result["timing"]["component_A_slowdown"] == pytest.approx(0.04)
    assert result["timing"]["component_B_slowdown"] == pytest.approx(0.05)
    assert result["timing"]["timing_basis"] == "component_monotonic"
    assert result["timing"]["control_A_component_wall_s"] == 500.0
    assert result["timing"]["concurrent_component_makespan_wall_s"] == 840.5
    assert result["timing"]["operational_process_wall_s"]["concurrent_makespan"] == 1000.5
    assert result["timing"]["operational_process_wall_s"]["control_A"] == 600.0
    assert result["validation_cache_speedup_included"] is False
    assert result["validation_replay_scope"] == "not_executed_not_timed"
    assert result["component_A_scope"] == "fused_preflight_only"
    assert load("perf48_dual_pipeline_harness").load_hashed_json(
        Path(args.root) / "DUAL_PIPELINE_RESULT.json"
    ) == result


def test_independent_compile_cache_and_exact_gpu_command(tmp_path):
    benchmark = load("perf48_dual_pipeline_benchmark")
    source = tmp_path / "source"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        out.mkdir()
        for subdir in ("tmp", "cache", "wandb", "jax_compilation"):
            (out / subdir).mkdir()
    args = SimpleNamespace(
        source=str(source),
        python="python",
        harness="harness.py",
        manifest="manifest.json",
        config="fast.yaml",
        source_commit="source",
        stage="early",
        repeat=0,
        gpu_a_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
        gpu_b_uuid="GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",
    )
    env_a = benchmark._component_env(args, "A", out_a)
    env_b = benchmark._component_env(args, "B", out_b)
    assert env_a["CUDA_VISIBLE_DEVICES"] == args.gpu_a_uuid
    assert env_b["CUDA_VISIBLE_DEVICES"] == args.gpu_b_uuid
    assert env_a["JAX_COMPILATION_CACHE_DIR"] != env_b["JAX_COMPILATION_CACHE_DIR"]
    assert "--xla_gpu_deterministic_ops=true" in env_a["XLA_FLAGS"]
    assert benchmark._command(args, "A", out_a)[-3:] == ["A", "--mode", "run"]


def test_gpu_app_classification_accepts_only_component_descendants(monkeypatch):
    benchmark = load("perf48_dual_pipeline_benchmark")
    uuid = "GPU-8df11537-ab79-722d-606f-411966196c4c"

    def classify(pid, roots):
        assert roots == [100]
        return {
            "pid": pid,
            "ancestry": [pid, 100] if pid == 101 else [pid, 50],
            "classification": "owned_descendant" if pid == 101 else "external",
        }

    monkeypatch.setattr(benchmark._pair, "classify_pid", classify)
    rows, violations = benchmark._classify_component_apps(
        f"101, python, 100 MiB, {uuid}\n202, python, 100 MiB, {uuid}",
        100,
        uuid,
    )
    assert [row["classification"] for row in rows] == [
        "owned_descendant",
        "external",
    ]
    assert violations == [f"202, python, 100 MiB, {uuid}"]


def test_child_ready_go_barrier_success_tamper_and_timeout(tmp_path):
    harness = load("perf48_dual_pipeline_harness")

    def args_for(name, timeout=1.0):
        return SimpleNamespace(
            ready_path=str(tmp_path / f"READY_{name}.json"),
            go_path=str(tmp_path / f"GO_{name}.json"),
            barrier_id=f"barrier-{name}",
            barrier_timeout_s=timeout,
            component="A",
        )

    args = args_for("ok")

    def publish_go():
        while not Path(args.ready_path).is_file():
            time.sleep(0.005)
        harness.atomic_json(
            args.go_path,
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": args.barrier_id,
                "components": ["A", "B"],
                "go_monotonic_ns": time.monotonic_ns(),
                "llm_api_calls": 0,
            },
        )

    thread = threading.Thread(target=publish_go)
    thread.start()
    receipt = harness._wait_for_go(args, {"verified": True})
    thread.join(timeout=1)
    assert receipt["enabled"] is True
    assert receipt["ready_monotonic_ns"] <= receipt["go_observed_monotonic_ns"]

    tampered = args_for("tampered")

    def publish_tampered():
        while not Path(tampered.ready_path).is_file():
            time.sleep(0.005)
        Path(tampered.go_path).write_text(
            json.dumps(
                {
                    "classification": harness.CLASSIFICATION,
                    "barrier_id": tampered.barrier_id,
                    "components": ["A", "B"],
                    "go_monotonic_ns": 1,
                    "llm_api_calls": 0,
                    "result_sha256": "tampered",
                }
            ),
            encoding="utf-8",
        )

    thread = threading.Thread(target=publish_tampered)
    thread.start()
    with pytest.raises(ValueError, match="hash mismatch"):
        harness._wait_for_go(tampered, {"verified": True})
    thread.join(timeout=1)

    timeout = args_for("timeout", timeout=0.01)
    with pytest.raises(TimeoutError, match="GO barrier timeout"):
        harness._wait_for_go(timeout, {"verified": True})


def test_parent_releases_go_only_after_two_valid_ready_receipts(tmp_path, monkeypatch):
    benchmark = load("perf48_dual_pipeline_benchmark")
    harness = load("perf48_dual_pipeline_harness")
    barrier = {
        "barrier_id": "parent-barrier",
        "ready_paths": {
            "A": tmp_path / "READY_A.json",
            "B": tmp_path / "READY_B.json",
        },
        "go_path": tmp_path / "GO.json",
        "timeout_s": 0.5,
    }

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    processes = {"A": Process(101), "B": Process(202)}
    evidence = {}
    monitors = {}
    for component in ("A", "B"):
        out = tmp_path / component
        out.mkdir()
        evidence[component] = {"out": str(out)}
        monitors[component] = (None, None, [])
        harness.atomic_json(
            barrier["ready_paths"][component],
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": barrier["barrier_id"],
                "component": component,
                "pid": processes[component].pid,
                "ready_monotonic_ns": time.monotonic_ns(),
                "runtime_source_evidence_sha256": "source",
                "llm_api_calls": 0,
            },
        )
    monkeypatch.setattr(benchmark._pair, "fatal_in", lambda paths: None)
    monkeypatch.setattr(benchmark._pair, "arm_gpu_metrics", lambda path: (0, 12_000))
    released = benchmark._release_barrier(
        barrier, processes, evidence, monitors
    )
    assert harness.load_hashed_json(barrier["go_path"]) == released["go"]
    assert set(released["ready"]) == {"A", "B"}

    bad_barrier = dict(barrier)
    bad_barrier["go_path"] = tmp_path / "BAD_GO.json"
    bad_barrier["ready_paths"] = dict(barrier["ready_paths"])
    document = json.loads(Path(bad_barrier["ready_paths"]["B"]).read_text())
    document["pid"] = 999
    Path(bad_barrier["ready_paths"]["B"]).write_text(
        json.dumps(document), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark._release_barrier(bad_barrier, processes, evidence, monitors)


def test_parent_child_barrier_link_tamper_fails_closed():
    benchmark = load("perf48_dual_pipeline_benchmark")
    child = {
        "barrier": {
            "barrier_id": "id",
            "ready_sha256": "ready-A",
            "ready_monotonic_ns": 10,
            "go_sha256": "go",
            "go_monotonic_ns": 20,
        }
    }
    parent = {
        "barrier_id": "id",
        "ready": {
            "A": {
                "result_sha256": "ready-A",
                "ready_monotonic_ns": 10,
            }
        },
        "go": {
            "result_sha256": "go",
            "go_monotonic_ns": 20,
            "ready_sha256": {"A": "ready-A"},
        },
    }
    benchmark._verify_barrier_link(child, parent, "A")
    parent["go"]["ready_sha256"]["A"] = "tampered"
    with pytest.raises(RuntimeError, match="parent/child barrier mismatch"):
        benchmark._verify_barrier_link(child, parent, "A")


def _component_run_fixture(tmp_path, component):
    harness = load("perf48_dual_pipeline_harness")
    code = "class Env:\n    pass\n"
    stage = {
        "name": "early",
        "task_ids": ["one"],
        "tasks": [
            {
                "id": "one",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        ],
        "repeats": [{"rng": [7, 8]}],
        "graph": {"path": "/frozen/graph", "sha256": "graph"},
        "checkpoint": {"path": "/frozen/checkpoint/100", "sha256": "checkpoint"},
        "conditioning": {
            "path": "/frozen/conditioning.npy",
            "sha256": "conditioning-file",
            "shape": [2, 67],
            "dtype": "float32",
        },
        "embedding": {"hash": "embedding"},
        "global_step": 100,
        "initial_env_steps": 100 * 1024 * 128,
    }
    manifest = {"manifest_sha256": "manifest", "stages": [stage]}
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "validation": {"rollout_updates": 40, "num_envs": 1024, "num_steps": 128},
            "performance": {"learnability_fused_preflight_summary": True},
            "dicode_manager": {"score_function": "learnability"},
            "gen_manager": {"graph_path": "/unused"},
        }
    )
    args = SimpleNamespace(
        component=component,
        source_commit="source",
        required_gpu_uuid=(
            "GPU-8df11537-ab79-722d-606f-411966196c4c"
            if component == "A"
            else "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"
        ),
        stage="early",
        repeat=0,
        barrier_receipt={"enabled": False, "mode": "control_direct"},
    )

    class Archive:
        def __init__(self):
            self.mutated = False

        def get_task_codes(self, ids):
            return {task_id: code for task_id in ids}

    return harness, manifest, config, stage, args, Archive(), tmp_path / component


def test_run_component_a_executes_fused_score_and_route_signature(tmp_path, monkeypatch):
    harness, manifest, config, stage, args, archive, out = _component_run_fixture(
        tmp_path, "A"
    )
    out.mkdir()
    conditioning = np.zeros((2, 67), dtype=np.float32)
    monkeypatch.setattr(
        harness._fast,
        "_verify_conditioning",
        lambda loaded: (conditioning, "embedding"),
    )
    calls = {"evaluate": 0, "route": 0}
    train_state = SimpleNamespace(params="params", opt_state="optimizer")

    def evaluate(config_arg, rng, state, ids, archive_arg, provider, **kwargs):
        assert config_arg is config and state is train_state and archive_arg is archive
        assert ids == ["one"]
        assert kwargs == {
            "preloaded_task_classes": ["TaskClass"],
            "preloaded_task_ids": ["one"],
        }
        calls["evaluate"] += 1
        return {
            "learnability_summary": {
                "finished_counts": np.array([4], dtype=np.int32),
                "success_counts": np.array([2], dtype=np.int32),
            }
        }

    def preflight_route(scores, ids, kept, archive_arg, route, tracker):
        assert scores == {"0": {"sr": 0.5, "priority_score": 0.25}}
        assert ids == ["one"] and tracker is None and archive_arg is archive
        kept.append("one")
        archive_arg.mutated = True
        calls["route"] += 1

    runtime = {
        "_load_agent_state": lambda cfg, path: train_state,
        "array_rng": lambda value: np.asarray(value, dtype=np.uint32),
        "split_rng": lambda value: (value + 1, value + 2),
        "rng_hash": harness._fingerprint,
        "state_hash": harness._fingerprint,
        "reconstruct_archive": lambda path: archive,
        "archive_hash": lambda value: "after" if value.mutated else "before",
        "archive_get_codes": lambda value, ids: value.get_task_codes(ids),
        "load_tasks_from_env_codes": lambda value, ids: (["TaskClass"], list(ids)),
        "evaluate_new_tasks": evaluate,
        "require_learnability_fused_contract": lambda score: None,
        "device_get": lambda value: value,
        "learnability_scores_from_counts": (
            lambda finished, successes, count: {
                "0": {"sr": 0.5, "priority_score": 0.25}
            }
        ),
        "preflight_route": preflight_route,
        "route": object(),
        "source_evidence": lambda loaded: {"verified": True},
        "env_evidence": lambda: {"jax_version": "fake"},
    }
    result = harness.run_component_a(
        manifest,
        config,
        {"path": "config", "sha256": "config"},
        runtime,
        out,
        args,
    )
    assert calls == {"evaluate": 1, "route": 1}
    assert result["component_scope"] == "fused_preflight_only"
    assert result["candidate_task_load_ids"] == ["one"]
    assert result["accepted_ids"] == ["one"] and result["rejected_ids"] == []
    assert result["preflight_env_steps"] == 40 * 1024 * 128
    assert result["validation_cache_speedup_included"] is False
    assert harness.load_hashed_json(out / "RESULT.json") == result


def test_run_component_b_executes_100_updates_and_checkpoint_reload(tmp_path, monkeypatch):
    harness, manifest, config, stage, args, archive, out = _component_run_fixture(
        tmp_path, "B"
    )
    out.mkdir()
    conditioning = np.zeros((2, 67), dtype=np.float32)
    monkeypatch.setattr(
        harness._fast,
        "_verify_conditioning",
        lambda loaded: (conditioning, "embedding"),
    )
    before = SimpleNamespace(params="params-before", opt_state="optimizer-before")
    after = SimpleNamespace(params="params-after", opt_state="optimizer-after")
    calls = {"training": 0, "save": 0, "reload": 0}
    saved = {}

    class Manager:
        def save(self, step, state):
            assert step == 200 and state is after
            saved["state"] = state
            calls["save"] += 1

        def wait_until_finished(self):
            assert saved["state"] is after

        def close(self):
            pass

    class Ocp:
        class PyTreeCheckpointer:
            pass

        class CheckpointManagerOptions:
            def __init__(self, **kwargs):
                assert kwargs == {"create": True, "max_to_keep": 1}

        @staticmethod
        def CheckpointManager(path, checkpointer, options):
            return Manager()

    class Wandb:
        log = None

        @staticmethod
        def init(**kwargs):
            assert kwargs["mode"] == "offline"

        @staticmethod
        def finish():
            pass

    def load_state(cfg, path):
        if path == stage["checkpoint"]["path"]:
            return before
        calls["reload"] += 1
        assert path.endswith("checkpoint\\200") or path.endswith("checkpoint/200")
        return saved["state"]

    def train(config_arg, rng, task_classes, **kwargs):
        assert config_arg is config and task_classes == ["TaskClass", "OriginalTask"]
        assert kwargs["num_training_updates"] == 100
        assert kwargs["train_state"] is before
        assert kwargs["global_update_step"] == 100
        assert kwargs["task_embeddings"].shape == (2, 67)
        calls["training"] += 1
        return {
            "train_state": after,
            "metrics": {
                "num_updates_done": 100,
                "num_env_steps_done": 13_107_200,
                "scoring_window_data": np.array([1], dtype=np.int32),
            },
        }

    fake_jax = SimpleNamespace(
        tree_util=SimpleNamespace(tree_leaves=lambda value: []),
        block_until_ready=lambda value: value,
        device_get=lambda value: value,
    )
    runtime = {
        "TaskArchive": lambda cfg: archive,
        "archive_get_codes": lambda value, ids: value.get_task_codes(ids),
        "load_tasks_from_env_codes": lambda value, ids: (["TaskClass"], list(ids)),
        "OriginalTask": "OriginalTask",
        "_calculate_task_distribution": lambda cfg, count: np.array([0.5, 0.5]),
        "_load_agent_state": load_state,
        "array_rng": lambda value: np.asarray(value, dtype=np.uint32),
        "split_rng": lambda value: (value + 1, value + 2),
        "rng_hash": harness._fingerprint,
        "state_hash": harness._fingerprint,
        "jnp": SimpleNamespace(asarray=np.asarray),
        "jax": fake_jax,
        "wandb": Wandb,
        "run_training_session": train,
        "_create_achievement_masks": lambda classes: (
            np.ones((2, 1), dtype=bool),
            np.ones((2, 1), dtype=bool),
        ),
        "calculate_scores_from_snapshot": (
            lambda payload, count, mask, completed, cfg, forced: {"0": {"sr": 1.0}}
        ),
        "ocp": Ocp,
        "source_evidence": lambda loaded: {"verified": True},
        "env_evidence": lambda: {"jax_version": "fake"},
    }
    result = harness.run_component_b(
        manifest,
        config,
        {"path": "config", "sha256": "config"},
        runtime,
        out,
        args,
    )
    assert calls == {"training": 1, "save": 1, "reload": 1}
    assert result["updates"] == 100 and result["env_steps"] == 13_107_200
    assert result["global_update_step"] == 200
    assert result["checkpoint_loadable"] is True
    assert result["checkpoint_reloaded_params_sha256"] == result["params_sha256_after"]
    assert harness.load_hashed_json(out / "RESULT.json") == result


def test_sources_have_no_eager_jax_import_and_lock_research_contract():
    for name in ("perf48_dual_pipeline_harness", "perf48_dual_pipeline_benchmark"):
        source = (PERF / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "jax" not in imported
    harness_source = (PERF / "perf48_dual_pipeline_harness.py").read_text()
    benchmark_source = (PERF / "perf48_dual_pipeline_benchmark.py").read_text()
    assert "num_training_updates=100" in harness_source
    assert 'summary_mode != "fused"' in harness_source
    assert "monitor_interval_s=2.0" in benchmark_source
    assert "_strict_monitor_popen" in benchmark_source
    assert "stop.wait(2.0)" in benchmark_source
    assert "_pair.stop_owned" in benchmark_source
    assert "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE" in harness_source
