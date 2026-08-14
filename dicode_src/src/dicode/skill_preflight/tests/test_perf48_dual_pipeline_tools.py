"""CPU-only contract tests for the dual-pipeline research tools."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
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
            candidate_validation_ids=["one", "two"],
            candidate_validation_sha256="validation-order",
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
            validation_cache_enabled=True,
            validation_cache_exercised=False,
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


def test_training_runtime_source_binding_detects_post_manifest_tamper(tmp_path):
    harness = load("perf48_dual_pipeline_harness")
    module_path = tmp_path / "src" / "dicode" / "training.py"
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
                "src/dicode/training.py": {
                    "path": str(module_path.resolve()),
                    "sha256": harness._file_sha256(module_path),
                }
            }
        }
    }
    evidence = harness._bind_sources(
        {"run_training_session": module.run_training_session},
        manifest,
        {"run_training_session": "src/dicode/training.py"},
    )
    assert evidence["verified"]
    assert evidence["expected_relatives"]["run_training_session"] == (
        "src/dicode/training.py"
    )
    module_path.write_text(
        "def run_training_session():\n    return 2\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source binding mismatch"):
        harness._bind_sources(
            {"run_training_session": module.run_training_session},
            manifest,
            {"run_training_session": "src/dicode/training.py"},
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


def _execution(component: str, wall: float, start: int, peak: int = 1000):
    result = _result(component, "execution")
    result["component_started_monotonic_ns"] = start
    result["component_ended_monotonic_ns"] = start + int(wall * 1e9)
    result["component_wall_s"] = wall
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
            return {"A": _execution("A", 500.0, 0)}
        if label == "control_B":
            return {"B": _execution("B", 800.0, 0)}
        return {
            "A": _execution("A", 520.0, int(1e9)),
            "B": _execution("B", 840.0, int(1.5e9)),
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
