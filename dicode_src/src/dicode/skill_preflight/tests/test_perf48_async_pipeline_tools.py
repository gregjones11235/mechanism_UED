from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf


PERF = Path(__file__).parents[4] / "experiments" / "performance"
SOURCE_ROOT = PERF.parents[1]


def load(name: str):
    path = PERF / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _args(stage="early", repeat=0):
    return SimpleNamespace(
        manifest_sha256="manifest",
        source_commit="f" * 40,
        stage=stage,
        repeat=repeat,
        gpu2_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
        gpu3_uuid="GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",
        source=str(SOURCE_ROOT),
    )


def _projection():
    return [
        {
            "task_index": 0,
            "task_id": "one",
            "sr": 0.5,
            "priority_score": 0.25,
        }
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _async_source_evidence():
    benchmark = load("perf48_async_pipeline_benchmark")
    return {
        name: {
            "path": str((SOURCE_ROOT / relative).resolve()),
            "sha256": _sha(SOURCE_ROOT / relative),
            "relative": relative,
        }
        for name, relative in benchmark.ASYNC_RUNTIME_SOURCES.items()
    }


def _reference_source_evidence():
    benchmark = load("perf48_async_pipeline_benchmark")
    path = (SOURCE_ROOT / "src/dicode/skill_preflight/async_preflight.py").resolve()
    return {
        "verified": True,
        "paths": {name: str(path) for name in benchmark.REFERENCE_RUNTIME_SOURCES},
        "hashes": {name: _sha(path) for name in benchmark.REFERENCE_RUNTIME_SOURCES},
    }


def _async_result(**changes):
    harness = load("perf48_async_pipeline_harness")
    projection = _projection()
    result = {
        "classification": harness.CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": "ASYNC",
        "manifest_sha256": "manifest",
        "source_commit": "f" * 40,
        "stage": "early",
        "repeat": 0,
        "gpu_uuid": "GPU-8df11537-ab79-722d-606f-411966196c4c",
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "validation_replay_reference": "6f0625d_external_not_timed",
        "task_ids": ["one"],
        "task_code_rows": [{"task_id": "one", "code_sha256": "code"}],
        "input_rng_sha256": "input",
        "main_rng_after_split_sha256": "main",
        "preflight_rng_sha256": "pf",
        "checkpoint_input_path": "/checkpoint/1",
        "checkpoint_input_sha256": "checkpoint",
        "graph_input_path": "/graph",
        "graph_input_sha256": "graph",
        "config_evidence": {"path": "/config", "sha256": "config"},
        "session_N": {
            "fresh_ids": ["one"],
            "training_new_ids": [],
            "launch_ids": ["one"],
        },
        "session_N1": {
            "fresh_ids": [],
            "delayed_kept_ids": ["one"],
            "training_new_ids": ["one"],
            "launch_ids": [],
        },
        "double_apply_rejected": True,
        "route_calls": 1,
        "barrier": {"enabled": False, "mode": "control_direct"},
        "score_projection": projection,
        "score_fingerprint": harness.fingerprint(projection),
        "kept_ids": ["one"],
        "rejected_ids": [],
        "archive_before_sha256": "before",
        "archive_after_sha256": "after",
        "worker_gpu_preflight": {
            "uuid": "GPU-8df11537-ab79-722d-606f-411966196c4c",
            "index": 2,
            "free_mib": 45000,
            "external": [],
        },
        "worker_jax_backend": "gpu",
        "worker_jax_device_count": 1,
        "controller_backend": "cpu",
        "worker_route_calls": 0,
        "main_route_calls": 1,
        "receipt_sha256": {name: name.lower() for name in ("JOB", "RUNNING", "RESULT", "APPLYING", "APPLIED")},
        "runtime_source_evidence": _async_source_evidence(),
        "component_started_monotonic_ns": 1,
        "worker_launched_monotonic_ns": 2,
        "component_ended_monotonic_ns": 10,
        "component_wall_s": 9e-9,
        **{marker: False for marker in harness.RUNTIME_MARKERS},
    }
    result.update(changes)
    return harness._dual._hashed_document(result)


def _reference_result(**changes):
    harness = load("perf48_async_pipeline_harness")
    projection = _projection()
    result = {
        "classification": harness.CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": "REFERENCE",
        "manifest_sha256": "manifest",
        "source_commit": "f" * 40,
        "stage": "early",
        "repeat": 0,
        "gpu_uuid": "GPU-8df11537-ab79-722d-606f-411966196c4c",
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "task_ids": ["one"],
        "task_code_rows": [{"task_id": "one", "code_sha256": "code"}],
        "input_rng_sha256": "input",
        "preflight_rng_sha256": "pf",
        "checkpoint_input_path": "/checkpoint/1",
        "checkpoint_input_sha256": "checkpoint",
        "graph_input_path": "/graph",
        "graph_input_sha256": "graph",
        "config_evidence": {"path": "/config", "sha256": "config"},
        "score_projection": projection,
        "score_fingerprint": harness.fingerprint(projection),
        "kept_ids": ["one"],
        "rejected_ids": [],
        "archive_before_sha256": "before",
        "archive_after_sha256": "after",
        "summary_mode": "fused",
        "return_payload_bytes": 8,
        "route_calls": 1,
        "barrier": {"enabled": False, "mode": "control_direct"},
        "runtime_source_evidence": _reference_source_evidence(),
        "env_evidence": {"jax_version": "fake"},
        "component_started_monotonic_ns": 1,
        "component_ended_monotonic_ns": 10,
        "component_wall_s": 9e-9,
        **{marker: False for marker in harness.RUNTIME_MARKERS},
    }
    result.update(changes)
    return harness._dual._hashed_document(result)


def _derived_parent(tmp_path: Path):
    dual_tests_path = Path(__file__).with_name("test_perf48_dual_pipeline_tools.py")
    spec = importlib.util.spec_from_file_location("async_parent_dual_tests", dual_tests_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    manifest, config, source, parent, _, commit = module._derived_dual_manifest(tmp_path)
    deploy = load("perf48_async_pipeline_deploy")
    for relative in deploy.ASYNC_REQUIRED_SOURCE_FILES:
        path = source / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# async bound {relative}\n", encoding="utf-8")
    return manifest, config, source, parent, commit


def test_deploy_real_temp_manifest_and_tamper_fail_closed(tmp_path):
    deploy = load("perf48_async_pipeline_deploy")
    harness = load("perf48_async_pipeline_harness")
    parent, _, source, _, commit = _derived_parent(tmp_path)
    out = tmp_path / "async"
    evidence = deploy.build_deploy(
        dual_manifest=parent,
        source=source,
        source_commit=commit,
        out=out,
    )
    manifest = deploy._manifest.load_manifest(out / "manifest.json")
    metadata = harness.verify_async_manifest(manifest, commit)
    assert metadata["matrix"]["count"] == 6
    assert metadata["validation_cache"]["exercised"] is False
    assert set(deploy.ASYNC_REQUIRED_SOURCE_FILES).issubset(
        manifest["source_config"]["source"]
    )
    assert deploy.load_deploy_evidence(out / "async_pipeline_deploy_evidence.json") == evidence
    document = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    document["async_pipeline"]["source_commit"] = "0" * 40
    (out / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256"):
        deploy._manifest.load_manifest(out / "manifest.json")


def test_manifest_source_and_parent_tamper_fail_closed(tmp_path):
    deploy = load("perf48_async_pipeline_deploy")
    harness = load("perf48_async_pipeline_harness")
    parent, _, source, parent_parent, commit = _derived_parent(tmp_path)
    out = tmp_path / "async"
    deploy.build_deploy(dual_manifest=parent, source=source, source_commit=commit, out=out)
    manifest = deploy._manifest.load_manifest(out / "manifest.json")
    target = source / "src/dicode/skill_preflight/async_preflight.py"
    target.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        harness.verify_async_manifest(manifest, commit)
    target.write_text("# async bound src/dicode/skill_preflight/async_preflight.py\n", encoding="utf-8")
    parent.write_text(parent.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent file binding"):
        harness.verify_async_manifest(manifest, commit)
    assert parent_parent.exists()


def test_apply_once_route_exact_once_and_double_apply_rejected():
    harness = load("perf48_async_pipeline_harness")
    guard = harness.ApplyOnce()
    calls = []

    class Manager:
        def poll_and_apply(self, **kwargs):
            kept = []
            kwargs["route_apply_fn"]({}, ["one"], kept, object(), kwargs["route_fn"])
            return kept

    def apply(scores, ids, kept, archive, route):
        calls.append(ids)
        kept.extend(ids)

    assert guard.poll(Manager(), archive=object(), session=1, route_fn=object(), route_apply_fn=apply) == ["one"]
    assert guard.route_calls == 1 and calls == [["one"]]
    with pytest.raises(RuntimeError, match="double apply rejected"):
        guard.poll(Manager(), archive=object(), session=1, route_fn=object(), route_apply_fn=apply)
    assert guard.route_calls == 1


def test_fake_controller_executes_N_N1_receipts_and_route(tmp_path, monkeypatch):
    harness = load("perf48_async_pipeline_harness")
    from dicode.skill_preflight.async_preflight import plan_async_session

    code = "class Env:\n    pass\n"
    code_sha = hashlib.sha256(code.encode()).hexdigest()
    checkpoint = tmp_path / "checkpoint" / "1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state").write_text("ok", encoding="utf-8")
    stage = {
        "name": "early",
        "task_ids": ["one"],
        "tasks": [{"id": "one", "code_sha256": code_sha}],
        "repeats": [{"rng": [7, 8]}],
        "graph": {"path": str(tmp_path / "graph"), "sha256": "graph"},
        "checkpoint": {"path": str(checkpoint), "sha256": "checkpoint"},
        "global_step": 1,
    }
    manifest = {"manifest_sha256": "manifest", "stages": [stage]}
    config = OmegaConf.create(
        {
            "training": {"use_wandb": False},
            "performance": {
                "async_preflight_pipeline": False,
                "learnability_fused_preflight_summary": True,
                "preflight_reuse_loaded_tasks": True,
            },
            "skill_preflight": {"use_preflight": True},
            "dicode_manager": {"score_function": "learnability"},
        }
    )

    class Archive:
        mutated = False

        def get_task_codes(self, ids):
            return {task_id: code for task_id in ids}

    archive = Archive()
    projection = _projection()

    class Manager:
        def __init__(self, cfg, **kwargs):
            self.root = Path(cfg.performance.async_preflight_root)

        def launch(self, **kwargs):
            assert "JAX_PLATFORMS" not in os.environ
            job = self.root / "job"
            job.mkdir(parents=True)
            harness.atomic_json(job / "JOB.json", {"kind": "job"})
            harness.atomic_json(job / "RUNNING.json", {"kind": "running"})
            harness.atomic_json(
                job / "RESULT.json",
                {
                    "score_projection": projection,
                    "score_fingerprint": harness.fingerprint(projection),
                    "jax_backend": "gpu",
                    "jax_device_count": 1,
                    "route_calls": 0,
                    "gpu_preflight": {
                        "uuid": kwargs.get("gpu_uuid", "GPU-test"),
                        "index": 2,
                        "free_mib": 45000,
                        "external": [],
                    },
                },
            )
            self.job = job
            return job

        def poll_and_apply(self, **kwargs):
            kept = []
            kwargs["route_apply_fn"](
                {"0": {"sr": 0.5, "priority_score": 0.25}},
                ["one"], kept, archive, kwargs["route_fn"],
            )
            harness.atomic_json(self.job / "APPLYING.json", {"kind": "applying"})
            harness.atomic_json(self.job / "APPLIED.json", {"kind": "applied", "route_calls": 1})
            return kept

    def apply(scores, ids, kept, live, route):
        live.mutated = True
        kept.extend(ids)

    runtime = {
        "reconstruct_archive": lambda path: archive,
        "archive_hash": lambda value: "after" if value.mutated else "before",
        "array_rng": lambda value: np.asarray(value, dtype=np.uint32),
        "split_rng": lambda value: (value + 1, value + 2),
        "rng_hash": harness.fingerprint,
        "plan_async_session": plan_async_session,
        "AsyncPreflightManager": Manager,
        "load_async_receipt": harness.load_hashed_json,
        "route": object(),
        "preflight_route": apply,
        "source_evidence": {"verified": True},
        "controller_backend": "cpu",
    }
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(
        stage="early",
        repeat=0,
        required_gpu_uuid="GPU-test",
        source_commit="source",
        source=str(tmp_path),
        result_timeout_s=1,
        barrier_receipt={"enabled": False, "mode": "control_direct"},
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    result = harness.run_async_controller(
        manifest, config, {"path": "config", "sha256": "config"}, runtime, out, args
    )
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert result["controller_backend"] == "cpu"
    assert result["session_N"]["training_new_ids"] == []
    assert result["session_N1"]["training_new_ids"] == ["one"]
    assert result["route_calls"] == 1 and result["double_apply_rejected"]
    assert set(result["receipt_sha256"]) == {"JOB", "RUNNING", "RESULT", "APPLYING", "APPLIED"}


def test_async_and_reference_semantics_and_receipt_tamper(tmp_path):
    benchmark = load("perf48_async_pipeline_benchmark")
    harness = load("perf48_async_pipeline_harness")
    async_result = _async_result()
    reference = _reference_result()
    assert benchmark.validate_async_result(async_result, _args()) == async_result
    assert benchmark.validate_reference_result(reference, _args()) == reference
    assert benchmark.compare_async_reference(async_result, reference)["ok"]
    changed = _reference_result(archive_after_sha256="different")
    assert not benchmark.compare_async_reference(async_result, changed)["ok"]
    path = tmp_path / "RESULT.json"
    path.write_text(json.dumps(async_result), encoding="utf-8")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["kept_ids"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        harness.load_hashed_json(path)
    bad_source = _async_result(runtime_source_evidence={"verified": True})
    with pytest.raises(RuntimeError, match="source evidence incomplete"):
        benchmark.validate_async_result(bad_source, _args())


@pytest.mark.parametrize(
    "change, message",
    [
        ({"worker_gpu_preflight": {"uuid": "GPU-wrong", "index": 2, "free_mib": 45000, "external": []}}, "GPU preflight"),
        ({"worker_gpu_preflight": {"uuid": "GPU-8df11537-ab79-722d-606f-411966196c4c", "index": 2, "free_mib": 3000, "external": []}}, "GPU preflight"),
        ({"oom": True}, "runtime marker"),
        ({"validation_cache_exercised": True}, "validation_cache_exercised"),
    ],
)
def test_gpu_fatal_and_validation_cache_fail_closed(change, message):
    benchmark = load("perf48_async_pipeline_benchmark")
    with pytest.raises(RuntimeError, match=message):
        benchmark.validate_async_result(_async_result(**change), _args())


def test_runtime_memory_gate_and_exact_conclusions():
    benchmark = load("perf48_async_pipeline_benchmark")
    good = dict(
        peak_deltas_mib={"A": 512, "B": -1},
        minimum_free_mib=[4096, 5000],
        fatal_markers=[None, None],
        violations=[],
    )
    assert benchmark.runtime_gate(**good)
    for changed in (
        {"peak_deltas_mib": {"A": 513}},
        {"minimum_free_mib": [4095]},
        {"fatal_markers": ["Traceback"]},
        {"violations": ["external"]},
    ):
        values = dict(good)
        values.update(changed)
        assert not benchmark.runtime_gate(**values)
    assert benchmark.classify_conclusion(runtime_ok=True, semantic_ok=True, schedule_ok=True) == "ASYNC_PIPELINE_PASS"
    assert benchmark.classify_conclusion(runtime_ok=False, semantic_ok=True, schedule_ok=True) == "REJECTED_RUNTIME"
    assert benchmark.classify_conclusion(runtime_ok=True, semantic_ok=False, schedule_ok=True) == "REJECTED_SEMANTIC"
    assert benchmark.classify_conclusion(runtime_ok=True, semantic_ok=True, schedule_ok=False) == "REJECTED_SCHEDULE"


def test_timing_uses_component_makespan_and_separates_process_wall():
    benchmark = load("perf48_async_pipeline_benchmark")

    def run(start, end, wall, process_start, process_end, process_wall):
        return {
            "started_monotonic_ns": process_start,
            "ended_monotonic_ns": process_end,
            "process_wall_s": process_wall,
            "result": {
                "component_started_monotonic_ns": start,
                "component_ended_monotonic_ns": end,
                "component_wall_s": wall,
            },
        }

    async_run = run(10_000_000_000, 18_000_000_000, 8.0, 1, 30_000_000_001, 30.0)
    concurrent_b = run(11_000_000_000, 20_000_000_000, 9.0, 2, 29_000_000_002, 29.0)
    reference = run(0, 12_000_000_000, 12.0, 0, 14_000_000_000, 14.0)
    control_b = run(0, 10_000_000_000, 10.0, 0, 11_000_000_000, 11.0)
    timing = benchmark.calculate_timing(async_run, concurrent_b, reference, control_b)
    assert timing["concurrent_component_makespan_s"] == 10.0
    assert timing["component_start_skew_s"] == 1.0
    assert timing["concurrent_component_overlap_s"] == 7.0
    assert timing["overlap_hidden_wall_s"] == 12.0
    assert timing["slowdown"] == {
        "async_vs_sync_reference": -1 / 3,
        "B_concurrent_vs_control": -0.1,
    }
    assert timing["operational_process_makespan_s"] == 30.0
    assert timing["hidden_wall_formula"].startswith("sync_reference_component_wall_s")


def test_schedule_threshold_boundaries_and_epsilon_reject():
    benchmark = load("perf48_async_pipeline_benchmark")
    result = _async_result()
    timing = {
        "concurrent_component_overlap_s": 0.0,
        "component_start_skew_s": 2.0,
        "overlap_hidden_wall_s": 400.0,
        "slowdown": {
            "async_vs_sync_reference": 0.05,
            "B_concurrent_vs_control": 0.05,
        },
    }
    assert benchmark.schedule_gate(timing, result)[0]
    for path, value in (
        (("component_start_skew_s",), 2.000001),
        (("overlap_hidden_wall_s",), 399.999999),
        (("slowdown", "async_vs_sync_reference"), 0.050001),
        (("slowdown", "B_concurrent_vs_control"), 0.050001),
        (("concurrent_component_overlap_s",), -0.000001),
    ):
        changed = json.loads(json.dumps(timing))
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert not benchmark.schedule_gate(changed, result)[0]
    changed_result = dict(result, double_apply_rejected=False)
    assert not benchmark.schedule_gate(timing, changed_result)[0]


def test_concurrent_launch_order_alternates_by_repeat():
    benchmark = load("perf48_async_pipeline_benchmark")
    assert benchmark.concurrent_launch_order(0) == ("ASYNC", "B")
    assert benchmark.concurrent_launch_order(1) == ("B", "ASYNC")
    with pytest.raises(ValueError, match="repeat"):
        benchmark.concurrent_launch_order(2)


def test_concurrent_command_carries_tool_owned_barrier(tmp_path):
    benchmark = load("perf48_async_pipeline_benchmark")
    args = SimpleNamespace(
        python="python", harness="harness", manifest="manifest", config="config",
        source="source", source_commit="f" * 40, stage="early", repeat=0,
        result_timeout_s=10, gpu2_uuid=benchmark.GPU2[1], gpu3_uuid=benchmark.GPU3[1],
        active_barrier={
            "ready_paths": {"ASYNC": tmp_path / "READY_ASYNC", "B": tmp_path / "READY_B"},
            "go_path": tmp_path / "GO", "barrier_id": "id", "timeout_s": 5,
        },
    )
    command = benchmark.command(args, "ASYNC", tmp_path / "out")
    assert command[command.index("--barrier-id") + 1] == "id"
    assert command[command.index("--ready-path") + 1].endswith("READY_ASYNC")
    args.active_barrier = None
    assert "--barrier-id" not in benchmark.command(args, "B", tmp_path / "out-b")


def _cell(stage, repeat, conclusion="ASYNC_PIPELINE_PASS", overlap=10.0):
    return {
        "stage": stage,
        "repeat": repeat,
        "conclusion": conclusion,
        "xla_mode": "deterministic_gate_not_production_timing",
        "timing": {
            "overlap_hidden_wall_s": overlap,
            "slowdown": {
                "async_vs_sync_reference": 0.01,
                "B_concurrent_vs_control": 0.02,
            },
        },
    }


def test_aggregate_requires_exact_six_and_projects_only_model():
    benchmark = load("perf48_async_pipeline_benchmark")
    cells = [_cell(stage, repeat) for stage in ("early", "mid", "late") for repeat in (0, 1)]
    result = benchmark.aggregate_exact_six(cells)
    assert result["conclusion"] == "ASYNC_PIPELINE_PASS"
    assert result["cell_count"] == 6
    assert result["timing"]["overlap_hidden_wall_s_total"] == 60.0
    assert result["timing"]["overlap_hidden_wall_s_median"] == 10.0
    assert result["timing"]["async_slowdown_median"] == 0.01
    assert result["timing"]["B_slowdown_median"] == 0.02
    assert result["model_projection"]["projected_attempt06_savings_s"] == 220.0
    assert result["xla_mode"] == "deterministic_gate_not_production_timing"
    assert result["validation_cache_exercised"] is False
    with pytest.raises(RuntimeError, match="exact unique"):
        benchmark.aggregate_exact_six(cells[:-1])
    rejected = list(cells)
    rejected[0] = _cell("early", 0, "REJECTED_SEMANTIC")
    assert benchmark.aggregate_exact_six(rejected)["conclusion"] == "REJECTED_SEMANTIC"
    unknown = list(cells)
    unknown[0] = _cell("early", 0, "MISSING")
    with pytest.raises(RuntimeError, match="unknown conclusion"):
        benchmark.aggregate_exact_six(unknown)
    wrong_xla = list(cells)
    wrong_xla[0] = {**wrong_xla[0], "xla_mode": "production"}
    with pytest.raises(RuntimeError, match="XLA mode"):
        benchmark.aggregate_exact_six(wrong_xla)


def test_fake_B_delegation_preserves_100_update_result(monkeypatch, tmp_path):
    harness = load("perf48_async_pipeline_harness")
    calls = []

    def run_b(manifest, config, config_evidence, runtime, out, args):
        calls.append((args.component, args.barrier_receipt))
        return {"updates": 100, "env_steps": 13_107_200, "checkpoint_loadable": True}

    monkeypatch.setattr(harness._dual, "run_component_b", run_b)
    args = SimpleNamespace(component="unused")
    result = harness.run_component_b({}, {}, {}, {}, tmp_path, args)
    assert result == {"updates": 100, "env_steps": 13_107_200, "checkpoint_loadable": True}
    assert calls == [("B", {"enabled": False, "mode": "control_direct"})]
    receipt = {"enabled": True, "mode": "ready_go", "barrier_id": "id"}
    args = SimpleNamespace(component="unused", barrier_receipt=receipt)
    harness.run_component_b({}, {}, {}, {}, tmp_path, args)
    assert calls[-1] == ("B", receipt)


def test_component_env_exact_gpu_and_independent_caches(tmp_path):
    benchmark = load("perf48_async_pipeline_benchmark")
    args = SimpleNamespace(
        source=str(SOURCE_ROOT),
        gpu2_uuid=benchmark.GPU2[1],
        gpu3_uuid=benchmark.GPU3[1],
    )
    outs = []
    for name in ("async", "reference", "B"):
        out = tmp_path / name
        out.mkdir()
        for subdir in ("tmp", "cache", "wandb", "jax_compilation"):
            (out / subdir).mkdir()
        outs.append(out)
    env_a = benchmark.component_env(args, "ASYNC", outs[0])
    env_r = benchmark.component_env(args, "REFERENCE", outs[1])
    env_b = benchmark.component_env(args, "B", outs[2])
    assert env_a["CUDA_VISIBLE_DEVICES"] == ""
    assert env_a["JAX_PLATFORMS"] == "cpu"
    assert env_r["CUDA_VISIBLE_DEVICES"] == benchmark.GPU2[1]
    assert "JAX_PLATFORMS" not in env_r
    assert env_b["CUDA_VISIBLE_DEVICES"] == benchmark.GPU3[1]
    assert len({env_a["JAX_COMPILATION_CACHE_DIR"], env_r["JAX_COMPILATION_CACHE_DIR"], env_b["JAX_COMPILATION_CACHE_DIR"]}) == 3


def test_worker_platform_override_restores_parent_even_on_failure(monkeypatch):
    harness = load("perf48_async_pipeline_harness")
    seen = []

    class Manager:
        def launch(self, **kwargs):
            seen.append(os.environ.get("JAX_PLATFORMS"))
            if kwargs.get("fail"):
                raise RuntimeError("launch failed")
            return Path("job")

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert harness.launch_worker_without_cpu_platform(Manager()) == Path("job")
    assert seen == [None] and os.environ["JAX_PLATFORMS"] == "cpu"
    with pytest.raises(RuntimeError, match="launch failed"):
        harness.launch_worker_without_cpu_platform(Manager(), fail=True)
    assert seen == [None, None] and os.environ["JAX_PLATFORMS"] == "cpu"


def test_cuda_plugin_traceback_remains_fatal(tmp_path):
    benchmark = load("perf48_async_pipeline_benchmark")
    log = tmp_path / "stderr"
    log.write_text("CUDA plugin init failed\nTraceback (most recent call last):\n", encoding="utf-8")
    assert benchmark._pair.fatal_in([log]) == "Traceback (most recent call last):"


def _barrier_args(tmp_path, timeout=1.0):
    return SimpleNamespace(
        ready_path=str(tmp_path / "READY_ASYNC.json"),
        go_path=str(tmp_path / "GO.json"),
        barrier_id="barrier-id",
        barrier_timeout_s=timeout,
    )


def test_harness_ready_go_success_and_control(tmp_path):
    harness = load("perf48_async_pipeline_harness")
    args = _barrier_args(tmp_path)
    source = {"source": "evidence"}

    def release():
        ready_path = Path(args.ready_path)
        deadline = time.monotonic() + 1
        while not ready_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        ready = harness.load_hashed_json(ready_path)
        harness.atomic_json(
            args.go_path,
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": args.barrier_id,
                "components": ["ASYNC", "B"],
                "ready_sha256": {
                    "ASYNC": ready["result_sha256"], "B": "b" * 64,
                },
                "go_monotonic_ns": time.monotonic_ns(),
                "llm_api_calls": 0,
            },
        )

    thread = threading.Thread(target=release)
    thread.start()
    receipt = harness.wait_for_async_go(args, "ASYNC", source)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert receipt["enabled"] and receipt["mode"] == "ready_go"
    assert receipt["runtime_source_evidence_sha256"] == harness.fingerprint(source)
    assert receipt["go_observed_monotonic_ns"] >= receipt["go_monotonic_ns"]
    control = SimpleNamespace(ready_path=None, go_path=None, barrier_id=None)
    assert harness.wait_for_async_go(control, "REFERENCE", source) == {
        "enabled": False, "mode": "control_direct",
    }


def test_harness_go_tamper_and_timeout_fail_closed(tmp_path):
    harness = load("perf48_async_pipeline_harness")
    args = _barrier_args(tmp_path / "tamper")
    Path(args.ready_path).parent.mkdir()

    def bad_release():
        while not Path(args.ready_path).exists():
            time.sleep(0.005)
        harness.atomic_json(
            args.go_path,
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": args.barrier_id,
                "components": ["ASYNC", "B"],
                "ready_sha256": {"ASYNC": "0" * 64, "B": "b" * 64},
                "go_monotonic_ns": time.monotonic_ns(),
                "llm_api_calls": 0,
            },
        )

    thread = threading.Thread(target=bad_release)
    thread.start()
    with pytest.raises(RuntimeError, match="GO barrier receipt invalid"):
        harness.wait_for_async_go(args, "ASYNC", {"source": "evidence"})
    thread.join(timeout=1)
    timeout_root = tmp_path / "timeout"
    timeout_root.mkdir()
    with pytest.raises(TimeoutError, match="GO barrier timeout"):
        harness.wait_for_async_go(
            _barrier_args(timeout_root, timeout=0.01), "ASYNC", {"source": "evidence"}
        )


def _parent_barrier(harness, source, component="ASYNC"):
    source_sha = harness.fingerprint(source)
    ready = harness._dual._hashed_document(
        {
            "classification": harness.CLASSIFICATION,
            "barrier_id": "barrier-id",
            "component": component,
            "pid": 101,
            "ready_monotonic_ns": 10,
            "runtime_source_evidence_sha256": source_sha,
            "llm_api_calls": 0,
        }
    )
    other = "B" if component == "ASYNC" else "ASYNC"
    other_ready = harness._dual._hashed_document(
        {
            "classification": harness.CLASSIFICATION,
            "barrier_id": "barrier-id",
            "component": other,
            "pid": 102,
            "ready_monotonic_ns": 11,
            "runtime_source_evidence_sha256": "b" * 64,
            "llm_api_calls": 0,
        }
    )
    ready_map = {component: ready, other: other_ready}
    go = harness._dual._hashed_document(
        {
            "classification": harness.CLASSIFICATION,
            "barrier_id": "barrier-id",
            "components": ["ASYNC", "B"],
            "ready_sha256": {name: row["result_sha256"] for name, row in ready_map.items()},
            "go_monotonic_ns": 20,
            "llm_api_calls": 0,
        }
    )
    return {"barrier_id": "barrier-id", "components": ["ASYNC", "B"], "ready": ready_map, "go": go, "timeout_s": 1.0}


def test_parent_child_barrier_link_and_workload_after_go():
    harness = load("perf48_async_pipeline_harness")
    benchmark = load("perf48_async_pipeline_benchmark")
    source = {"source": "evidence"}
    parent = _parent_barrier(harness, source)
    ready = parent["ready"]["ASYNC"]
    go = parent["go"]
    result = {
        "runtime_source_evidence": source,
        "component_started_monotonic_ns": 31,
        "barrier": {
            "enabled": True,
            "barrier_id": "barrier-id",
            "ready_sha256": ready["result_sha256"],
            "ready_monotonic_ns": ready["ready_monotonic_ns"],
            "runtime_source_evidence_sha256": ready["runtime_source_evidence_sha256"],
            "go_sha256": go["result_sha256"],
            "go_monotonic_ns": go["go_monotonic_ns"],
            "go_observed_monotonic_ns": 30,
        },
    }
    benchmark.verify_async_barrier_link(result, parent, "ASYNC")
    early = json.loads(json.dumps(result))
    early["component_started_monotonic_ns"] = 29
    with pytest.raises(RuntimeError, match="parent/child"):
        benchmark.verify_async_barrier_link(early, parent, "ASYNC")
    tampered = json.loads(json.dumps(result))
    tampered["barrier"]["go_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="parent/child"):
        benchmark.verify_async_barrier_link(tampered, parent, "ASYNC")
    source_mismatch = json.loads(json.dumps(result))
    source_mismatch["barrier"]["runtime_source_evidence_sha256"] = "a" * 64
    source_mismatch_parent = json.loads(json.dumps(parent))
    source_mismatch_parent["ready"]["ASYNC"]["runtime_source_evidence_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="parent/child"):
        benchmark.verify_async_barrier_link(
            source_mismatch, source_mismatch_parent, "ASYNC"
        )


def test_parent_ready_validation_pid_source_tamper_and_early_exit(tmp_path, monkeypatch):
    harness = load("perf48_async_pipeline_harness")
    benchmark = load("perf48_async_pipeline_benchmark")
    args = SimpleNamespace(
        manifest_sha256="manifest", source_commit="f" * 40,
        stage="early", repeat=0, barrier_timeout_s=0.05,
    )
    barrier = benchmark.make_async_barrier(args, tmp_path, "concurrent")
    evidence = {}

    class Process:
        def __init__(self, pid, rc=None): self.pid, self.rc = pid, rc
        def poll(self): return self.rc

    processes = {"ASYNC": Process(101), "B": Process(102)}
    monitors = {name: (None, None, []) for name in ("ASYNC", "B")}
    for component in ("ASYNC", "B"):
        out = tmp_path / component
        out.mkdir()
        evidence[component] = {"out": str(out)}
    monkeypatch.setattr(benchmark._pair, "fatal_in", lambda paths: None)
    monkeypatch.setattr(benchmark._pair, "arm_gpu_metrics", lambda path: (0, 5000))

    def write_ready(component, pid, source_sha="a" * 64):
        harness.atomic_json(
            barrier["ready_paths"][component],
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": barrier["barrier_id"],
                "component": component,
                "pid": pid,
                "ready_monotonic_ns": time.monotonic_ns(),
                "runtime_source_evidence_sha256": source_sha,
                "llm_api_calls": 0,
            },
        )

    write_ready("ASYNC", 101)
    write_ready("B", 102)
    parent = benchmark.release_async_barrier(barrier, processes, evidence, monitors)
    assert parent["go"]["components"] == ["ASYNC", "B"]

    for label, pid, source_sha, message in (
        ("pid", 999, "a" * 64, "READY receipt invalid"),
        ("source", 101, "bad", "READY receipt invalid"),
    ):
        case = tmp_path / label
        case.mkdir()
        case_barrier = benchmark.make_async_barrier(args, case, "concurrent")
        case_evidence = {}
        for component in ("ASYNC", "B"):
            out = case / component
            out.mkdir()
            case_evidence[component] = {"out": str(out)}
        target = "ASYNC"
        harness.atomic_json(
            case_barrier["ready_paths"][target],
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": case_barrier["barrier_id"],
                "component": target,
                "pid": pid,
                "ready_monotonic_ns": 1,
                "runtime_source_evidence_sha256": source_sha,
                "llm_api_calls": 0,
            },
        )
        write_processes = {"ASYNC": Process(101), "B": Process(102, 1)}
        with pytest.raises(RuntimeError, match=message):
            benchmark.release_async_barrier(
                case_barrier, write_processes, case_evidence, monitors
            )

    early = tmp_path / "early"
    early.mkdir()
    early_barrier = benchmark.make_async_barrier(args, early, "concurrent")
    early_evidence = {}
    for component in ("ASYNC", "B"):
        out = early / component
        out.mkdir()
        early_evidence[component] = {"out": str(out)}
    with pytest.raises(RuntimeError, match="exited before READY"):
        benchmark.release_async_barrier(
            early_barrier,
            {"ASYNC": Process(101, 1), "B": Process(102)},
            early_evidence,
            monitors,
        )


def test_parent_ready_selfhash_tamper_and_timeout(tmp_path, monkeypatch):
    harness = load("perf48_async_pipeline_harness")
    benchmark = load("perf48_async_pipeline_benchmark")
    args = SimpleNamespace(
        manifest_sha256="manifest", source_commit="f" * 40,
        stage="early", repeat=0, barrier_timeout_s=0.01,
    )

    class Process:
        def __init__(self, pid): self.pid = pid
        def poll(self): return None

    processes = {"ASYNC": Process(101), "B": Process(102)}
    monitors = {name: (None, None, []) for name in ("ASYNC", "B")}
    monkeypatch.setattr(benchmark._pair, "fatal_in", lambda paths: None)
    monkeypatch.setattr(benchmark._pair, "arm_gpu_metrics", lambda path: (0, 5000))

    timeout_barrier = benchmark.make_async_barrier(args, tmp_path, "timeout")
    timeout_evidence = {}
    for component in ("ASYNC", "B"):
        out = tmp_path / f"timeout_{component}"
        out.mkdir()
        timeout_evidence[component] = {"out": str(out)}
    with pytest.raises(TimeoutError, match="READY barrier timeout"):
        benchmark.release_async_barrier(
            timeout_barrier, processes, timeout_evidence, monitors
        )

    tamper_root = tmp_path / "tamper"
    tamper_root.mkdir()
    tamper_barrier = benchmark.make_async_barrier(args, tamper_root, "concurrent")
    tamper_evidence = {}
    for component, pid in (("ASYNC", 101), ("B", 102)):
        out = tamper_root / component
        out.mkdir()
        tamper_evidence[component] = {"out": str(out)}
        document = harness.atomic_json(
            tamper_barrier["ready_paths"][component],
            {
                "classification": harness.CLASSIFICATION,
                "barrier_id": tamper_barrier["barrier_id"],
                "component": component,
                "pid": pid,
                "ready_monotonic_ns": 1,
                "runtime_source_evidence_sha256": "a" * 64,
                "llm_api_calls": 0,
            },
        )
        if component == "ASYNC":
            document["pid"] = 999
            Path(tamper_barrier["ready_paths"][component]).write_text(
                json.dumps(document), encoding="utf-8"
            )
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark.release_async_barrier(
            tamper_barrier, processes, tamper_evidence, monitors
        )


def test_benchmark_prepared_output_is_accepted_but_finalized_output_is_not(tmp_path):
    benchmark = load("perf48_async_pipeline_benchmark")
    harness = load("perf48_async_pipeline_harness")
    out = benchmark._prepare(tmp_path, "control", "B")
    assert out.is_dir() and (out / "jax_compilation").is_dir()
    assert harness.prepare_output(out) == out
    (out / "RESULT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already finalized"):
        harness.prepare_output(out)


def test_sources_lock_default_research_and_no_eager_jax_import():
    deploy = load("perf48_async_pipeline_deploy")
    for name in ("perf48_async_pipeline_harness", "perf48_async_pipeline_benchmark"):
        source = (PERF / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        eager = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                eager.add(node.module.split(".")[0])
        assert "jax" not in eager
        assert "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE" in source
        assert "validation_cache_exercised" in source
    assert {
        "experiments/training/run_dicode.py",
        "src/dicode/skill_preflight/async_preflight.py",
        "src/dicode/training.py",
        "src/dicode/ppo_tr.py",
        "src/dicode/evaluation/online_evaluation.py",
        "src/dicode/scoring.py",
        "src/dicode/wrappers_cl.py",
    }.issubset(deploy.ASYNC_REQUIRED_SOURCE_FILES)
