from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
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


def test_fake_controller_executes_N_N1_receipts_and_route(tmp_path):
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
    )
    result = harness.run_async_controller(
        manifest, config, {"path": "config", "sha256": "config"}, runtime, out, args
    )
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
    assert timing["concurrent_component_overlap_s"] == 7.0
    assert timing["overlap_hidden_wall_s"] == 12.0
    assert timing["slowdown"] == {
        "async_vs_sync_reference": -1 / 3,
        "B_concurrent_vs_control": -0.1,
    }
    assert timing["operational_process_makespan_s"] == 30.0
    assert timing["hidden_wall_formula"].startswith("sync_reference_component_wall_s")


def _cell(stage, repeat, conclusion="ASYNC_PIPELINE_PASS", overlap=10.0):
    return {
        "stage": stage,
        "repeat": repeat,
        "conclusion": conclusion,
        "timing": {
            "overlap_hidden_wall_s": overlap,
            "slowdown": {
                "async_vs_sync_reference": 0.1,
                "B_concurrent_vs_control": 0.2,
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
    assert result["model_projection"]["projected_attempt06_savings_s"] == 220.0
    assert result["validation_cache_exercised"] is False
    with pytest.raises(RuntimeError, match="exact unique"):
        benchmark.aggregate_exact_six(cells[:-1])
    rejected = list(cells)
    rejected[0] = _cell("early", 0, "REJECTED_SEMANTIC")
    assert benchmark.aggregate_exact_six(rejected)["conclusion"] == "REJECTED_SEMANTIC"


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
    assert env_r["CUDA_VISIBLE_DEVICES"] == benchmark.GPU2[1]
    assert env_b["CUDA_VISIBLE_DEVICES"] == benchmark.GPU3[1]
    assert len({env_a["JAX_COMPILATION_CACHE_DIR"], env_r["JAX_COMPILATION_CACHE_DIR"], env_b["JAX_COMPILATION_CACHE_DIR"]}) == 3


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
