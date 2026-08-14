#!/usr/bin/env python3
"""Strict six-cell benchmark for the frozen async N/N+1 acceptance schedule."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_harness = _load_sibling("perf48_async_harness_benchmark", "perf48_async_pipeline_harness.py")
_dual_benchmark = _load_sibling("perf48_async_dual_benchmark", "perf48_dual_pipeline_benchmark.py")
_pair = _dual_benchmark._pair

CLASSIFICATION = "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE"
if CLASSIFICATION != _harness.CLASSIFICATION:
    raise RuntimeError("async benchmark classification binding mismatch")
CONCLUSIONS = _harness.CONCLUSIONS
GPU2 = (2, "GPU-8df11537-ab79-722d-606f-411966196c4c")
GPU3 = (3, "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd")
ASYNC_REFERENCE_FIELDS = (
    "task_ids",
    "task_code_rows",
    "input_rng_sha256",
    "preflight_rng_sha256",
    "checkpoint_input_path",
    "checkpoint_input_sha256",
    "graph_input_path",
    "graph_input_sha256",
    "score_projection",
    "score_fingerprint",
    "kept_ids",
    "rejected_ids",
    "archive_before_sha256",
    "archive_after_sha256",
)
ASYNC_RUNTIME_SOURCES = {
    "AsyncPreflightManager": "src/dicode/skill_preflight/async_preflight.py",
    "plan_async_session": "src/dicode/skill_preflight/async_preflight.py",
    "preflight_route": "src/dicode/skill_preflight/preflight_route.py",
}
REFERENCE_RUNTIME_SOURCES = {
    "TaskArchive",
    "load_tasks_from_env_codes",
    "evaluate_new_tasks",
    "run_evaluation_rollouts",
    "calculate_scores_from_snapshot",
    "learnability_scores_from_counts",
    "learnability_summary_contract",
    "_load_agent_state",
    "route",
    "preflight_route",
    "resolve_preloaded_tasks",
    "heldout_eval",
    "wrappers_cl",
}


def validate_runtime_source_evidence(
    evidence: Any, source_root: str | Path, *, async_component: bool,
) -> None:
    root = Path(source_root).resolve()
    if async_component:
        if not isinstance(evidence, Mapping) or set(evidence) != set(ASYNC_RUNTIME_SOURCES):
            raise RuntimeError("async runtime source evidence incomplete")
        for name, relative in ASYNC_RUNTIME_SOURCES.items():
            row = evidence.get(name)
            expected = (root / relative).resolve()
            if (
                not isinstance(row, Mapping)
                or row.get("relative") != relative
                or Path(row.get("path", "")).resolve() != expected
                or not expected.is_file()
                or _harness.file_sha256(expected) != row.get("sha256")
            ):
                raise RuntimeError(f"async runtime source mismatch: {name}")
        return
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("verified") is not True
        or set(evidence.get("paths", {})) != REFERENCE_RUNTIME_SOURCES
        or set(evidence.get("hashes", {})) != REFERENCE_RUNTIME_SOURCES
    ):
        raise RuntimeError("reference runtime source evidence incomplete")
    for name in REFERENCE_RUNTIME_SOURCES:
        path = Path(evidence["paths"][name]).resolve()
        if (
            not path.is_file()
            or path != root and root not in path.parents
            or _harness.file_sha256(path) != evidence["hashes"][name]
        ):
            raise RuntimeError(f"reference runtime source mismatch: {name}")


def validate_async_result(document: Mapping[str, Any], args: Any) -> dict[str, Any]:
    expected = {
        "classification": CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": "ASYNC",
        "manifest_sha256": args.manifest_sha256,
        "source_commit": args.source_commit,
        "stage": args.stage,
        "repeat": args.repeat,
        "gpu_uuid": args.gpu2_uuid,
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "double_apply_rejected": True,
        "route_calls": 1,
        "worker_route_calls": 0,
        "main_route_calls": 1,
        "worker_jax_backend": "gpu",
        "worker_jax_device_count": 1,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"async result mismatch: {key}")
    if _harness._dual._hashed_document(document)["result_sha256"] != document.get("result_sha256"):
        raise RuntimeError("async result self-hash mismatch")
    task_ids = document.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("async task IDs invalid")
    if document.get("session_N") != {
        "fresh_ids": task_ids,
        "training_new_ids": [],
        "launch_ids": task_ids,
    }:
        raise RuntimeError("session N exclusion mismatch")
    kept = document.get("kept_ids")
    if document.get("session_N1") != {
        "fresh_ids": [],
        "delayed_kept_ids": kept,
        "training_new_ids": kept,
        "launch_ids": [],
    }:
        raise RuntimeError("session N+1 delayed inclusion mismatch")
    if set(kept) | set(document.get("rejected_ids", [])) != set(task_ids):
        raise RuntimeError("async route partition mismatch")
    receipts = document.get("receipt_sha256")
    if not isinstance(receipts, Mapping) or set(receipts) != {"JOB", "RUNNING", "RESULT", "APPLYING", "APPLIED"} or any(not value for value in receipts.values()):
        raise RuntimeError("async receipt chain incomplete")
    gpu = document.get("worker_gpu_preflight")
    if not isinstance(gpu, Mapping) or gpu.get("uuid") != args.gpu2_uuid or gpu.get("free_mib", 0) < 4096 or gpu.get("external") != []:
        raise RuntimeError("async worker GPU preflight invalid")
    if any(bool(document.get(marker)) for marker in _harness.RUNTIME_MARKERS):
        raise RuntimeError("async runtime marker present")
    validate_runtime_source_evidence(
        document.get("runtime_source_evidence"), args.source, async_component=True
    )
    return dict(document)


def validate_reference_result(document: Mapping[str, Any], args: Any) -> dict[str, Any]:
    expected = {
        "classification": CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": "REFERENCE",
        "manifest_sha256": args.manifest_sha256,
        "source_commit": args.source_commit,
        "stage": args.stage,
        "repeat": args.repeat,
        "gpu_uuid": args.gpu2_uuid,
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "summary_mode": "fused",
        "route_calls": 1,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"reference result mismatch: {key}")
    if _harness._dual._hashed_document(document)["result_sha256"] != document.get("result_sha256"):
        raise RuntimeError("reference result self-hash mismatch")
    if any(bool(document.get(marker)) for marker in _harness.RUNTIME_MARKERS):
        raise RuntimeError("reference runtime marker present")
    validate_runtime_source_evidence(
        document.get("runtime_source_evidence"), args.source, async_component=False
    )
    return dict(document)


def compare_async_reference(async_result: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    differences = {
        field: {"async": async_result.get(field), "reference": reference.get(field)}
        for field in ASYNC_REFERENCE_FIELDS
        if async_result.get(field) != reference.get(field)
    }
    return {"ok": not differences, "fields": list(ASYNC_REFERENCE_FIELDS), "differences": differences}


def classify_conclusion(*, runtime_ok: bool, semantic_ok: bool, schedule_ok: bool) -> str:
    if not runtime_ok:
        return "REJECTED_RUNTIME"
    if not semantic_ok:
        return "REJECTED_SEMANTIC"
    if not schedule_ok:
        return "REJECTED_SCHEDULE"
    return "ASYNC_PIPELINE_PASS"


def runtime_gate(
    *, peak_deltas_mib: Mapping[str, int], minimum_free_mib: list[int | None],
    fatal_markers: list[str | None], violations: list[str],
) -> bool:
    """Shared fail-closed GPU/runtime acceptance gate."""
    return bool(
        peak_deltas_mib
        and max(peak_deltas_mib.values()) <= 512
        and minimum_free_mib
        and all(value is not None and value >= 4096 for value in minimum_free_mib)
        and not any(fatal_markers)
        and not violations
    )


def calculate_timing(
    async_run: Mapping[str, Any], concurrent_b: Mapping[str, Any],
    reference: Mapping[str, Any], control_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep component timing and operational process timing in separate ledgers."""
    async_result = async_run["result"]
    concurrent_b_result = concurrent_b["result"]
    reference_result = reference["result"]
    control_b_result = control_b["result"]
    async_wall = float(async_result["component_wall_s"])
    concurrent_b_wall = float(concurrent_b_result["component_wall_s"])
    reference_wall = float(reference_result["component_wall_s"])
    control_b_wall = float(control_b_result["component_wall_s"])
    if min(async_wall, concurrent_b_wall, reference_wall, control_b_wall) <= 0:
        raise RuntimeError("component wall must be positive")
    component_makespan = (
        max(
            int(async_result["component_ended_monotonic_ns"]),
            int(concurrent_b_result["component_ended_monotonic_ns"]),
        )
        - min(
            int(async_result["component_started_monotonic_ns"]),
            int(concurrent_b_result["component_started_monotonic_ns"]),
        )
    ) / 1e9
    process_makespan = (
        max(int(async_run["ended_monotonic_ns"]), int(concurrent_b["ended_monotonic_ns"]))
        - min(int(async_run["started_monotonic_ns"]), int(concurrent_b["started_monotonic_ns"]))
    ) / 1e9
    component_overlap = async_wall + concurrent_b_wall - component_makespan
    hidden_wall = reference_wall + control_b_wall - component_makespan
    return {
        "timing_basis": "component_monotonic",
        "async_component_wall_s": async_wall,
        "sync_reference_component_wall_s": reference_wall,
        "B_control_component_wall_s": control_b_wall,
        "B_concurrent_component_wall_s": concurrent_b_wall,
        "concurrent_component_makespan_s": component_makespan,
        "concurrent_component_overlap_s": component_overlap,
        "overlap_hidden_wall_s": hidden_wall,
        "hidden_wall_formula": (
            "sync_reference_component_wall_s + B_control_component_wall_s - "
            "concurrent_component_makespan_s"
        ),
        "slowdown": {
            "async_vs_sync_reference": (async_wall - reference_wall) / reference_wall,
            "B_concurrent_vs_control": (concurrent_b_wall - control_b_wall) / control_b_wall,
        },
        "operational_process_makespan_s": process_makespan,
        "operational_process_wall_s": {
            "async": float(async_run["process_wall_s"]),
            "sync_reference": float(reference["process_wall_s"]),
            "B_control": float(control_b["process_wall_s"]),
            "B_concurrent": float(concurrent_b["process_wall_s"]),
        },
    }


def _prepare(root: Path, label: str, component: str) -> Path:
    out = root / label / component
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    out.mkdir(parents=True)
    for name in ("tmp", "cache", "wandb", "jax_compilation"):
        (out / name).mkdir()
    return out


def component_env(args: Any, component: str, out: Path) -> dict[str, str]:
    env = dict(os.environ)
    uuid = "" if component == "ASYNC" else (args.gpu2_uuid if component == "REFERENCE" else args.gpu3_uuid)
    env.update(
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONPATH=str(Path(args.source) / "src"),
        PYTHONNOUSERSITE="1",
        WANDB_MODE="offline",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TMPDIR=str(out / "tmp"),
        TMP=str(out / "tmp"),
        TEMP=str(out / "tmp"),
        XDG_CACHE_HOME=str(out / "cache"),
        WANDB_DIR=str(out / "wandb"),
        JAX_COMPILATION_CACHE_DIR=str(out / "jax_compilation"),
    )
    env.pop("JAX_PLATFORMS", None)
    env["XLA_FLAGS"] = _pair.append_xla_flag(env.get("XLA_FLAGS"))
    return env


def command(args: Any, component: str, out: Path) -> list[str]:
    uuid = args.gpu2_uuid if component in ("ASYNC", "REFERENCE") else args.gpu3_uuid
    return [
        args.python,
        args.harness,
        "--manifest", args.manifest,
        "--config", args.config,
        "--source", args.source,
        "--source-commit", args.source_commit,
        "--out", str(out),
        "--component", component,
        "--required-gpu-uuid", uuid,
        "--stage", args.stage,
        "--repeat", str(args.repeat),
        "--result-timeout-s", str(args.result_timeout_s),
    ]


def _run_group(args: Any, root: Path, label: str, components: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    processes: dict[str, subprocess.Popen[Any]] = {}
    streams: dict[str, tuple[Any, Any]] = {}
    monitors: dict[str, tuple[Any, Any, list[str]]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    try:
        for component in components:
            index, uuid = GPU2 if component in ("ASYNC", "REFERENCE") else GPU3
            actual = (args.gpu2_index, args.gpu2_uuid) if index == 2 else (args.gpu3_index, args.gpu3_uuid)
            if actual != (index, uuid):
                raise RuntimeError(f"exact GPU contract mismatch: {component}")
            _pair.assert_gpu_free(index, uuid)
        for component in components:
            out = _prepare(root, label, component)
            stdout = (out / "harness.stdout").open("w", encoding="utf-8")
            stderr = (out / "harness.stderr").open("w", encoding="utf-8")
            streams[component] = (stdout, stderr)
            cmd = command(args, component, out)
            started = time.monotonic_ns()
            process = subprocess.Popen(cmd, env=component_env(args, component, out), stdout=stdout, stderr=stderr, text=True, start_new_session=True)
            processes[component] = process
            index, uuid = GPU2 if component in ("ASYNC", "REFERENCE") else GPU3
            monitors[component] = _dual_benchmark._strict_monitor_popen(process, index, uuid, out)
            evidence[component] = {"pid": process.pid, "argv": cmd, "out": str(out), "started_monotonic_ns": started, "compile_cache_dir": str(out / "jax_compilation")}
        while any(process.poll() is None for process in processes.values()):
            for component, process in processes.items():
                out = Path(evidence[component]["out"])
                fatal = _pair.fatal_in([out / "harness.stdout", out / "harness.stderr"])
                _, minimum = _pair.arm_gpu_metrics(out / "gpu_memory.csv")
                violations = monitors[component][2]
                if fatal or violations or (minimum is not None and minimum < 4096) or process.poll() not in (None, 0):
                    raise RuntimeError(f"{component} runtime safety failure: fatal={fatal}, violations={violations}, min_free={minimum}, rc={process.poll()}")
            time.sleep(0.2)
        for component, process in processes.items():
            process.wait(timeout=10)
            evidence[component]["ended_monotonic_ns"] = time.monotonic_ns()
            evidence[component]["process_wall_s"] = (evidence[component]["ended_monotonic_ns"] - evidence[component]["started_monotonic_ns"]) / 1e9
            out = Path(evidence[component]["out"])
            peak, minimum = _pair.arm_gpu_metrics(out / "gpu_memory.csv")
            fatal = _pair.fatal_in([out / "harness.stdout", out / "harness.stderr"])
            if process.returncode or fatal or minimum is None or minimum < 4096 or monitors[component][2]:
                raise RuntimeError(f"{component} final runtime safety failure")
            evidence[component].update(gpu_peak_memory_mib=peak, gpu_min_free_mib=minimum, fatal_marker=fatal, monitor_interval_s=2.0)
            document = _harness.load_hashed_json(out / "RESULT.json")
            if component == "ASYNC":
                evidence[component]["result"] = validate_async_result(document, args)
            elif component == "REFERENCE":
                evidence[component]["result"] = validate_reference_result(document, args)
            else:
                evidence[component]["result"] = _dual_benchmark.validate_component_result(document, component="B", manifest_sha256=args.manifest_sha256, source_commit=args.source_commit, gpu_uuid=args.gpu3_uuid, stage=args.stage, repeat=args.repeat, expected_barrier=False)
        return evidence
    finally:
        for process in processes.values():
            if process.poll() is None:
                _pair.stop_owned(process.pid)
        for stop, thread, _ in monitors.values():
            stop.set()
            thread.join(timeout=5)
        for stdout, stderr in streams.values():
            stdout.close()
            stderr.close()


def static_gate(args: Any) -> dict[str, Any]:
    manifest, _, config_evidence = _harness.load_inputs(args.manifest, args.config, args.source_commit)
    metadata = _harness.verify_async_manifest(manifest, args.source_commit)
    args.manifest_sha256 = manifest["manifest_sha256"]
    expected_harness = (Path(args.source) / "experiments/performance/perf48_async_pipeline_harness.py").resolve()
    if Path(args.harness).resolve() != expected_harness or not expected_harness.is_file():
        raise RuntimeError("async harness/source binding mismatch")
    expected_benchmark = (Path(args.source) / "experiments/performance/perf48_async_pipeline_benchmark.py").resolve()
    benchmark_entry = manifest["source_config"]["source"].get("experiments/performance/perf48_async_pipeline_benchmark.py")
    if (
        Path(__file__).resolve() != expected_benchmark
        or not isinstance(benchmark_entry, Mapping)
        or Path(benchmark_entry.get("path", "")).resolve() != expected_benchmark
        or _harness.file_sha256(expected_benchmark) != benchmark_entry.get("sha256")
    ):
        raise RuntimeError("async benchmark/source binding mismatch")
    if Path(metadata["source_root"]).resolve() != Path(args.source).resolve():
        raise RuntimeError("async manifest/source-root mismatch")
    if (args.gpu2_index, args.gpu2_uuid) != GPU2 or (args.gpu3_index, args.gpu3_uuid) != GPU3:
        raise RuntimeError("exact GPU index/UUID mismatch")
    return {"manifest_sha256": args.manifest_sha256, "config_evidence": config_evidence, "async_manifest": metadata, "harness_path": str(expected_harness), "harness_sha256": _harness.file_sha256(expected_harness), "benchmark_path": str(expected_benchmark), "benchmark_sha256": _harness.file_sha256(expected_benchmark)}


def run_cell(args: Any, root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"cell root exists: {root}")
    root.mkdir(parents=True)
    static = static_gate(args)
    order = ["B_CONTROL", "CONCURRENT", "REFERENCE"] if args.repeat == 0 else ["REFERENCE", "B_CONTROL", "CONCURRENT"]
    runs: dict[str, Any] = {}
    for label in order:
        if label == "B_CONTROL":
            runs[label] = _run_group(args, root, "b_control", ("B",))["B"]
        elif label == "REFERENCE":
            runs[label] = _run_group(args, root, "sync_reference", ("REFERENCE",))["REFERENCE"]
        else:
            runs[label] = _run_group(args, root, "session_N_concurrent", ("ASYNC", "B"))
    async_run = runs["CONCURRENT"]["ASYNC"]
    concurrent_b = runs["CONCURRENT"]["B"]
    control_b = runs["B_CONTROL"]
    reference = runs["REFERENCE"]
    semantic = compare_async_reference(async_run["result"], reference["result"])
    b_semantic = _dual_benchmark.compare_component_semantics(control_b["result"], concurrent_b["result"], "B")
    timing = calculate_timing(async_run, concurrent_b, reference, control_b)
    memory_deltas = {"GPU2_async_minus_reference_peak_mib": async_run["gpu_peak_memory_mib"] - reference["gpu_peak_memory_mib"], "GPU3_B_concurrent_minus_control_peak_mib": concurrent_b["gpu_peak_memory_mib"] - control_b["gpu_peak_memory_mib"]}
    runtime_ok = runtime_gate(
        peak_deltas_mib=memory_deltas,
        minimum_free_mib=[async_run["gpu_min_free_mib"], reference["gpu_min_free_mib"], concurrent_b["gpu_min_free_mib"], control_b["gpu_min_free_mib"]],
        fatal_markers=[async_run["fatal_marker"], reference["fatal_marker"], concurrent_b["fatal_marker"], control_b["fatal_marker"]],
        violations=[],
    )
    schedule_ok = timing["concurrent_component_overlap_s"] >= 0 and async_run["result"]["session_N"]["training_new_ids"] == [] and async_run["result"]["session_N1"]["training_new_ids"] == async_run["result"]["kept_ids"]
    conclusion = classify_conclusion(runtime_ok=runtime_ok, semantic_ok=semantic["ok"] and b_semantic["ok"], schedule_ok=schedule_ok)
    result = {
        "classification": CLASSIFICATION,
        "conclusion": conclusion,
        "alternate_conclusions": list(CONCLUSIONS),
        "manifest_sha256": args.manifest_sha256,
        "source_commit": args.source_commit,
        "stage": args.stage,
        "repeat": args.repeat,
        "execution_order": order,
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "validation_replay_reference": "6f0625d_external_not_timed",
        "static_evidence": static,
        "semantic": {"async_vs_sync": semantic, "B_control_vs_concurrent": b_semantic},
        "schedule": {"ok": schedule_ok, "session_N": async_run["result"]["session_N"], "session_N1": async_run["result"]["session_N1"], "double_apply_rejected": async_run["result"]["double_apply_rejected"]},
        "runtime": {"ok": runtime_ok, "memory_peak_deltas_mib": memory_deltas, "peak_delta_limit_mib": 512, "min_free_mib": 4096, "monitor_interval_s": 2.0},
        "timing": timing,
        "model_projection": {"kind": "linear_schedule_model_not_full_budget_measurement", "formula": "overlap_hidden_wall_s * 22 design sessions", "projected_attempt06_savings_s": timing["overlap_hidden_wall_s"] * 22.0, "dual_overlap_not_added_elsewhere": True},
        "runs": runs,
    }
    return _harness.atomic_json(root / "ASYNC_PIPELINE_RESULT.json", result)


def aggregate_exact_six(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(stage, repeat) for stage in ("early", "mid", "late") for repeat in (0, 1)}
    keys = [(str(item.get("stage")), int(item.get("repeat", -1))) for item in results]
    if len(results) != 6 or set(keys) != expected or len(set(keys)) != 6:
        raise RuntimeError("aggregate requires exact unique early/mid/late x repeat0/1")
    conclusion = "ASYNC_PIPELINE_PASS"
    for rejected in ("REJECTED_RUNTIME", "REJECTED_SEMANTIC", "REJECTED_SCHEDULE"):
        if any(item.get("conclusion") == rejected for item in results):
            conclusion = rejected
            break
    overlaps = [float(item["timing"]["overlap_hidden_wall_s"]) for item in results]
    b_slow = [float(item["timing"]["slowdown"]["B_concurrent_vs_control"]) for item in results]
    a_slow = [float(item["timing"]["slowdown"]["async_vs_sync_reference"]) for item in results]
    return {
        "classification": CLASSIFICATION,
        "conclusion": conclusion,
        "alternate_conclusions": list(CONCLUSIONS),
        "cell_count": 6,
        "cells": [{"stage": key[0], "repeat": key[1]} for key in sorted(expected)],
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "timing": {"overlap_hidden_wall_s_total": sum(overlaps), "overlap_hidden_wall_s_mean": sum(overlaps) / 6, "async_slowdown_mean": sum(a_slow) / 6, "B_slowdown_mean": sum(b_slow) / 6},
        "model_projection": {"kind": "linear_model_not_full_budget_measurement", "formula": "mean_overlap_hidden_wall_s * 22", "projected_attempt06_savings_s": sum(overlaps) / 6 * 22.0},
    }


def run_matrix(args: Any) -> dict[str, Any]:
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(f"matrix root exists: {root}")
    root.mkdir(parents=True)
    results = []
    stages = ("early", "mid", "late") if args.stage == "all" else (args.stage,)
    repeats = (0, 1) if args.repeat == "all" else (int(args.repeat),)
    for stage in stages:
        for repeat in repeats:
            cell_args = SimpleNamespace(**vars(args))
            cell_args.stage, cell_args.repeat = stage, repeat
            results.append(run_cell(cell_args, root / f"{stage}_repeat{repeat}"))
    if len(results) == 6:
        result = aggregate_exact_six(results)
    else:
        result = {"classification": CLASSIFICATION, "conclusion": results[0]["conclusion"] if len(results) == 1 else "REJECTED_SCHEDULE", "cell_count": len(results), "cells": results, "llm_api_calls": 0, "validation_cache_exercised": False, "validation_cache_speedup_included": False}
    return _harness.atomic_json(root / "ASYNC_PIPELINE_MATRIX_RESULT.json", result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--stage", choices=("all", "early", "mid", "late"), default="all")
    parser.add_argument("--repeat", choices=("all", "0", "1"), default="all")
    parser.add_argument("--gpu2-index", type=int, default=2)
    parser.add_argument("--gpu2-uuid", default=GPU2[1])
    parser.add_argument("--gpu3-index", type=int, default=3)
    parser.add_argument("--gpu3-uuid", default=GPU3[1])
    parser.add_argument("--result-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    try:
        run_matrix(args)
    except Exception as exc:
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        _harness.atomic_json(root / "FAILURE.json", {"classification": CLASSIFICATION, "conclusion": "REJECTED_RUNTIME", "error_class": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "llm_api_calls": 0})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
