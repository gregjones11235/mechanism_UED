#!/usr/bin/env python3
"""Fail-closed control/concurrent benchmark for the dual-GPU research schedule."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_harness = _load_sibling(
    "perf48_dual_pipeline_harness_benchmark", "perf48_dual_pipeline_harness.py"
)
_pair = _load_sibling("perf48_dual_pair_base", "perf48_pair_benchmark.py")

CLASSIFICATION = _harness.CLASSIFICATION
EXPECTED_GPU = {
    "A": (2, "GPU-8df11537-ab79-722d-606f-411966196c4c"),
    "B": (3, "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"),
}
RUNTIME_MARKERS = _harness.RUNTIME_MARKERS
COMMON_FIELDS = (
    "classification",
    "not_semantic_mainline",
    "component",
    "manifest_sha256",
    "source_commit",
    "gpu_uuid",
    "stage",
    "repeat",
    "llm_api_calls",
    "validation_cache_speedup_included",
    "validation_replay_scope",
    "task_ids",
    "task_assignment_sha256",
    "task_code_hashes",
    "input_rng_sha256",
    "frozen_rng",
    "checkpoint_input_path",
    "checkpoint_input_sha256",
    "conditioning_path",
    "conditioning_file_sha256",
    "embedding_hash",
    "conditioning_type",
    "conditioning_shape",
    "conditioning_dtype",
    "config_evidence",
    "runtime_source_evidence",
    "env_evidence",
)
A_SEMANTIC_FIELDS = COMMON_FIELDS + (
    "component_scope",
    "candidate_task_load_ids",
    "candidate_task_load_sha256",
    "preflight_rng_sha256",
    "params_sha256_before",
    "optimizer_sha256_before",
    "score_projection",
    "scoring_fingerprint",
    "accepted_ids",
    "rejected_ids",
    "archive_before_sha256",
    "archive_after_sha256",
    "preflight_env_steps",
    "preflight_summary_mode",
    "preflight_return_payload_bytes",
)
B_SEMANTIC_FIELDS = COMMON_FIELDS + (
    "params_sha256_before",
    "params_sha256_after",
    "optimizer_sha256_before",
    "optimizer_sha256_after",
    "checkpoint_reloaded_params_sha256",
    "checkpoint_reloaded_optimizer_sha256",
    "checkpoint_loadable",
    "train_rng_sha256",
    "outer_rng_after_sha256",
    "input_global_update_step",
    "global_update_step",
    "global_env_steps",
    "updates",
    "env_steps",
    "scoring_fingerprint",
)
SHARED_INPUT_FIELDS = (
    "manifest_sha256",
    "source_commit",
    "stage",
    "repeat",
    "task_ids",
    "task_assignment_sha256",
    "task_code_hashes",
    "input_rng_sha256",
    "frozen_rng",
    "checkpoint_input_path",
    "checkpoint_input_sha256",
    "conditioning_path",
    "conditioning_file_sha256",
    "embedding_hash",
    "conditioning_type",
    "conditioning_shape",
    "conditioning_dtype",
    "config_evidence",
)
CONCLUSIONS = (
    "PASS",
    "REJECTED_RUNTIME_FAILURE",
    "REJECTED_SHARED_INPUT_MISMATCH",
    "REJECTED_SEMANTIC_COMPONENT_A",
    "REJECTED_SEMANTIC_COMPONENT_B",
    "REJECTED_START_SKEW",
    "REJECTED_COMPONENT_A_SLOWDOWN",
    "REJECTED_COMPONENT_B_SLOWDOWN",
    "NO_HIDDEN_WALL",
)
A_RUNTIME_SOURCES = {
    "TaskArchive": "src/dicode/dreaming/gen_manager.py",
    "load_tasks_from_env_codes": "src/dicode/task_utils.py",
    "evaluate_new_tasks": "src/dicode/evaluation/online_evaluation.py",
    "learnability_scores_from_counts": "experiments/training/run_dicode.py",
    "_load_agent_state": "src/dicode/setup.py",
    "route": "src/dicode/skill_preflight/preflight.py",
    "preflight_route": "src/dicode/skill_preflight/preflight_route.py",
    "learnability_summary_contract": (
        "src/dicode/skill_preflight/learnability_summary.py"
    ),
}
B_RUNTIME_SOURCES = {
    "TaskArchive": "src/dicode/dreaming/gen_manager.py",
    "load_tasks_from_env_codes": "src/dicode/task_utils.py",
    "run_training_session": "src/dicode/ppo_tr.py",
    "_calculate_task_distribution": "src/dicode/training.py",
    "_create_achievement_masks": "src/dicode/training.py",
    "calculate_scores_from_snapshot": "src/dicode/scoring.py",
    "_load_agent_state": "src/dicode/setup.py",
    "wrappers_cl": "src/dicode/wrappers_cl.py",
}


def _classify_component_apps(
    text: str, component_pid: int, uuid: str
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pid_text = line.split(",", 1)[0].strip()
        if not pid_text.isdigit() or uuid not in line:
            violations.append(line)
            continue
        row = _pair.classify_pid(int(pid_text), [component_pid])
        row["gpu_line"] = line
        rows.append(row)
        if row["classification"] == "external":
            violations.append(line)
    return rows, violations


def _strict_monitor_popen(
    process: subprocess.Popen[Any], index: int, uuid: str, out: Path
):
    stop = threading.Event()
    violations: list[str] = []
    csv_path = out / "gpu_memory.csv"
    evidence_path = out / "gpu_evidence.log"
    csv_path.write_text(_pair.CSV_HEADER + "\n", encoding="utf-8")

    def loop() -> None:
        while not stop.is_set():
            timestamp = time.time()
            try:
                snapshot = _pair.gpu_snapshot(index, uuid)
                apps = _pair.gpu_apps(index)
                gpu_uuid, util, used, free, gpu_index = _pair._snapshot_values(snapshot)
                rows, bad = _classify_component_apps(apps, process.pid, uuid)
                app_pid = rows[0]["pid"] if rows else ""
                app_class = rows[0]["classification"] if rows else "none"
                with csv_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{timestamp:.6f},{gpu_index},{gpu_uuid},{used},{free},"
                        f"{util},{app_pid},{app_class}\n"
                    )
                if bad or gpu_uuid != uuid:
                    found = bad or ["uuid_change"]
                    violations.extend(found)
                    with evidence_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"{timestamp:.6f} violation={found!r}\n")
            except Exception as exc:
                violations.append(str(exc))
                with evidence_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{timestamp:.6f} error={exc}\n")
            stop.wait(2.0)

    thread = threading.Thread(
        target=loop, name=f"dual-pipeline-gpu-{index}", daemon=True
    )
    thread.start()
    return stop, thread, violations


def validate_component_result(
    document: Mapping[str, Any],
    *,
    component: str,
    manifest_sha256: str,
    source_commit: str,
    gpu_uuid: str,
    stage: str,
    repeat: int,
    expected_barrier: bool = False,
) -> dict[str, Any]:
    if component not in EXPECTED_GPU:
        raise ValueError("invalid component")
    expected = {
        "classification": CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": component,
        "manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
        "gpu_uuid": gpu_uuid,
        "stage": stage,
        "repeat": repeat,
        "llm_api_calls": 0,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"invalid {component} result {key}")
    if _harness._hashed_document(document).get("result_sha256") != document.get(
        "result_sha256"
    ):
        raise RuntimeError("component result self-hash mismatch")
    fields = A_SEMANTIC_FIELDS if component == "A" else B_SEMANTIC_FIELDS
    missing = [field for field in fields if field not in document]
    if missing:
        raise RuntimeError(f"component {component} missing fields: {missing}")
    if any(bool(document.get(marker)) for marker in RUNTIME_MARKERS):
        raise RuntimeError("component runtime marker present")
    source = document.get("runtime_source_evidence")
    if not isinstance(source, Mapping) or source.get("verified") is not True:
        raise RuntimeError("runtime source evidence invalid")
    expected_sources = A_RUNTIME_SOURCES if component == "A" else B_RUNTIME_SOURCES
    if (
        set(source.get("paths", {})) != set(expected_sources)
        or set(source.get("hashes", {})) != set(expected_sources)
        or source.get("expected_relatives") != expected_sources
    ):
        raise RuntimeError("runtime source evidence incomplete")
    env = document.get("env_evidence")
    if not isinstance(env, Mapping) or not env.get("jax_version"):
        raise RuntimeError("environment evidence invalid")
    if not document.get("compile_cache_dir"):
        raise RuntimeError("independent compile cache evidence missing")
    if document.get("validation_replay_scope") != "not_executed_not_timed":
        raise RuntimeError("validation replay timing scope mismatch")
    barrier = document.get("barrier")
    if not isinstance(barrier, Mapping) or barrier.get("enabled") is not expected_barrier:
        raise RuntimeError("component barrier evidence mismatch")
    if expected_barrier:
        required_barrier = (
            "barrier_id",
            "ready_sha256",
            "ready_monotonic_ns",
            "go_sha256",
            "go_monotonic_ns",
            "go_observed_monotonic_ns",
        )
        if any(not barrier.get(field) for field in required_barrier):
            raise RuntimeError("component concurrent barrier receipt incomplete")
    elif barrier.get("mode") != "control_direct":
        raise RuntimeError("component control barrier mode mismatch")
    started = document.get("component_started_monotonic_ns")
    ended = document.get("component_ended_monotonic_ns")
    wall = document.get("component_wall_s")
    if (
        not isinstance(started, int)
        or not isinstance(ended, int)
        or ended <= started
        or not isinstance(wall, (int, float))
        or abs(float(wall) - (ended - started) / 1e9) > 1e-9
    ):
        raise RuntimeError("component monotonic timing evidence invalid")
    if expected_barrier and started < barrier["go_observed_monotonic_ns"]:
        raise RuntimeError("component workload started before GO observation")
    if component == "A":
        if document.get("component_scope") != "fused_preflight_only":
            raise RuntimeError("component A scope mismatch")
        if document.get("candidate_task_load_ids") != document.get("task_ids"):
            raise RuntimeError("component A task-load order mismatch")
        if document.get("preflight_summary_mode") != "fused":
            raise RuntimeError("component A did not use fused preflight")
        if document.get("preflight_env_steps") != 40 * 1024 * 128:
            raise RuntimeError("component A env-step mismatch")
        if document.get("validation_cache_speedup_included") is not False:
            raise RuntimeError("component A validation-cache claim mismatch")
        accepted = document.get("accepted_ids", [])
        rejected = document.get("rejected_ids", [])
        if (
            len(accepted) != len(set(accepted))
            or len(rejected) != len(set(rejected))
            or set(accepted) & set(rejected)
            or set(accepted) | set(rejected) != set(document["task_ids"])
            or [row.get("task_id") for row in document["score_projection"]]
            != document["task_ids"]
        ):
            raise RuntimeError("component A score/route partition mismatch")
    else:
        if document.get("updates") != 100 or document.get("env_steps") != 13_107_200:
            raise RuntimeError("component B 100-update accounting mismatch")
        if document.get("checkpoint_loadable") is not True:
            raise RuntimeError("component B checkpoint not loadable")
        if document.get("params_sha256_after") != document.get(
            "checkpoint_reloaded_params_sha256"
        ) or document.get("optimizer_sha256_after") != document.get(
            "checkpoint_reloaded_optimizer_sha256"
        ):
            raise RuntimeError("component B checkpoint semantic mismatch")
        if document.get("global_update_step") != document.get(
            "input_global_update_step"
        ) + 100:
            raise RuntimeError("component B global-step mismatch")
    return dict(document)


def compare_component_semantics(
    control: Mapping[str, Any], concurrent: Mapping[str, Any], component: str
) -> dict[str, Any]:
    fields = A_SEMANTIC_FIELDS if component == "A" else B_SEMANTIC_FIELDS
    differences = {
        field: {"control": control.get(field), "concurrent": concurrent.get(field)}
        for field in fields
        if control.get(field) != concurrent.get(field)
    }
    return {"ok": not differences, "differences": differences, "fields": list(fields)}


def compare_shared_inputs(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    differences = {
        field: {"A": a.get(field), "B": b.get(field)}
        for field in SHARED_INPUT_FIELDS
        if a.get(field) != b.get(field)
    }
    return {
        "ok": not differences,
        "differences": differences,
        "fields": list(SHARED_INPUT_FIELDS),
    }


def judge(
    *,
    runtime_ok: bool,
    shared_ok: bool,
    semantic_a_ok: bool,
    semantic_b_ok: bool,
    start_skew_s: float,
    slowdown_a: float,
    slowdown_b: float,
    hidden_wall_s: float,
) -> str:
    if not runtime_ok:
        return "REJECTED_RUNTIME_FAILURE"
    if not shared_ok:
        return "REJECTED_SHARED_INPUT_MISMATCH"
    if not semantic_a_ok:
        return "REJECTED_SEMANTIC_COMPONENT_A"
    if not semantic_b_ok:
        return "REJECTED_SEMANTIC_COMPONENT_B"
    if start_skew_s > 2.0:
        return "REJECTED_START_SKEW"
    if slowdown_a > 0.10:
        return "REJECTED_COMPONENT_A_SLOWDOWN"
    if slowdown_b > 0.10:
        return "REJECTED_COMPONENT_B_SLOWDOWN"
    if hidden_wall_s < 400.0:
        return "NO_HIDDEN_WALL"
    return "PASS"


def _prepare_output(root: Path, label: str, component: str) -> Path:
    out = root / label / component
    if out.exists():
        raise FileExistsError(f"component output exists: {out}")
    out.mkdir(parents=True)
    for subdir in ("tmp", "cache", "wandb", "jax_compilation"):
        (out / subdir).mkdir()
    return out


def _component_env(args: Any, component: str, out: Path) -> dict[str, str]:
    env = dict(os.environ)
    uuid = args.gpu_a_uuid if component == "A" else args.gpu_b_uuid
    env.update(
        CUDA_VISIBLE_DEVICES=uuid,
        WANDB_MODE="offline",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(Path(args.source) / "src"),
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


def _command(args: Any, component: str, out: Path) -> list[str]:
    uuid = args.gpu_a_uuid if component == "A" else args.gpu_b_uuid
    command = [
        args.python,
        args.harness,
        "--manifest",
        args.manifest,
        "--config",
        args.config,
        "--out",
        str(out),
        "--required-gpu-uuid",
        uuid,
        "--source-commit",
        args.source_commit,
        "--stage",
        args.stage,
        "--repeat",
        str(args.repeat),
        "--component",
        component,
        "--mode",
        "run",
    ]
    barrier = getattr(args, "active_barrier", None)
    if barrier is not None:
        command.extend(
            [
                "--ready-path",
                str(barrier["ready_paths"][component]),
                "--go-path",
                str(barrier["go_path"]),
                "--barrier-id",
                barrier["barrier_id"],
                "--barrier-timeout-s",
                str(barrier["timeout_s"]),
            ]
        )
    return command


def _make_barrier(args: Any, root: Path, label: str) -> dict[str, Any]:
    barrier_dir = root / label / "barrier"
    barrier_dir.mkdir(parents=True)
    barrier_id = _harness._fingerprint(
        {
            "root": str(root.resolve()),
            "label": label,
            "manifest_sha256": args.manifest_sha256,
            "source_commit": args.source_commit,
            "stage": args.stage,
            "repeat": args.repeat,
        }
    )
    return {
        "barrier_id": barrier_id,
        "ready_paths": {
            component: barrier_dir / f"READY_{component}.json"
            for component in ("A", "B")
        },
        "go_path": barrier_dir / "GO.json",
        "timeout_s": float(args.barrier_timeout_s),
    }


def _release_barrier(
    barrier: Mapping[str, Any],
    processes: Mapping[str, subprocess.Popen[Any]],
    evidence: Mapping[str, dict[str, Any]],
    monitors: Mapping[str, tuple[Any, Any, list[str]]],
) -> dict[str, Any]:
    deadline = time.monotonic() + float(barrier["timeout_s"])
    ready: dict[str, dict[str, Any]] = {}
    while len(ready) != 2:
        if time.monotonic() >= deadline:
            raise TimeoutError("parent READY barrier timeout")
        for component in ("A", "B"):
            if component in ready:
                continue
            process = processes[component]
            if process.poll() is not None:
                raise RuntimeError(f"component {component} exited before READY")
            if monitors[component][2]:
                raise RuntimeError(f"component {component} GPU violation before READY")
            out = Path(evidence[component]["out"])
            fatal = _pair.fatal_in([out / "harness.stdout", out / "harness.stderr"])
            if fatal:
                raise RuntimeError(f"component {component} fatal before READY: {fatal}")
            _, minimum_free = _pair.arm_gpu_metrics(out / "gpu_memory.csv")
            if minimum_free is not None and minimum_free < 4096:
                raise RuntimeError(f"component {component} below 4GiB before READY")
            path = Path(barrier["ready_paths"][component])
            if not path.is_file():
                continue
            document = _harness.load_hashed_json(path)
            if (
                document.get("classification") != CLASSIFICATION
                or document.get("barrier_id") != barrier["barrier_id"]
                or document.get("component") != component
                or document.get("pid") != process.pid
                or document.get("llm_api_calls") != 0
                or not isinstance(document.get("ready_monotonic_ns"), int)
            ):
                raise RuntimeError(f"component {component} READY receipt invalid")
            ready[component] = document
        time.sleep(0.05)
    go_path = Path(barrier["go_path"])
    if go_path.exists():
        raise FileExistsError("GO barrier already exists")
    go_ns = time.monotonic_ns()
    go = _harness.atomic_json(
        go_path,
        {
            "classification": CLASSIFICATION,
            "barrier_id": barrier["barrier_id"],
            "components": ["A", "B"],
            "go_monotonic_ns": go_ns,
            "ready_sha256": {
                component: ready[component]["result_sha256"]
                for component in ("A", "B")
            },
            "llm_api_calls": 0,
        },
    )
    return {
        "barrier_id": barrier["barrier_id"],
        "ready": ready,
        "go": go,
        "timeout_s": barrier["timeout_s"],
    }


def _verify_barrier_link(
    result: Mapping[str, Any], parent: Mapping[str, Any], component: str
) -> None:
    child = result.get("barrier", {})
    ready = parent.get("ready", {}).get(component, {})
    go = parent.get("go", {})
    if (
        child.get("barrier_id") != parent.get("barrier_id")
        or child.get("ready_sha256") != ready.get("result_sha256")
        or child.get("ready_monotonic_ns") != ready.get("ready_monotonic_ns")
        or child.get("go_sha256") != go.get("result_sha256")
        or child.get("go_monotonic_ns") != go.get("go_monotonic_ns")
        or go.get("ready_sha256", {}).get(component) != ready.get("result_sha256")
    ):
        raise RuntimeError(f"component {component} parent/child barrier mismatch")


def _launch_group(
    args: Any, root: Path, label: str, components: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    processes: dict[str, subprocess.Popen[Any]] = {}
    streams: dict[str, tuple[Any, Any]] = {}
    monitors: dict[str, tuple[Any, Any, list[str]]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    runtime_error: str | None = None
    barrier = _make_barrier(args, root, label) if len(components) == 2 else None
    args.active_barrier = barrier
    try:
        for component in components:
            index, uuid = EXPECTED_GPU[component]
            actual_index = args.gpu_a_index if component == "A" else args.gpu_b_index
            actual_uuid = args.gpu_a_uuid if component == "A" else args.gpu_b_uuid
            if (actual_index, actual_uuid) != (index, uuid):
                raise RuntimeError(f"component {component} exact GPU contract mismatch")
            _pair.assert_gpu_free(actual_index, actual_uuid)
        for component in components:
            out = _prepare_output(root, label, component)
            stdout = (out / "harness.stdout").open("w", encoding="utf-8")
            stderr = (out / "harness.stderr").open("w", encoding="utf-8")
            streams[component] = (stdout, stderr)
            command = _command(args, component, out)
            started_ns = time.monotonic_ns()
            process = subprocess.Popen(
                command,
                env=_component_env(args, component, out),
                text=True,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            processes[component] = process
            index = args.gpu_a_index if component == "A" else args.gpu_b_index
            uuid = args.gpu_a_uuid if component == "A" else args.gpu_b_uuid
            monitors[component] = _strict_monitor_popen(process, index, uuid, out)
            evidence[component] = {
                "component": component,
                "pid": process.pid,
                "argv": command,
                "started_monotonic_ns": started_ns,
                "out": str(out),
                "compile_cache_dir": str(out / "jax_compilation"),
            }
        if barrier is not None:
            parent_barrier = _release_barrier(
                barrier, processes, evidence, monitors
            )
            for component in components:
                evidence[component]["parent_barrier"] = parent_barrier
        while any(process.poll() is None for process in processes.values()):
            for component, process in processes.items():
                if (
                    process.poll() is not None
                    and "ended_monotonic_ns" not in evidence[component]
                ):
                    evidence[component]["ended_monotonic_ns"] = time.monotonic_ns()
                out = Path(evidence[component]["out"])
                stop, thread, violations = monitors[component]
                fatal = _pair.fatal_in(
                    [out / "harness.stdout", out / "harness.stderr"]
                )
                _, minimum_free = _pair.arm_gpu_metrics(out / "gpu_memory.csv")
                if violations:
                    runtime_error = f"{component} GPU violation: {violations}"
                elif minimum_free is not None and minimum_free < 4096:
                    runtime_error = f"{component} GPU free memory below 4GiB"
                elif fatal:
                    runtime_error = f"{component} fatal output: {fatal}"
                elif process.poll() not in (None, 0):
                    runtime_error = f"{component} nonzero exit: {process.returncode}"
                if runtime_error:
                    break
            if runtime_error:
                for process in processes.values():
                    if process.poll() is None:
                        _pair.stop_owned(process.pid)
                break
            time.sleep(0.2)
        for component, process in processes.items():
            process.wait(timeout=10)
            evidence[component].setdefault("ended_monotonic_ns", time.monotonic_ns())
            evidence[component]["process_wall_s"] = (
                evidence[component]["ended_monotonic_ns"]
                - evidence[component]["started_monotonic_ns"]
            ) / 1e9
            evidence[component]["returncode"] = process.returncode
            out = Path(evidence[component]["out"])
            fatal = _pair.fatal_in([out / "harness.stdout", out / "harness.stderr"])
            peak, minimum = _pair.arm_gpu_metrics(out / "gpu_memory.csv")
            evidence[component].update(
                gpu_peak_memory_mib=peak,
                gpu_min_free_mib=minimum,
                fatal_marker=fatal,
                monitor_interval_s=2.0,
            )
            if process.returncode or fatal or minimum is None or minimum < 4096:
                runtime_error = runtime_error or f"{component} runtime safety gate failed"
            violations = monitors[component][2]
            if violations:
                runtime_error = runtime_error or f"{component} GPU violations: {violations}"
        if runtime_error:
            raise RuntimeError(runtime_error)
        for component in components:
            out = Path(evidence[component]["out"])
            result_path = out / "RESULT.json"
            if not result_path.is_file():
                raise RuntimeError(f"component {component} missing RESULT.json")
            document = _harness.load_hashed_json(result_path)
            uuid = args.gpu_a_uuid if component == "A" else args.gpu_b_uuid
            evidence[component]["result"] = validate_component_result(
                document,
                component=component,
                manifest_sha256=args.manifest_sha256,
                source_commit=args.source_commit,
                gpu_uuid=uuid,
                stage=args.stage,
                repeat=args.repeat,
                expected_barrier=barrier is not None,
            )
            if barrier is not None:
                _verify_barrier_link(
                    evidence[component]["result"],
                    evidence[component]["parent_barrier"],
                    component,
                )
        return evidence
    finally:
        args.active_barrier = None
        for process in processes.values():
            if process.poll() is None:
                _pair.stop_owned(process.pid)
        for stop, thread, violations in monitors.values():
            stop.set()
            thread.join(timeout=5)
        for stdout, stderr in streams.values():
            stdout.close()
            stderr.close()


def _static_gate(args: Any) -> dict[str, Any]:
    manifest, config, config_evidence = _harness.load_inputs(args.manifest, args.config)
    dual_metadata = _harness.verify_dual_manifest(manifest, args.source_commit)
    _harness._stage_repeat(manifest, args.stage, args.repeat)
    args.manifest_sha256 = manifest["manifest_sha256"]
    harness = Path(args.harness).resolve()
    source = Path(args.source).resolve()
    expected = (source / "experiments/performance/perf48_dual_pipeline_harness.py").resolve()
    if harness != expected or not harness.is_file():
        raise RuntimeError("dual harness/source binding mismatch")
    if Path(dual_metadata["source_root"]).resolve() != source:
        raise RuntimeError("dual manifest/source root mismatch")
    if args.gpu_a_index == args.gpu_b_index or args.gpu_a_uuid == args.gpu_b_uuid:
        raise RuntimeError("dual components require distinct GPUs")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": str(Path(args.manifest).resolve()),
        "config": config_evidence,
        "harness_path": str(harness),
        "harness_sha256": _harness._file_sha256(harness),
        "source_root": str(source),
        "dual_manifest": dual_metadata,
    }


def run_benchmark(args: Any) -> dict[str, Any]:
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(f"benchmark root exists: {root}")
    root.mkdir(parents=True)
    static: dict[str, Any] = {}
    try:
        static = _static_gate(args)
        control_a = _launch_group(args, root, "control_A", ("A",))["A"]
        control_b = _launch_group(args, root, "control_B", ("B",))["B"]
        concurrent = _launch_group(args, root, "concurrent", ("A", "B"))
        semantic_a = compare_component_semantics(
            control_a["result"], concurrent["A"]["result"], "A"
        )
        semantic_b = compare_component_semantics(
            control_b["result"], concurrent["B"]["result"], "B"
        )
        shared = compare_shared_inputs(control_a["result"], control_b["result"])
        shared_concurrent = compare_shared_inputs(
            concurrent["A"]["result"], concurrent["B"]["result"]
        )
        shared_ok = shared["ok"] and shared_concurrent["ok"]
        launch_skew_s = abs(
            concurrent["A"]["started_monotonic_ns"]
            - concurrent["B"]["started_monotonic_ns"]
        ) / 1e9
        start_skew_s = abs(
            concurrent["A"]["result"]["component_started_monotonic_ns"]
            - concurrent["B"]["result"]["component_started_monotonic_ns"]
        ) / 1e9
        concurrent_wall_s = (
            max(
                concurrent["A"]["ended_monotonic_ns"],
                concurrent["B"]["ended_monotonic_ns"],
            )
            - min(
                concurrent["A"]["started_monotonic_ns"],
                concurrent["B"]["started_monotonic_ns"],
            )
        ) / 1e9
        control_a_wall = float(control_a["result"]["component_wall_s"])
        control_b_wall = float(control_b["result"]["component_wall_s"])
        concurrent_a_wall = float(concurrent["A"]["result"]["component_wall_s"])
        concurrent_b_wall = float(concurrent["B"]["result"]["component_wall_s"])
        concurrent_component_wall_s = (
            max(
                concurrent["A"]["result"]["component_ended_monotonic_ns"],
                concurrent["B"]["result"]["component_ended_monotonic_ns"],
            )
            - min(
                concurrent["A"]["result"]["component_started_monotonic_ns"],
                concurrent["B"]["result"]["component_started_monotonic_ns"],
            )
        ) / 1e9
        slowdown_a = (concurrent_a_wall - control_a_wall) / control_a_wall
        slowdown_b = (concurrent_b_wall - control_b_wall) / control_b_wall
        hidden_wall_s = control_a_wall + control_b_wall - concurrent_component_wall_s
        memory_deltas = {
            "A": concurrent["A"]["gpu_peak_memory_mib"]
            - control_a["gpu_peak_memory_mib"],
            "B": concurrent["B"]["gpu_peak_memory_mib"]
            - control_b["gpu_peak_memory_mib"],
        }
        runtime_ok = (
            max(memory_deltas.values()) <= 512
            and min(
                control_a["gpu_min_free_mib"],
                control_b["gpu_min_free_mib"],
                concurrent["A"]["gpu_min_free_mib"],
                concurrent["B"]["gpu_min_free_mib"],
            )
            >= 4096
        )
        conclusion = judge(
            runtime_ok=runtime_ok,
            shared_ok=shared_ok,
            semantic_a_ok=semantic_a["ok"],
            semantic_b_ok=semantic_b["ok"],
            start_skew_s=start_skew_s,
            slowdown_a=slowdown_a,
            slowdown_b=slowdown_b,
            hidden_wall_s=hidden_wall_s,
        )
        result = {
            "classification": CLASSIFICATION,
            "conclusion": conclusion,
            "alternate_conclusions": list(CONCLUSIONS),
            "manifest_sha256": args.manifest_sha256,
            "source_commit": args.source_commit,
            "stage": args.stage,
            "repeat": args.repeat,
            "llm_api_calls": 0,
            "validation_cache_speedup_included": False,
            "validation_replay_scope": "not_executed_not_timed",
            "component_A_scope": "fused_preflight_only",
            "static_evidence": static,
            "controls": {"A": control_a, "B": control_b},
            "concurrent": concurrent,
            "semantic": {"A": semantic_a, "B": semantic_b},
            "shared_inputs": {"control": shared, "concurrent": shared_concurrent},
            "timing": {
                "start_skew_s": start_skew_s,
                "component_start_skew_s": start_skew_s,
                "process_launch_skew_s": launch_skew_s,
                "start_skew_limit_s": 2.0,
                "timing_basis": "component_monotonic",
                "control_A_component_wall_s": control_a_wall,
                "control_B_component_wall_s": control_b_wall,
                "concurrent_A_component_wall_s": concurrent_a_wall,
                "concurrent_B_component_wall_s": concurrent_b_wall,
                "concurrent_component_makespan_wall_s": concurrent_component_wall_s,
                "component_A_slowdown": slowdown_a,
                "component_B_slowdown": slowdown_b,
                "slowdown_limit": 0.10,
                "hidden_wall_s": hidden_wall_s,
                "hidden_wall_formula": (
                    "control_A_component_wall_s + control_B_component_wall_s - "
                    "concurrent_component_makespan_wall_s"
                ),
                "hidden_wall_min_s": 400.0,
                "operational_process_wall_s": {
                    "control_A": float(control_a["process_wall_s"]),
                    "control_B": float(control_b["process_wall_s"]),
                    "concurrent_A": float(concurrent["A"]["process_wall_s"]),
                    "concurrent_B": float(concurrent["B"]["process_wall_s"]),
                    "concurrent_makespan": concurrent_wall_s,
                },
            },
            "barrier": concurrent["A"].get("parent_barrier"),
            "gpu_safety": {
                "ok": runtime_ok,
                "concurrent_minus_control_peak_mib": memory_deltas,
                "max_peak_delta_mib": 512,
                "min_free_mib": 4096,
                "monitor_interval_s": 2.0,
            },
            "pass_contract": (
                "PASS iff skew<=2s, each slowdown<=10%, hidden_wall>=400s, "
                "semantic/shared-input/runtime/GPU gates all pass"
            ),
        }
        return _harness.atomic_json(root / "DUAL_PIPELINE_RESULT.json", result)
    except Exception as exc:
        _harness.atomic_json(
            root / "FAILURE.json",
            {
                "classification": CLASSIFICATION,
                "conclusion": "REJECTED_RUNTIME_FAILURE",
                "error_class": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "static_evidence": static,
                "llm_api_calls": 0,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=("early", "mid", "late"), required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu-a-index", type=int, required=True)
    parser.add_argument("--gpu-a-uuid", required=True)
    parser.add_argument("--gpu-b-index", type=int, required=True)
    parser.add_argument("--gpu-b-uuid", required=True)
    parser.add_argument("--barrier-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
