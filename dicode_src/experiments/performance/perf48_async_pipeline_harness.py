#!/usr/bin/env python3
"""Frozen N/N+1 acceptance harness for the production async preflight manager."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_dual = _load_sibling("perf48_async_dual", "perf48_dual_pipeline_harness.py")
_deploy = _load_sibling("perf48_async_deploy", "perf48_async_pipeline_deploy.py")

CLASSIFICATION = "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE"
COMPONENTS = ("ASYNC", "REFERENCE", "B")
CONCLUSIONS = (
    "ASYNC_PIPELINE_PASS",
    "REJECTED_SEMANTIC",
    "REJECTED_RUNTIME",
    "REJECTED_SCHEDULE",
)
RUNTIME_MARKERS = _dual.RUNTIME_MARKERS
atomic_json = _dual.atomic_json
load_hashed_json = _dual.load_hashed_json
fingerprint = _dual._fingerprint
file_sha256 = _dual._file_sha256


def verify_async_manifest(manifest: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    metadata = manifest.get("async_pipeline")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("async-pipeline manifest metadata missing")
    if metadata.get("classification") != CLASSIFICATION or metadata.get("source_commit") != source_commit:
        raise RuntimeError("async-pipeline identity mismatch")
    entries = manifest.get("source_config", {}).get("source", {})
    required = metadata.get("required_source_files")
    if not isinstance(required, list) or set(required) != set(entries):
        raise RuntimeError("async-pipeline source set mismatch")
    if not set(_deploy.ASYNC_REQUIRED_SOURCE_FILES).issubset(entries):
        raise RuntimeError("async-pipeline source binding incomplete")
    for relative, entry in entries.items():
        path = Path(entry.get("path", "")).resolve()
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"async source hash mismatch: {relative}")
    parent = metadata.get("parent_manifest")
    if not isinstance(parent, Mapping):
        raise RuntimeError("async parent binding missing")
    parent_path = Path(parent.get("path", "")).resolve()
    if not parent_path.is_file() or file_sha256(parent_path) != parent.get("file_sha256"):
        raise RuntimeError("async parent file binding mismatch")
    loaded_parent = _dual._fast.load_manifest(parent_path)
    if loaded_parent.get("manifest_sha256") != parent.get("manifest_sha256"):
        raise RuntimeError("async parent manifest hash mismatch")
    if _deploy.canonical_manifest_sha256(loaded_parent) != parent.get("canonical_sha256"):
        raise RuntimeError("async parent canonical hash mismatch")
    matrix = metadata.get("matrix")
    if matrix != {"stages": ["early", "mid", "late"], "repeats": [0, 1], "count": 6}:
        raise RuntimeError("async exact-six matrix mismatch")
    cache = metadata.get("validation_cache")
    if not isinstance(cache, Mapping) or cache.get("exercised") is not False or cache.get("speedup_included") is not False:
        raise RuntimeError("validation-cache scope mismatch")
    return dict(metadata)


def load_inputs(manifest_path: str | Path, config_path: str | Path, source_commit: str):
    manifest, config, config_evidence = _dual.load_inputs(manifest_path, config_path)
    verify_async_manifest(manifest, source_commit)
    return manifest, config, config_evidence


def stage_repeat(manifest: Mapping[str, Any], stage: str, repeat: int):
    return _dual._stage_repeat(manifest, stage, repeat)


def candidate_rows(stage: Mapping[str, Any], archive: Any) -> list[dict[str, str]]:
    task_ids = [str(value) for value in stage["task_ids"]]
    code_map = archive.get_task_codes(task_ids)
    if list(code_map) != task_ids:
        raise RuntimeError("candidate code order mismatch")
    expected = {str(row["id"]): str(row["code_sha256"]) for row in stage["tasks"]}
    rows = []
    for task_id in task_ids:
        actual = hashlib.sha256(code_map[task_id].encode()).hexdigest()
        if actual != expected.get(task_id):
            raise RuntimeError(f"candidate code hash mismatch: {task_id}")
        rows.append({"task_id": task_id, "code_sha256": actual})
    return rows


def _source_binding(obj: Any, manifest: Mapping[str, Any], relative: str) -> dict[str, str]:
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    entry = manifest["source_config"]["source"].get(relative)
    if not isinstance(entry, Mapping) or path != Path(entry.get("path", "")).resolve() or file_sha256(path) != entry.get("sha256"):
        raise RuntimeError(f"runtime source binding mismatch: {relative}")
    return {"path": str(path), "sha256": entry["sha256"], "relative": relative}


class FrozenCheckpointManager:
    def __init__(self, checkpoint_root: str | Path, expected_step: int):
        self.root = Path(checkpoint_root)
        self.expected_step = int(expected_step)

    def wait_until_finished(self):
        return None

    def check_for_errors(self):
        return None

    def item_metadata(self, step: int):
        path = self.root / str(int(step))
        if int(step) != self.expected_step or not path.is_dir() or not any(item.is_file() for item in path.rglob("*")):
            raise RuntimeError("exact frozen checkpoint unavailable")
        return {"step": int(step), "path": str(path.resolve())}


class ApplyOnce:
    """Acceptance-controller guard around the production poll/apply operation."""

    def __init__(self):
        self.applied = False
        self.route_calls = 0

    def poll(self, manager: Any, *, archive: Any, session: int, route_fn: Callable[..., Any], route_apply_fn: Callable[..., Any]):
        if self.applied:
            raise RuntimeError("async result double apply rejected")

        def counted(scores, ids, kept, live, decision):
            self.route_calls += 1
            return route_apply_fn(scores, ids, kept, live, decision)

        kept = manager.poll_and_apply(
            archive=archive,
            current_session_idx=session,
            route_apply_fn=counted,
            route_fn=route_fn,
        )
        if kept is not None:
            self.applied = True
        return kept


def _async_config(config: Any, root: Path, gpu_uuid: str) -> Any:
    from omegaconf import OmegaConf

    updated = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    updated.training.use_wandb = False
    updated.performance.async_preflight_pipeline = True
    updated.performance.async_preflight_gpu_uuid = gpu_uuid
    updated.performance.async_preflight_root = str((root / "jobs").resolve())
    updated.performance.async_preflight_result_timeout_s = 0
    updated.performance.async_preflight_shutdown_timeout_s = 120
    return updated


def production_async_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    import jax
    from dicode.skill_preflight.async_preflight import (
        AsyncPreflightManager,
        load_hashed_json as load_async_receipt,
        plan_async_session,
    )
    from dicode.skill_preflight.preflight import route
    from dicode.skill_preflight.preflight_route import preflight_route

    controller_backend = jax.default_backend()
    if controller_backend != "cpu":
        raise RuntimeError("async controller runtime must be CPU")
    return {
        "jax": jax,
        "controller_backend": controller_backend,
        "array_rng": lambda value: jax.numpy.asarray(np.asarray(value, dtype=np.uint32)),
        "split_rng": jax.random.split,
        "rng_hash": _dual._fast.rng_hash,
        "reconstruct_archive": _dual._fast._reconstruct_archive,
        "archive_hash": _dual._fast._graph_sha,
        "AsyncPreflightManager": AsyncPreflightManager,
        "load_async_receipt": load_async_receipt,
        "plan_async_session": plan_async_session,
        "route": route,
        "preflight_route": preflight_route,
        "source_evidence": {
            "AsyncPreflightManager": _source_binding(AsyncPreflightManager, manifest, "src/dicode/skill_preflight/async_preflight.py"),
            "plan_async_session": _source_binding(plan_async_session, manifest, "src/dicode/skill_preflight/async_preflight.py"),
            "preflight_route": _source_binding(preflight_route, manifest, "src/dicode/skill_preflight/preflight_route.py"),
        },
    }


def launch_worker_without_cpu_platform(manager: Any, **kwargs: Any) -> Path:
    """Let only the fresh worker discover its exact visible GPU, then restore parent env."""
    missing = object()
    parent_platform = os.environ.pop("JAX_PLATFORMS", missing)
    try:
        return manager.launch(**kwargs)
    finally:
        if parent_platform is missing:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = str(parent_platform)


def run_async_controller(manifest: Mapping[str, Any], config: Any, config_evidence: Mapping[str, str], runtime: Mapping[str, Any], out: Path, args: Any) -> dict[str, Any]:
    stage, repeat = stage_repeat(manifest, args.stage, args.repeat)
    started = time.monotonic_ns()
    archive = runtime["reconstruct_archive"](stage["graph"]["path"])
    rows = candidate_rows(stage, archive)
    task_ids = [row["task_id"] for row in rows]
    archive_before = runtime["archive_hash"](archive)
    input_rng = runtime["array_rng"](repeat["rng"])
    main_rng, pf_rng = runtime["split_rng"](input_rng)
    plan_n = runtime["plan_async_session"](
        async_enabled=True, delayed_ids=[], fresh_ids=task_ids, pending=False
    )
    if plan_n != {"training_new_ids": [], "launch_ids": task_ids}:
        raise RuntimeError("session N scheduling contract mismatch")
    async_config = _async_config(config, out, args.required_gpu_uuid)
    manager = runtime["AsyncPreflightManager"](
        async_config,
        source_root=Path(args.source),
    )
    if runtime.get("controller_backend") != "cpu":
        raise RuntimeError("async controller backend must be CPU")
    checkpoint = Path(stage["checkpoint"]["path"])
    launch_ns = time.monotonic_ns()
    job_dir = launch_worker_without_cpu_platform(
        manager,
        session_idx=0,
        global_update_step=int(stage["global_step"]),
        task_ids=task_ids,
        pf_rng=pf_rng,
        archive=archive,
        rl_ckpt_manager=FrozenCheckpointManager(checkpoint.parent, int(stage["global_step"])),
        rl_ckpt_path=checkpoint.parent,
    )
    guard = ApplyOnce()
    kept = None
    deadline = time.monotonic() + float(args.result_timeout_s)
    while kept is None:
        kept = guard.poll(
            manager,
            archive=archive,
            session=1,
            route_fn=runtime["route"],
            route_apply_fn=runtime["preflight_route"],
        )
        if kept is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("async worker result timeout")
            time.sleep(0.2)
    plan_n1 = runtime["plan_async_session"](
        async_enabled=True, delayed_ids=kept, fresh_ids=[], pending=False
    )
    if plan_n1 != {"training_new_ids": list(kept), "launch_ids": []}:
        raise RuntimeError("session N+1 scheduling contract mismatch")
    double_apply_rejected = False
    try:
        guard.poll(
            manager,
            archive=archive,
            session=1,
            route_fn=runtime["route"],
            route_apply_fn=runtime["preflight_route"],
        )
    except RuntimeError as exc:
        double_apply_rejected = "double apply rejected" in str(exc)
    if not double_apply_rejected or guard.route_calls != 1:
        raise RuntimeError("double-apply/route-once contract mismatch")
    receipts = {
        name: runtime["load_async_receipt"](job_dir / f"{name}.json")
        for name in ("JOB", "RUNNING", "RESULT", "APPLYING", "APPLIED")
    }
    result = receipts["RESULT"]
    if result.get("jax_backend") != "gpu" or result.get("jax_device_count") != 1:
        raise RuntimeError("async worker gpu/device1 contract mismatch")
    if result.get("route_calls") != 0 or receipts["APPLIED"].get("route_calls") != 1:
        raise RuntimeError("worker/main route contract mismatch")
    ended = time.monotonic_ns()
    document = {
        "classification": CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": "ASYNC",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "stage": args.stage,
        "repeat": args.repeat,
        "gpu_uuid": args.required_gpu_uuid,
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "validation_replay_reference": "6f0625d_external_not_timed",
        "task_ids": task_ids,
        "task_code_rows": rows,
        "input_rng_sha256": runtime["rng_hash"](input_rng),
        "main_rng_after_split_sha256": runtime["rng_hash"](main_rng),
        "preflight_rng_sha256": runtime["rng_hash"](pf_rng),
        "checkpoint_input_path": stage["checkpoint"]["path"],
        "checkpoint_input_sha256": stage["checkpoint"]["sha256"],
        "graph_input_path": stage["graph"]["path"],
        "graph_input_sha256": stage["graph"]["sha256"],
        "config_evidence": dict(config_evidence),
        "session_N": {"fresh_ids": task_ids, **plan_n},
        "session_N1": {"fresh_ids": [], "delayed_kept_ids": list(kept), **plan_n1},
        "double_apply_rejected": True,
        "route_calls": guard.route_calls,
        "score_projection": result["score_projection"],
        "score_fingerprint": result["score_fingerprint"],
        "kept_ids": list(kept),
        "rejected_ids": [task_id for task_id in task_ids if task_id not in kept],
        "archive_before_sha256": archive_before,
        "archive_after_sha256": runtime["archive_hash"](archive),
        "worker_gpu_preflight": result["gpu_preflight"],
        "controller_backend": runtime["controller_backend"],
        "worker_jax_backend": result["jax_backend"],
        "worker_jax_device_count": result["jax_device_count"],
        "worker_route_calls": result["route_calls"],
        "main_route_calls": receipts["APPLIED"]["route_calls"],
        "receipt_sha256": {name: value["result_sha256"] for name, value in receipts.items()},
        "runtime_source_evidence": runtime["source_evidence"],
        "component_started_monotonic_ns": started,
        "worker_launched_monotonic_ns": launch_ns,
        "component_ended_monotonic_ns": ended,
        "component_wall_s": (ended - started) / 1e9,
        **{marker: False for marker in RUNTIME_MARKERS},
    }
    return atomic_json(out / "RESULT.json", document)


def production_reference_runtime(manifest: Mapping[str, Any], out: Path) -> dict[str, Any]:
    runtime = _dual._production_runtime_a(manifest, out)
    return runtime


def run_sync_reference(manifest: Mapping[str, Any], config: Any, config_evidence: Mapping[str, str], runtime: Mapping[str, Any], out: Path, args: Any) -> dict[str, Any]:
    stage, repeat = stage_repeat(manifest, args.stage, args.repeat)
    started = time.monotonic_ns()
    train_state = runtime["_load_agent_state"](config, stage["checkpoint"]["path"])
    input_rng = runtime["array_rng"](repeat["rng"])
    _, pf_rng = runtime["split_rng"](input_rng)
    archive = runtime["reconstruct_archive"](stage["graph"]["path"])
    rows = candidate_rows(stage, archive)
    task_ids = [row["task_id"] for row in rows]
    before = runtime["archive_hash"](archive)
    classes, ok_ids = runtime["load_tasks_from_env_codes"](archive, task_ids)
    if list(ok_ids) != task_ids:
        raise RuntimeError("reference task load/order mismatch")
    raw = runtime["evaluate_new_tasks"](
        config, pf_rng, train_state, ok_ids, archive, None,
        preloaded_task_classes=classes, preloaded_task_ids=ok_ids,
    )
    scores, mode, payload_bytes = _dual._fast._score_preflight_result(
        raw, config, len(ok_ids), runtime, None
    )
    if mode != "fused":
        raise RuntimeError("reference did not use fused summary")
    projection = [
        {"task_index": index, **row}
        for index, row in enumerate(_dual._fast._score_projection(scores, task_ids))
    ]
    kept: list[str] = []
    runtime["preflight_route"](scores, ok_ids, kept, archive, runtime["route"], tracker=None)
    ended = time.monotonic_ns()
    document = {
        "classification": CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": "REFERENCE",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "stage": args.stage,
        "repeat": args.repeat,
        "gpu_uuid": args.required_gpu_uuid,
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_included": False,
        "task_ids": task_ids,
        "task_code_rows": rows,
        "input_rng_sha256": runtime["rng_hash"](input_rng),
        "preflight_rng_sha256": runtime["rng_hash"](pf_rng),
        "checkpoint_input_path": stage["checkpoint"]["path"],
        "checkpoint_input_sha256": stage["checkpoint"]["sha256"],
        "graph_input_path": stage["graph"]["path"],
        "graph_input_sha256": stage["graph"]["sha256"],
        "config_evidence": dict(config_evidence),
        "score_projection": projection,
        "score_fingerprint": fingerprint(projection),
        "kept_ids": list(kept),
        "rejected_ids": [task_id for task_id in task_ids if task_id not in kept],
        "archive_before_sha256": before,
        "archive_after_sha256": runtime["archive_hash"](archive),
        "summary_mode": mode,
        "return_payload_bytes": payload_bytes,
        "route_calls": 1,
        "runtime_source_evidence": runtime["source_evidence"](manifest),
        "env_evidence": runtime["env_evidence"](),
        "component_started_monotonic_ns": started,
        "component_ended_monotonic_ns": ended,
        "component_wall_s": (ended - started) / 1e9,
        **{marker: False for marker in RUNTIME_MARKERS},
    }
    return atomic_json(out / "RESULT.json", document)


def run_component_b(manifest: Mapping[str, Any], config: Any, config_evidence: Mapping[str, str], runtime: Mapping[str, Any], out: Path, args: Any):
    args.component = "B"
    args.barrier_receipt = {"enabled": False, "mode": "control_direct"}
    return _dual.run_component_b(manifest, config, config_evidence, runtime, out, args)


def prepare_output(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RESULT.json").exists() or (out / "FAILURE.json").exists():
        raise FileExistsError("async component output already finalized")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--required-gpu-uuid", required=True)
    parser.add_argument("--stage", choices=("early", "mid", "late"), required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--result-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    out = prepare_output(args.out)
    try:
        manifest, config, config_evidence = load_inputs(args.manifest, args.config, args.source_commit)
        if Path(manifest["async_pipeline"]["source_root"]).resolve() != Path(args.source).resolve():
            raise RuntimeError("async manifest/source-root mismatch")
        if args.component == "ASYNC":
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
                raise RuntimeError("async controller must be CPU-only")
            runtime = production_async_runtime(manifest)
            if runtime["jax"].default_backend() != "cpu":
                raise RuntimeError("async controller backend must be CPU")
            run_async_controller(manifest, config, config_evidence, runtime, out, args)
        elif args.component == "REFERENCE":
            _dual._verify_gpu(args.required_gpu_uuid)
            run_sync_reference(manifest, config, config_evidence, production_reference_runtime(manifest, out), out, args)
        else:
            _dual._verify_gpu(args.required_gpu_uuid)
            run_component_b(manifest, config, config_evidence, _dual._production_runtime_b(manifest), out, args)
    except Exception as exc:
        atomic_json(out / "FAILURE.json", {"classification": CLASSIFICATION, "component": args.component, "conclusion": "REJECTED_RUNTIME", "error_class": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "llm_api_calls": 0})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
