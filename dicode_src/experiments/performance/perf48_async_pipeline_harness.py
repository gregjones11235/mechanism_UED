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
import threading
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


class PureNetworkXArchive:
    """TaskArchive-compatible control-plane view without importing gen_manager/JAX."""

    def __init__(self, graph_path: str | Path):
        import networkx as nx

        self.graph = nx.read_graphml(graph_path)
        self._lock = threading.Lock()
        self.active_task_count = 0
        for _, data in self.graph.nodes(data=True):
            if "performance_history" in data:
                try:
                    data["performance_history"] = json.loads(data["performance_history"])
                except (json.JSONDecodeError, TypeError):
                    data["performance_history"] = []
            active = data.get("is_active") not in (None, "false", False)
            data["is_active"] = bool(active)
            self.active_task_count += int(active)
            data["priority_score"] = float(
                data.get("priority_score", data.get("learnability_score", 0.0))
            )
            data["session_last_trained"] = int(data.get("session_last_trained", -1))

    def get_task_codes(self, task_ids: list[str]) -> dict[str, str]:
        with self._lock:
            return {
                task_id: self.graph.nodes[task_id].get("code", "")
                for task_id in task_ids
                if self.graph.has_node(task_id)
            }

    def update_node_learnability(self, task_id: str, score: float) -> None:
        with self._lock:
            if not self.graph.has_node(task_id):
                raise RuntimeError(f"archive task missing: {task_id}")
            self.graph.nodes[task_id]["learnability_score"] = (
                float(score) if np.isfinite(score) else 0.0
            )

    def update_node_status(self, task_id: str, status: str) -> None:
        with self._lock:
            if not self.graph.has_node(task_id):
                raise RuntimeError(f"archive task missing: {task_id}")
            self.graph.nodes[task_id]["status"] = status

    def set_task_active_status(self, task_id: str, is_active: bool) -> None:
        with self._lock:
            if not self.graph.has_node(task_id):
                raise RuntimeError(f"archive task missing: {task_id}")
            current = bool(self.graph.nodes[task_id].get("is_active", False))
            requested = bool(is_active)
            self.graph.nodes[task_id]["is_active"] = requested
            if requested and not current:
                self.active_task_count += 1
            elif current and not requested:
                self.active_task_count = max(0, self.active_task_count - 1)


def pure_graph_sha(archive: Any) -> str:
    """Match the frozen reference graph fingerprint without TaskArchive imports."""
    graph = archive.graph
    parts = [
        str(node)
        + ":"
        + json.dumps(_dual._fast.canonical(dict(graph.nodes[node])), sort_keys=True)
        for node in sorted(graph.nodes())
    ]
    parts.extend(f"{left}->{right}" for left, right in sorted(sorted(edge) for edge in graph.edges()))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def numpy_rng_hash(value: Any) -> str:
    """Match the reference single-array RNG hash without importing JAX."""
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def rng_receipt(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "values": array.tolist(),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": numpy_rng_hash(array),
    }


def validate_rng_receipt(receipt: Any) -> np.ndarray:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "values", "dtype", "shape", "sha256"
    }:
        raise RuntimeError("async RNG receipt shape mismatch")
    array = np.asarray(receipt["values"], dtype=str(receipt["dtype"]))
    if list(array.shape) != receipt["shape"] or numpy_rng_hash(array) != receipt["sha256"]:
        raise RuntimeError("async RNG receipt hash mismatch")
    return array


def controller_import_state() -> dict[str, bool]:
    names = tuple(sys.modules)
    return {
        "jax_loaded": any(name == "jax" or name.startswith("jax.") for name in names),
        "jaxlib_loaded": any(
            name == "jaxlib" or name.startswith("jaxlib.") for name in names
        ),
        "gen_manager_loaded": "dicode.dreaming.gen_manager" in sys.modules,
    }


def assert_controller_import_clean(label: str) -> dict[str, bool]:
    state = controller_import_state()
    if any(state.values()):
        raise RuntimeError(f"async controller forbidden import at {label}: {state}")
    return state


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
    import_gate_pre = assert_controller_import_clean("before_runtime")
    from dicode.skill_preflight.async_preflight import (
        AsyncPreflightManager,
        load_hashed_json as load_async_receipt,
        plan_async_session,
    )
    from dicode.skill_preflight.preflight import route
    from dicode.skill_preflight.preflight_route import preflight_route

    import_gate_post = assert_controller_import_clean("after_runtime")
    return {
        "controller_backend": "pure_python_no_jax",
        "controller_import_gate": {
            "before_runtime": import_gate_pre,
            "after_runtime": import_gate_post,
        },
        "rng_hash": numpy_rng_hash,
        "reconstruct_archive": PureNetworkXArchive,
        "archive_hash": pure_graph_sha,
        "AsyncPreflightManager": AsyncPreflightManager,
        "load_async_receipt": load_async_receipt,
        "plan_async_session": plan_async_session,
        "route": route,
        "preflight_route": preflight_route,
        "source_evidence": {
            "AsyncPreflightManager": _source_binding(AsyncPreflightManager, manifest, "src/dicode/skill_preflight/async_preflight.py"),
            "plan_async_session": _source_binding(plan_async_session, manifest, "src/dicode/skill_preflight/async_preflight.py"),
            "preflight_route": _source_binding(preflight_route, manifest, "src/dicode/skill_preflight/preflight_route.py"),
            "PureNetworkXArchive": _source_binding(
                PureNetworkXArchive,
                manifest,
                "experiments/performance/perf48_async_pipeline_harness.py",
            ),
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


def load_async_input(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
    config_evidence: Mapping[str, str],
    args: Any,
) -> dict[str, Any]:
    document = load_hashed_json(path)
    reference_path = Path(document.get("reference_result_path", "")).resolve()
    if not reference_path.is_file():
        raise RuntimeError("async input reference result missing")
    reference = load_hashed_json(reference_path)
    expected = {
        "classification": CLASSIFICATION,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "source_root": str(Path(args.source).resolve()),
        "stage": args.stage,
        "repeat": args.repeat,
        "config_evidence": dict(config_evidence),
        "reference_result_sha256": reference["result_sha256"],
        "reference_result_file_sha256": file_sha256(reference_path),
        "llm_api_calls": 0,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"async input binding mismatch: {key}")
    if reference.get("component") != "REFERENCE":
        raise RuntimeError("async input did not bind a reference result")
    fields = {
        "task_ids": reference.get("task_ids"),
        "task_code_rows": reference.get("task_code_rows"),
        "input_rng": reference.get("input_rng"),
        "main_rng_after_split": reference.get("main_rng_after_split"),
        "preflight_rng": reference.get("preflight_rng"),
        "checkpoint_input_path": reference.get("checkpoint_input_path"),
        "checkpoint_input_sha256": reference.get("checkpoint_input_sha256"),
        "graph_input_path": reference.get("graph_input_path"),
        "graph_input_sha256": reference.get("graph_input_sha256"),
        "archive_before_sha256": reference.get("archive_before_sha256"),
    }
    if document.get("reference_contract") != fields:
        raise RuntimeError("async input/reference contract mismatch")
    if document.get("reference_contract_sha256") != fingerprint(fields):
        raise RuntimeError("async input reference contract hash mismatch")
    for name in ("input_rng", "main_rng_after_split", "preflight_rng"):
        validate_rng_receipt(fields[name])
    task_ids = fields["task_ids"]
    rows = fields["task_code_rows"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or len(task_ids) != len(set(task_ids))
        or not isinstance(rows, list)
        or [row.get("task_id") for row in rows] != task_ids
    ):
        raise RuntimeError("async input task order invalid")
    return {"document": document, "reference": reference, **fields}


def prepare_async_controller(
    manifest: Mapping[str, Any],
    config: Any,
    config_evidence: Mapping[str, str],
    runtime: Mapping[str, Any],
    out: Path,
    args: Any,
) -> dict[str, Any]:
    assert_controller_import_clean("before_prepare")
    stage, _ = stage_repeat(manifest, args.stage, args.repeat)
    bound = load_async_input(
        args.async_input,
        manifest=manifest,
        config_evidence=config_evidence,
        args=args,
    )
    archive = runtime["reconstruct_archive"](stage["graph"]["path"])
    rows = candidate_rows(stage, archive)
    if rows != bound["task_code_rows"]:
        raise RuntimeError("async input/live archive candidate mismatch")
    archive_before = runtime["archive_hash"](archive)
    if archive_before != bound["archive_before_sha256"]:
        raise RuntimeError("async input/live archive fingerprint mismatch")
    task_ids = [row["task_id"] for row in rows]
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
    assert_controller_import_clean("after_prepare")
    return {
        "stage": stage,
        "archive": archive,
        "rows": rows,
        "task_ids": task_ids,
        "archive_before": archive_before,
        "input_rng": validate_rng_receipt(bound["input_rng"]),
        "main_rng": validate_rng_receipt(bound["main_rng_after_split"]),
        "pf_rng": validate_rng_receipt(bound["preflight_rng"]),
        "input_rng_receipt": bound["input_rng"],
        "main_rng_receipt": bound["main_rng_after_split"],
        "pf_rng_receipt": bound["preflight_rng"],
        "plan_n": plan_n,
        "manager": manager,
        "async_input": bound["document"],
    }


def wait_for_async_go(
    args: Any,
    component: str,
    runtime_source_evidence: Mapping[str, Any],
    *,
    controller_import_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = (args.ready_path, args.go_path, args.barrier_id)
    if not any(values):
        return {"enabled": False, "mode": "control_direct"}
    if not all(values) or component not in ("ASYNC", "B"):
        raise RuntimeError("incomplete or invalid async barrier arguments")
    ready_path = Path(args.ready_path)
    go_path = Path(args.go_path)
    if ready_path.exists() or go_path.exists():
        raise FileExistsError("async barrier artifact already exists")
    source_sha = fingerprint(runtime_source_evidence)
    ready_ns = time.monotonic_ns()
    ready_document = {
        "classification": CLASSIFICATION,
        "barrier_id": args.barrier_id,
        "component": component,
        "pid": os.getpid(),
        "ready_monotonic_ns": ready_ns,
        "runtime_source_evidence_sha256": source_sha,
        "llm_api_calls": 0,
    }
    if component == "ASYNC":
        if not isinstance(controller_import_gate, Mapping):
            raise RuntimeError("async controller import gate missing before READY")
        states = list(controller_import_gate.values())
        if not states or any(
            not isinstance(state, Mapping) or any(bool(value) for value in state.values())
            for state in states
        ):
            raise RuntimeError("async controller import gate failed before READY")
        ready_document["controller_forbidden_imports"] = copy.deepcopy(
            dict(controller_import_gate)
        )
    ready = atomic_json(
        ready_path,
        ready_document,
    )
    deadline = time.monotonic() + float(args.barrier_timeout_s)
    while not go_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("async GO barrier timeout")
        time.sleep(0.05)
    go = load_hashed_json(go_path)
    ready_hashes = go.get("ready_sha256")
    if (
        go.get("classification") != CLASSIFICATION
        or go.get("barrier_id") != args.barrier_id
        or go.get("components") != ["ASYNC", "B"]
        or not isinstance(ready_hashes, Mapping)
        or set(ready_hashes) != {"ASYNC", "B"}
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in ready_hashes.values()
        )
        or ready_hashes.get(component) != ready["result_sha256"]
        or not isinstance(go.get("go_monotonic_ns"), int)
        or go.get("llm_api_calls") != 0
    ):
        raise RuntimeError("async GO barrier receipt invalid")
    return {
        "enabled": True,
        "mode": "ready_go",
        "barrier_id": args.barrier_id,
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": ready["result_sha256"],
        "ready_monotonic_ns": ready_ns,
        "runtime_source_evidence_sha256": source_sha,
        "go_path": str(go_path.resolve()),
        "go_sha256": go["result_sha256"],
        "go_monotonic_ns": go["go_monotonic_ns"],
        "go_observed_monotonic_ns": time.monotonic_ns(),
    }


def run_async_controller(
    manifest: Mapping[str, Any],
    config: Any,
    config_evidence: Mapping[str, str],
    runtime: Mapping[str, Any],
    out: Path,
    args: Any,
    *,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = dict(
        prepared
        or prepare_async_controller(
            manifest, config, config_evidence, runtime, out, args
        )
    )
    started = time.monotonic_ns()
    stage = prepared["stage"]
    archive = prepared["archive"]
    rows = prepared["rows"]
    task_ids = prepared["task_ids"]
    archive_before = prepared["archive_before"]
    input_rng = prepared["input_rng"]
    main_rng = prepared["main_rng"]
    pf_rng = prepared["pf_rng"]
    plan_n = prepared["plan_n"]
    manager = prepared["manager"]
    if runtime.get("controller_backend") != "pure_python_no_jax":
        raise RuntimeError("async controller backend must be pure Python")
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
    import_gate_after_result = assert_controller_import_clean("after_result")
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
        "input_rng": copy.deepcopy(prepared["input_rng_receipt"]),
        "main_rng_after_split": copy.deepcopy(prepared["main_rng_receipt"]),
        "preflight_rng": copy.deepcopy(prepared["pf_rng_receipt"]),
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
        "controller_forbidden_imports": {
            **copy.deepcopy(runtime["controller_import_gate"]),
            "after_result": import_gate_after_result,
        },
        "async_input_sha256": prepared["async_input"]["result_sha256"],
        "reference_result_sha256": prepared["async_input"][
            "reference_result_sha256"
        ],
        "worker_jax_backend": result["jax_backend"],
        "worker_jax_device_count": result["jax_device_count"],
        "worker_route_calls": result["route_calls"],
        "main_route_calls": receipts["APPLIED"]["route_calls"],
        "receipt_sha256": {name: value["result_sha256"] for name, value in receipts.items()},
        "barrier": dict(args.barrier_receipt),
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
    main_rng, pf_rng = runtime["split_rng"](input_rng)
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
        "input_rng": rng_receipt(input_rng),
        "main_rng_after_split": rng_receipt(main_rng),
        "preflight_rng": rng_receipt(pf_rng),
        "input_rng_sha256": runtime["rng_hash"](input_rng),
        "main_rng_after_split_sha256": runtime["rng_hash"](main_rng),
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
        "barrier": dict(args.barrier_receipt),
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
    if not hasattr(args, "barrier_receipt"):
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
    parser.add_argument("--async-input")
    parser.add_argument("--ready-path")
    parser.add_argument("--go-path")
    parser.add_argument("--barrier-id")
    parser.add_argument("--barrier-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    out = prepare_output(args.out)
    try:
        manifest, config, config_evidence = load_inputs(args.manifest, args.config, args.source_commit)
        if Path(manifest["async_pipeline"]["source_root"]).resolve() != Path(args.source).resolve():
            raise RuntimeError("async manifest/source-root mismatch")
        if args.component == "ASYNC":
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
                raise RuntimeError("async controller must be CPU-only")
            if os.environ.get("JAX_PLATFORMS") is not None:
                raise RuntimeError("async controller must not set JAX_PLATFORMS")
            if not args.async_input:
                raise RuntimeError("async controller input receipt missing")
            assert_controller_import_clean("main_entry")
            runtime = production_async_runtime(manifest)
            prepared = prepare_async_controller(
                manifest, config, config_evidence, runtime, out, args
            )
            runtime["controller_import_gate"]["after_prepare"] = (
                assert_controller_import_clean("before_ready")
            )
            args.barrier_receipt = wait_for_async_go(
                args,
                "ASYNC",
                runtime["source_evidence"],
                controller_import_gate=runtime["controller_import_gate"],
            )
            run_async_controller(
                manifest,
                config,
                config_evidence,
                runtime,
                out,
                args,
                prepared=prepared,
            )
        elif args.component == "REFERENCE":
            _dual._verify_gpu(args.required_gpu_uuid)
            runtime = production_reference_runtime(manifest, out)
            args.barrier_receipt = wait_for_async_go(
                args, "REFERENCE", runtime["source_evidence"](manifest)
            )
            run_sync_reference(manifest, config, config_evidence, runtime, out, args)
        else:
            _dual._verify_gpu(args.required_gpu_uuid)
            runtime = _dual._production_runtime_b(manifest)
            args.barrier_receipt = wait_for_async_go(
                args, "B", runtime["source_evidence"](manifest)
            )
            run_component_b(manifest, config, config_evidence, runtime, out, args)
    except Exception as exc:
        atomic_json(out / "FAILURE.json", {"classification": CLASSIFICATION, "component": args.component, "conclusion": "REJECTED_RUNTIME", "error_class": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "llm_api_calls": 0})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
