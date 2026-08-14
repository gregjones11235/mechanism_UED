#!/usr/bin/env python3
"""Research harness for the two independent halves of a dual-GPU schedule.

Component A validates the frozen candidates and runs the fused learnability
preflight/route path.  Component B runs the frozen 100-update training kernel
and checkpoint round trip.  This is deliberately classified as a scheduling
research experiment, not as a semantic-mainline DiCode result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_fast = _load_sibling("perf48_dual_fastpath_base", "perf48_fastpath_harness.py")
_train = _load_sibling(
    "perf48_dual_training_base", "perf48_training_kernel_harness.py"
)

CLASSIFICATION = "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE"
DUAL_SOURCE_FILES = (
    "src/dicode/training.py",
    "experiments/performance/perf48_dual_pipeline_harness.py",
    "experiments/performance/perf48_dual_pipeline_benchmark.py",
    "experiments/performance/perf48_dual_pipeline_deploy.py",
)
COMPONENTS = ("A", "B")
RUNTIME_MARKERS = (
    "runtime_failure",
    "fatal_error",
    "oom",
    "xid",
    "checkpoint_error",
    "gpu_violation",
)


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Inf"
        if np.isneginf(value):
            return "-Inf"
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("result_sha256", None)
    result["result_sha256"] = _fingerprint(result)
    return result


def atomic_json(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = _hashed_document(document)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_canonical(result), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return result


def load_hashed_json(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = document.get("result_sha256")
    if not expected or _hashed_document(document)["result_sha256"] != expected:
        raise ValueError(f"result hash mismatch: {path}")
    return document


def _wait_for_go(
    args: Any, runtime_source_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    paths = (args.ready_path, args.go_path, args.barrier_id)
    if not any(paths):
        return {"enabled": False, "mode": "control_direct"}
    if not all(paths):
        raise RuntimeError("incomplete concurrent barrier arguments")
    ready_path = Path(args.ready_path)
    go_path = Path(args.go_path)
    if ready_path.exists() or go_path.exists():
        raise FileExistsError("concurrent barrier artifact already exists")
    ready_ns = time.monotonic_ns()
    ready = atomic_json(
        ready_path,
        {
            "classification": CLASSIFICATION,
            "barrier_id": args.barrier_id,
            "component": args.component,
            "pid": os.getpid(),
            "ready_monotonic_ns": ready_ns,
            "runtime_source_evidence_sha256": _fingerprint(
                runtime_source_evidence
            ),
            "llm_api_calls": 0,
        },
    )
    deadline = time.monotonic() + float(args.barrier_timeout_s)
    while not go_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("concurrent GO barrier timeout")
        time.sleep(0.05)
    go = load_hashed_json(go_path)
    if (
        go.get("classification") != CLASSIFICATION
        or go.get("barrier_id") != args.barrier_id
        or go.get("components") != list(COMPONENTS)
        or go.get("llm_api_calls") != 0
        or not isinstance(go.get("go_monotonic_ns"), int)
    ):
        raise RuntimeError("concurrent GO barrier receipt invalid")
    return {
        "enabled": True,
        "mode": "ready_go",
        "barrier_id": args.barrier_id,
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": ready["result_sha256"],
        "ready_monotonic_ns": ready_ns,
        "go_path": str(go_path.resolve()),
        "go_sha256": go["result_sha256"],
        "go_monotonic_ns": go["go_monotonic_ns"],
        "go_observed_monotonic_ns": time.monotonic_ns(),
    }


def _stage_repeat(
    manifest: Mapping[str, Any], stage_name: str, repeat_index: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    try:
        stage = next(item for item in manifest["stages"] if item["name"] == stage_name)
        repeat = stage["repeats"][repeat_index]
    except (KeyError, IndexError, StopIteration, TypeError) as exc:
        raise RuntimeError("invalid frozen stage/repeat") from exc
    return stage, repeat


def _config_binding(
    manifest: Mapping[str, Any], config_path: str | Path
) -> dict[str, str]:
    resolved = Path(config_path).resolve()
    entry = next(
        (
            value
            for value in manifest.get("source_config", {}).get("config", {}).values()
            if Path(value.get("path", "")).resolve() == resolved
        ),
        None,
    )
    if entry is None:
        raise RuntimeError("config is not bound by frozen manifest")
    actual = _file_sha256(resolved)
    if actual != entry.get("sha256"):
        raise RuntimeError("frozen config hash mismatch")
    return {"path": str(resolved), "sha256": actual}


def load_inputs(
    manifest_path: str | Path, config_path: str | Path
) -> tuple[dict[str, Any], Any, dict[str, str]]:
    manifest = _fast.load_manifest(manifest_path)
    config = _fast._load_config(config_path)
    _fast._config_contract(config)
    _fast._arm_contract(config, "FINAL_COMBO", "FAST_COMBO")
    _train._config_contract(config)
    compact, score = _train._arm_values(config)
    if compact or score != "learnability":
        raise RuntimeError("dual pipeline requires non-compact learnability training")
    return manifest, config, _config_binding(manifest, config_path)


def verify_dual_manifest(
    manifest: Mapping[str, Any], source_commit: str
) -> dict[str, Any]:
    metadata = manifest.get("dual_pipeline")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("dual-pipeline manifest metadata missing")
    if metadata.get("classification") != CLASSIFICATION:
        raise RuntimeError("dual-pipeline manifest classification mismatch")
    if metadata.get("source_commit") != source_commit:
        raise RuntimeError("dual-pipeline source commit mismatch")
    required = metadata.get("required_source_files")
    entries = manifest.get("source_config", {}).get("source", {})
    if not isinstance(required, list) or set(required) != set(entries):
        raise RuntimeError("dual-pipeline source set mismatch")
    if not set(DUAL_SOURCE_FILES).issubset(entries):
        raise RuntimeError("dual-pipeline required source binding missing")
    parent = metadata.get("parent_manifest")
    if not isinstance(parent, Mapping):
        raise RuntimeError("dual-pipeline parent manifest evidence missing")
    parent_path = Path(parent.get("path", ""))
    if not parent_path.is_file() or _file_sha256(parent_path) != parent.get(
        "file_sha256"
    ):
        raise RuntimeError("dual-pipeline parent manifest binding mismatch")
    loaded_parent = _fast.load_manifest(parent_path)
    if loaded_parent.get("manifest_sha256") != parent.get("manifest_sha256"):
        raise RuntimeError("dual-pipeline parent manifest identity mismatch")
    return dict(metadata)


def _bind_sources(
    runtime: Mapping[str, Any],
    manifest: Mapping[str, Any],
    required: Mapping[str, str],
) -> dict[str, Any]:
    entries = manifest.get("source_config", {}).get("source", {})
    evidence = {"verified": True, "paths": {}, "hashes": {}}
    for name, expected_relative in required.items():
        obj = runtime.get(name)
        source = inspect.getsourcefile(obj) if obj is not None else None
        if not source:
            raise RuntimeError(f"runtime source unavailable: {name}")
        resolved = Path(source).resolve()
        entry = entries.get(expected_relative)
        if not isinstance(entry, Mapping) or Path(entry.get("path", "")).resolve() != resolved:
            entry = None
        if entry is None or _file_sha256(resolved) != entry.get("sha256"):
            raise RuntimeError(f"runtime source binding mismatch: {name}")
        evidence["paths"][name] = str(resolved)
        evidence["hashes"][name] = entry["sha256"]
        evidence.setdefault("expected_relatives", {})[name] = expected_relative
    return evidence


def _verify_frozen_tasks(
    stage: Mapping[str, Any], archive: Any, runtime: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    task_ids = [str(value) for value in stage["task_ids"]]
    code_map = runtime["archive_get_codes"](archive, task_ids)
    if list(code_map) != task_ids:
        raise RuntimeError("archive candidate code order mismatch")
    code_hashes = []
    for candidate in stage["tasks"]:
        actual = hashlib.sha256(code_map.get(candidate["id"], "").encode()).hexdigest()
        if actual != candidate["code_sha256"]:
            raise RuntimeError(f"candidate code hash mismatch: {candidate['id']}")
        code_hashes.append(actual)
    return task_ids, code_hashes


def _common_result(
    *,
    args: Any,
    manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    repeat: Mapping[str, Any],
    task_ids: list[str],
    code_hashes: list[str],
    config_evidence: Mapping[str, str],
    input_rng_sha256: str,
) -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "not_semantic_mainline": True,
        "component": args.component,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "gpu_uuid": args.required_gpu_uuid,
        "stage": args.stage,
        "repeat": args.repeat,
        "llm_api_calls": 0,
        "validation_cache_speedup_included": False,
        "validation_replay_scope": "not_executed_not_timed",
        "task_ids": task_ids,
        "task_assignment_sha256": _fingerprint(task_ids),
        "task_code_hashes": code_hashes,
        "input_rng_sha256": input_rng_sha256,
        "frozen_rng": list(repeat["rng"]),
        "checkpoint_input_path": stage["checkpoint"]["path"],
        "checkpoint_input_sha256": stage["checkpoint"]["sha256"],
        "conditioning_path": stage["conditioning"]["path"],
        "conditioning_file_sha256": stage["conditioning"]["sha256"],
        "embedding_hash": stage["embedding"]["hash"],
        "conditioning_type": "one_hot",
        "conditioning_shape": stage["conditioning"]["shape"],
        "conditioning_dtype": stage["conditioning"]["dtype"],
        "config_evidence": dict(config_evidence),
        "compile_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "barrier": dict(args.barrier_receipt),
        "runtime_failure": False,
        "fatal_error": False,
        "oom": False,
        "xid": False,
        "checkpoint_error": False,
        "gpu_violation": False,
    }


def _production_runtime_a(manifest: Mapping[str, Any], out: Path) -> dict[str, Any]:
    import jax

    runtime = _fast._real_runtime(manifest, out)
    runtime.update(
        array_rng=lambda value: jax.numpy.asarray(np.asarray(value, dtype=np.uint32)),
        split_rng=lambda value: jax.random.split(value),
        state_hash=_fast.state_hash,
        rng_hash=_fast.rng_hash,
        env_evidence=_fast.env_evidence,
    )
    required = {
        "TaskArchive": "src/dicode/dreaming/gen_manager.py",
        "load_tasks_from_env_codes": "src/dicode/task_utils.py",
        "evaluate_new_tasks": "src/dicode/evaluation/online_evaluation.py",
        "learnability_scores_from_counts": "experiments/training/run_dicode.py",
        "_load_agent_state": "src/dicode/setup.py",
        "route": "src/dicode/skill_preflight/preflight.py",
        "preflight_route": "src/dicode/skill_preflight/preflight_route.py",
        "learnability_summary_contract": "src/dicode/skill_preflight/learnability_summary.py",
    }
    runtime["source_evidence"] = lambda loaded: _bind_sources(runtime, loaded, required)
    return runtime


def run_component_a(
    manifest: Mapping[str, Any],
    config: Any,
    config_evidence: Mapping[str, str],
    runtime: Mapping[str, Any],
    out: Path,
    args: Any,
) -> dict[str, Any]:
    stage, repeat = _stage_repeat(manifest, args.stage, args.repeat)
    started = time.monotonic_ns()
    train_state = runtime["_load_agent_state"](config, stage["checkpoint"]["path"])
    conditioning, conditioning_hash = _fast._verify_conditioning(stage)
    if conditioning_hash != stage["embedding"]["hash"]:
        raise RuntimeError("conditioning compatibility hash mismatch")
    input_rng = runtime["array_rng"](repeat["rng"])
    preflight_rng, _ = runtime["split_rng"](input_rng)
    archive = runtime["reconstruct_archive"](stage["graph"]["path"])
    archive_before = runtime["archive_hash"](archive)
    task_ids, code_hashes = _verify_frozen_tasks(stage, archive, runtime)

    task_load_started = time.monotonic_ns()
    classes, ok_ids = runtime["load_tasks_from_env_codes"](archive, task_ids)
    candidate_task_load_wall_s = (time.monotonic_ns() - task_load_started) / 1e9
    if list(ok_ids) != task_ids:
        raise RuntimeError("candidate task load/order mismatch")

    preflight_started = time.monotonic_ns()
    raw = runtime["evaluate_new_tasks"](
        config,
        preflight_rng,
        train_state,
        ok_ids,
        archive,
        _fast._FrozenEmbeddingProvider(conditioning),
        preloaded_task_classes=classes,
        preloaded_task_ids=ok_ids,
    )
    scores, summary_mode, payload_bytes = _fast._score_preflight_result(
        raw, config, len(ok_ids), runtime, None
    )
    if summary_mode != "fused":
        raise RuntimeError("component A requires fused preflight summary")
    projection = _fast._score_projection(scores, task_ids)
    kept: list[str] = []
    route_started = time.monotonic_ns()
    runtime["preflight_route"](
        scores, ok_ids, kept, archive, runtime["route"], tracker=None
    )
    route_wall_s = (time.monotonic_ns() - route_started) / 1e9
    archive_after = runtime["archive_hash"](archive)
    preflight_wall_s = (time.monotonic_ns() - preflight_started) / 1e9
    rollout_updates = int(config.validation.rollout_updates)
    if rollout_updates != 40:
        raise RuntimeError("component A requires exactly 40 preflight updates")
    env_steps = rollout_updates * int(config.validation.num_envs) * int(
        config.validation.num_steps
    )
    common = _common_result(
        args=args,
        manifest=manifest,
        stage=stage,
        repeat=repeat,
        task_ids=task_ids,
        code_hashes=code_hashes,
        config_evidence=config_evidence,
        input_rng_sha256=runtime["rng_hash"](input_rng),
    )
    ended = time.monotonic_ns()
    result = {
        **common,
        "component_scope": "fused_preflight_only",
        "candidate_task_load_ids": list(ok_ids),
        "candidate_task_load_sha256": _fingerprint(list(ok_ids)),
        "preflight_rng_sha256": runtime["rng_hash"](preflight_rng),
        "params_sha256_before": runtime["state_hash"](train_state.params),
        "optimizer_sha256_before": runtime["state_hash"](train_state.opt_state),
        "score_projection": projection,
        "scoring_fingerprint": _fingerprint(projection),
        "accepted_ids": sorted(kept),
        "rejected_ids": sorted(task_id for task_id in task_ids if task_id not in kept),
        "archive_before_sha256": archive_before,
        "archive_after_sha256": archive_after,
        "preflight_env_steps": env_steps,
        "candidate_task_load_wall_s": candidate_task_load_wall_s,
        "preflight_wall_s": preflight_wall_s,
        "route_wall_s": route_wall_s,
        "component_started_monotonic_ns": started,
        "component_ended_monotonic_ns": ended,
        "component_wall_s": (ended - started) / 1e9,
        "preflight_summary_mode": summary_mode,
        "preflight_return_payload_bytes": payload_bytes,
        "runtime_source_evidence": runtime["source_evidence"](manifest),
        "env_evidence": runtime["env_evidence"](),
    }
    return atomic_json(out / "RESULT.json", result)


def _production_runtime_b(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _train._runtime_imports()
    jax = runtime["jax"]
    runtime.update(
        array_rng=lambda value: runtime["jnp"].asarray(
            np.asarray(value, dtype=np.uint32)
        ),
        split_rng=lambda value: jax.random.split(value),
        state_hash=_train.state_hash,
        rng_hash=_train.rng_hash,
        env_evidence=_train._env_evidence,
        archive_get_codes=lambda archive, ids: archive.get_task_codes(ids),
    )
    required = {
        "TaskArchive": "src/dicode/dreaming/gen_manager.py",
        "load_tasks_from_env_codes": "src/dicode/task_utils.py",
        "run_training_session": "src/dicode/training.py",
        "_calculate_task_distribution": "src/dicode/training.py",
        "_create_achievement_masks": "src/dicode/training.py",
        "calculate_scores_from_snapshot": "src/dicode/scoring.py",
        "_load_agent_state": "src/dicode/setup.py",
        "wrappers_cl": "src/dicode/wrappers_cl.py",
    }
    runtime["source_evidence"] = lambda loaded: _bind_sources(runtime, loaded, required)
    return runtime


def run_component_b(
    manifest: Mapping[str, Any],
    config: Any,
    config_evidence: Mapping[str, str],
    runtime: Mapping[str, Any],
    out: Path,
    args: Any,
) -> dict[str, Any]:
    stage, repeat = _stage_repeat(manifest, args.stage, args.repeat)
    started = time.monotonic_ns()
    config.gen_manager.graph_path = stage["graph"]["path"]
    archive = runtime["TaskArchive"](config.gen_manager)
    task_ids, code_hashes = _verify_frozen_tasks(stage, archive, runtime)
    classes, ok_ids = runtime["load_tasks_from_env_codes"](archive, task_ids)
    if list(ok_ids) != task_ids:
        raise RuntimeError("training task validation/order mismatch")
    task_classes = classes + [runtime["OriginalTask"]]
    conditioning, conditioning_hash = _fast._verify_conditioning(stage)
    proportions = runtime["_calculate_task_distribution"](config, len(classes))
    train_state = runtime["_load_agent_state"](config, stage["checkpoint"]["path"])
    params_before = runtime["state_hash"](train_state.params)
    optimizer_before = runtime["state_hash"](train_state.opt_state)
    input_rng = runtime["array_rng"](repeat["rng"])
    train_rng, outer_rng_after = runtime["split_rng"](input_rng)

    wandb = runtime["wandb"]
    wandb.init(mode="offline", dir=str(out / "wandb"), project=CLASSIFICATION)
    wandb.log = lambda *args, **kwargs: None
    training_started = time.monotonic_ns()
    try:
        raw = runtime["run_training_session"](
            config,
            train_rng,
            task_classes,
            num_training_updates=100,
            task_embeddings=runtime["jnp"].asarray(conditioning),
            train_state=train_state,
            task_distribution_proportions=proportions,
            global_update_step=int(stage["global_step"]),
            current_original_return=0.0,
        )
        for leaf in runtime["jax"].tree_util.tree_leaves(raw):
            runtime["jax"].block_until_ready(leaf)
    finally:
        wandb.finish()
    training_wall_s = (time.monotonic_ns() - training_started) / 1e9
    train_state = raw.get("train_state", raw) if isinstance(raw, dict) else raw
    metrics = raw["metrics"]
    updates = int(metrics["num_updates_done"])
    env_steps = int(metrics["num_env_steps_done"])
    if updates != 100 or env_steps != 13_107_200:
        raise RuntimeError("unexpected 100-update training accounting")
    task_mask, completed = runtime["_create_achievement_masks"](task_classes)
    scoring_payload = runtime["jax"].device_get(metrics["scoring_window_data"])
    scores = runtime["calculate_scores_from_snapshot"](
        scoring_payload,
        len(task_classes),
        np.asarray(task_mask),
        np.asarray(completed),
        config,
        [len(task_classes) - 1],
    )

    checkpoint_dir = out / "checkpoint"
    manager = runtime["ocp"].CheckpointManager(
        str(checkpoint_dir),
        runtime["ocp"].PyTreeCheckpointer(),
        options=runtime["ocp"].CheckpointManagerOptions(create=True, max_to_keep=1),
    )
    save_step = int(stage["global_step"]) + updates
    checkpoint_started = time.monotonic_ns()
    try:
        manager.save(save_step, train_state)
        manager.wait_until_finished()
    finally:
        manager.close()
    checkpoint_wall_s = (time.monotonic_ns() - checkpoint_started) / 1e9
    reloaded = runtime["_load_agent_state"](
        config, str(checkpoint_dir / str(save_step))
    )
    params_after = runtime["state_hash"](train_state.params)
    optimizer_after = runtime["state_hash"](train_state.opt_state)
    reloaded_params = runtime["state_hash"](reloaded.params)
    reloaded_optimizer = runtime["state_hash"](reloaded.opt_state)
    if params_after != reloaded_params or optimizer_after != reloaded_optimizer:
        raise RuntimeError("training checkpoint reload mismatch")
    common = _common_result(
        args=args,
        manifest=manifest,
        stage=stage,
        repeat=repeat,
        task_ids=task_ids,
        code_hashes=code_hashes,
        config_evidence=config_evidence,
        input_rng_sha256=runtime["rng_hash"](input_rng),
    )
    ended = time.monotonic_ns()
    result = {
        **common,
        "params_sha256_before": params_before,
        "params_sha256_after": params_after,
        "optimizer_sha256_before": optimizer_before,
        "optimizer_sha256_after": optimizer_after,
        "checkpoint_reloaded_params_sha256": reloaded_params,
        "checkpoint_reloaded_optimizer_sha256": reloaded_optimizer,
        "checkpoint_loadable": True,
        "checkpoint_output_path": str(checkpoint_dir / str(save_step)),
        "train_rng_sha256": runtime["rng_hash"](train_rng),
        "outer_rng_after_sha256": runtime["rng_hash"](outer_rng_after),
        "input_global_update_step": int(stage["global_step"]),
        "global_update_step": save_step,
        "global_env_steps": int(stage["initial_env_steps"]) + env_steps,
        "updates": updates,
        "env_steps": env_steps,
        "scoring_fingerprint": _train.scoring_fingerprint(scores),
        "training_wall_s": training_wall_s,
        "checkpoint_wall_s": checkpoint_wall_s,
        "component_started_monotonic_ns": started,
        "component_ended_monotonic_ns": ended,
        "component_wall_s": (ended - started) / 1e9,
        "runtime_source_evidence": runtime["source_evidence"](manifest),
        "env_evidence": runtime["env_evidence"](),
    }
    if conditioning_hash != common["embedding_hash"]:
        raise RuntimeError("training conditioning hash mismatch")
    return atomic_json(out / "RESULT.json", result)


def _verify_gpu(required_uuid: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != required_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exact component GPU UUID")
    import jax

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("component must see exactly one GPU")
    device = jax.devices()[0]
    if required_uuid not in str(device) and required_uuid not in os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    ):
        raise RuntimeError("JAX GPU UUID mismatch")


def _preflight(
    args: Any,
    manifest: Mapping[str, Any],
    config: Any,
    config_evidence: Mapping[str, str],
    out: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "classification": CLASSIFICATION,
        "component": args.component,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "gpu_uuid": args.required_gpu_uuid,
        "stage": args.stage,
        "repeat": args.repeat,
        "config_evidence": dict(config_evidence),
        "llm_api_calls": 0,
        "runtime_source_evidence": runtime["source_evidence"](manifest),
        "pass": True,
    }
    return atomic_json(out / "PREFLIGHT.json", result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--required-gpu-uuid", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=("early", "mid", "late"), required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--ready-path")
    parser.add_argument("--go-path")
    parser.add_argument("--barrier-id")
    parser.add_argument("--barrier-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RESULT.json").exists() or (out / "FAILURE.json").exists():
        raise FileExistsError("component output already finalized")
    try:
        manifest, config, config_evidence = load_inputs(args.manifest, args.config)
        verify_dual_manifest(manifest, args.source_commit)
        _stage_repeat(manifest, args.stage, args.repeat)
        _verify_gpu(args.required_gpu_uuid)
        runtime = (
            _production_runtime_a(manifest, out)
            if args.component == "A"
            else _production_runtime_b(manifest)
        )
        runtime_source_evidence = runtime["source_evidence"](manifest)
        args.barrier_receipt = _wait_for_go(args, runtime_source_evidence)
        if args.mode == "preflight":
            _preflight(args, manifest, config, config_evidence, out, runtime)
        elif args.component == "A":
            run_component_a(
                manifest,
                config,
                config_evidence,
                runtime,
                out,
                args,
            )
        else:
            run_component_b(
                manifest,
                config,
                config_evidence,
                runtime,
                out,
                args,
            )
    except Exception:
        atomic_json(
            out / "FAILURE.json",
            {
                "classification": CLASSIFICATION,
                "component": args.component,
                "error_class": sys.exc_info()[0].__name__ if sys.exc_info()[0] else "Error",
                "error": traceback.format_exc(),
                "llm_api_calls": 0,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
