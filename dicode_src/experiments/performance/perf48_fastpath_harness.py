#!/usr/bin/env python3
"""Frozen replay harness for B4_SINGLE and FINAL_COMBO.

The harness reuses the combo manifest's frozen checkpoints, conditioning,
candidate order and RNG. It never runs evolution, so validation-cache evidence
is explicitly recorded as ``validation_cache_exercised=false`` even when the
FAST_COMBO overlay enables that cache.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_base = _load_sibling("perf48_combo_harness_base", "perf48_combo_harness.py")
_cfg = _load_sibling("perf48_fastpath_config", "perf48_fastpath_config.py")

CLASSIFICATION = "PERF48_FASTPATH_BENCHMARK"
COMPARISONS = _cfg.COMPARISONS
ARMS = _cfg.ARMS

sha256_bytes = _base.sha256_bytes
sha256_file = _base.sha256_file
canonical = _base.canonical
fingerprint = _base.fingerprint
state_hash = _base.state_hash
rng_hash = _base.rng_hash
env_evidence = _base.env_evidence
verify_selection_semantics = _base.verify_selection_semantics
atomic_json = _base.atomic_json
load_manifest = _base.load_manifest
_load_config = _base._load_config
_config_get = _base._config_get
_reconstruct_archive = _base._reconstruct_archive
_graph_sha = _base._graph_sha
_verify_conditioning = _base._verify_conditioning
_FrozenEmbeddingProvider = _base._FrozenEmbeddingProvider
_heldout_embedding = _base._heldout_embedding
_verify_gpu = _base._verify_gpu


def _config_contract(config) -> None:
    _base._config_contract(config)
    score = _config_get(config, "dicode_manager.score_function")
    if score is None:
        score = _config_get(config, "training.score_function")
    if score != "learnability":
        raise RuntimeError("fastpath benchmark requires score_function=learnability")


def _arm_contract(config, comparison: str, arm: str) -> None:
    if comparison not in COMPARISONS or arm not in COMPARISONS[comparison]:
        raise RuntimeError(f"invalid comparison/arm: {comparison}/{arm}")
    perf = _config_get(config, "performance")
    if perf is None:
        raise RuntimeError("performance section missing")
    for key, expected in _cfg.EXPECTED_FLAGS[arm].items():
        if bool(perf.get(key, False)) is not expected:
            raise RuntimeError(f"{arm} requires performance.{key}={expected}")
    for key in _cfg.FORCED_FALSE:
        if bool(perf.get(key, False)):
            raise RuntimeError(f"{arm} requires performance.{key}=false")
    if not bool(_config_get(config, "runtime_profiling.enabled", False)):
        raise RuntimeError("fastpath benchmark requires runtime profiling")
    if bool(perf.get("learnability_fused_preflight_summary", False)):
        score = _config_get(config, "dicode_manager.score_function")
        if score != "learnability":
            raise RuntimeError("fused fastpath is only valid for learnability")


def _runtime_source_evidence(runtime: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = manifest.get("source_config", {}).get("source", {})
    required = {
        "TaskArchive": "src/dicode/dreaming/gen_manager.py",
        "load_tasks_from_env_codes": "src/dicode/task_utils.py",
        "evaluate_new_tasks": "src/dicode/evaluation/online_evaluation.py",
        "run_evaluation_rollouts": "src/dicode/ppo_tr.py",
        "calculate_scores_from_snapshot": "src/dicode/scoring.py",
        "learnability_scores_from_counts": "experiments/training/run_dicode.py",
        "learnability_summary_contract": "src/dicode/skill_preflight/learnability_summary.py",
        "_load_agent_state": "src/dicode/setup.py",
        "route": "src/dicode/skill_preflight/preflight.py",
        "preflight_route": "src/dicode/skill_preflight/preflight_route.py",
        "resolve_preloaded_tasks": "src/dicode/skill_preflight/reuse_loaded_tasks.py",
        "heldout_eval": "src/dicode/craftax_evaluation.py",
        "wrappers_cl": "src/dicode/wrappers_cl.py",
    }
    evidence = {"verified": True, "paths": {}, "hashes": {}}
    for key, relative in required.items():
        obj = runtime.get(key)
        source = inspect.getsourcefile(obj) if obj is not None else None
        if not source:
            raise RuntimeError(f"runtime source unavailable: {key}")
        resolved = Path(source).resolve()
        entry = next(
            (
                value for value in entries.values()
                if Path(value["path"]).resolve() == resolved
            ),
            None,
        )
        if entry is None:
            raise RuntimeError(f"manifest source entry missing: {relative}")
        actual = sha256_file(resolved)
        if actual != entry["sha256"]:
            raise RuntimeError(f"runtime source hash mismatch: {key}")
        evidence["paths"][key] = str(resolved)
        evidence["hashes"][key] = actual
    return evidence


def _span(tracker, name: str):
    return tracker.span(name) if getattr(tracker, "enabled", False) else nullcontext()


def _score_preflight_result(
    raw: Mapping[str, Any], config, num_tasks: int, runtime: Mapping[str, Any], tracker
) -> tuple[dict[str, Any], str, int]:
    """Score either the fused summary or the legacy trajectory, fail-closed."""
    fused = bool(
        (_config_get(config, "performance", {}) or {}).get(
            "learnability_fused_preflight_summary", False
        )
    )
    if fused:
        runtime["require_learnability_fused_contract"](
            _config_get(config, "dicode_manager.score_function")
        )
        summary = raw.get("learnability_summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError("fused evaluate_new_tasks returned no learnability_summary")
        finished = summary.get("finished_counts")
        successes = summary.get("success_counts")
        if finished is None or successes is None:
            raise RuntimeError("fused learnability summary missing counters")
        with _span(tracker, "scoring_transfer"):
            finished, successes = runtime["device_get"]((finished, successes))
        with _span(tracker, "scoring_cpu"):
            scores = runtime["learnability_scores_from_counts"](
                finished, successes, num_tasks
            )
        payload_bytes = int(np.asarray(finished).nbytes + np.asarray(successes).nbytes)
        return scores, "fused", payload_bytes

    scoring_window = raw.get("scoring_window_data")
    if scoring_window is None:
        raise RuntimeError("legacy evaluate_new_tasks returned no scoring_window_data")
    scores = runtime["calculate_scores_from_snapshot"](
        scoring_window,
        num_tasks,
        np.asarray(raw["task_achievement_mask"]),
        np.asarray(raw["task_completed_mask"]),
        config,
    )
    return scores, "legacy", -1


def _score_projection(scores: Mapping[str, Any], task_ids: list[str]) -> list[dict[str, Any]]:
    projection = []
    for index, task_id in enumerate(task_ids):
        row = scores.get(str(index), {})
        projection.append(
            {
                "task_id": task_id,
                "sr": float(row.get("sr", -1.0)),
                "priority_score": float(row.get("priority_score", 0.0)),
            }
        )
    return projection


def _phase_total(events: list[Mapping[str, Any]], phase: str) -> float:
    return float(sum(float(event.get("duration_s", 0.0)) for event in events if event.get("phase") == phase))


def _run_arm(
    manifest: Mapping[str, Any], config, runtime: Mapping[str, Any], out: Path, args: Any
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from dicode.runtime_analysis import tracker

    stage = next(item for item in manifest["stages"] if item["name"] == args.stage)
    repeat = stage["repeats"][args.repeat]
    task_ids = [str(value) for value in stage["task_ids"]]
    out.mkdir(parents=True, exist_ok=True)

    tracker.configure(enabled=True, output_jsonl=str(out / "events.jsonl"), reset=True)
    session_name = f"{args.comparison}_{args.stage}_{args.repeat}_{args.arm}"
    tracker.set_session(session_name)
    session_start_ns = time.monotonic_ns()

    with tracker.span("checkpoint_load"):
        train_state = runtime["_load_agent_state"](config, stage["checkpoint"]["path"])
    params_before = state_hash(train_state.params)
    optimizer_before = state_hash(train_state.opt_state)

    with tracker.span("conditioning_verify"):
        conditioning, conditioning_hash = _verify_conditioning(stage)
    if conditioning_hash != stage["embedding"]["hash"]:
        raise RuntimeError("conditioning compatibility hash mismatch")

    input_rng = jnp.asarray(np.asarray(repeat["rng"], dtype=np.uint32))
    preflight_rng, heldout_rng = jax.random.split(input_rng)
    with tracker.span("archive_copy_or_load"):
        archive = runtime["reconstruct_archive"](stage["graph"]["path"])
    archive_before = runtime["archive_hash"](archive)

    with tracker.span("candidate_code_load"):
        code_map = runtime["archive_get_codes"](archive, task_ids)
        if list(code_map) != task_ids:
            raise RuntimeError("archive candidate code order mismatch")
        for candidate in stage["tasks"]:
            if sha256_bytes(code_map.get(candidate["id"], "").encode()) != candidate["code_sha256"]:
                raise RuntimeError(f"candidate code hash mismatch: {candidate['id']}")

    preflight_start_ns = time.monotonic_ns()
    with tracker.span("preflight_wall"):
        with tracker.span("preflight_task_load"):
            classes, ok_ids = runtime["load_tasks_from_env_codes"](archive, task_ids)
        if list(ok_ids) != task_ids:
            raise RuntimeError("preflight first-load id mismatch")
        perf = _config_get(config, "performance", {}) or {}
        reuse = bool(perf.get("preflight_reuse_loaded_tasks", False))
        raw = runtime["evaluate_new_tasks"](
            config,
            preflight_rng,
            train_state,
            ok_ids,
            archive,
            _FrozenEmbeddingProvider(conditioning),
            preloaded_task_classes=(classes if reuse else None),
            preloaded_task_ids=(ok_ids if reuse else None),
        )
        score_start_ns = time.monotonic_ns()
        scores, summary_mode, payload_bytes = _score_preflight_result(
            raw, config, len(ok_ids), runtime, tracker
        )
        scoring_wall_s = (time.monotonic_ns() - score_start_ns) / 1e9
        with tracker.span("route"):
            kept: list[str] = []
            runtime["preflight_route"](
                scores, ok_ids, kept, archive, runtime["route"], tracker=tracker
            )
        archive_after = runtime["archive_hash"](archive)
    preflight_wall_s = (time.monotonic_ns() - preflight_start_ns) / 1e9

    eval_embedding = _heldout_embedding(int(config.evaluation.num_envs))
    runtime["clear_eval_cache"]()
    eval_start_ns = time.monotonic_ns()
    metrics_first = runtime["heldout_eval"](
        config, heldout_rng, train_state=train_state, eval_embedding=eval_embedding, detail=False
    )
    jax.clear_caches()
    metrics_second = runtime["heldout_eval"](
        config, heldout_rng, train_state=train_state, eval_embedding=eval_embedding, detail=False
    )
    eval_wall_s = (time.monotonic_ns() - eval_start_ns) / 1e9

    import orbax.checkpoint as ocp

    checkpoint_dir = out / "checkpoint"
    manager = ocp.CheckpointManager(
        str(checkpoint_dir),
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(create=True, max_to_keep=1),
    )
    save_step = int(stage["global_step"]) + 100
    checkpoint_start_ns = time.monotonic_ns()
    try:
        manager.save(save_step, train_state)
        manager.wait_until_finished()
    finally:
        manager.close()
    checkpoint_wall_s = (time.monotonic_ns() - checkpoint_start_ns) / 1e9
    reloaded = runtime["_load_agent_state"](config, str(checkpoint_dir / str(save_step)))
    reloaded_params = state_hash(reloaded.params)
    reloaded_optimizer = state_hash(reloaded.opt_state)
    if reloaded_params != state_hash(train_state.params) or reloaded_optimizer != state_hash(train_state.opt_state):
        raise RuntimeError("checkpoint reload hash mismatch")

    tracker.record("session_wall", session_start_ns, session=session_name)
    tracker.derive_reports()
    events_path = out / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if events_path.exists() else []
    phases = {event.get("phase") for event in events}
    eval_compile_spans = [event for event in events if event.get("phase") == "eval_compile"]
    eval_execute_spans = [event for event in events if event.get("phase") == "eval_execute"]

    def metrics_projection(metrics):
        return {key: float(np.asarray(value)) for key, value in metrics.items() if not key.startswith("_")}

    evaluation_metrics = metrics_projection(metrics_first)
    evaluation_second = metrics_projection(metrics_second)
    first_fp = fingerprint(evaluation_metrics)
    second_fp = fingerprint(evaluation_second)
    if not args.perf and first_fp != second_fp:
        raise RuntimeError("heldout metrics differ between replayed sessions")

    rollout_updates = int(config.validation.rollout_updates)
    preflight_env_steps = rollout_updates * int(config.validation.num_envs) * int(config.validation.num_steps)
    heldout_env_steps = 2 * int(config.evaluation.num_steps) * int(config.evaluation.num_envs)
    session_wall_s = (time.monotonic_ns() - session_start_ns) / 1e9
    projection = _score_projection(scores, task_ids)
    runtime_markers = {
        "runtime_failure": False,
        "fatal_error": False,
        "oom": False,
        "xid": False,
        "checkpoint_error": False,
        "gpu_violation": False,
    }

    result = {
        "classification": CLASSIFICATION,
        "comparison": args.comparison,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "gpu_uuid": args.required_gpu_uuid,
        "stage": args.stage,
        "repeat": args.repeat,
        "arm": args.arm,
        "llm_api_calls": 0,
        "validation_cache_enabled": bool(
            (_config_get(config, "performance", {}) or {}).get("validation_cache", False)
        ),
        "validation_cache_exercised": False,
        "validation_cache_speedup_claimed": False,
        "params_sha256_before": params_before,
        "params_sha256_after": state_hash(train_state.params),
        "optimizer_sha256_before": optimizer_before,
        "optimizer_sha256_after": state_hash(train_state.opt_state),
        "checkpoint_reloaded_params_sha256": reloaded_params,
        "checkpoint_reloaded_optimizer_sha256": reloaded_optimizer,
        "checkpoint_loadable": True,
        "checkpoint_path": str(checkpoint_dir / str(save_step)),
        "input_rng_sha256": rng_hash(input_rng),
        "rng_sha256_before": rng_hash(input_rng),
        "preflight_rng_sha256": rng_hash(preflight_rng),
        "heldout_rng_sha256": rng_hash(heldout_rng),
        "task_ids": task_ids,
        "task_assignment_sha256": fingerprint(task_ids),
        "task_code_hashes": [candidate["code_sha256"] for candidate in stage["tasks"]],
        "embedding_hash": stage["embedding"]["hash"],
        "conditioning_type": "one_hot",
        "conditioning_shape": stage["conditioning"]["shape"],
        "conditioning_dtype": stage["conditioning"]["dtype"],
        "reset_selection_semantics": verify_selection_semantics(),
        "global_update_step": int(stage["global_step"]),
        "score_function": "learnability",
        "score_projection": projection,
        "scoring_fingerprint": fingerprint(projection),
        "accepted_ids": sorted(kept),
        "rejected_ids": sorted(task_id for task_id in task_ids if task_id not in kept),
        "archive_before_sha256": archive_before,
        "archive_after_sha256": archive_after,
        "evaluation_metrics": evaluation_metrics,
        "evaluation_metrics_sha256": first_fp,
        "evaluation_metrics_second_sha256": second_fp,
        "evaluation_metrics_equal_across_sessions": first_fp == second_fp,
        "preflight_summary_mode": summary_mode,
        "preflight_return_payload_bytes": payload_bytes,
        "compact_scoring_payload": False,
        "session_wall_s": session_wall_s,
        "preflight_wall_s": preflight_wall_s,
        "preflight_build_s": _phase_total(events, "preflight_eval_build"),
        "preflight_lower_compile_s": _phase_total(events, "preflight_eval_lower_compile"),
        "preflight_execute_s": _phase_total(events, "preflight_eval_execute"),
        "preflight_transfer_s": _phase_total(events, "scoring_transfer"),
        "preflight_scoring_cpu_s": _phase_total(events, "scoring_cpu"),
        "scoring_wall_s": scoring_wall_s,
        "route_wall_s": _phase_total(events, "route"),
        "eval_wall_s": eval_wall_s,
        "checkpoint_wall_s": checkpoint_wall_s,
        "preflight_env_steps": preflight_env_steps,
        "heldout_env_steps": heldout_env_steps,
        "total_env_steps": preflight_env_steps + heldout_env_steps,
        "preflight_throughput_env_s": preflight_env_steps / max(preflight_wall_s, 1e-9),
        "eval_throughput_env_s": heldout_env_steps / max(eval_wall_s, 1e-9),
        "session_throughput_env_s": (preflight_env_steps + heldout_env_steps) / max(session_wall_s, 1e-9),
        "preflight_task_reload_occurred": "preflight_task_reload" in phases,
        "preflight_task_reload_explicit_absent": "preflight_task_reload" not in phases,
        "eval_compile_span_count": len(eval_compile_spans),
        "eval_cache_hit_count": sum(1 for event in eval_execute_spans if event.get("cache_hit")),
        "eval_first_cache_miss": bool(
            eval_execute_spans and not eval_execute_spans[0].get("cache_hit", True)
        ),
        "gpu_peak_memory_mib": None,
        "gpu_min_free_mib": None,
        "runtime_source_evidence": runtime["source_evidence"](manifest),
        "env_evidence": env_evidence(),
        "profiling": {
            "enabled": True,
            "event_count": len(events),
            "events_csv_sha256": sha256_file(out / "events.csv") if (out / "events.csv").exists() else None,
            "critical_path_sha256": sha256_file(out / "critical_path.json") if (out / "critical_path.json").exists() else None,
        },
        **runtime_markers,
    }
    atomic_json(out / "RESULT.json", result)
    return result


def _load_run_dicode_module():
    path = Path(__file__).parents[2] / "experiments" / "training" / "run_dicode.py"
    spec = importlib.util.spec_from_file_location("perf48_fastpath_run_dicode", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _real_runtime(manifest: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    import jax
    from dicode.dreaming.gen_manager import TaskArchive
    from dicode.task_utils import load_tasks_from_env_codes
    from dicode.evaluation import evaluate_new_tasks
    from dicode.ppo_tr import run_evaluation_rollouts
    from dicode.scoring import calculate_scores_from_snapshot
    from dicode.setup import _load_agent_state
    from dicode.skill_preflight.learnability_summary import require_learnability_fused_contract
    from dicode.skill_preflight.preflight import route
    from dicode.skill_preflight.preflight_route import preflight_route
    from dicode.skill_preflight.reuse_loaded_tasks import resolve_preloaded_tasks
    from dicode import craftax_evaluation
    import dicode.wrappers_cl as wrappers_cl

    driver = _load_run_dicode_module()
    runtime = {
        "reconstruct_archive": _reconstruct_archive,
        "archive_hash": _graph_sha,
        "archive_get_codes": lambda archive, ids: archive.get_task_codes(ids),
        "_load_agent_state": _load_agent_state,
        "load_tasks_from_env_codes": load_tasks_from_env_codes,
        "evaluate_new_tasks": evaluate_new_tasks,
        "run_evaluation_rollouts": run_evaluation_rollouts,
        "calculate_scores_from_snapshot": calculate_scores_from_snapshot,
        "learnability_scores_from_counts": driver._learnability_scores_from_counts,
        "require_learnability_fused_contract": require_learnability_fused_contract,
        "device_get": jax.device_get,
        "TaskArchive": TaskArchive,
        "route": route,
        "preflight_route": preflight_route,
        "resolve_preloaded_tasks": resolve_preloaded_tasks,
        "heldout_eval": craftax_evaluation.main,
        "clear_eval_cache": craftax_evaluation.clear_compiled_evaluator_cache,
        "craftax_evaluation": craftax_evaluation,
        "wrappers_cl": wrappers_cl,
        "learnability_summary_contract": require_learnability_fused_contract,
    }
    runtime["source_evidence"] = lambda loaded: _runtime_source_evidence(runtime, loaded)
    return runtime


def _preflight(args, manifest: Mapping[str, Any]) -> dict[str, Any]:
    import jax

    _verify_gpu(args)
    config = _load_config(args.config)
    _config_contract(config)
    _arm_contract(config, args.comparison, args.arm)
    runtime = _real_runtime(manifest, Path(args.out))
    result = {
        "classification": CLASSIFICATION,
        "comparison": args.comparison,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "gpu_uuid": args.required_gpu_uuid,
        "stage": args.stage,
        "repeat": args.repeat,
        "arm": args.arm,
        "jax_backend": jax.default_backend(),
        "jax_device_count": len(jax.devices()),
        "validation_cache_exercised": False,
        "llm_api_calls": 0,
        "runtime_source_evidence": runtime["source_evidence"](manifest),
        "env_evidence": env_evidence(),
        "pass": True,
    }
    atomic_json(Path(args.out) / "PREFLIGHT.json", result)
    return result


def _exception_text() -> str:
    import traceback

    return traceback.format_exc()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--required-gpu-uuid", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--comparison", choices=tuple(COMPARISONS), required=True)
    parser.add_argument("--stage", choices=("early", "mid", "late"), required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--perf", action="store_true")
    args = parser.parse_args(argv)
    if args.arm not in COMPARISONS[args.comparison]:
        parser.error(f"{args.arm} does not belong to {args.comparison}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    if args.mode == "preflight":
        _preflight(args, manifest)
        return 0
    _verify_gpu(args)
    config = _load_config(args.config)
    _config_contract(config)
    _arm_contract(config, args.comparison, args.arm)
    try:
        _run_arm(manifest, config, _real_runtime(manifest, out), out, args)
    except Exception:
        atomic_json(out / "FAILURE.json", {"error": _exception_text()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
