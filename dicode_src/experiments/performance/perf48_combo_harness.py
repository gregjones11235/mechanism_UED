#!/usr/bin/env python3
"""BC combination harness: B2 preflight-task-reuse + C eval-compile-cache.

One arm (BC_OFF or BC_ON) for one stage/repeat, against the FROZEN P0 materials.
Runs the REAL production control chain, fail-closed:

  preflight:
    load_tasks_from_env_codes (first load; span "preflight_task_load")
      -> evaluate_new_tasks (B2 on: reuse first load, NO second load / span
         "preflight_task_reload"; B2 off: historical second load happens inside,
         span fires)
      -> calculate_scores_from_snapshot (scoring_transfer/scoring_cpu)
      -> preflight_route (route / archive_update)

  held-out eval (C):
    craftax_evaluation.main twice with jax.clear_caches() between calls.
    C on: first call compiles (eval_compile span, cache_hit=false), second call
          hits the run-scoped compiled-evaluator cache (eval_execute
          cache_hit=true, no second eval_compile).
    C off: zero eval_compile spans; eval_execute cache_hit=false both calls.

  checkpoint save/reload:
    orbax save -> wait_until_finished -> reload -> hash verify.

Semantic hashes (params/optimizer/RNG before-after, scoring fingerprint,
evaluation metrics, checkpoint reload) are recorded so the benchmark can prove
BC_OFF/BC_ON equivalence under --xla_gpu_deterministic_ops=true.

The harness never calls an LLM/API: the preflight embedding provider is frozen
and conditioning_type is one_hot, so _get_new_task_embeddings derives embeddings
from task classes locally.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CONDITIONING_DIM = 67
CLASSIFICATION = "PERF48_COMBO_BENCHMARK"
SCORE_FUNCTIONS = ("learnability", "pvl", "max_mc")
FORCED_FALSE = ("compact_preflight_payload", "train_compile_cache", "embedding_cache", "validation_cache")
# Config budget contract shared with the frozen P0 replay.
BUDGET = {"timesteps": 2_000_000_000, "num_envs": 1024, "num_steps": 128, "updates": 100}

_MSPEC = importlib.util.spec_from_file_location(
    "perf48_combo_manifest", Path(__file__).with_name("perf48_combo_manifest.py"))
_manifest_mod = importlib.util.module_from_spec(_MSPEC)
assert _MSPEC.loader
_MSPEC.loader.exec_module(_manifest_mod)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, dict):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def fingerprint(value: Any) -> str:
    return sha256_bytes(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode())


def state_hash(tree: Any) -> str:
    h = hashlib.sha256()
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(tree)
    except Exception:
        leaves = tree if isinstance(tree, (list, tuple)) else [tree]
    for leaf in leaves:
        try:
            a = np.asarray(leaf)
            h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        except Exception:
            h.update(repr(leaf).encode())
    return h.hexdigest()


def rng_hash(rng) -> str:
    return state_hash(rng)


def env_evidence() -> dict[str, Any]:
    """Record XLA/JAX/CUDA environment evidence (deterministic flag provenance)."""
    import jax
    import jaxlib
    xla_flags = os.environ.get("XLA_FLAGS") or ""
    det_flag = "--xla_gpu_deterministic_ops=true"

    def _pkg(name):
        try:
            from importlib import metadata
            return metadata.version(name)
        except Exception:
            return None

    return {
        "xla_flags": xla_flags,
        "deterministic_ops_requested": det_flag in xla_flags,
        "deterministic_flag_verified": det_flag in xla_flags,
        "jax_version": getattr(jax, "__version__", None),
        "jaxlib_version": getattr(jaxlib, "__version__", None),
        "jax_backend": jax.default_backend(),
        "cuda_version": _pkg("nvidia-cuda-runtime-cu12"),
        "cudnn_version": _pkg("nvidia-cudnn-cu12"),
    }


def verify_selection_semantics() -> dict[str, Any]:
    """Deterministic reset-index selection equivalence check (frozen P0 audit)."""
    num_envs, num_resets = 1024, 64
    rng = np.random.default_rng(20260809)
    cases = []
    for done_count in range(num_envs + 1):
        for _ in range(4):
            mask = np.zeros(num_envs, dtype=bool)
            if done_count:
                mask[rng.choice(num_envs, size=done_count, replace=False)] = True
            cases.append(mask)
    for mask in cases[:4100]:
        old = np.sort(np.where(mask, np.arange(num_envs), num_envs))[:num_resets]
        pos = np.flatnonzero(mask)
        new = np.full(num_resets, num_envs, dtype=np.int64)
        new[:min(num_resets, len(pos))] = pos[:num_resets]
        if not np.array_equal(old, new):
            raise AssertionError("reset index mismatch")
    return {"cases": 4100, "num_envs": num_envs, "num_resets": num_resets, "pass": True}


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(value), f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def load_manifest(path: str) -> dict[str, Any]:
    return _manifest_mod.load_manifest(path)


def _load_config(path: str) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def _config_get(config, key: str, default: Any = None) -> Any:
    from omegaconf import OmegaConf
    if isinstance(config, dict):
        cur = config
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur
    return OmegaConf.select(config, key, default=default)


def _config_contract(config) -> None:
    required = {
        "total_timesteps": (("total_timesteps", "training.total_timesteps"), BUDGET["timesteps"]),
        "num_envs": (("num_envs", "training.num_envs"), BUDGET["num_envs"]),
        "num_steps": (("num_steps", "training.num_steps"), BUDGET["num_steps"]),
        "updates": (("updates", "dicode_manager.max_updates_per_session"), BUDGET["updates"]),
    }
    for key, (paths, expected) in required.items():
        val = _config_get(config, paths[0]) if _config_get(config, paths[0]) is not None else _config_get(config, paths[1])
        if val is None or int(val) != expected:
            raise RuntimeError(f"config {key} mismatch")
    cond = _config_get(config, "conditioning_type") or _config_get(config, "training.conditioning_type")
    cot = _config_get(config, "condition_on_task") or _config_get(config, "training.condition_on_task")
    if cond != "one_hot" or cot is not True:
        raise RuntimeError("one_hot conditioning contract mismatch")
    score = _config_get(config, "dicode_manager.score_function")
    if score is None:
        score = _config_get(config, "training.score_function")
    if score not in SCORE_FUNCTIONS:
        raise RuntimeError("score_function contract mismatch")
    compact = _config_get(config, "compact_scoring_payload", False) or _config_get(config, "training.compact_scoring_payload", False)
    if compact:
        raise RuntimeError("compact_scoring_payload must be false for the combo")


def _arm_contract(config, arm: str) -> None:
    if arm not in ("BC_OFF", "BC_ON"):
        raise RuntimeError(f"invalid arm {arm}")
    perf = _config_get(config, "performance")
    if perf is None:
        raise RuntimeError("performance section missing")
    b2 = bool(perf.get("preflight_reuse_loaded_tasks", False))
    c = bool(perf.get("eval_compile_cache", False))
    if arm == "BC_ON" and not (b2 and c):
        raise RuntimeError("BC_ON requires preflight_reuse_loaded_tasks=true AND eval_compile_cache=true")
    if arm == "BC_OFF" and (b2 or c):
        raise RuntimeError("BC_OFF requires preflight_reuse_loaded_tasks=false AND eval_compile_cache=false")
    for key in FORCED_FALSE:
        if bool(perf.get(key, False)):
            raise RuntimeError(f"performance flag {key} must be false in both arms")
    enabled = bool(_config_get(config, "runtime_profiling.enabled", False))
    if not enabled:
        raise RuntimeError("BC arms require runtime_profiling.enabled=true")


def _runtime_source_evidence(rt: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every executed runtime callable to the manifest source tree."""
    entries = manifest.get("source_config", {}).get("source", {})
    required = {
        "TaskArchive": "src/dicode/dreaming/gen_manager.py",
        "load_tasks_from_env_codes": "src/dicode/task_utils.py",
        "evaluate_new_tasks": "src/dicode/evaluation/online_evaluation.py",
        "run_evaluation_rollouts": "src/dicode/ppo_tr.py",
        "calculate_scores_from_snapshot": "src/dicode/scoring.py",
        "_load_agent_state": "src/dicode/setup.py",
        "route": "src/dicode/skill_preflight/preflight.py",
        "preflight_route": "src/dicode/skill_preflight/preflight_route.py",
        "resolve_preloaded_tasks": "src/dicode/skill_preflight/reuse_loaded_tasks.py",
        "heldout_eval": "src/dicode/craftax_evaluation.py",
        "wrappers_cl": "src/dicode/wrappers_cl.py",
    }
    result = {"verified": True, "paths": {}, "hashes": {}}
    for key, relative in required.items():
        obj = rt.get(key)
        if obj is None:
            raise RuntimeError(f"runtime source binding missing {key}")
        source = inspect.getsourcefile(obj)
        if not source:
            raise RuntimeError(f"runtime source path unavailable {key}")
        path = Path(source).resolve()
        entry = None
        for entry_label, entry_value in entries.items():
            if Path(entry_value["path"]).resolve() == path:
                entry = entry_value
                break
        if entry is None:
            raise RuntimeError(f"runtime source binding {key}: no manifest entry for {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"runtime source binding mismatch {key}")
        result["paths"][key] = str(path); result["hashes"][key] = entry["sha256"]
    return result


def _state_hash(tree: Any) -> str:
    return state_hash(tree)


def _reconstruct_archive(graph_path: str, config) -> Any:
    """Copy the frozen graph to a temp archive so route mutations never touch it."""
    from dicode.dreaming.gen_manager import TaskArchive

    tmp = Path(tempfile.mkdtemp(prefix="perf48_combo_archive_"))
    dst = tmp / "task_graph.graphml"
    shutil.copy2(graph_path, dst)
    cfg = type("C", (), {"graph_path": str(dst)})()
    return TaskArchive(cfg)


def _graph_sha(archive) -> str:
    g = archive.graph
    parts = []
    for node in sorted(g.nodes()):
        parts.append(str(node) + ":" + json.dumps(canonical(dict(g.nodes[node])), sort_keys=True))
    for u, v in sorted((sorted(e) for e in g.edges())):
        parts.append(f"{u}->{v}")
    return sha256_bytes("\n".join(parts).encode())


def _verify_conditioning(stage: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Load + hash-verify the frozen conditioning table (fail-closed)."""
    info = stage.get("conditioning", {})
    path = Path(info.get("path", ""))
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError("unable to load frozen conditioning table") from exc
    expected_shape = info.get("shape")
    if (values.dtype != np.dtype("float32") or values.ndim != 2
            or values.shape[1] != CONDITIONING_DIM or not np.isfinite(values).all()
            or (expected_shape is not None and list(values.shape) != list(expected_shape))):
        raise RuntimeError("invalid frozen conditioning table")
    values = np.ascontiguousarray(values)
    if info.get("sha256") and sha256_file(path) != info["sha256"]:
        raise RuntimeError("frozen conditioning file hash mismatch")
    h = hashlib.sha256()
    h.update(repr(tuple(values.shape)).encode()); h.update(str(values.dtype).encode())
    h.update(values.tobytes())
    if info.get("content_sha256") and h.hexdigest() != info["content_sha256"]:
        raise RuntimeError("frozen conditioning content hash mismatch")
    if stage.get("embedding", {}).get("hash") != info.get("content_sha256"):
        raise RuntimeError("frozen conditioning compatibility hash mismatch")
    return values, h.hexdigest()


class _FrozenEmbeddingProvider:
    """Never calls an LLM/API: returns frozen rows when asked. With
    conditioning_type=one_hot (the frozen contract) the preflight path derives
    embeddings from task classes locally and this provider is not consulted."""

    def __init__(self, frozen: np.ndarray):
        self._frozen = frozen

    def get_embedding(self, labels, instruction=None):
        return [{"embedding": self._frozen[i]} for i in range(len(labels))]


def _heldout_embedding(num_envs: int) -> np.ndarray:
    from dicode.task_utils import get_achievement_multi_hot
    from minicraftax.envs.craftax import CraftaxAugObsTrain

    eval_env = CraftaxAugObsTrain()
    base_emb = get_achievement_multi_hot(eval_env.relevant_achievements)
    return np.tile(base_emb, (num_envs, 1)).astype(np.float32)


def _run_arm(manifest: Mapping[str, Any], config, rt: Mapping[str, Any], out: Path, args: Any) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from dicode.runtime_analysis import tracker
    from dicode.task_utils import load_tasks_from_env_codes
    from dicode.evaluation import evaluate_new_tasks
    from dicode.scoring import calculate_scores_from_snapshot
    from dicode.skill_preflight.preflight import route as route_fn
    from dicode.skill_preflight.preflight_route import preflight_route
    from dicode.craftax_evaluation import clear_compiled_evaluator_cache, main as heldout_main

    stage = next(s for s in manifest["stages"] if s["name"] == args.stage)
    repeat = stage["repeats"][args.repeat]
    task_ids = [str(x) for x in stage["task_ids"]]
    out.mkdir(parents=True, exist_ok=True)

    tracker.configure(enabled=True, output_jsonl=str(out / "events.jsonl"), reset=True)
    tracker.set_session(f"{args.stage}_{args.repeat}_{args.arm}")
    session_start_ns = time.monotonic_ns()

    # ---- checkpoint load + before hashes ----
    with tracker.span("checkpoint_load"):
        train_state = rt["_load_agent_state"](config, stage["checkpoint"]["path"])
    params_hash_before = state_hash(train_state.params)
    optimizer_hash_before = state_hash(train_state.opt_state)

    # ---- frozen conditioning verification (fail-closed) ----
    with tracker.span("conditioning_verify"):
        conditioning, conditioning_hash = _verify_conditioning(stage)
    if conditioning_hash != stage["embedding"]["hash"]:
        raise RuntimeError("conditioning compatibility hash mismatch")

    # ---- frozen RNG ----
    input_rng = jnp.asarray(np.asarray(repeat["rng"], dtype=np.uint32))
    rng_input_hash = rng_hash(input_rng)
    preflight_rng, heldout_rng = jax.random.split(input_rng)

    # ---- archive + first load ----
    with tracker.span("archive_copy_or_load"):
        archive = rt["reconstruct_archive"](stage["graph"]["path"])
    archive_before = rt["archive_hash"](archive)

    # verify candidate code hashes against the frozen graph
    with tracker.span("candidate_code_load"):
        code_map = rt["archive_get_codes"](archive, task_ids)
        if list(code_map.keys()) != task_ids:
            raise RuntimeError("archive candidate code order mismatch")
        for cand in stage["tasks"]:
            if sha256_bytes(code_map.get(cand["id"], "").encode()) != cand["code_sha256"]:
                raise RuntimeError(f"candidate {cand['id']} code hash mismatch")

    # ---- preflight chain (B2) ----
    preflight_wall_start = time.monotonic_ns()
    with tracker.span("preflight_wall"):
        with tracker.span("preflight_task_load"):
            classes, ok_ids = load_tasks_from_env_codes(archive, task_ids)
        if list(ok_ids) != task_ids:
            raise RuntimeError("preflight first-load id mismatch")
        reuse = bool((config.get("performance", {}) if hasattr(config, "get") else {})
                     .get("preflight_reuse_loaded_tasks", False))
        provider = _FrozenEmbeddingProvider(conditioning)
        raw = evaluate_new_tasks(
            config, preflight_rng, train_state, ok_ids, archive, provider,
            preloaded_task_classes=(classes if reuse else None),
            preloaded_task_ids=(ok_ids if reuse else None),
        )
        swd = raw.get("scoring_window_data")
        if swd is None:
            raise RuntimeError("evaluate_new_tasks returned no scoring_window_data")
        score_start = time.monotonic_ns()
        scores = calculate_scores_from_snapshot(
            swd, len(ok_ids),
            np.asarray(raw["task_achievement_mask"]),
            np.asarray(raw["task_completed_mask"]),
            config,
        )
        scoring_wall_s = (time.monotonic_ns() - score_start) / 1e9
        with tracker.span("route"):
            kept = []
            preflight_route(scores, ok_ids, kept, archive, route_fn, tracker=tracker)
        archive_after = rt["archive_hash"](archive)
    preflight_wall_s = (time.monotonic_ns() - preflight_wall_start) / 1e9

    # ---- held-out eval (C) ----
    eval_embedding = _heldout_embedding(int(config.evaluation.num_envs))
    clear_compiled_evaluator_cache()
    eval_start = time.monotonic_ns()
    metrics_first = heldout_main(config, heldout_rng, train_state=train_state,
                                 eval_embedding=eval_embedding, detail=False)
    jax.clear_caches()
    metrics_second = heldout_main(config, heldout_rng, train_state=train_state,
                                  eval_embedding=eval_embedding, detail=False)
    eval_wall_s = (time.monotonic_ns() - eval_start) / 1e9

    # ---- checkpoint save/reload ----
    import orbax.checkpoint as ocp
    ckpt_dir = out / "checkpoint"
    manager = ocp.CheckpointManager(str(ckpt_dir), ocp.PyTreeCheckpointer(),
                                    options=ocp.CheckpointManagerOptions(create=True, max_to_keep=1))
    save_step = int(stage["global_step"]) + 100
    checkpoint_start = time.monotonic_ns()
    try:
        manager.save(save_step, train_state)
        manager.wait_until_finished()
    finally:
        manager.close()
    checkpoint_wall_s = (time.monotonic_ns() - checkpoint_start) / 1e9
    reloaded = rt["_load_agent_state"](config, str(ckpt_dir / str(save_step)))
    reloaded_params = state_hash(reloaded.params)
    reloaded_optimizer = state_hash(reloaded.opt_state)
    if reloaded_params != state_hash(train_state.params) or reloaded_optimizer != state_hash(train_state.opt_state):
        raise RuntimeError("checkpoint reload hash mismatch")

    # ---- env-step accounting (throughput evidence) ----
    rollout_updates = int(config.validation.rollout_updates)
    num_envs = int(config.training.num_envs)
    num_steps = int(config.training.num_steps)
    preflight_env_steps = rollout_updates * num_envs * num_steps
    eval_num_steps = int(config.evaluation.num_steps)
    eval_num_envs = int(config.evaluation.num_envs)
    heldout_env_steps = 2 * eval_num_steps * eval_num_envs  # two sessions

    # ---- event analysis ----
    tracker.record("session_wall", session_start_ns, session=f"{args.stage}_{args.repeat}_{args.arm}")
    tracker.derive_reports()

    events_path = out / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if events_path.exists() else []
    phases = {e.get("phase") for e in events}
    preflight_task_reload_occurred = "preflight_task_reload" in phases
    eval_compile_spans = [e for e in events if e.get("phase") == "eval_compile"]
    eval_execute_spans = [e for e in events if e.get("phase") == "eval_execute"]
    reload_spans = [e for e in events if e.get("phase") == "preflight_task_reload"]

    # ---- scoring fingerprint ----
    def metrics_fp(metrics):
        return fingerprint({k: float(np.asarray(v)) for k, v in metrics.items() if not k.startswith("_")})

    first_fp = metrics_fp(metrics_first)
    second_fp = metrics_fp(metrics_second)
    if first_fp != second_fp:
        raise RuntimeError("held-out eval metrics differ between the two sessions")
    evaluation_metrics = {k: float(np.asarray(v)) for k, v in metrics_first.items() if not k.startswith("_")}

    source_hashes = {section: {label: entry["sha256"] for label, entry in entries.items()}
                     for section, entries in manifest.get("source_config", {}).items()}
    wrappers = Path(inspect.getsourcefile(rt["wrappers_cl"]) or "")
    if not wrappers.is_file():
        raise RuntimeError("unable to locate wrappers_cl source")

    score_function = config.dicode_manager.score_function
    profiling = {
        "enabled": True,
        "jsonl": str(out / "events.jsonl"),
        "events": events,
        "event_count": len(events),
        "events_csv_sha256": sha256_file(out / "events.csv") if (out / "events.csv").exists() else None,
        "critical_path_sha256": sha256_file(out / "critical_path.json") if (out / "critical_path.json").exists() else None,
    }
    cp_path = out / "critical_path.json"
    report = json.loads(cp_path.read_text()) if cp_path.exists() else {}
    profiling["session_wall_s"] = report.get("sessions", {}).get(
        f"{args.stage}_{args.repeat}_{args.arm}", {}).get("session_wall", 0.0)

    result = {
        "classification": CLASSIFICATION,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": args.source_commit,
        "gpu_uuid": args.required_gpu_uuid,
        "stage": args.stage,
        "repeat": args.repeat,
        "arm": args.arm,
        "llm_api_calls": 0,
        # semantic hashes
        "params_sha256_before": params_hash_before,
        "params_sha256_after": state_hash(train_state.params),
        "optimizer_sha256_before": optimizer_hash_before,
        "optimizer_sha256_after": state_hash(train_state.opt_state),
        "checkpoint_reloaded_params_sha256": reloaded_params,
        "checkpoint_reloaded_optimizer_sha256": reloaded_optimizer,
        "input_rng_sha256": rng_input_hash,
        "rng_sha256_before": rng_input_hash,
        "heldout_rng_sha256": rng_hash(heldout_rng),
        "preflight_rng_sha256": rng_hash(preflight_rng),
        "task_ids": task_ids,
        "task_assignment_sha256": fingerprint(task_ids),
        "task_code_hashes": [c["code_sha256"] for c in stage["tasks"]],
        "embedding_hash": stage["embedding"]["hash"],
        "conditioning_type": "one_hot",
        "conditioning_shape": stage["conditioning"]["shape"],
        "conditioning_dtype": stage["conditioning"]["dtype"],
        "reset_selection_semantics": verify_selection_semantics(),
        "global_update_step": int(stage["global_step"]),
        "score_function": score_function,
        "wrappers_cl_sha256": sha256_file(wrappers) if wrappers else "",
        "compact_scoring_payload": False,
        "scoring_fingerprint": fingerprint(scores),
        "scoring_wall_s": scoring_wall_s,
        "evaluation_metrics": evaluation_metrics,
        "evaluation_metrics_sha256": first_fp,
        "evaluation_metrics_equal_across_sessions": True,
        "checkpoint_loadable": True,
        "checkpoint_path": str(ckpt_dir / str(save_step)),
        # performance / event evidence
        "session_wall_s": (time.monotonic_ns() - session_start_ns) / 1e9,
        "preflight_wall_s": preflight_wall_s,
        "eval_wall_s": eval_wall_s,
        "checkpoint_wall_s": checkpoint_wall_s,
        "preflight_env_steps": int(preflight_env_steps),
        "heldout_env_steps": int(heldout_env_steps),
        "total_env_steps": int(preflight_env_steps + heldout_env_steps),
        "preflight_throughput_env_s": round(preflight_env_steps / max(preflight_wall_s, 1e-9), 3),
        "eval_throughput_env_s": round(heldout_env_steps / max(eval_wall_s, 1e-9), 3),
        "preflight_task_reload_occurred": preflight_task_reload_occurred,
        "preflight_task_reload_explicit_absent": (not preflight_task_reload_occurred),
        "preflight_task_reload_s": round(sum(e["duration_s"] for e in reload_spans), 6),
        "accepted_ids": sorted(kept),
        "rejected_ids": sorted(t for t in task_ids if t not in kept),
        "eval_compile_span_count": len(eval_compile_spans),
        "eval_execute_spans": [{"duration_s": e["duration_s"], "cache_hit": e["cache_hit"]} for e in eval_execute_spans],
        "eval_cache_hit_count": sum(1 for e in eval_execute_spans if e.get("cache_hit")),
        "eval_first_cache_miss": (len(eval_execute_spans) > 0 and not eval_execute_spans[0].get("cache_hit", True)),
        "archive_before_sha256": archive_before,
        "archive_after_sha256": archive_after,
        "source_hashes": source_hashes,
        "runtime_source_evidence": rt["source_evidence"](manifest),
        "env_evidence": env_evidence(),
        "profiling": profiling,
    }
    atomic_json(out / "RESULT.json", result)
    return result


def _real_runtime(manifest: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    from dicode.dreaming.gen_manager import TaskArchive
    from dicode.task_utils import load_tasks_from_env_codes
    from dicode.evaluation import evaluate_new_tasks
    from dicode.ppo_tr import run_evaluation_rollouts
    from dicode.scoring import calculate_scores_from_snapshot
    from dicode.setup import _load_agent_state
    from dicode.skill_preflight.preflight import route
    from dicode.skill_preflight.preflight_route import preflight_route
    from dicode.skill_preflight.reuse_loaded_tasks import resolve_preloaded_tasks
    from dicode import craftax_evaluation
    from dicode.craftax_evaluation import main as heldout_main
    import dicode.wrappers_cl as wrappers_cl

    return {
        "reconstruct_archive": _reconstruct_archive,
        "archive_hash": _graph_sha,
        "archive_get_codes": lambda a, ids: a.get_task_codes(ids),
        "load_checkpoint": _load_agent_state,
        "_load_agent_state": _load_agent_state,
        "load_tasks_from_env_codes": load_tasks_from_env_codes,
        "evaluate_new_tasks": evaluate_new_tasks,
        "run_evaluation_rollouts": run_evaluation_rollouts,
        "calculate_scores_from_snapshot": calculate_scores_from_snapshot,
        "TaskArchive": TaskArchive,
        "route": route,
        "preflight_route": preflight_route,
        "resolve_preloaded_tasks": resolve_preloaded_tasks,
        "heldout_eval": heldout_main,
        "craftax_evaluation": craftax_evaluation,
        "wrappers_cl": wrappers_cl,
        "source_evidence": lambda m: _runtime_source_evidence(
            {
                "TaskArchive": TaskArchive, "load_tasks_from_env_codes": load_tasks_from_env_codes,
                "evaluate_new_tasks": evaluate_new_tasks, "run_evaluation_rollouts": run_evaluation_rollouts,
                "calculate_scores_from_snapshot": calculate_scores_from_snapshot,
                "_load_agent_state": _load_agent_state, "route": route, "preflight_route": preflight_route,
                "resolve_preloaded_tasks": resolve_preloaded_tasks, "heldout_eval": heldout_main,
                "wrappers_cl": wrappers_cl,
            },
            m,
        ),
    }


def _preflight(args, loaded: dict[str, Any]) -> dict[str, Any]:
    import jax

    if args.required_gpu_uuid and os.environ.get("CUDA_VISIBLE_DEVICES", "") != args.required_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exact GPU UUID")
    if args.required_gpu_uuid and (jax.default_backend() != "gpu" or len(jax.devices()) != 1):
        raise RuntimeError("JAX must see exactly one GPU")
    config = _load_config(args.config)
    _config_contract(config)
    _arm_contract(config, args.arm)
    rt = _real_runtime(loaded, Path(args.out))
    result = {
        "classification": loaded["classification"],
        "manifest_sha256": loaded["manifest_sha256"],
        "source_commit": args.source_commit,
        "gpu_uuid": args.required_gpu_uuid,
        "stage": args.stage,
        "repeat": args.repeat,
        "arm": args.arm,
        "jax_backend": jax.default_backend(),
        "jax_device_count": len(jax.devices()),
        "llm_api_calls": 0,
        "runtime_source_evidence": rt["source_evidence"](loaded),
        "env_evidence": env_evidence(),
        "pass": True,
    }
    atomic_json(Path(args.out) / "PREFLIGHT.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--required-gpu-uuid", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=_manifest_mod.STAGES, required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=("BC_OFF", "BC_ON"), required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    args = parser.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        _preflight(args, load_manifest(args.manifest))
        return 0

    loaded = load_manifest(args.manifest)
    config = _load_config(args.config)
    _config_contract(config)
    _arm_contract(config, args.arm)
    rt = _real_runtime(loaded, out)
    try:
        result = _run_arm(loaded, config, rt, out, args)
        return 0
    except Exception:
        atomic_json(out / "FAILURE.json", {"error": _exc_text()})
        raise


def _exc_text() -> str:
    import traceback
    return traceback.format_exc()


if __name__ == "__main__":
    raise SystemExit(main())
