"""Deterministic E3v2 training-kernel harness (no LLM/API calls)."""
from __future__ import annotations
import argparse, hashlib, importlib.util, inspect, json, os, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_MSPEC = importlib.util.spec_from_file_location("e3v2_replay_manifest", Path(__file__).with_name("e3v2_replay_manifest.py"))
manifest = importlib.util.module_from_spec(_MSPEC); _MSPEC.loader.exec_module(manifest)

def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256();
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray): return canonical(value.tolist())
    if isinstance(value, np.generic): return canonical(value.item())
    if isinstance(value, float):
        if np.isnan(value): return "NaN"
        if np.isposinf(value): return "Inf"
        if np.isneginf(value): return "-Inf"
    if isinstance(value, dict): return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
    return value

def fingerprint(value: Any) -> str:
    return sha256_bytes(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode())

def canonical_scoring_metrics(metrics: Any) -> Any:
    """Semantic scoring canonicalization; arrays/scalars and NaN/Inf are stable."""
    return canonical(metrics)

def scoring_fingerprint(metrics: Any) -> str:
    return fingerprint(canonical_scoring_metrics(metrics))

def state_hash(tree: Any) -> str:
    h = hashlib.sha256()
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(tree)
    except Exception: leaves = tree if isinstance(tree, (list, tuple)) else [tree]
    for leaf in leaves:
        try:
            a = np.asarray(leaf); h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
        except Exception: h.update(repr(leaf).encode())
    return h.hexdigest()

def rng_hash(rng): return state_hash(rng)
def load_frozen_conditioning(stage):
    info = stage.get("conditioning", {})
    path = Path(info.get("path", ""))
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError("unable to load frozen conditioning table") from exc
    if values.dtype != np.dtype("float32") or values.ndim != 2 or values.shape[1] != manifest.CONDITIONING_DIM or list(values.shape) != list(info.get("shape", [])) or not np.isfinite(values).all():
        raise RuntimeError("invalid frozen conditioning table")
    values = np.ascontiguousarray(values)
    if info.get("sha256") and sha256_file(path) != info["sha256"]:
        raise RuntimeError("frozen conditioning file hash mismatch")
    h = hashlib.sha256(); h.update(repr(tuple(values.shape)).encode()); h.update(str(values.dtype).encode()); h.update(values.tobytes())
    if h.hexdigest() != info.get("content_sha256") or h.hexdigest() != stage.get("embedding", {}).get("hash"):
        raise RuntimeError("frozen conditioning hash mismatch")
    return values, h.hexdigest()

def real_jax_debug_callback(fn, *args):
    """Keep the production JAX callback primitive available (never monkeypatched)."""
    import jax
    return jax.debug.callback(fn, *args)

def verify_selection_semantics():
    num_envs, num_resets = 1024, 64; rng = np.random.default_rng(20260809); cases = []
    for done_count in range(num_envs + 1):
        for _ in range(4):
            mask = np.zeros(num_envs, dtype=bool)
            if done_count: mask[rng.choice(num_envs, size=done_count, replace=False)] = True
            cases.append(mask)
    for mask in cases[:4100]:
        old = np.sort(np.where(mask, np.arange(num_envs), num_envs))[:num_resets]; pos = np.flatnonzero(mask)
        new = np.full(num_resets, num_envs, dtype=np.int64); new[:min(num_resets, len(pos))] = pos[:num_resets]
        if not np.array_equal(old, new): raise AssertionError("reset index mismatch")
    return {"cases": 4100, "num_envs": num_envs, "num_resets": num_resets, "pass": True}

def _atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f: json.dump(canonical(value), f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_manifest(path): return manifest.load_manifest(path)

def _runtime_source_evidence(rt, loaded, arm):
    """Bind every runtime callable to the selected arm's manifest source tree."""
    root = {"E0": "e0", "E3V2": "e3v2"}[arm]
    entries = loaded.get("source_config", {}).get("source", {})
    required = {
        "run_training_session": ("run_training_session", "src/dicode/ppo_tr.py"),
        "calculate_scores_from_snapshot": ("calculate_scores_from_snapshot", "src/dicode/scoring.py"),
        "wrappers_cl": ("wrappers_cl", "src/dicode/wrappers_cl.py"),
        "TaskArchive": ("TaskArchive", "src/dicode/dreaming/gen_manager.py"),
        "load_tasks_from_env_codes": ("load_tasks_from_env_codes", "src/dicode/task_utils.py"),
        "_load_agent_state": ("_load_agent_state", "src/dicode/setup.py"),
    }
    if "_calculate_task_distribution" in rt:
        required["_calculate_task_distribution"] = ("_calculate_task_distribution", "src/dicode/training.py")
    elif "_create_achievement_masks" in rt:
        required["_create_achievement_masks"] = ("_create_achievement_masks", "src/dicode/training.py")
    result = {"verified": True, "paths": {}, "hashes": {}}
    for key, (name, relative) in required.items():
        obj = rt.get(name)
        if obj is None:
            raise RuntimeError(f"runtime source binding missing {name}")
        target = obj
        source = inspect.getsourcefile(target)
        if not source:
            raise RuntimeError(f"runtime source path unavailable {name}")
        path = Path(source).resolve()
        label = f"{root}/{relative}"
        entry = entries.get(label)
        if entry is None or Path(entry["path"]).resolve() != path or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"runtime source binding mismatch {name}")
        result["paths"][key] = str(path); result["hashes"][key] = entry["sha256"]
    return result

def _config_contract(config):
    def get(*paths):
        if hasattr(config, "get") and not isinstance(config, dict):
            try:
                from omegaconf import OmegaConf
                for path in paths:
                    val = OmegaConf.select(config, path)
                    if val is not None: return val
                return None
            except Exception: pass
        for path in paths:
            cur = config
            for key in path.split("."):
                if not isinstance(cur, dict) or key not in cur: cur = None; break
                cur = cur[key]
            if cur is not None: return cur
        return None
    required = {"total_timesteps": (("total_timesteps", "training.total_timesteps"), 2_000_000_000), "num_envs": (("num_envs", "training.num_envs"), 1024), "num_steps": (("num_steps", "training.num_steps"), 128), "updates": (("updates", "dicode_manager.max_updates_per_session"), 100)}
    for key, (paths, expected) in required.items():
        val = get(*paths)
        if val is None or int(val) != expected: raise RuntimeError(f"config {key} mismatch")
    cond = get("conditioning_type", "training.conditioning_type")
    condition_on_task = get("condition_on_task", "training.condition_on_task")
    if cond != "one_hot" or condition_on_task is not True: raise RuntimeError("one_hot conditioning contract mismatch")

def _load_config(path):
    if isinstance(path, dict): return path
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(path)
    except Exception as exc: raise RuntimeError("unable to load config") from exc

def _arm_contract(config, arm):
    compact, score = _arm_values(config)
    if arm == "E0" and compact: raise RuntimeError("E0 requires compact_scoring_payload false/absent")
    if arm == "E3V2" and (not compact or score not in ("learnability", "pvl", "max_mc")):
        raise RuntimeError("E3V2 scoring contract mismatch")

def _arm_values(config):
    if not isinstance(config, dict):
        from omegaconf import OmegaConf
        score = OmegaConf.select(config, "training.score_function")
        if score is None: score = OmegaConf.select(config, "dicode_manager.score_function")
        return bool(OmegaConf.select(config, "training.compact_scoring_payload", default=False)), score
    score = config.get("score_function", config.get("training", {}).get("score_function"))
    if score is None: score = config.get("dicode_manager", {}).get("score_function")
    return bool(config.get("compact_scoring_payload", config.get("training", {}).get("compact_scoring_payload", False))), score

def _runtime_imports():
    try:
        import jax, jax.numpy as jnp, orbax.checkpoint as ocp, wandb
        from dicode.dreaming.gen_manager import TaskArchive
        from dicode.task_utils import load_tasks_from_env_codes
        from dicode.training import _calculate_task_distribution, _create_achievement_masks, run_training_session
        from dicode.setup import _load_agent_state
        from dicode.scoring import calculate_scores_from_snapshot
        from minicraftax.tasks.seed_tasks.original import Env as OriginalTask
        import dicode.wrappers_cl as wrappers_cl
        return locals()
    except ImportError as exc: raise RuntimeError("training runtime dependencies unavailable") from exc

def preflight(args, loaded):
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != args.required_gpu_uuid: raise RuntimeError("CUDA_VISIBLE_DEVICES must be exact GPU UUID")
    try:
        import jax
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1: raise RuntimeError("JAX must see exactly one GPU")
    except ImportError as exc: raise RuntimeError("JAX is required for preflight") from exc
    cfg = _load_config(args.config); _config_contract(cfg); _arm_contract(cfg, args.arm)
    rt = _runtime_imports(); runtime_source_evidence = _runtime_source_evidence(rt, loaded, args.arm)
    result = {"classification": loaded["classification"], "manifest_sha256": loaded["manifest_sha256"], "source_commit": args.source_commit,
              "gpu_uuid": args.required_gpu_uuid, "jax_backend": jax.default_backend(), "jax_device_count": 1, "llm_api_calls": 0, "runtime_source_evidence": runtime_source_evidence, "pass": True}
    _atomic_json(Path(args.out) / "PREFLIGHT.json", result); return result

def run(args, loaded):
    stage = next(s for s in loaded["stages"] if s["name"] == args.stage); repeat = stage["repeats"][args.repeat]
    preflight_result = preflight(args, loaded)
    rt = _runtime_imports(); jax, jnp, ocp, wandb = rt["jax"], rt["jnp"], rt["ocp"], rt["wandb"]
    runtime_source_evidence = _runtime_source_evidence(rt, loaded, args.arm)
    config = _load_config(args.config); _config_contract(config); _arm_contract(config, args.arm)
    ids = [str(x) for x in stage["task_ids"]]
    graph_path = stage["graph"]["path"]
    config.gen_manager.graph_path = graph_path
    archive = rt["TaskArchive"](config.gen_manager)
    classes, ok_ids = rt["load_tasks_from_env_codes"](archive, ids)
    if list(ok_ids) != ids: raise RuntimeError("task compilation/load mismatch")
    task_classes = classes + [rt["OriginalTask"]]
    conditioning, conditioning_hash = load_frozen_conditioning(stage)
    proportions = rt["_calculate_task_distribution"](config, len(classes))
    train_state = rt["_load_agent_state"](config, stage["checkpoint"]["path"])
    before, opt_before = state_hash(train_state.params), state_hash(train_state.opt_state)
    rng = jnp.asarray(np.asarray(repeat["rng"], dtype=np.uint32)); train_rng, outer_after = jax.random.split(rng)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); wandb.init(mode="offline", dir=str(out / "wandb"), project="TRAINING_KERNEL_BENCHMARK"); wandb.log = lambda *a, **k: None
    started = time.monotonic_ns()
    result_state = rt["run_training_session"](config, train_rng, task_classes, num_training_updates=100, task_embeddings=jnp.asarray(conditioning), train_state=train_state, task_distribution_proportions=proportions, global_update_step=int(stage["global_step"]), current_original_return=0.0)
    leaves = jax.tree_util.tree_leaves(result_state)
    for leaf in leaves:
        jax.block_until_ready(leaf)
    train_wall_s = (time.monotonic_ns() - started) / 1e9
    train_state = result_state.get("train_state", result_state) if isinstance(result_state, dict) else result_state
    metrics = result_state["metrics"]
    updates = int(metrics["num_updates_done"]); env_steps = int(metrics["num_env_steps_done"])
    if updates != 100 or env_steps != 13107200: raise RuntimeError("unexpected session accounting")
    task_mask, completed = rt["_create_achievement_masks"](task_classes)
    payload = metrics["scoring_window_data"]
    transfer_start = time.monotonic_ns(); metrics_cpu = jax.device_get(payload); scoring_transfer_s = (time.monotonic_ns() - transfer_start) / 1e9
    score_start = time.monotonic_ns(); score = rt["calculate_scores_from_snapshot"](metrics_cpu, len(task_classes), np.asarray(task_mask), np.asarray(completed), config, [len(task_classes) - 1]); scoring_cpu_s = (time.monotonic_ns() - score_start) / 1e9
    ckpt_dir = out / "checkpoint"; manager = ocp.CheckpointManager(str(ckpt_dir), ocp.PyTreeCheckpointer(), options=ocp.CheckpointManagerOptions(create=True, max_to_keep=1)); save_step = int(stage["global_step"]) + 100; checkpoint_start = time.monotonic_ns()
    try:
        manager.save(save_step, train_state); manager.wait_until_finished()
    finally:
        manager.close()
    checkpoint_s = (time.monotonic_ns() - checkpoint_start) / 1e9
    trained, trained_opt = state_hash(train_state.params), state_hash(train_state.opt_state)
    reloaded = rt["_load_agent_state"](config, str(ckpt_dir / str(save_step))); after, opt_after = state_hash(reloaded.params), state_hash(reloaded.opt_state)
    if trained != after or trained_opt != opt_after: raise RuntimeError("checkpoint reload hash mismatch")
    source_hashes = {section: {label: entry["sha256"] for label, entry in entries.items()} for section, entries in loaded.get("source_config", {}).items()}
    wrappers = Path(inspect.getsourcefile(rt["wrappers_cl"]) or "")
    if not wrappers.is_file(): raise RuntimeError("unable to locate wrappers_cl source")
    compact_flag, score_function = _arm_values(config)
    result = {"classification": loaded["classification"], "not_end_to_end_ued": True, "llm_api_calls": 0, "manifest_sha256": loaded["manifest_sha256"], "source_commit": args.source_commit, "gpu_uuid": args.required_gpu_uuid, "runtime_source_evidence": runtime_source_evidence, "stage": args.stage, "repeat": args.repeat, "arm": args.arm, "global_update_step": save_step, "global_env_steps": int(stage["initial_env_steps"]) + env_steps, "updates": updates, "env_steps": env_steps, "params_sha256_before": before, "params_sha256_after": trained, "checkpoint_reloaded_params_sha256": after, "optimizer_sha256_before": opt_before, "optimizer_sha256_after": trained_opt, "checkpoint_reloaded_optimizer_sha256": opt_after, "checkpoint_path": str(ckpt_dir / str(save_step)), "checkpoint_exists": (ckpt_dir / str(save_step)).exists(), "checkpoint_s": checkpoint_s, "input_rng_sha256": rng_hash(rng), "rng_sha256_before": rng_hash(rng), "rng_sha256_after": rng_hash(outer_after), "train_rng_sha256": rng_hash(train_rng), "outer_rng_after_sha256": rng_hash(outer_after), "train_wall_s": train_wall_s, "scoring_transfer_s": scoring_transfer_s, "scoring_cpu_s": scoring_cpu_s, "scoring_fingerprint": scoring_fingerprint(score), "checkpoint_loadable": True, "task_ids": ids, "task_assignment_sha256": fingerprint(ids), "task_code_hashes": [x["code_sha256"] for x in stage["tasks"]], "embedding_hash": conditioning_hash, "conditioning_type": "one_hot", "conditioning_shape": list(conditioning.shape), "conditioning_dtype": str(conditioning.dtype), "score_function": score_function, "compact_scoring_payload": compact_flag, "source_hashes": source_hashes, "reset_selection_semantics": verify_selection_semantics(), "wrappers_cl_sha256": sha256_file(wrappers) if wrappers else ""}
    wandb.finish()
    _atomic_json(out / "RESULT.json", result)
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--config", required=True); p.add_argument("--out", required=True); p.add_argument("--required-gpu-uuid", required=True); p.add_argument("--source-commit", required=True); p.add_argument("--stage", choices=manifest.STAGES, required=True); p.add_argument("--repeat", type=int, choices=(0,1), required=True); p.add_argument("--arm", choices=("E0", "E3V2"), required=True); p.add_argument("--mode", choices=("preflight", "run"), required=True)
    a = p.parse_args(); loaded = load_manifest(a.manifest)
    if a.mode == "preflight": preflight(a, loaded)
    else: run(a, loaded)

if __name__ == "__main__": main()
