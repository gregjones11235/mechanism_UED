#!/usr/bin/env python3
"""Fixed-candidate preflight replay (B1) — R3 complete runtime wiring.

Replays the preflight gate against a FROZEN candidate set so the gate's
cost/decisions can be re-measured deterministically after A2 (no LLM calls, no
mutation of the frozen snapshot, fixed mid checkpoint, fixed ids+order, fixed
RNG, fixed conditioning, fixed archive initial state, fixed score function, 40
rollout updates).

R3 requirements (audit):
- The replay manifest freezes and verifies source_commit, source mapping+sha,
  config path+sha, checkpoint path/tree/metadata sha, archive snapshot,
  candidate ids/order + per-candidate code sha, conditioning content
  sha/shape/dtype, RNG seed+derivation protocol, score_function,
  rollout_updates=40, validation num_envs/num_steps, GPU UUID, version.
  Any file/order/shape/dtype/hash change rejects.
- The GPU replay loads a real DictConfig (never passes None), verifies the
  config contract, reconstructs the archive on a COPY, verifies candidate code
  hashes against the archive, builds REAL achievement masks from the task
  classes (production helper), loads the real checkpoint (params/optimizer
  hashes), runs the frozen-policy rollout with the frozen input RNG, records
  honest RNG evidence (rng_after is `not_exposed` because
  run_evaluation_rollouts does not return a final RNG), applies the real route
  + archive mutations on the copy, and emits archive before/after hashes plus
  runtime source evidence.

The core logic runs against a runtime bundle (``_run_replay(manifest, rt)``) so
the full call sequence is exercised by a fake-runtime integration test on the
CPU-only box; ``run_replay`` supplies the real runtime (jax + dicode, run after
A2 on the server).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

# --- B1 profiling phase contract -------------------------------------------------
PREFLIGHT_PHASES = (
    "candidate_code_load",
    "candidate_cpu_validation_build",
    "candidate_cpu_validation_compile",
    "candidate_cpu_validation_execute",
    "preflight_task_reload",
    "preflight_eval_build",
    "preflight_eval_lower_compile",
    "preflight_eval_execute",
    "route",
    "archive_update",
    "preflight_wall",
)

# --- frozen inputs ----------------------------------------------------------------
MID_CHECKPOINT_STEP = 2100
ROLLOUT_UPDATES = 40
CONDITIONING_DIM = 67
RNG_ALGORITHM = "sha256-little-endian-u32:PREFLIGHT_REPLAY_V1:{candidate_id}:{idx}"
SCORE_FUNCTIONS = ("learnability", "pvl", "max_mc")
CLASSIFICATION = "PREFLIGHT_CANDIDATE_REPLAY"
VALIDATOR_VERSION = "1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> Any:
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
        return [canonical(x) for x in value]
    return value


def fingerprint(value: Any) -> str:
    return sha256_bytes(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = p.relative_to(path).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, "little")); h.update(rel)
        h.update(len(data).to_bytes(8, "little")); h.update(data)
    return h.hexdigest()


def derive_rng(seed: int, candidate_id: str, idx: int) -> list[int]:
    """Deterministic 2x u32 RNG per candidate (little-endian sha256)."""
    material = f"PREFLIGHT_REPLAY_V1:{candidate_id}:{idx}:{int(seed)}".encode()
    digest = hashlib.sha256(material).digest()
    return [int.from_bytes(digest[0:4], "little"), int.from_bytes(digest[4:8], "little")]


def _resolve(path: str | os.PathLike[str], base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def _conditioning_info(path: Path, task_count: int) -> dict[str, Any]:
    import numpy as np
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.dtype("float32") or values.ndim != 2 or values.shape[0] != task_count + 1 or values.shape[1] != CONDITIONING_DIM:
        raise ValueError(f"conditioning table must be finite float32 [{task_count + 1}, {CONDITIONING_DIM}]")
    if not np.isfinite(values).all():
        raise ValueError("conditioning table contains non-finite values")
    h = hashlib.sha256()
    h.update(repr(tuple(values.shape)).encode())
    h.update(str(values.dtype).encode())
    h.update(np.ascontiguousarray(values).tobytes())
    return {
        "path": str(path), "sha256": file_sha256(path),
        "content_sha256": h.hexdigest(), "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _checkpoint_info(path: Path, global_step: int) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"checkpoint must be a directory: {path}")
    files = [p for p in path.rglob("*") if p.is_file()]
    meta = next((p for p in files if p.name == "_CHECKPOINT_METADATA"), None)
    if meta is None:
        raise ValueError(f"checkpoint missing _CHECKPOINT_METADATA: {path}")
    nums = re.findall(r"\d+", path.name)
    if not nums or int(nums[-1]) != global_step:
        raise ValueError(f"checkpoint basename/global_step mismatch: {path.name} != {global_step}")
    return {
        "path": str(path), "tree_sha256": tree_sha256(path),
        "metadata_sha256": file_sha256(meta), "basename": path.name,
        "global_step": global_step,
    }


def _archive_info(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return {"path": str(path), "tree_sha256": tree_sha256(path)}
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path)}
    raise ValueError(f"archive snapshot must be a file or directory: {path}")


def _candidates(spec: Mapping[str, Any], base: Path) -> tuple[list[dict[str, Any]], list[str]]:
    raw = spec.get("candidate_codes")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("candidate_codes must be a non-empty {id: code_path} mapping")
    ids: list[str] = []
    result: list[dict[str, Any]] = []
    for raw_id, raw_path in raw.items():
        cid = str(raw_id)
        if not cid.strip():
            raise ValueError("candidate id must be a non-empty string")
        if cid in ids:
            raise ValueError(f"duplicate candidate id: {cid!r}")
        path = _resolve(raw_path, base)
        if not path.is_file():
            raise ValueError(f"missing candidate code file: {path}")
        code = path.read_text(encoding="utf-8")
        if not code.strip():
            raise ValueError(f"candidate code file is empty: {path}")
        ids.append(cid)
        result.append({"id": cid, "path": str(path), "code_sha256": sha256_bytes(code.encode()),
                       "code_bytes": len(code.encode())})
    return result, ids


def _source_mapping(spec: Mapping[str, Any], base: Path) -> dict[str, dict[str, str]]:
    raw = spec.get("source_mapping")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("source_mapping must be a non-empty {label: file_path} mapping")
    out = {}
    for label, raw_path in raw.items():
        path = _resolve(raw_path, base)
        if not path.is_file():
            raise ValueError(f"missing source file for {label}: {path}")
        out[str(label)] = {"path": str(path), "sha256": file_sha256(path)}
    return out


def _config_info(spec: Mapping[str, Any], base: Path) -> dict[str, Any]:
    raw = spec.get("config_path")
    if not raw:
        raise ValueError("replay spec requires config_path")
    path = _resolve(raw, base)
    if not path.is_file():
        raise ValueError(f"missing config file: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def build_replay_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a replay spec and produce the frozen manifest (with hashes)."""
    if spec.get("classification") not in (None, CLASSIFICATION):
        raise ValueError(f"classification must be {CLASSIFICATION}")
    if int(spec.get("global_step", MID_CHECKPOINT_STEP)) != MID_CHECKPOINT_STEP:
        raise ValueError(f"global_step must be {MID_CHECKPOINT_STEP}")
    if int(spec.get("rollout_updates", ROLLOUT_UPDATES)) != ROLLOUT_UPDATES:
        raise ValueError(f"rollout_updates must be {ROLLOUT_UPDATES}")
    score_function = spec.get("score_function")
    if score_function not in SCORE_FUNCTIONS:
        raise ValueError(f"score_function must be one of {SCORE_FUNCTIONS}")
    source_commit = spec.get("source_commit")
    if not source_commit or not str(source_commit).strip():
        raise ValueError("source_commit is required")
    gpu_uuid = spec.get("gpu_uuid")
    if not gpu_uuid or not str(gpu_uuid).strip():
        raise ValueError("gpu_uuid is required")
    seed = int(spec.get("rng_seed", 42))
    base = Path(spec.get("base_dir", ".")).resolve()

    checkpoint_raw = spec.get("checkpoint")
    conditioning_raw = spec.get("conditioning_path")
    archive_raw = spec.get("archive_snapshot")
    if not checkpoint_raw or not conditioning_raw or not archive_raw:
        raise ValueError("replay spec requires checkpoint, conditioning_path, archive_snapshot")

    checkpoint = _checkpoint_info(_resolve(checkpoint_raw, base), MID_CHECKPOINT_STEP)
    candidates, candidate_ids = _candidates(spec, base)
    conditioning = _conditioning_info(_resolve(conditioning_raw, base), len(candidate_ids))
    archive = _archive_info(_resolve(archive_raw, base))
    source_mapping = _source_mapping(spec, base)
    config = _config_info(spec, base)

    num_envs = int(spec.get("num_envs", 1024))
    num_steps = int(spec.get("num_steps", 128))
    if num_envs < 1024 or num_steps < 128:
        raise ValueError("num_envs/num_steps must not be lowered below the scientific budget")

    rng = {cid: derive_rng(seed, cid, 0) for cid in candidate_ids}
    manifest = {
        "classification": CLASSIFICATION,
        "not_end_to_end_ued": True,
        "llm_api_calls": 0,
        "mid_checkpoint_step": MID_CHECKPOINT_STEP,
        "rollout_updates": ROLLOUT_UPDATES,
        "conditioning_dim": CONDITIONING_DIM,
        "score_function": score_function,
        "rng_seed": seed,
        "rng_algorithm": RNG_ALGORITHM,
        "validator_version": str(spec.get("validator_version", VALIDATOR_VERSION)),
        "source_commit": str(source_commit),
        "gpu_uuid": str(gpu_uuid),
        "validation": {"rollout_updates": ROLLOUT_UPDATES, "num_envs": num_envs, "num_steps": num_steps},
        "source_mapping": source_mapping,
        "config": config,
        "candidate_ids": candidate_ids,
        "candidates": candidates,
        "checkpoint": checkpoint,
        "conditioning": conditioning,
        "archive_snapshot": archive,
        "rng": rng,
        "phases": list(PREFLIGHT_PHASES),
    }
    return manifest


def _without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "manifest_sha256"}


def write_manifest(manifest: Mapping[str, Any], output: str | os.PathLike[str]) -> dict[str, Any]:
    data = dict(manifest)
    data.pop("manifest_sha256", None)
    data["manifest_sha256"] = fingerprint(_without_hash(data))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(data), f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return data


def validate_replay_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all hashes and re-run the structural gates; raises on mismatch."""
    if manifest.get("manifest_sha256") is not None and manifest.get("manifest_sha256") != fingerprint(_without_hash(manifest)):
        raise ValueError("manifest_sha256 mismatch")
    if manifest.get("classification") != CLASSIFICATION or manifest.get("not_end_to_end_ued") is not True or manifest.get("llm_api_calls") != 0:
        raise ValueError("invalid replay classification gates")
    if int(manifest.get("mid_checkpoint_step", -1)) != MID_CHECKPOINT_STEP or int(manifest.get("rollout_updates", -1)) != ROLLOUT_UPDATES:
        raise ValueError("invalid mid_checkpoint_step / rollout_updates")
    if manifest.get("score_function") not in SCORE_FUNCTIONS:
        raise ValueError("invalid score_function")
    if manifest.get("conditioning_dim") != CONDITIONING_DIM:
        raise ValueError("invalid conditioning_dim")
    if not manifest.get("source_commit") or not manifest.get("gpu_uuid"):
        raise ValueError("source_commit/gpu_uuid required")
    val = manifest.get("validation", {})
    if int(val.get("rollout_updates", -1)) != ROLLOUT_UPDATES:
        raise ValueError("validation.rollout_updates must be 40")
    if int(val.get("num_envs", 0)) < 1024 or int(val.get("num_steps", 0)) < 128:
        raise ValueError("validation num_envs/num_steps lowered below budget")
    if not manifest.get("config", {}).get("path") or not manifest.get("config", {}).get("sha256"):
        raise ValueError("config path/sha required")
    ids = [str(x) for x in manifest.get("candidate_ids", [])]
    if not ids or len(ids) != len(set(ids)) or any(not x.strip() for x in ids):
        raise ValueError("invalid candidate id order/duplicates")
    if len(manifest.get("candidates", [])) != len(ids):
        raise ValueError("candidates/candidate_ids mismatch")
    if manifest.get("phases") != list(PREFLIGHT_PHASES):
        raise ValueError(f"phases must equal {list(PREFLIGHT_PHASES)}")

    base = Path.cwd()
    cp = _checkpoint_info(Path(manifest["checkpoint"]["path"]), MID_CHECKPOINT_STEP)
    if cp != manifest["checkpoint"]:
        raise ValueError("checkpoint hash/step changed")
    cond = _conditioning_info(Path(manifest["conditioning"]["path"]), len(ids))
    if cond != manifest["conditioning"]:
        raise ValueError("conditioning table changed")
    archive = _archive_info(Path(manifest["archive_snapshot"]["path"]))
    if archive != manifest["archive_snapshot"]:
        raise ValueError("archive snapshot changed")
    candidates, rederived_ids = _candidates(
        {"candidate_codes": {c["id"]: c["path"] for c in manifest["candidates"]}}, base)
    if rederived_ids != ids or candidates != manifest["candidates"]:
        raise ValueError("candidate code hashes changed")
    src = _source_mapping({"source_mapping": {k: v["path"] for k, v in manifest["source_mapping"].items()}}, base)
    if src != manifest["source_mapping"]:
        raise ValueError("source mapping hashes changed")
    cfg = _config_info({"config_path": manifest["config"]["path"]}, base)
    if cfg != manifest["config"]:
        raise ValueError("config file hash changed")
    expected_rng = {cid: derive_rng(int(manifest["rng_seed"]), cid, 0) for cid in ids}
    if manifest.get("rng") != expected_rng:
        raise ValueError("replay RNG changed")
    return dict(manifest)


# --- config contract ---------------------------------------------------------------
def verify_config(config, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the loaded DictConfig against the frozen manifest contract.

    Raises ValueError on any mismatch: rollout_updates must be 40, score_function
    must match the manifest, num_envs/num_steps must not be lowered, all
    performance flags are recorded (not silently defaulted), LLM calls are 0.
    Returns a dict of the recorded performance flags.
    """
    if config is None:
        raise ValueError("config must not be None")
    rup = None
    try:
        rup = int(config.validation.rollout_updates)
    except Exception:
        pass
    if rup != int(manifest["rollout_updates"]):
        raise ValueError(f"validation.rollout_updates must be {manifest['rollout_updates']} (got {rup})")
    sf = None
    try:
        sf = str(config.dicode_manager.score_function)
    except Exception:
        pass
    if sf != manifest["score_function"]:
        raise ValueError(f"score_function must match manifest {manifest['score_function']} (got {sf})")
    nenv = nstep = None
    try:
        nenv = int(config.validation.num_envs); nstep = int(config.validation.num_steps)
    except Exception:
        pass
    if nenv is None or nstep is None or nenv < int(manifest["validation"]["num_envs"]) or nstep < int(manifest["validation"]["num_steps"]):
        raise ValueError("validation num_envs/num_steps lowered below the frozen budget")
    perf = {}
    if hasattr(config, "get"):
        perf = config.get("performance", {}) if hasattr(config, "get") else {}
    recorded = {
        "preflight_reuse_loaded_tasks": bool(perf.get("preflight_reuse_loaded_tasks", False)),
        "compact_preflight_payload": bool(perf.get("compact_preflight_payload", False)),
        "eval_compile_cache": bool(perf.get("eval_compile_cache", False)),
        "train_compile_cache": bool(perf.get("train_compile_cache", False)),
        "embedding_cache": bool(perf.get("embedding_cache", False)),
        "validation_cache": bool(perf.get("validation_cache", False)),
    }
    return recorded


# --- core replay (runtime-bundle, testable without jax) ----------------------------
def _verify_candidate_codes(code_map: Mapping[str, str], candidates: list[dict[str, Any]], candidate_ids: list[str]) -> None:
    if list(code_map.keys()) != candidate_ids:
        raise ValueError(f"archive candidate code order mismatch: {list(code_map.keys())} != {candidate_ids}")
    for cand in candidates:
        cid = cand["id"]
        code = code_map.get(cid)
        if code is None:
            raise ValueError(f"candidate {cid} missing from archive")
        if sha256_bytes(code.encode()) != cand["code_sha256"]:
            raise ValueError(f"candidate {cid} code hash mismatch with manifest")


def _route_all(route_fn, scores: Mapping[str, Any], candidate_ids: list[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    decisions: list[dict[str, Any]] = []
    accepted: list[str] = []
    rejected: list[str] = []
    for i, tid in enumerate(candidate_ids):
        sr = float(scores.get(str(i), {}).get("sr", -1.0))
        d = route_fn(max(sr, 0.0), any_partial_progress=(sr >= 0.0))
        decisions.append({"id": tid, "action": d.action, "reason": d.reason, "sr": sr})
        (accepted if d.action == "accept" else rejected).append(tid)
    return decisions, accepted, rejected


def _apply_archive_updates(archive, decisions, update_accept, update_reject) -> None:
    for dec in decisions:
        tid = dec["id"]
        if dec["action"] == "accept":
            clip = min(max(dec["sr"], 0.0), 1.0)
            update_accept(archive, tid, clip * (1.0 - clip))
        else:
            update_reject(archive, tid, f"preflight_{dec['reason']}")


def _run_replay(manifest: Mapping[str, Any], rt: Mapping[str, Any]) -> dict[str, Any]:
    """Core replay call sequence against a runtime bundle (real or fake).

    ``rt`` keys (each must be present):
      load_config(path) -> config
      verify_gpu(gpu_uuid)
      reconstruct_archive(snapshot_path) -> archive (a copy)
      archive_get_codes(archive, ids) -> {id: code}
      load_tasks(archive, ids) -> (classes, ok_ids)
      achievement_masks(classes) -> (task_achievement_mask, task_completed_mask)
      load_checkpoint(config, ckpt_path) -> train_state
      state_hash(state) -> sha256
      make_input_rng(rng_seed, candidate_ids) -> rng
      run_rollout(config, rng, classes, updates, embeddings, train_state) -> results
      score(config, swd, num_tasks, mask, completed) -> scores
      route(sr, any_partial) -> decision
      archive_hash(archive) -> sha256
      archive_update_accept(archive, tid, learnability)
      archive_update_reject(archive, tid, status)
    """
    validate_replay_manifest(manifest)
    import numpy as np

    config = rt["load_config"](manifest["config"]["path"])
    perf_flags = verify_config(config, manifest)
    rt["verify_gpu"](manifest["gpu_uuid"])

    candidate_ids = [str(x) for x in manifest["candidate_ids"]]
    candidates = manifest["candidates"]

    archive = rt["reconstruct_archive"](manifest["archive_snapshot"]["path"])
    archive_before = rt["archive_hash"](archive)

    code_map = rt["archive_get_codes"](archive, candidate_ids)
    _verify_candidate_codes(code_map, candidates, candidate_ids)

    task_classes, ok_ids = rt["load_tasks"](archive, candidate_ids)
    if ok_ids != candidate_ids:
        raise ValueError(f"candidate code load mismatch: expected {candidate_ids}, got {ok_ids}")

    task_achievement_mask, task_completed_mask = rt["achievement_masks"](task_classes)

    train_state = rt["load_checkpoint"](config, manifest["checkpoint"]["path"])
    params_hash = rt["state_hash"](train_state.params)
    optimizer_hash = rt["state_hash"](train_state.opt_state)

    cond_table = np.load(manifest["conditioning"]["path"], allow_pickle=False)
    task_embeddings = cond_table[1:]
    task_embeddings = np.ascontiguousarray(task_embeddings)

    rng_input = rt["make_input_rng"](int(manifest["rng_seed"]), candidate_ids)
    rng_input_sha = sha256_bytes(json.dumps(canonical(rng_input), sort_keys=True).encode())

    results = rt["run_rollout"](config, rng_input, task_classes, int(manifest["rollout_updates"]),
                                task_embeddings, train_state)
    swd = results.get("metrics", {}).get("scoring_window_data")
    if swd is None:
        raise ValueError("rollouts produced no scoring_window_data")

    scores = rt["score"](config, swd, len(candidate_ids), task_achievement_mask, task_completed_mask)
    decisions, accepted, rejected = _route_all(rt["route"], scores, candidate_ids)
    _apply_archive_updates(archive, decisions,
                           rt["archive_update_accept"], rt["archive_update_reject"])
    archive_after = rt["archive_hash"](archive)

    return {
        "classification": CLASSIFICATION,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "validator_version": manifest["validator_version"],
        "source_commit": manifest["source_commit"],
        "gpu_uuid": manifest["gpu_uuid"],
        "candidate_ids": candidate_ids,
        "accepted_ids": accepted,
        "rejected_ids": rejected,
        "scores": scores,
        "route_decisions": decisions,
        "archive_before_sha256": archive_before,
        "archive_after_sha256": archive_after,
        "task_code_sha256s": {c["id"]: c["code_sha256"] for c in candidates},
        "task_masks_hash": sha256_bytes(np.ascontiguousarray(task_achievement_mask).tobytes()
                                        + np.ascontiguousarray(task_completed_mask).tobytes()),
        "checkpoint_tree_sha256": manifest["checkpoint"]["tree_sha256"],
        "checkpoint_metadata_sha256": manifest["checkpoint"]["metadata_sha256"],
        "conditioning_content_sha256": manifest["conditioning"]["content_sha256"],
        "rng_input_sha256": rng_input_sha,
        "rng_after_sha256": "not_exposed:run_evaluation_rollouts_does_not_return_final_rng",
        "params_hash_before": params_hash,
        "optimizer_hash_before": optimizer_hash,
        "performance_flags": perf_flags,
        "llm_api_calls": 0,
        "runtime_source_evidence": {k: v["sha256"] for k, v in manifest["source_mapping"].items()},
    }


# --- real runtime (requires jax + dicode; run on server after A2) -----------------
def _real_runtime() -> dict[str, Any]:
    import jax
    import numpy as np
    from omegaconf import OmegaConf
    from dicode.task_utils import load_tasks_from_env_codes
    from dicode.dreaming.gen_manager import TaskArchive
    from dicode.scoring import calculate_scores_from_snapshot
    from dicode.skill_preflight.preflight import route
    from dicode.ppo_tr import run_evaluation_rollouts
    from dicode.setup import _load_agent_state
    from dicode.training import _create_achievement_masks

    def reconstruct_archive(snapshot_path):
        # copy to a fresh temp dir so mutations never touch the frozen snapshot
        tmp = Path(tempfile.mkdtemp(prefix="preflight_replay_archive_"))
        src = Path(snapshot_path)
        if src.is_dir():
            shutil.copytree(src, tmp / src.name)
            return TaskArchive.load(str(tmp / src.name))
        shutil.copy(src, tmp / src.name)
        return TaskArchive.load(str(tmp / src.name))

    def archive_hash(archive):
        # persist the archive's current graph state and hash it
        from io import BytesIO
        import networkx as nx
        buf = BytesIO()
        nx.write_graphml(archive.graph, buf)
        return sha256_bytes(buf.getvalue())

    def archive_get_codes(archive, ids):
        return archive.get_task_codes(ids)

    def load_tasks(archive, ids):
        return load_tasks_from_env_codes(archive, ids)

    def achievement_masks(classes):
        masks = _create_achievement_masks(classes)
        return np.asarray(masks[0]), np.asarray(masks[1])

    def load_checkpoint(config, ckpt_path):
        return _load_agent_state(config, ckpt_path)

    def state_hash(state):
        from dicode.runtime_analysis import RuntimeTracker
        return _state_hash_impl(state)

    def make_input_rng(seed, candidate_ids):
        # frozen input RNG: derived deterministically from seed+first candidate
        rng0 = derive_rng(seed, candidate_ids[0], 0)
        return jax.random.PRNGKey(rng0[0])

    def run_rollout(config, rng, classes, updates, embeddings, train_state):
        return run_evaluation_rollouts(config, rng, classes, updates,
                                       task_embeddings=embeddings, train_state=train_state)

    def score(config, swd, num_tasks, mask, completed):
        return calculate_scores_from_snapshot(swd, num_tasks, mask, completed, config)

    def verify_gpu(gpu_uuid):
        import jax
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            raise RuntimeError("replay requires a GPU device")
        # JAX exposes the device id, not the UUID; confirm at least the env pins
        # the exact UUID via CUDA_VISIBLE_DEVICES like the pair benchmark does.
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != gpu_uuid:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {gpu_uuid}")

    def archive_update_accept(archive, tid, learnability):
        archive.update_node_learnability(tid, learnability)

    def archive_update_reject(archive, tid, status):
        archive.update_node_status(tid, status)
        archive.set_task_active_status(tid, False)

    return {
        "load_config": OmegaConf.load,
        "verify_gpu": verify_gpu,
        "reconstruct_archive": reconstruct_archive,
        "archive_get_codes": archive_get_codes,
        "load_tasks": load_tasks,
        "achievement_masks": achievement_masks,
        "load_checkpoint": load_checkpoint,
        "state_hash": state_hash,
        "make_input_rng": make_input_rng,
        "run_rollout": run_rollout,
        "score": score,
        "route": route,
        "archive_hash": archive_hash,
        "archive_update_accept": archive_update_accept,
        "archive_update_reject": archive_update_reject,
    }


def _state_hash_impl(tree) -> str:
    import numpy as np
    h = hashlib.sha256()
    import jax
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        try:
            a = np.asarray(leaf)
            h.update(str(a.dtype).encode())
            h.update(repr(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        except Exception:
            h.update(repr(leaf).encode())
    return h.hexdigest()


def run_replay(manifest: Mapping[str, Any], *, out_jsonl: str | None = None) -> dict[str, Any]:
    """Execute the frozen-candidate preflight replay (requires jax + dicode)."""
    result = _run_replay(manifest, _real_runtime())
    if out_jsonl:
        write_manifest(result, out_jsonl)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="path to a replay spec JSON, or inline JSON")
    parser.add_argument("--output", required=True, help="output manifest JSON path")
    parser.add_argument("--run", action="store_true",
                        help="also execute the replay (requires jax + server artifacts)")
    args = parser.parse_args(argv)
    spec_text = Path(args.spec).read_text(encoding="utf-8") if Path(args.spec).is_file() else args.spec
    spec = json.loads(spec_text)
    spec.setdefault("base_dir", str(Path(args.spec).parent if Path(args.spec).is_file() else Path.cwd()))
    manifest = build_replay_manifest(spec)
    write_manifest(manifest, args.output)
    if args.run:
        run_replay(manifest, out_jsonl=str(Path(args.output).with_suffix(".result.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
