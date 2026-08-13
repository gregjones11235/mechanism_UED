#!/usr/bin/env python3
"""Fixed-candidate preflight replay (B1) — R3 second repair.

Replays the preflight gate against a FROZEN candidate set through the REAL
production control chain (TaskArchive -> evaluate_new_tasks ->
calculate_scores_from_snapshot -> shared preflight_route), using a FROZEN RNG
artifact (not a re-derived key) and the production RuntimeTracker for phase
timing. Source evidence fails closed (no success RESULT unless every executed
runtime object is verified). RESULT is atomically written exactly once with a
canonical result_sha256 and reloaded/recomputed.

Frozen inputs (all hash-verified in the manifest):
  checkpoint, conditioning, archive snapshot, candidate codes, frozen RNG
  artifact (path/file_sha256/content_sha256/shape/dtype), config, source
  mapping, score_function, rollout_updates=40, num_envs/num_steps, GPU UUID.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

MID_CHECKPOINT_STEP = 2100
ROLLOUT_UPDATES = 40
CONDITIONING_DIM = 67
SCORE_FUNCTIONS = ("learnability", "pvl", "max_mc")
CLASSIFICATION = "PREFLIGHT_CANDIDATE_REPLAY"
VALIDATOR_VERSION = "3"


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
    if hasattr(value, "tolist") and not isinstance(value, (list, dict, str)):
        try:
            return canonical(value.tolist())
        except Exception:
            pass
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


def _resolve(path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


# ---- frozen RNG artifact -------------------------------------------------------
def rng_evidence(rng) -> dict[str, Any]:
    """Stable evidence for a JAX PRNGKey: device_get + shape/dtype + raw sha."""
    import numpy as np
    try:
        import jax
        if hasattr(rng, "aval") or hasattr(rng, "device"):
            arr = jax.device_get(rng)
        else:
            arr = np.asarray(rng)
    except Exception:
        arr = np.asarray(rng)
    a = np.ascontiguousarray(arr)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_bytes(a.tobytes())}


def _rng_artifact_info(path: Path) -> dict[str, Any]:
    """Validate a frozen PRNGKey artifact file (.npy) and record its evidence."""
    import numpy as np
    if not path.is_file():
        raise ValueError(f"rng_path is not a file: {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 1 or arr.dtype != np.dtype("uint32") or arr.shape[0] != 2:
        raise ValueError(f"rng artifact must be a uint32[2] PRNGKey, got {arr.shape}/{arr.dtype}")
    h = hashlib.sha256()
    h.update(repr(tuple(arr.shape)).encode()); h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return {"path": str(path), "file_sha256": file_sha256(path),
            "content_sha256": h.hexdigest(), "shape": list(arr.shape), "dtype": str(arr.dtype)}


def load_frozen_rng(manifest_rng: Mapping[str, Any]):
    """Rebuild the exact frozen JAX PRNGKey from the artifact (no synthesis)."""
    import numpy as np
    import jax
    path = Path(manifest_rng["path"])
    arr = np.load(path, allow_pickle=False)
    if list(arr.shape) != manifest_rng["shape"] or str(arr.dtype) != manifest_rng["dtype"]:
        raise ValueError("rng artifact shape/dtype changed since the manifest")
    if file_sha256(path) != manifest_rng["file_sha256"]:
        raise ValueError("rng artifact file changed")
    return jax.device_put(jax.numpy.asarray(arr))


# ---- manifest build / validate --------------------------------------------------
def _conditioning_info(path: Path, task_count: int) -> dict[str, Any]:
    import numpy as np
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.dtype("float32") or values.ndim != 2 or values.shape[0] != task_count + 1 or values.shape[1] != CONDITIONING_DIM:
        raise ValueError(f"conditioning table must be finite float32 [{task_count + 1}, {CONDITIONING_DIM}]")
    if not np.isfinite(values).all():
        raise ValueError("conditioning table contains non-finite values")
    h = hashlib.sha256()
    h.update(repr(tuple(values.shape)).encode()); h.update(str(values.dtype).encode())
    h.update(np.ascontiguousarray(values).tobytes())
    return {"path": str(path), "sha256": file_sha256(path), "content_sha256": h.hexdigest(),
            "shape": list(values.shape), "dtype": str(values.dtype)}


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
    return {"path": str(path), "tree_sha256": tree_sha256(path),
            "metadata_sha256": file_sha256(meta), "basename": path.name, "global_step": global_step}


def _archive_info(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return {"path": str(path), "tree_sha256": tree_sha256(path)}
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path)}
    raise ValueError(f"archive snapshot must be a file or directory: {path}")


def _candidates(spec, base: Path):
    raw = spec.get("candidate_codes")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("candidate_codes must be a non-empty {id: code_path} mapping")
    ids, result = [], []
    for raw_id, raw_path in raw.items():
        cid = str(raw_id)
        if not cid.strip() or cid in ids:
            raise ValueError(f"invalid/duplicate candidate id: {cid!r}")
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


def _source_mapping(spec, base: Path) -> dict[str, dict[str, str]]:
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


def _config_info(spec, base: Path) -> dict[str, Any]:
    raw = spec.get("config_path")
    if not raw:
        raise ValueError("replay spec requires config_path")
    path = _resolve(raw, base)
    if not path.is_file():
        raise ValueError(f"missing config file: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def build_replay_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("classification") not in (None, CLASSIFICATION):
        raise ValueError(f"classification must be {CLASSIFICATION}")
    if int(spec.get("global_step", MID_CHECKPOINT_STEP)) != MID_CHECKPOINT_STEP:
        raise ValueError(f"global_step must be {MID_CHECKPOINT_STEP}")
    if int(spec.get("rollout_updates", ROLLOUT_UPDATES)) != ROLLOUT_UPDATES:
        raise ValueError(f"rollout_updates must be {ROLLOUT_UPDATES}")
    score_function = spec.get("score_function")
    if score_function not in SCORE_FUNCTIONS:
        raise ValueError(f"score_function must be one of {SCORE_FUNCTIONS}")
    if not spec.get("source_commit") or not spec.get("gpu_uuid"):
        raise ValueError("source_commit and gpu_uuid are required")
    base = Path(spec.get("base_dir", ".")).resolve()
    checkpoint = _checkpoint_info(_resolve(spec["checkpoint"], base), MID_CHECKPOINT_STEP)
    candidates, candidate_ids = _candidates(spec, base)
    conditioning = _conditioning_info(_resolve(spec["conditioning_path"], base), len(candidate_ids))
    archive = _archive_info(_resolve(spec["archive_snapshot"], base))
    source_mapping = _source_mapping(spec, base)
    config = _config_info(spec, base)
    rng = _rng_artifact_info(_resolve(spec["rng_path"], base))
    num_envs = int(spec.get("num_envs", 1024)); num_steps = int(spec.get("num_steps", 128))
    if num_envs < 1024 or num_steps < 128:
        raise ValueError("num_envs/num_steps must not be lowered below the scientific budget")
    return {
        "classification": CLASSIFICATION, "not_end_to_end_ued": True, "llm_api_calls": 0,
        "mid_checkpoint_step": MID_CHECKPOINT_STEP, "rollout_updates": ROLLOUT_UPDATES,
        "conditioning_dim": CONDITIONING_DIM, "score_function": score_function,
        "validator_version": str(spec.get("validator_version", VALIDATOR_VERSION)),
        "source_commit": str(spec["source_commit"]), "gpu_uuid": str(spec["gpu_uuid"]),
        "validation": {"rollout_updates": ROLLOUT_UPDATES, "num_envs": num_envs, "num_steps": num_steps},
        "source_mapping": source_mapping, "config": config,
        "candidate_ids": candidate_ids, "candidates": candidates,
        "checkpoint": checkpoint, "conditioning": conditioning, "archive_snapshot": archive,
        "rng": rng,
    }


def _without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "manifest_sha256"}


def write_manifest(manifest: Mapping[str, Any], output) -> dict[str, Any]:
    data = dict(manifest)
    data.pop("manifest_sha256", None)
    data["manifest_sha256"] = fingerprint(_without_hash(data))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(data), f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return data


def validate_replay_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("manifest_sha256") is not None and manifest.get("manifest_sha256") != fingerprint(_without_hash(manifest)):
        raise ValueError("manifest_sha256 mismatch")
    if manifest.get("classification") != CLASSIFICATION or manifest.get("llm_api_calls") != 0:
        raise ValueError("invalid replay classification gates")
    if int(manifest.get("mid_checkpoint_step", -1)) != MID_CHECKPOINT_STEP or int(manifest.get("rollout_updates", -1)) != ROLLOUT_UPDATES:
        raise ValueError("invalid mid_checkpoint_step / rollout_updates")
    if manifest.get("score_function") not in SCORE_FUNCTIONS or manifest.get("conditioning_dim") != CONDITIONING_DIM:
        raise ValueError("invalid score_function / conditioning_dim")
    if not manifest.get("source_commit") or not manifest.get("gpu_uuid"):
        raise ValueError("source_commit/gpu_uuid required")
    val = manifest.get("validation", {})
    if int(val.get("rollout_updates", -1)) != ROLLOUT_UPDATES or int(val.get("num_envs", 0)) < 1024 or int(val.get("num_steps", 0)) < 128:
        raise ValueError("validation budget mismatch")
    ids = [str(x) for x in manifest.get("candidate_ids", [])]
    if not ids or len(ids) != len(set(ids)) or any(not x.strip() for x in ids):
        raise ValueError("invalid candidate id order/duplicates")
    base = Path.cwd()
    if _checkpoint_info(Path(manifest["checkpoint"]["path"]), MID_CHECKPOINT_STEP) != manifest["checkpoint"]:
        raise ValueError("checkpoint hash/step changed")
    if _conditioning_info(Path(manifest["conditioning"]["path"]), len(ids)) != manifest["conditioning"]:
        raise ValueError("conditioning table changed")
    if _archive_info(Path(manifest["archive_snapshot"]["path"])) != manifest["archive_snapshot"]:
        raise ValueError("archive snapshot changed")
    candidates, rederived = _candidates({"candidate_codes": {c["id"]: c["path"] for c in manifest["candidates"]}}, base)
    if rederived != ids or candidates != manifest["candidates"]:
        raise ValueError("candidate code hashes changed")
    if _source_mapping({"source_mapping": {k: v["path"] for k, v in manifest["source_mapping"].items()}}, base) != manifest["source_mapping"]:
        raise ValueError("source mapping hashes changed")
    if _config_info({"config_path": manifest["config"]["path"]}, base) != manifest["config"]:
        raise ValueError("config file hash changed")
    if _rng_artifact_info(Path(manifest["rng"]["path"])) != manifest["rng"]:
        raise ValueError("rng artifact changed")
    return dict(manifest)


def verify_config(config, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if config is None:
        raise ValueError("config must not be None")
    rup = getattr(getattr(config, "validation", None), "rollout_updates", None)
    if int(rup) != int(manifest["rollout_updates"]):
        raise ValueError(f"validation.rollout_updates must be {manifest['rollout_updates']} (got {rup})")
    sf = getattr(getattr(config, "dicode_manager", None), "score_function", None)
    if sf != manifest["score_function"]:
        raise ValueError(f"score_function must match manifest {manifest['score_function']} (got {sf})")
    nenv = getattr(getattr(config, "validation", None), "num_envs", None)
    nstep = getattr(getattr(config, "validation", None), "num_steps", None)
    if nenv is None or nstep is None or int(nenv) < int(manifest["validation"]["num_envs"]) or int(nstep) < int(manifest["validation"]["num_steps"]):
        raise ValueError("validation num_envs/num_steps lowered below the frozen budget")
    perf = config.get("performance", {}) if hasattr(config, "get") else {}
    return {k: bool(perf.get(k, False)) for k in
            ("preflight_reuse_loaded_tasks", "compact_preflight_payload", "eval_compile_cache",
             "train_compile_cache", "embedding_cache", "validation_cache")}


# ---- source evidence (fail-closed) ----------------------------------------------
class SourceEvidenceError(RuntimeError):
    """Raised when a replay runtime source object cannot be verified against the
    manifest; the replay must fail closed (no success RESULT)."""


def runtime_source_evidence(manifest: Mapping[str, Any], source_objects: Mapping[str, Any]) -> dict[str, Any]:
    """Verify each ACTUAL executed runtime object against its manifest entry.

    ``source_objects`` maps a label (e.g. "TaskArchive") to the real imported
    object. For every object: resolve the source file, sha256 it, require an
    exact manifest source_mapping entry whose resolved path equals the object's
    real source path AND whose sha256 matches. Any missing/wrong/mismatched
    entry or an uninspectable object raises ``SourceEvidenceError`` (fail-closed).
    """
    expected = manifest["source_mapping"]
    result: dict[str, Any] = {"verified": True, "objects": {}}
    for key, obj in source_objects.items():
        src = inspect.getsourcefile(obj)
        if not src:
            raise SourceEvidenceError(f"{key}: no inspectable source file")
        path = Path(src).resolve()
        sha = file_sha256(path)
        match = None
        for label, entry in expected.items():
            if Path(entry["path"]).resolve() == path:
                match = (label, entry)
                break
        if match is None:
            raise SourceEvidenceError(
                f"{key} source {path} has no matching manifest source_mapping entry")
        if match[1]["sha256"] != sha:
            raise SourceEvidenceError(
                f"{key} source {path} sha256 {sha} != manifest {match[1]['sha256']}")
        result["objects"][key] = {
            "module": getattr(obj, "__module__", ""), "qualname": getattr(obj, "__qualname__", getattr(obj, "__name__", "")),
            "label": match[0], "source_path": str(path), "source_sha256": sha, "verified": True,
        }
    return result


# ---- real runtime (production tracker + real chain) ------------------------------
def _real_runtime(manifest: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    import jax
    import numpy as np
    from omegaconf import OmegaConf
    from dicode.runtime_analysis import tracker
    from dicode.dreaming.gen_manager import TaskArchive
    from dicode.task_utils import load_tasks_from_env_codes
    from dicode.scoring import calculate_scores_from_snapshot
    from dicode.skill_preflight.preflight import route
    from dicode.skill_preflight.preflight_route import preflight_route
    from dicode.evaluation import evaluate_new_tasks
    from dicode.setup import _load_agent_state
    from dicode.ppo_tr import run_evaluation_rollouts

    tracker.configure(enabled=True, output_jsonl=str(out_dir / "events.jsonl"), reset=True)
    tracker.set_session("replay")

    def reconstruct_archive(snapshot_path):
        tmp = Path(tempfile.mkdtemp(prefix="preflight_replay_archive_"))
        src = Path(snapshot_path)
        if src.is_dir():
            shutil.copytree(src, tmp / src.name)
            graphml = next((p for p in (tmp / src.name).rglob("*.graphml")), None)
            if graphml is None:
                raise ValueError(f"archive snapshot dir has no .graphml: {src}")
            return TaskArchive(type("C", (), {"graph_path": str(graphml)})())
        shutil.copy(src, tmp / src.name)
        return TaskArchive(type("C", (), {"graph_path": str(tmp / src.name)})())

    def _graph_sha(archive):
        g = archive.graph
        parts = []
        for node in sorted(g.nodes()):
            parts.append(str(node) + ":" + json.dumps(canonical(dict(g.nodes[node])), sort_keys=True))
        for u, v in sorted((sorted(e) for e in g.edges())):
            parts.append(f"{u}->{v}")
        return sha256_bytes("\n".join(parts).encode())

    class FrozenEmbeddingProvider:
        """Returns only the frozen embedding table; never calls an LLM/API."""

        def __init__(self, frozen_table):
            self._frozen = frozen_table

        def get_embedding(self, labels, instruction=None):
            return [{"embedding": self._frozen[i]} for i in range(len(labels))]

    def evaluate_and_score(config, rng, train_state, ids, archive):
        # B2 reuse off: the production evaluate_new_tasks performs the second
        # task reload itself (production path). one_hot -> production one-hot
        # generation inside evaluate_new_tasks (no LLM). For embedding
        # conditioning a frozen provider would be used; one_hot is the frozen
        # default here and needs no provider.
        raw = evaluate_new_tasks(config, rng, train_state, ids, archive,
                                 FrozenEmbeddingProvider(np.zeros((len(ids), CONDITIONING_DIM), dtype=np.float32)))
        swd = raw.get("scoring_window_data")
        if swd is None:
            raise ValueError("evaluate_new_tasks returned no scoring_window_data")
        scores = calculate_scores_from_snapshot(
            swd, len(ids), np.asarray(raw["task_achievement_mask"]),
            np.asarray(raw["task_completed_mask"]), config)
        return raw, scores

    def preflight_route_fn(scores, ids, archive):
        kept = []
        preflight_route(scores, ids, kept, archive, route, tracker=tracker)
        return kept

    def source_evidence(manifest_):
        objs = {
            "TaskArchive": TaskArchive, "load_tasks": load_tasks_from_env_codes,
            "evaluate_new_tasks": evaluate_new_tasks, "run_evaluation_rollouts": run_evaluation_rollouts,
            "scoring": calculate_scores_from_snapshot, "checkpoint_loader": _load_agent_state,
            "route": route, "preflight_route": preflight_route,
        }
        return runtime_source_evidence(manifest_, objs)

    def verify_gpu(gpu_uuid):
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != gpu_uuid:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {gpu_uuid}")

    return {
        "event_sink": tracker,
        "load_config": OmegaConf.load,
        "verify_gpu": verify_gpu,
        "reconstruct_archive": reconstruct_archive,
        "archive_hash": _graph_sha,
        "archive_get_codes": lambda a, ids: a.get_task_codes(ids),
        "load_checkpoint": _load_agent_state,
        "load_frozen_rng": load_frozen_rng,
        "evaluate_and_score": evaluate_and_score,
        "preflight_route": preflight_route_fn,
        "state_hash": lambda tree: _state_hash(tree),
        "source_evidence": source_evidence,
    }


def _state_hash(tree) -> str:
    import numpy as np
    import jax
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            a = np.asarray(leaf)
            h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        except Exception:
            h.update(repr(leaf).encode())
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(value), f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _run_replay(manifest: Mapping[str, Any], rt: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Core replay through the real production chain. Fail-closed on any error."""
    import numpy as np
    validate_replay_manifest(manifest)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sink = rt["event_sink"]
    candidate_ids = [str(x) for x in manifest["candidate_ids"]]
    candidates = manifest["candidates"]
    try:
        with sink.span("replay_wall"):
            config = rt["load_config"](manifest["config"]["path"])
            perf_flags = verify_config(config, manifest)
            rt["verify_gpu"](manifest["gpu_uuid"])

            with sink.span("archive_copy_or_load"):
                archive = rt["reconstruct_archive"](manifest["archive_snapshot"]["path"])
                archive_before = rt["archive_hash"](archive)

            with sink.span("candidate_code_load"):
                code_map = rt["archive_get_codes"](archive, candidate_ids)
                if list(code_map.keys()) != candidate_ids:
                    raise ValueError("archive candidate code order mismatch")
                for cand in candidates:
                    if sha256_bytes(code_map.get(cand["id"], "").encode()) != cand["code_sha256"]:
                        raise ValueError(f"candidate {cand['id']} code hash mismatch")

            with sink.span("checkpoint_load"):
                train_state = rt["load_checkpoint"](config, manifest["checkpoint"]["path"])
            params_hash = rt["state_hash"](train_state.params)
            optimizer_hash = rt["state_hash"](train_state.opt_state)

            rng_input = rt["load_frozen_rng"](manifest["rng"])
            rng_input_evidence = rng_evidence(rng_input)

            # real production chain: evaluate_new_tasks -> calculate_scores ->
            # shared preflight_route (route/archive_update spans fire via tracker)
            raw, scores = rt["evaluate_and_score"](config, rng_input, train_state, candidate_ids, archive)
            with sink.span("route"):
                kept = rt["preflight_route"](scores, candidate_ids, archive)
            accepted = kept
            rejected = [t for t in candidate_ids if t not in kept]
            archive_after = rt["archive_hash"](archive)

            masks = np.asarray(raw["task_achievement_mask"])
            completed = np.asarray(raw["task_completed_mask"])
            task_masks_hash = sha256_bytes(masks.tobytes() + completed.tobytes())

        # source evidence fail-closed BEFORE writing any RESULT
        source_evidence = rt["source_evidence"](manifest)
        if not source_evidence.get("verified", False):
            raise SourceEvidenceError("source evidence not verified")

        result = {
            "classification": CLASSIFICATION,
            "input_manifest_sha256": manifest.get("manifest_sha256"),
            "validator_version": manifest["validator_version"],
            "source_commit": manifest["source_commit"],
            "gpu_uuid": manifest["gpu_uuid"],
            "candidate_ids": candidate_ids,
            "accepted_ids": accepted,
            "rejected_ids": rejected,
            "scores": scores,
            "archive_before_sha256": archive_before,
            "archive_after_sha256": archive_after,
            "task_code_sha256s": {c["id"]: c["code_sha256"] for c in candidates},
            "task_masks_hash": task_masks_hash,
            "checkpoint_tree_sha256": manifest["checkpoint"]["tree_sha256"],
            "checkpoint_metadata_sha256": manifest["checkpoint"]["metadata_sha256"],
            "conditioning_content_sha256": manifest["conditioning"]["content_sha256"],
            "rng_input_evidence": rng_input_evidence,
            "rng_after_sha256": "not_exposed:run_evaluation_rollouts_does_not_return_final_rng",
            "params_hash_before": params_hash,
            "optimizer_hash_before": optimizer_hash,
            "performance_flags": perf_flags,
            "llm_api_calls": 0,
            "runtime_source_evidence": source_evidence,
        }
        # atomic publish exactly once, then reload + recompute
        with sink.span("result_write"):
            result["result_sha256"] = fingerprint({k: v for k, v in result.items() if k != "result_sha256"})
            atomic_json(out / "RESULT.json", result)
        written = json.loads((out / "RESULT.json").read_text(encoding="utf-8"))
        recomputed = fingerprint({k: v for k, v in written.items() if k != "result_sha256"})
        if recomputed != written["result_sha256"]:
            raise SourceEvidenceError("result_sha256 recompute mismatch after publish")
        return result
    except Exception:
        atomic_json(out / "FAILURE.json", {"error": _exc_text()})
        raise


def _exc_text():
    import traceback
    return traceback.format_exc()


def run_replay(manifest: Mapping[str, Any], *, out_dir, enabled_events: bool = True) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rt = _real_runtime(manifest, out)
    try:
        return _run_replay(manifest, rt, out)
    finally:
        # finalize the production tracker reports (events.csv / critical_path.json)
        # in finally so error events are also captured
        try:
            from dicode.runtime_analysis import tracker
            tracker.derive_reports()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    spec_text = Path(args.spec).read_text(encoding="utf-8") if Path(args.spec).is_file() else args.spec
    spec = json.loads(spec_text)
    spec.setdefault("base_dir", str(Path(args.spec).parent if Path(args.spec).is_file() else Path.cwd()))
    manifest = build_replay_manifest(spec)
    written = write_manifest(manifest, args.output)
    reloaded = json.loads(Path(args.output).read_text(encoding="utf-8"))
    validate_replay_manifest(reloaded)
    if args.run:
        run_replay(reloaded, out_dir=args.out_dir or str(Path(args.output).with_suffix(".run")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
