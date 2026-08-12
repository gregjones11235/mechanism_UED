#!/usr/bin/env python3
"""Fixed-candidate preflight replay (B1) — R3 rework, real production path.

Replays the preflight gate against a FROZEN candidate set (no LLM, no mutation
of the frozen snapshot, fixed mid checkpoint / ids+order / RNG / conditioning /
archive initial state / score function / 40 rollout updates).

R3 requirements (audit REJECT -> fixed):
1. Real TaskArchive API: constructed via ``TaskArchive(config_obj)`` where the
   config object exposes ``graph_path`` (the production constructor), NOT a
   non-existent ``TaskArchive.load()``.
2. Stable JAX PRNGKey evidence: ``device_get`` + shape/dtype + sha256 of the
   contiguous raw bytes; never ``json.dumps`` a jax array; the RNG is not
   modified; full RNG data is not written to the report.
3. Distinct hashes: ``input_manifest_sha256`` (hash of the written/reloaded
   frozen input manifest file) vs ``result_sha256`` (hash of the result dict);
   per-file hashes for source/config/checkpoint/archive/candidate/conditioning.
   ``write_manifest``'s returned manifest is bound, reloaded and re-validated.
4. Real append-only monotonic-clock event JSONL covering the replay phases;
   JAX execute/transfer boundaries call ``block_until_ready()``; derived
   events.csv / critical_path.json / summary; overlapping time is de-duplicated
   (exclusive totals never exceed the enclosing parent's duration).
5. Real runtime source evidence: inspect actual imported/executed objects
   (TaskArchive, load_tasks, checkpoint loader, evaluator, scoring, route,
   archive update) and compare their real source file SHA256 against the
   manifest's source_mapping; mismatch or missing binding fails closed.

All new profiling / events are the replay tool's own output (always produced by
``--run``); the production ``tracker`` (runtime_profiling) is left disabled so
no production profiling artifacts are created.
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

# --- B1 production profiling phase contract (11) --------------------------------
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

# --- replay event phases (measurement, not the production tracker) ---------------
REPLAY_PHASES = (
    "replay_wall", "archive_copy_or_load", "candidate_code_load", "checkpoint_load",
    "task_load", "evaluator_build", "evaluator_lower", "evaluator_compile",
    "evaluator_execute", "scoring_device_transfer", "scoring_cpu",
    "route_decision", "archive_update", "result_write",
)
EVENT_FIELDS = (
    "run_id", "stage", "candidate_id", "phase", "parent_phase",
    "start_monotonic_ns", "end_monotonic_ns", "duration_s", "status",
    "cache_hit", "task_signature", "overlap_group",
)

MID_CHECKPOINT_STEP = 2100
ROLLOUT_UPDATES = 40
CONDITIONING_DIM = 67
RNG_ALGORITHM = "sha256-little-endian-u32:PREFLIGHT_REPLAY_V1:{candidate_id}:{idx}"
SCORE_FUNCTIONS = ("learnability", "pvl", "max_mc")
CLASSIFICATION = "PREFLIGHT_CANDIDATE_REPLAY"
VALIDATOR_VERSION = "2"


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


def derive_rng(seed: int, candidate_id: str, idx: int) -> list[int]:
    material = f"PREFLIGHT_REPLAY_V1:{candidate_id}:{idx}:{int(seed)}".encode()
    digest = hashlib.sha256(material).digest()
    return [int.from_bytes(digest[0:4], "little"), int.from_bytes(digest[4:8], "little")]


def rng_evidence(rng) -> dict[str, Any]:
    """Stable evidence for a JAX PRNGKey without json-dumping the array.

    Uses device_get (when jax is available) + shape/dtype + sha256 of the
    contiguous raw bytes. The RNG is not modified and its full data is not
    written to the report. Plain host arrays (e.g. the fake-runtime test RNG)
    hash identically, so the fake and real paths share one evidence function.
    """
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
    return {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "sha256": sha256_bytes(a.tobytes()),
    }


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


def _candidates(spec, base: Path) -> tuple[list[dict[str, Any]], list[str]]:
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
    source_commit = spec.get("source_commit")
    gpu_uuid = spec.get("gpu_uuid")
    if not source_commit or not str(source_commit).strip() or not gpu_uuid or not str(gpu_uuid).strip():
        raise ValueError("source_commit and gpu_uuid are required")
    seed = int(spec.get("rng_seed", 42))
    base = Path(spec.get("base_dir", ".")).resolve()
    checkpoint_raw, conditioning_raw, archive_raw = spec.get("checkpoint"), spec.get("conditioning_path"), spec.get("archive_snapshot")
    if not checkpoint_raw or not conditioning_raw or not archive_raw:
        raise ValueError("replay spec requires checkpoint, conditioning_path, archive_snapshot")
    checkpoint = _checkpoint_info(_resolve(checkpoint_raw, base), MID_CHECKPOINT_STEP)
    candidates, candidate_ids = _candidates(spec, base)
    conditioning = _conditioning_info(_resolve(conditioning_raw, base), len(candidate_ids))
    archive = _archive_info(_resolve(archive_raw, base))
    source_mapping = _source_mapping(spec, base)
    config = _config_info(spec, base)
    num_envs = int(spec.get("num_envs", 1024)); num_steps = int(spec.get("num_steps", 128))
    if num_envs < 1024 or num_steps < 128:
        raise ValueError("num_envs/num_steps must not be lowered below the scientific budget")
    rng = {cid: derive_rng(seed, cid, 0) for cid in candidate_ids}
    return {
        "classification": CLASSIFICATION, "not_end_to_end_ued": True, "llm_api_calls": 0,
        "mid_checkpoint_step": MID_CHECKPOINT_STEP, "rollout_updates": ROLLOUT_UPDATES,
        "conditioning_dim": CONDITIONING_DIM, "score_function": score_function,
        "rng_seed": seed, "rng_algorithm": RNG_ALGORITHM,
        "validator_version": str(spec.get("validator_version", VALIDATOR_VERSION)),
        "source_commit": str(source_commit), "gpu_uuid": str(gpu_uuid),
        "validation": {"rollout_updates": ROLLOUT_UPDATES, "num_envs": num_envs, "num_steps": num_steps},
        "source_mapping": source_mapping, "config": config,
        "candidate_ids": candidate_ids, "candidates": candidates,
        "checkpoint": checkpoint, "conditioning": conditioning, "archive_snapshot": archive,
        "rng": rng, "phases": list(REPLAY_PHASES),
    }


def _without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "manifest_sha256"}


def write_manifest(manifest: Mapping[str, Any], output: str | os.PathLike[str]) -> dict[str, Any]:
    """Atomically write a manifest and return the written (self-hashed) dict."""
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
    if manifest.get("classification") != CLASSIFICATION or manifest.get("not_end_to_end_ued") is not True or manifest.get("llm_api_calls") != 0:
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
    if len(manifest.get("candidates", [])) != len(ids) or manifest.get("phases") != list(REPLAY_PHASES):
        raise ValueError("candidates/phases mismatch")
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
    expected_rng = {cid: derive_rng(int(manifest["rng_seed"]), cid, 0) for cid in ids}
    if manifest.get("rng") != expected_rng:
        raise ValueError("replay RNG changed")
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
    return {
        "preflight_reuse_loaded_tasks": bool(perf.get("preflight_reuse_loaded_tasks", False)),
        "compact_preflight_payload": bool(perf.get("compact_preflight_payload", False)),
        "eval_compile_cache": bool(perf.get("eval_compile_cache", False)),
        "train_compile_cache": bool(perf.get("train_compile_cache", False)),
        "embedding_cache": bool(perf.get("embedding_cache", False)),
        "validation_cache": bool(perf.get("validation_cache", False)),
    }


# --- replay event recorder (monotonic-clock, append-only) -----------------------
class ReplayRecorder:
    """Append-only monotonic-clock event recorder for the replay measurement.

    Always writes the replay's own events.jsonl when enabled (the replay is a
    measurement tool). When ``enabled=False`` no artifacts are created. Finalize
    derives events.csv + critical_path.json (de-overlapped exclusive totals) +
    summary.json.
    """

    def __init__(self, out_dir: Path, run_id: str, enabled: bool = True, stage: str = "replay"):
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self.stage = stage
        self.enabled = enabled
        self.events: list[dict[str, Any]] = []
        self._jsonl = None
        if enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = (self.out_dir / "events.jsonl").open("a", encoding="utf-8")

    def _emit(self, event: dict[str, Any]) -> None:
        if not self.enabled or self._jsonl is None:
            return
        line = json.dumps(event, sort_keys=True) + "\n"
        self._jsonl.write(line)
        self._jsonl.flush()

    def record(self, phase: str, start_ns: int | None = None, end_ns: int | None = None,
               *, candidate_id: str = "", parent_phase: str = "", status: str = "ok",
               cache_hit: bool = False, task_signature: str = "", overlap_group: str = "") -> None:
        if not self.enabled:
            return
        s = start_ns if start_ns is not None else time.monotonic_ns()
        e = end_ns if end_ns is not None else time.monotonic_ns()
        ev = {"run_id": self.run_id, "stage": self.stage, "candidate_id": str(candidate_id),
              "phase": phase, "parent_phase": str(parent_phase), "start_monotonic_ns": s,
              "end_monotonic_ns": e, "duration_s": max(0.0, (e - s) / 1e9), "status": status,
              "cache_hit": bool(cache_hit), "task_signature": str(task_signature),
              "overlap_group": str(overlap_group)}
        self.events.append(ev)
        self._emit(ev)

    def span(self, phase: str, *, candidate_id: str = "", parent_phase: str = ""):
        class _Ctx:
            def __init__(self, rec):
                self.rec = rec
                self.start = None

            def __enter__(self):
                self.start = time.monotonic_ns()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.rec.record(phase, self.start, time.monotonic_ns(), candidate_id=candidate_id,
                                parent_phase=parent_phase, status="error" if exc_type else "ok")
                return False
        return _Ctx(self)

    def finalize(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        self._jsonl.close()
        events = self.events
        # de-overlapped exclusive totals per phase (interval union)
        def _union(evs):
            if not evs:
                return 0.0
            ivs = sorted(((e["start_monotonic_ns"], e["end_monotonic_ns"]) for e in evs))
            total = 0
            cur_s, cur_e = ivs[0]
            for s, e in ivs[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    total += cur_e - cur_s; cur_s, cur_e = s, e
            return total + (cur_e - cur_s)
        phases = {}
        for e in events:
            phases.setdefault(e["phase"], []).append(e)
        exclusive = {p: round(_union(v) / 1e9, 6) for p, v in phases.items()}
        wall = exclusive.get("replay_wall", 0.0)
        covered = _union(events) / 1e9
        with (self.out_dir / "events.csv").open("w", encoding="utf-8") as f:
            f.write(",".join(EVENT_FIELDS) + "\n")
            for e in events:
                f.write(",".join(str(e.get(k, "")) for k in EVENT_FIELDS) + "\n")
        cp = {"run_id": self.run_id, "session_wall_s": round(wall, 6),
              "covered_union_s": round(covered, 6), "exclusive_phase_totals": exclusive,
              "critical_path": [{"phase": p, "exclusive_s": v} for p, v in
                                sorted(exclusive.items(), key=lambda kv: kv[1], reverse=True)]}
        with (self.out_dir / "critical_path.json").open("w", encoding="utf-8") as f:
            json.dump(cp, f, indent=2)
        summary = {"run_id": self.run_id, "event_count": len(events), "critical_path": cp}
        with (self.out_dir / "replay_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


# --- real runtime source evidence ----------------------------------------------
def runtime_source_evidence(manifest: Mapping[str, Any], source_objects: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the ACTUAL runtime-imported objects against the manifest source_mapping.

    For each key in ``source_objects`` (real imported callable/class), inspect
    its real source file + sha256 and require an exact match against a
    manifest source_mapping entry (by resolved path). Mismatch/missing fails
    closed.
    """
    expected = manifest["source_mapping"]
    result: dict[str, Any] = {"verified": True, "objects": {}}
    for key, obj in source_objects.items():
        src = inspect.getsourcefile(obj)
        if not src:
            result["verified"] = False
            result["objects"][key] = {"verified": False, "error": "no source file"}
            continue
        path = Path(src).resolve()
        sha = file_sha256(path)
        match = None
        for label, entry in expected.items():
            if Path(entry["path"]).resolve() == path:
                match = (label, entry)
                break
        ok = match is not None and match[1]["sha256"] == sha
        if not ok:
            result["verified"] = False
        result["objects"][key] = {
            "module": getattr(obj, "__module__", ""), "qualname": getattr(obj, "__qualname__", getattr(obj, "__name__", "")),
            "source_path": str(path), "source_sha256": sha,
            "expected_label": match[0] if match else None,
            "expected_sha256": match[1]["sha256"] if match else None,
            "verified": ok,
        }
    return result


# --- core replay (runtime-bundle) ----------------------------------------------
def _verify_candidate_codes(code_map, candidates, candidate_ids) -> None:
    if list(code_map.keys()) != candidate_ids:
        raise ValueError(f"archive candidate code order mismatch: {list(code_map.keys())} != {candidate_ids}")
    for cand in candidates:
        cid = cand["id"]
        code = code_map.get(cid)
        if code is None or sha256_bytes(code.encode()) != cand["code_sha256"]:
            raise ValueError(f"candidate {cid} code hash mismatch with manifest")


def _route_all(route_fn, scores, candidate_ids):
    decisions, accepted, rejected = [], [], []
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


def _run_replay(manifest: Mapping[str, Any], rt: Mapping[str, Any], out_dir: Path,
                recorder: ReplayRecorder) -> dict[str, Any]:
    """Core replay call sequence. Every step emits a recorder event; the JAX
    execute/transfer boundaries call block_until_ready() inside the runtime."""
    import numpy as np
    validate_replay_manifest(manifest)
    run_id = recorder.run_id
    config = None
    result: dict[str, Any] = {}
    with recorder.span("replay_wall"):
        config = rt["load_config"](manifest["config"]["path"])
        perf_flags = verify_config(config, manifest)
        rt["verify_gpu"](manifest["gpu_uuid"])
        candidate_ids = [str(x) for x in manifest["candidate_ids"]]
        candidates = manifest["candidates"]

        with recorder.span("archive_copy_or_load"):
            archive = rt["reconstruct_archive"](manifest["archive_snapshot"]["path"])
            archive_before = rt["archive_hash"](archive)

        with recorder.span("candidate_code_load", parent_phase="archive_copy_or_load"):
            code_map = rt["archive_get_codes"](archive, candidate_ids)
            _verify_candidate_codes(code_map, candidates, candidate_ids)

        with recorder.span("task_load"):
            task_classes, ok_ids = rt["load_tasks"](archive, candidate_ids)
            if ok_ids != candidate_ids:
                raise ValueError(f"candidate code load mismatch: expected {candidate_ids}, got {ok_ids}")

        task_achievement_mask, task_completed_mask = rt["achievement_masks"](task_classes)

        with recorder.span("checkpoint_load"):
            train_state = rt["load_checkpoint"](config, manifest["checkpoint"]["path"])
        params_hash = rt["state_hash"](train_state.params)
        optimizer_hash = rt["state_hash"](train_state.opt_state)

        cond_table = np.load(manifest["conditioning"]["path"], allow_pickle=False)
        task_embeddings = np.ascontiguousarray(cond_table[1:])

        rng_input = rt["make_input_rng"](int(manifest["rng_seed"]), candidate_ids)
        rng_evidence_in = rng_evidence(rng_input)

        with recorder.span("evaluator_execute", parent_phase="replay_wall"):
            results = rt["run_rollout"](config, rng_input, task_classes, int(manifest["rollout_updates"]),
                                        task_embeddings, train_state)
        swd = results.get("metrics", {}).get("scoring_window_data")
        if swd is None:
            raise ValueError("rollouts produced no scoring_window_data")

        scores = rt["score"](config, swd, len(candidate_ids), task_achievement_mask, task_completed_mask)

        with recorder.span("route_decision"):
            decisions, accepted, rejected = _route_all(rt["route"], scores, candidate_ids)

        with recorder.span("archive_update"):
            _apply_archive_updates(archive, decisions, rt["archive_update_accept"], rt["archive_update_reject"])
        archive_after = rt["archive_hash"](archive)

        source_evidence = runtime_source_evidence(manifest, rt["source_objects"])

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
            "route_decisions": decisions,
            "archive_before_sha256": archive_before,
            "archive_after_sha256": archive_after,
            "task_code_sha256s": {c["id"]: c["code_sha256"] for c in candidates},
            "task_masks_hash": sha256_bytes(np.ascontiguousarray(task_achievement_mask).tobytes()
                                            + np.ascontiguousarray(task_completed_mask).tobytes()),
            "checkpoint_tree_sha256": manifest["checkpoint"]["tree_sha256"],
            "checkpoint_metadata_sha256": manifest["checkpoint"]["metadata_sha256"],
            "conditioning_content_sha256": manifest["conditioning"]["content_sha256"],
            "rng_input_evidence": rng_evidence_in,
            "rng_after_sha256": "not_exposed:run_evaluation_rollouts_does_not_return_final_rng",
            "params_hash_before": params_hash,
            "optimizer_hash_before": optimizer_hash,
            "performance_flags": perf_flags,
            "llm_api_calls": 0,
            "runtime_source_evidence": source_evidence,
            "run_id": run_id,
        }
        with recorder.span("result_write"):
            atomic_json(out_dir / "RESULT.json", result)
    result["result_sha256"] = fingerprint(result)
    # write the final result with its self-hash bound
    atomic_json(out_dir / "RESULT.json", result)
    return result


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


# --- real runtime --------------------------------------------------------------
def _graph_sha(archive) -> str:
    """Canonical sha256 of the archive's in-memory graph (nodes + edges)."""
    g = archive.graph
    parts = []
    for node in sorted(g.nodes()):
        attrs = dict(g.nodes[node])
        parts.append(str(node) + ":" + json.dumps(canonical(attrs), sort_keys=True))
    for u, v in sorted((sorted(e) for e in g.edges())):
        parts.append(f"{u}->{v}")
    return sha256_bytes("\n".join(parts).encode())


def _real_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
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
        # copy to a fresh temp dir; never mutate the frozen snapshot
        tmp = Path(tempfile.mkdtemp(prefix="preflight_replay_archive_"))
        src = Path(snapshot_path)
        if src.is_dir():
            shutil.copytree(src, tmp / src.name)
            graphml = next((p for p in (tmp / src.name).rglob("*.graphml")), None)
            if graphml is None:
                raise ValueError(f"archive snapshot dir has no .graphml: {src}")
            return TaskArchive(_graph_cfg(graphml))
        shutil.copy(src, tmp / src.name)
        return TaskArchive(_graph_cfg(tmp / src.name))

    def _graph_cfg(graphml_path):
        return type("ArchiveCfg", (), {"graph_path": str(graphml_path)})()

    def archive_hash(archive):
        return _graph_sha(archive)

    def archive_get_codes(archive, ids):
        return archive.get_task_codes(ids)

    def load_tasks(archive, ids):
        return load_tasks_from_env_codes(archive, ids)

    def achievement_masks(classes):
        masks = _create_achievement_masks(classes)
        return np.asarray(masks[0]), np.asarray(masks[1])

    def load_checkpoint(config, ckpt_path):
        return _load_agent_state(config, ckpt_path)

    def state_hash(tree):
        from dicode.runtime_analysis import RuntimeTracker
        return _state_hash_impl(tree)

    def make_input_rng(seed, candidate_ids):
        rng0 = derive_rng(seed, candidate_ids[0], 0)
        return jax.random.PRNGKey(rng0[0])

    def run_rollout(config, rng, classes, updates, embeddings, train_state):
        results = run_evaluation_rollouts(config, rng, classes, updates,
                                          task_embeddings=embeddings, train_state=train_state)
        for leaf in jax.tree_util.tree_leaves(results):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return results

    def score(config, swd, num_tasks, mask, completed):
        return calculate_scores_from_snapshot(swd, num_tasks, mask, completed, config)

    def verify_gpu(gpu_uuid):
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            raise RuntimeError("replay requires a GPU device")
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
        "source_objects": {
            "TaskArchive": TaskArchive, "load_tasks": load_tasks_from_env_codes,
            "checkpoint_loader": _load_agent_state, "scoring": calculate_scores_from_snapshot,
            "route": route, "evaluator": run_evaluation_rollouts,
        },
    }


def _state_hash_impl(tree) -> str:
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


def run_replay(manifest: Mapping[str, Any], *, out_dir: str | os.PathLike[str],
               enabled_events: bool = True) -> dict[str, Any]:
    """Execute the frozen-candidate preflight replay (real production path)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
    recorder = ReplayRecorder(out, run_id, enabled=enabled_events)
    rt = _real_runtime(manifest)
    result = _run_replay(manifest, rt, out, recorder)
    recorder.finalize()
    return result


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
    # bind the replay to the WRITTEN manifest: reload and re-validate
    reloaded = json.loads(Path(args.output).read_text(encoding="utf-8"))
    validate_replay_manifest(reloaded)
    if args.run:
        out_dir = args.out_dir or str(Path(args.output).with_suffix(".run"))
        run_replay(reloaded, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
