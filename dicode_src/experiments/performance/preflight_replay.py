#!/usr/bin/env python3
"""Fixed-candidate preflight replay (B1).

Replays the preflight gate against a FROZEN candidate set so that the gate's
cost/decisions can be re-measured deterministically after A2 (no LLM calls, no
archive mutation, fixed mid checkpoint, fixed ids+order, fixed RNG, fixed
conditioning, fixed archive initial state, fixed score function, 40 rollout
updates).

The module is importable on the CPU-only box: the pure-logic parts (phase
contract, manifest build/validate, RNG derivation, hashing) have no JAX
dependency and are unit-tested in
src/dicode/skill_preflight/tests/test_perf48_preflight_profiling_b1.py.
The actual GPU replay (`run_replay`) lazily imports jax + dicode and is marked
`# requires-jax-server` -- it runs after A2, not on this box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

# --- B1 profiling phase contract (must match the instrumentation sites) -------
# audit points:
#   candidate_code_load                 src/dicode/task_utils.py
#   candidate_cpu_validation_build      src/dicode/dreaming/gen_manager.py
#   candidate_cpu_validation_compile    src/dicode/dreaming/gen_manager.py
#   candidate_cpu_validation_execute    src/dicode/dreaming/gen_manager.py
#   preflight_task_reload               experiments/training/run_dicode.py + src/dicode/evaluation/online_evaluation.py
#   preflight_eval_build                src/dicode/ppo_tr.py
#   preflight_eval_lower_compile        src/dicode/ppo_tr.py
#   preflight_eval_execute              src/dicode/ppo_tr.py
#   route                               experiments/training/run_dicode.py
#   archive_update                      experiments/training/run_dicode.py
#   preflight_wall                      experiments/training/run_dicode.py
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

# --- frozen inputs ------------------------------------------------------------
MID_CHECKPOINT_STEP = 2100
ROLLOUT_UPDATES = 40
CONDITIONING_DIM = 67
RNG_ALGORITHM = "sha256-little-endian-u32:PREFLIGHT_REPLAY_V1:{candidate_id}:{idx}"
SCORE_FUNCTIONS = ("learnability", "pvl", "max_mc")
CLASSIFICATION = "PREFLIGHT_CANDIDATE_REPLAY"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> Any:
    """Canonicalize values before JSON hashing (including non-finite floats)."""
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
    return sha256_bytes(
        json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash sorted relative paths and bytes, preventing path/content collisions."""
    h = hashlib.sha256()
    for p in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = p.relative_to(path).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, "little")); h.update(rel)
        h.update(len(data).to_bytes(8, "little")); h.update(data)
    return h.hexdigest()


def derive_rng(seed: int, candidate_id: str, idx: int) -> list[int]:
    """Deterministic 2x u32 RNG per candidate (little-endian sha256).

    Mirrors perf48_replay_manifest._u32_rng. The replay's RNG is fully frozen so
    two runs on the same frozen inputs produce identical rollouts.
    """
    material = f"PREFLIGHT_REPLAY_V1:{candidate_id}:{idx}:{int(seed)}".encode()
    digest = hashlib.sha256(material).digest()
    return [int.from_bytes(digest[0:4], "little"), int.from_bytes(digest[4:8], "little")]


def _resolve(path: str | os.PathLike[str], base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def _conditioning_info(path: Path, task_count: int) -> dict[str, Any]:
    try:
        import numpy as np
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"unable to load conditioning table: {path}") from exc
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
    if not any(p.name == "_CHECKPOINT_METADATA" for p in files):
        raise ValueError(f"checkpoint missing _CHECKPOINT_METADATA: {path}")
    nums = re.findall(r"\d+", path.name)
    if not nums or int(nums[-1]) != global_step:
        raise ValueError(f"checkpoint basename/global_step mismatch: {path.name} != {global_step}")
    return {"path": str(path), "tree_sha256": tree_sha256(path),
            "basename": path.name, "global_step": global_step}


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
    """Recompute all hashes and re-run the structural gates; raises on mismatch.

    A freshly-built manifest (before write_manifest) has no manifest_sha256, so
    the self-check runs only when the key is present.
    """
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
    ids = [str(x) for x in manifest.get("candidate_ids", [])]
    if not ids or len(ids) != len(set(ids)) or any(not x.strip() for x in ids):
        raise ValueError("invalid candidate id order/duplicates")
    if len(manifest.get("candidates", [])) != len(ids):
        raise ValueError("candidates/candidate_ids mismatch")
    expected_phases = list(PREFLIGHT_PHASES)
    if manifest.get("phases") != expected_phases:
        raise ValueError(f"phases must equal {expected_phases}")

    # re-derive from the frozen files
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
    expected_rng = {cid: derive_rng(int(manifest["rng_seed"]), cid, 0) for cid in ids}
    if manifest.get("rng") != expected_rng:
        raise ValueError("replay RNG changed")
    return dict(manifest)


# --- GPU replay (requires jax + the full dicode stack; run after A2) ---------
def run_replay(manifest: Mapping[str, Any], *, out_jsonl: str | None = None) -> dict[str, Any]:
    """Execute the frozen-candidate preflight replay on GPU/server.

    requires-jax-server: lazily imports jax/dicode. Not exercised on the CPU-only
    box. Reconstructs the archive initial state, reloads the frozen candidate
    Env classes, runs 40 frozen-policy rollout updates via make_eval, scores via
    calculate_scores_from_snapshot, routes each candidate, and emits scores,
    route decisions, accepted/rejected ids+order, archive before/after hash,
    task code hashes, RNG before/after hash, checkpoint hash, and per-phase wall
    clocks.
    """
    validate_replay_manifest(manifest)  # fail closed before touching jax
    import jax
    import numpy as np

    # --- frozen inputs --------------------------------------------------------
    ckpt_path = Path(manifest["checkpoint"]["path"])
    score_function = manifest["score_function"]
    rollout_updates = int(manifest["rollout_updates"])
    candidate_ids = [str(x) for x in manifest["candidate_ids"]]
    rng_seed = int(manifest["rng_seed"])

    # --- lazy imports of the dicode machinery ---------------------------------
    from dicode.task_utils import load_tasks_from_env_codes
    from dicode.dreaming.gen_manager import TaskArchive
    from dicode.scoring import calculate_scores_from_snapshot
    from dicode.skill_preflight.preflight import route
    from dicode.ppo_tr import run_evaluation_rollouts
    from dicode.utils.general.train_state_utils import load_weights_only
    from minicraftax.envs.craftax import CraftaxAugObsTrain

    wall: dict[str, float] = {}

    t0 = time.monotonic()
    # Reconstruct the archive from the frozen snapshot (initial state).
    archive = _reconstruct_archive(Path(manifest["archive_snapshot"]["path"]))
    archive_before = tree_sha256(Path(manifest["archive_snapshot"]["path"]))
    wall["archive_reconstruct"] = time.monotonic() - t0

    t0 = time.monotonic()
    cond_table = np.load(manifest["conditioning"]["path"], allow_pickle=False)
    wall["conditioning_load"] = time.monotonic() - t0

    t0 = time.monotonic()
    dummy_env = CraftaxAugObsTrain(condition_on_task=True, conditioning_type="one_hot")
    train_state = load_weights_only(checkpoint_path=str(ckpt_path), env=dummy_env,
                                    env_params=dummy_env.default_params)
    wall["checkpoint_load"] = time.monotonic() - t0

    # Re-load the candidate Env classes from the frozen code (same source/semantics
    # as run_dicode's first load).
    t0 = time.monotonic()
    codes = {c["id"]: Path(c["path"]).read_text(encoding="utf-8") for c in manifest["candidates"]}
    task_classes, ok_ids = load_tasks_from_env_codes(archive, candidate_ids)
    wall["candidate_code_load"] = time.monotonic() - t0
    if ok_ids != candidate_ids:
        raise ValueError(f"candidate code load mismatch: expected {candidate_ids}, got {ok_ids}")

    # Frozen one-hot conditioning for the candidates (rows 1..N of the table).
    task_embeddings = cond_table[1:]
    task_embeddings = jax.numpy.asarray(task_embeddings)

    rng_before = derive_rng(rng_seed, candidate_ids[0], 0)
    rng = jax.random.PRNGKey(rng_before[0])

    t0 = time.monotonic()
    results = run_evaluation_rollouts(
        None, rng, task_classes, rollout_updates, task_embeddings=task_embeddings,
        train_state=train_state)
    wall["eval_rollouts"] = time.monotonic() - t0

    scoring_window_data = results.get("metrics", {}).get("scoring_window_data")
    if scoring_window_data is None:
        raise ValueError("rollouts produced no scoring_window_data")

    num_total_achievements = 67
    task_achievement_mask = np.zeros((len(candidate_ids), num_total_achievements), dtype=bool)
    task_completed_mask = np.zeros((len(candidate_ids), num_total_achievements), dtype=bool)

    t0 = time.monotonic()
    scores = calculate_scores_from_snapshot(
        scoring_window_data, len(candidate_ids), task_achievement_mask,
        task_completed_mask, None)
    wall["scoring"] = time.monotonic() - t0

    t0 = time.monotonic()
    decisions, accepted, rejected = [], [], []
    for i, tid in enumerate(candidate_ids):
        sr = float(scores.get(str(i), {}).get("sr", -1.0))
        d = route(max(sr, 0.0), any_partial_progress=(sr >= 0.0))
        decisions.append({"id": tid, "action": d.action, "reason": d.reason, "sr": sr})
        (accepted if d.action == "accept" else rejected).append(tid)
    wall["route"] = time.monotonic() - t0

    archive_after = tree_sha256(Path(manifest["archive_snapshot"]["path"]))
    rng_after = derive_rng(rng_seed, candidate_ids[0], 1)
    result = {
        "classification": CLASSIFICATION,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "candidate_ids": candidate_ids,
        "accepted_ids": accepted,
        "rejected_ids": rejected,
        "scores": scores,
        "route_decisions": decisions,
        "archive_before_sha256": archive_before,
        "archive_after_sha256": archive_after,
        "task_code_sha256s": {c["id"]: c["code_sha256"] for c in manifest["candidates"]},
        "rng_before_sha256": sha256_bytes(json.dumps(rng_before).encode()),
        "rng_after_sha256": sha256_bytes(json.dumps(rng_after).encode()),
        "checkpoint_sha256": manifest["checkpoint"]["tree_sha256"],
        "per_phase_wall_s": wall,
        "llm_api_calls": 0,
    }
    if out_jsonl:
        write_manifest(result, out_jsonl)
    return result


def _reconstruct_archive(snapshot: Path):
    """Reconstruct a TaskArchive from the frozen snapshot.

    requires-jax-server: TaskArchive construction needs the craftax stack. The
    snapshot is a directory (or GraphML) captured at the replay point; the exact
    reconstruction mirrors the A2/perf48 archive-reconstruction path and is
    finalized by the main agent after A2.
    """
    from dicode.dreaming.gen_manager import TaskArchive
    return TaskArchive.load(str(snapshot))


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
