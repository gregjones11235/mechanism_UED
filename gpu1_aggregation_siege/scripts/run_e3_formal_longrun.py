#!/usr/bin/env python3
"""E3 FORMAL LONGRUN controller (SESSION-ALIGNED CONSERVATIVE 16x128) — THIN
orchestration over the production chain.

Protocol: E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128.
  1 E3 window == 1 complete native DiCode curriculum session ==
  max_updates_per_session (100) outer updates.

Layout (FROZEN, conservative):
  num_envs = 1024, num_steps = 128  -> env_steps/update = 131072
  15 sampled curriculum tasks = 12 dynamic + 3 non-target anchors
  DiCode internally appends OriginalTask exactly once -> 16 total classes.
  original_task_proportion = 0.20.

Reuses the production E3 chain functions from ``run_e3_real_smoke`` (mount /
actual-N / two-LLM / canonical session runtime / compile+register / runstate
codec) and threads the selected Student forward across sessions:

    session 0: canonical checkpoint -> mount -> actual-N -> two-LLM ->
               compile 15+1 -> canonical DiCode SESSION (100 updates) ->
               full RunState
    session k: restore RunState(k-1) -> train_state + counters + memory ->
               mount (updated student) -> actual-N -> two-LLM ->
               compile 15+1 -> canonical DiCode SESSION (100 updates) ->
               full RunState

The frontier / actual-N / two-LLM / curriculum are regenerated ONCE per
session — never per update inside a session.  The OriginalTask is appended by
DiCode exactly once and is never in sampled_task_ids.

NO E3 algorithm is re-implemented here.  Every E3 step is delegated to the
production modules.  This file ONLY owns the session loop + counter threading
+ per-session evidence bookkeeping.

Usage (on server, in tmux):

    python run_e3_formal_longrun.py \
        --student=PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 \
        --sessions=<N> \
        --out=<RUN_DIR>

Env: source ~/.qwen_env; WANDB_MODE=offline; XLA_PYTHON_CLIENT_PREALLOCATE=false;
     CUDA_VISIBLE_DEVICES=<GPU>; PYTHONPATH=<repo>/gpu1_aggregation_siege/src
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import socket
import shutil
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIEGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(SIEGE_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

PASS, FAIL, BLOCKED = 0, 4, 5

PERSISTENT = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
RESET128 = "RESET128_RMT16_ORIGINAL_VTRACE_98304"
SLOWGRU = "SLOWGRU_PERSISTENT_CANONICAL_98304"

FORMAL_SOURCE_COMMIT = "SIGNED_RUNTIME_HEAD"
FORMAL_BRANCH = "henry/simulator-frontier-foundation-codex"

# E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128 resolved budget.
# num_envs = 1024, num_steps = 128 -> env_steps/update = 131072.
NUM_ENVS = 1024
NUM_STEPS = 128
ENV_STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS          # 131072
TOTAL_TIMESTEPS = 2_005_401_600
NATIVE_TOTAL_UPDATES = TOTAL_TIMESTEPS // ENV_STEPS_PER_UPDATE   # 15300
MAX_UPDATES_PER_SESSION_NATIVE = 100
# One formal window == one complete native DiCode curriculum session ==
# max_updates_per_session (100) outer updates.
UPDATES_PER_SESSION = MAX_UPDATES_PER_SESSION_NATIVE  # 100
# Reference experiment budget: selected Student is already past seed stage, so
# E3 starts from the curriculum part = 151 sessions x 100 = 15100 updates.
# Each session executes the full native session (never a for-loop of one-update
# calls).
REFERENCE_EXPERIMENT_UPDATES = 15100
REFERENCE_EXPERIMENT_SESSIONS = REFERENCE_EXPERIMENT_UPDATES // UPDATES_PER_SESSION  # 151
REFERENCE_EXPERIMENT_ENV_STEPS = REFERENCE_EXPERIMENT_UPDATES * ENV_STEPS_PER_UPDATE

# 15 sampled curriculum slots = 12 dynamic + 3 non-target anchors.
CURRICULUM_SLOT_COUNT_CONSERVATIVE = 15

# Session constants used by the smoke driver / E3 chain.
ACTUAL_N = 4
SEARCH_HORIZON = 16
REQUESTED_N_PER_SESSION = 12
SEED = 42

# Audit gate (sole-controller 2026-08-10): the full-budget formal longrun is
# NOT authorized until the audit items are closed and the sole controller signs
# a full-budget authorization manifest covering the current source commit.
# Until then this MUST stay False and the controller refuses full-budget
# launches (verification-scope runs with a signed verification manifest are
# still permitted).
E3_FORMAL_LONGRUN_AUTHORIZED = False

# Verification scope: sessions above this cap are only permitted when
# E3_FORMAL_LONGRUN_AUTHORIZED is True (i.e. the sole controller signed a
# full-budget authorization covering the current source commit).
VERIFICATION_SESSIONS_MAX = 3
EXPECTED_PHYSICAL_GPU_UUID = "GPU-3c7a2864-755b-7045-b293-6f80e748283f"
FORMAL_RUN_ROOT = "/media/数据磁盘2"

# Authorization material (runner-side, verification only — never the private
# key).  These are bound into the signed manifest and re-verified at runtime.
AUTH_DIR = os.path.join(SIEGE_ROOT, "auth")
AUTH_PUBLIC_KEY = os.path.join(AUTH_DIR, "e3_controller_public_key.bin")
AUTH_REGISTRY = os.path.join(AUTH_DIR, "formal_asset_registry.json")


def _log(msg: str) -> None:
    print(f"[e3-longrun-ctrl] {msg}", flush=True)


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=SIEGE_ROOT, timeout=30)
        return (proc.stdout or "").strip()
    except Exception:
        return "UNAVAILABLE"


def _git_worktree_clean() -> bool:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, cwd=SIEGE_ROOT, timeout=30)
        return proc.returncode == 0 and not (proc.stdout or "").strip()
    except Exception:
        return False


def _gpu_uuid() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        for line in out.splitlines():
            fields = [p.strip() for p in line.split(",")]
            if len(fields) >= 2 and fields[0] == cvd:
                return fields[1]
        return "UNKNOWN"
    except Exception as exc:
        return f"UNKNOWN:{exc!r}"


def _assert_formal_gpu_binding(*, expected_uuid: str = EXPECTED_PHYSICAL_GPU_UUID) -> str:
    """Require the physical GPU1 binding before any production work."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("BLOCKED_GPU_BINDING_CUDA_VISIBLE_DEVICES_MUST_BE_1")
    actual = _gpu_uuid()
    if actual != expected_uuid:
        raise RuntimeError(
            f"BLOCKED_GPU_BINDING_UUID_MISMATCH:{actual!r}!={expected_uuid!r}")
    try:
        import jax
        devices = list(jax.devices())
    except Exception as exc:
        raise RuntimeError(f"BLOCKED_GPU_BINDING_JAX_QUERY:{exc!r}") from exc
    if not devices or not any(str(getattr(device, "platform", "")) == "gpu"
                              for device in devices):
        raise RuntimeError("BLOCKED_GPU_BINDING_NO_GPU_JAX_DEVICE")
    return actual


def _assert_formal_disk_capacity(run_dir: str, *, estimated_checkpoint_size: int = 372_465_461) -> tuple[int, int]:
    """Require the full run to live on the data disk with safe headroom."""
    resolved = os.path.realpath(run_dir)
    root = os.path.realpath(FORMAL_RUN_ROOT)
    if not os.path.isdir(root):
        raise RuntimeError("BLOCKED_FORMAL_DATA_DISK_ROOT_UNAVAILABLE")
    if not (resolved == root or resolved.startswith(root + os.sep)):
        raise RuntimeError("BLOCKED_FORMAL_RUN_DIR_MUST_USE_DATA_DISK")
    if estimated_checkpoint_size <= 0:
        raise RuntimeError("BLOCKED_FORMAL_CHECKPOINT_SIZE_INVALID")
    # 151 checkpoints, 1.2x retained copies, plus a conservative 1 GiB
    # allowance for temporary state/meta files and logs; never lower than the
    # measured 70 GiB safety floor.
    required = max(70 * (1024 ** 3), int(151 * estimated_checkpoint_size * 1.2) + (1 << 30))
    # The claimed run directory is intentionally absent on a new run.  Probe
    # the existing filesystem root rather than forcing an early mkdir/claim.
    free = int(shutil.disk_usage(root).free)
    if free < required:
        raise RuntimeError(
            f"BLOCKED_FORMAL_DISK_CAPACITY:{free}<{required}")
    return free, required


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _params_hash(params) -> str:
    import jax
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256()
    for leaf in leaves:
        arr = jax.numpy.asarray(leaf)
        digest.update(arr.astype(jax.numpy.float32).tobytes())
    return digest.hexdigest()


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  default=str)
    os.replace(tmp, path)


def _numeric_leaves_finite(value: Any) -> bool:
    """Dependency-light recursive finite check for evidence containers."""
    import math
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_numeric_leaves_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_leaves_finite(v) for v in value)
    # NumPy/JAX arrays expose dtype/shape.  Do not silently treat these as
    # finite: materialize to host and inspect every numeric leaf.
    try:
        import numpy as np
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return bool(np.isfinite(arr).all())
    except (TypeError, ValueError, RuntimeError):
        return False
    return True


def _jax_tree_finite(value: Any) -> bool:
    """Check every JAX pytree numeric leaf after device_get."""
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(value)
        return all(_numeric_leaves_finite(jax.device_get(leaf))
                   for leaf in leaves)
    except (ImportError, TypeError, ValueError, RuntimeError):
        return False


def _assert_finite_training_artifacts(*, train_state: Any,
                                      receipt: Mapping[str, Any]) -> None:
    """Fail before checkpoint/report creation when any numeric artifact drifts."""
    params = getattr(train_state, "params", None)
    opt_state = getattr(train_state, "opt_state", None)
    if params is None or not _jax_tree_finite(params):
        raise RuntimeError("FINITE_GATE_PARAMS_NAN_OR_INF")
    if opt_state is None or not _jax_tree_finite(opt_state):
        raise RuntimeError("FINITE_GATE_OPT_STATE_NAN_OR_INF")
    for key in ("training_metrics", "evaluation_metrics", "architecture_memory"):
        if key in receipt and not _jax_tree_finite(receipt.get(key)):
            raise RuntimeError(f"FINITE_GATE_{key.upper()}_NAN_OR_INF")


@contextmanager
def run_lease(run_dir: str, *, metadata: dict):
    """Cross-process exclusive lease; OS lock is authoritative."""
    lock_path = os.path.join(run_dir + ".run.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "a+b")
    unlock = lambda: None
    try:
        try:
            import msvcrt
            if os.fstat(fh.fileno()).st_size == 0:
                fh.write(b"\0"); fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except ImportError:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        old = None
        try:
            fh.seek(0); raw = fh.read(); old = json.loads(raw[1:].decode("utf-8") if raw.startswith(b"\0") else raw.decode("utf-8"))
        except Exception:
            pass
        record = {"schema":"simulator_frontier.e3.run_lease/v1", "host":socket.gethostname(),
                  "pid":os.getpid(), "start_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "argv":list(sys.argv), "takeover_history": ([old] if isinstance(old, dict) else []), **metadata}
        fh.seek(0); fh.truncate(); fh.write(b"\0" + json.dumps(record, sort_keys=True).encode()); fh.flush(); os.fsync(fh.fileno())
        yield record
    except (BlockingIOError, OSError) as exc:
        raise RuntimeError("RUN_LEASE_BLOCKED: another process owns output") from exc
    finally:
        try: unlock()
        except Exception: pass
        fh.close()


def write_run_metadata_once(run_dir: str, payload: dict) -> None:
    path = Path(run_dir) / "RUN_METADATA.json"
    side = Path(run_dir) / "RUN_METADATA.sha256"
    if path.exists() or side.exists():
        raise ValueError("RUN_METADATA already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(raw); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)
    side.write_text(hashlib.sha256(raw).hexdigest(), encoding="ascii")


def verify_run_metadata(run_dir: str, expected: dict) -> dict:
    path = Path(run_dir) / "RUN_METADATA.json"; side = Path(run_dir) / "RUN_METADATA.sha256"
    raw = path.read_bytes()
    digest = side.read_text(encoding="ascii").strip()
    if len(digest) != 64 or digest != hashlib.sha256(raw).hexdigest():
        raise ValueError("RUN_METADATA sidecar/hash mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict): raise ValueError("RUN_METADATA must be mapping")
    for key, value in expected.items():
        if payload.get(key) != value: raise ValueError(f"RUN_METADATA mismatch: {key}")
    return payload


def append_resume_event(run_dir: str, event: dict) -> None:
    path = Path(run_dir) / "RESUME_EVENTS.jsonl"
    raw = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with open(path, "ab") as fh:
        fh.write(raw); fh.flush(); os.fsync(fh.fileno())


def _install_preseed_journal(*, preseed_path: str, run_dir: str, auth: Any,
                             source_commit: str, candidate_id: str,
                             client_hash: str) -> dict[str, Any]:
    """Validate and re-key exactly one prior diagnostician success."""
    if not auth.preseed_journal_sha256:
        raise ValueError("preseed journal is not signed")
    if _sha256_file(preseed_path) != auth.preseed_journal_sha256:
        raise ValueError("preseed journal SHA mismatch")
    import importlib.util
    journal_path = os.path.join(SRC_DIR, "dicode", "simulator_frontier",
                                "e3_durable_llm_journal.py")
    spec = importlib.util.spec_from_file_location("_e3_preseed_journal", journal_path)
    if spec is None or spec.loader is None:
        raise ValueError("preseed journal module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DurablePaidCallJournal = module.DurablePaidCallJournal
    source_journal = DurablePaidCallJournal(preseed_path)
    payload = source_journal._load()
    entries = payload.get("entries", {})
    if len(entries) not in (1, 2):
        raise ValueError("preseed journal must contain one or two successes")
    ordered = sorted(entries.items(), key=lambda item: str(item[1].get("role", "")))
    if {str(entry.get("role", "")) for _, entry in ordered} != {
            "frontier_evidence_diagnostician"} and len(ordered) == 1:
        raise ValueError("single preseed entry must be diagnostician")
    roles = [str(entry.get("role", "")) for _, entry in ordered]
    if len(ordered) == 2 and set(roles) != {
            "frontier_evidence_diagnostician", "curriculum_search_planner"}:
        raise ValueError("dual preseed roles must be diagnostician and planner")
    if any(int(entry.get("session", 0)) != 1 for _, entry in ordered):
        raise ValueError("preseed entries must be session 1")
    provenance = payload.get("preseed_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("preseed migration provenance missing")
    for field in ("source_run", "source_key", "source_journal_sha256",
                  "source_commit", "source_client_implementation_hash"):
        if not provenance.get(field):
            raise ValueError(f"preseed provenance missing {field}")
    target_path = os.path.join(run_dir, "LLM_PAID_CALL_JOURNAL.json")
    target = DurablePaidCallJournal(target_path)
    provenance = dict(provenance)
    provenance.update({"installed_source_keys": [key for key, _ in ordered],
                       "installed_source_journal_sha256": auth.preseed_journal_sha256})
    identities = []
    source_keys = []
    source_identity_anchor = None
    for source_key, source_entry in ordered:
        old_identity = dict(source_entry.get("key_identity", {}))
        if not old_identity:
            raise ValueError("preseed key identity missing")
        for field in ("source_commit", "candidate", "session", "evidence_hash",
                      "role", "provider", "requested_model",
                      "client_implementation_hash"):
            if source_entry.get(field) != old_identity.get(field):
                raise ValueError(f"preseed identity mismatch: {field}")
        if source_entry.get("returned_model") != source_entry.get("requested_model"):
            raise ValueError("preseed returned/requested model mismatch")
        if source_entry.get("candidate") != candidate_id:
            raise ValueError("preseed candidate mismatch")
        if source_identity_anchor is None:
            source_identity_anchor = old_identity
        else:
            for field in ("source_commit", "candidate", "provider",
                          "requested_model", "client_implementation_hash"):
                if old_identity.get(field) != source_identity_anchor.get(field):
                    raise ValueError(f"preseed cross-entry identity mismatch: {field}")
        signed_provider = str(getattr(auth, "provider", "dashscope") or "dashscope")
        signed_model = str(getattr(auth, "requested_model", "qwen-plus") or "qwen-plus")
        if old_identity.get("provider") != signed_provider or old_identity.get("requested_model") != signed_model:
            raise ValueError("preseed provider/model mismatch")
        identity = {
            "source_commit": source_commit, "candidate": candidate_id,
            "session": 1, "evidence_hash": old_identity.get("evidence_hash", ""),
            "role": str(source_entry.get("role", "")),
            "provider": str(old_identity.get("provider", "dashscope")),
            "requested_model": str(old_identity.get("requested_model", "")),
            "client_implementation_hash": client_hash,
        }
        if not identity["evidence_hash"] or not identity["requested_model"]:
            raise ValueError("preseed identity incomplete")
        identities.append(identity); source_keys.append(source_key)
    installed = target.install_preseed_entries(
        entries=[entry for _, entry in ordered], identities=identities,
        provenance=provenance)
    diag = next((entry for entry in installed
                 if entry["role"] == "frontier_evidence_diagnostician"), None)
    if diag is None:
        raise ValueError("preseed diagnostician missing")
    os.environ["E3_PRESEEDED_DIAGNOSTIC_KEY"] = diag["key"]
    result = {"sha256": auth.preseed_journal_sha256, "source_keys": source_keys,
              "installed_keys": [entry["key"] for entry in installed],
              "provenance": provenance}
    if len(installed) == 1:
        result.update({"source_key": source_keys[0], "installed_key": installed[0]["key"]})
    return result


def _metadata_payload(*, source_commit, runner_sha256, auth, candidate_id, sessions,
                      budget, mounted_file_sha, mounted_params_sha, gpu, pid, started_utc,
                      client_combo_hash, checkpoint_sha256, task_asset_manifest_sha256,
                      disk_free_bytes=0, disk_required_bytes=0,
                      preseed_info=None):
    return {"schema":"simulator_frontier.e3_formal_longrun/v2", "source_commit":source_commit,
            "runner_sha256":runner_sha256, "authorization_id":auth.authorization_id,
            "authorization_manifest_hash":auth.manifest_hash, "candidate_id":candidate_id,
            "sessions":sessions, "budget_semantics":budget.budget_semantics,
            "resolved_budget_env_steps":budget.resolved_env_steps,
            "client_implementation_hash":client_combo_hash,
            "checkpoint_sha256":checkpoint_sha256,
            "task_asset_manifest_sha256":task_asset_manifest_sha256,
            "mounted_file_sha256":mounted_file_sha, "mounted_params_sha256":mounted_params_sha,
            "gpu":gpu, "pid":pid, "started_utc":started_utc,
            "updates_per_session":UPDATES_PER_SESSION, "num_envs":NUM_ENVS, "num_steps":NUM_STEPS,
            "max_logical_calls":302, "max_output_tokens_per_call":4096,
            "max_total_tokens_per_call":20000, "retry_cap":0,
            "disk_free_bytes": disk_free_bytes,
            "disk_required_bytes": disk_required_bytes,
            "preseed_journal_sha256": getattr(auth, "preseed_journal_sha256", None) or "",
            "preseed_provenance": (preseed_info or {}).get("provenance", {})}


def _metadata_expected_static(meta: dict) -> dict:
    return {k:v for k,v in meta.items() if k not in {"mounted_file_sha256","mounted_params_sha256","gpu","pid","started_utc"}}


def _load_and_validate_completed_sessions(run_dir: str, *, candidate: str,
                                           source_commit: str, sessions: int,
                                           authorization_manifest_hash: str = "") -> tuple[list[dict], int, str | None]:
    evidence_dir = Path(run_dir) / "evidence"
    files = sorted(evidence_dir.glob("session_*.json"))
    if len(files) > int(sessions):
        raise ValueError("resume evidence exceeds requested session budget")
    reports = []
    expected_stems = set()
    for index, path in enumerate(files, 1):
        if path.name != f"session_{index:03d}.json":
            raise ValueError("resume evidence numbering gap or extra file")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "simulator_frontier.e3_formal_longrun_session/v1":
            raise ValueError("resume evidence schema mismatch")
        if (report.get("candidate_id") != candidate or report.get("source_commit") != source_commit
                or report.get("authorization_manifest_hash") != authorization_manifest_hash):
            raise ValueError("resume evidence identity mismatch")
        if report.get("protocol") != "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128":
            raise ValueError("resume evidence protocol mismatch")
        if report.get("session_idx") != index or report.get("current_session_idx") != index or report.get("num_updates_in_session") != 100:
            raise ValueError("resume evidence session/counter mismatch")
        expected_previous = None if index == 1 else reports[-1]["checkpoint_path"]
        if report.get("previous_checkpoint") != expected_previous:
            raise ValueError("resume previous checkpoint chain mismatch")
        if report.get("env_steps_per_update") != ENV_STEPS_PER_UPDATE:
            raise ValueError("resume evidence layout mismatch")
        if (report.get("start_global_update") != (index-1)*100
                or report.get("start_global_env_steps") != (index-1)*UPDATES_PER_SESSION*ENV_STEPS_PER_UPDATE
                or report.get("global_update_step") != index * UPDATES_PER_SESSION
                or report.get("global_env_steps") != index * UPDATES_PER_SESSION * ENV_STEPS_PER_UPDATE):
            raise ValueError("resume evidence global counter mismatch")
        if report.get("fresh_process_restore_equivalent") is not True:
            raise ValueError("resume evidence restore not verified")
        checkpoint = Path(str(report.get("checkpoint_path", "")))
        expected = Path(run_dir).resolve() / "runstate" / f"e3_canonical_runstate_s{index:03d}"
        if checkpoint.resolve() != expected:
            raise ValueError("resume checkpoint stem mismatch")
        state_file = Path(str(checkpoint) + ".state.pkl")
        meta_file = Path(str(checkpoint) + ".meta.json")
        if (not checkpoint.is_absolute() or Path(run_dir).resolve() not in checkpoint.resolve().parents
                or not state_file.exists() or not meta_file.exists()):
            raise ValueError("resume checkpoint path invalid")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        expected_stems.add(checkpoint.name)
        if meta.get("schema") != "simulator_frontier.canonical_runstate_checkpoint/v1":
            raise ValueError("resume checkpoint metadata schema mismatch")
        if meta.get("idempotency_token") != f"e3-longrun-s{index}":
            raise ValueError("resume checkpoint idempotency token mismatch")
        if (meta.get("source_commit") != source_commit
                or meta.get("current_session_idx") != index):
            raise ValueError("resume checkpoint identity mismatch")
        if (meta.get("global_update_step") != report.get("global_update_step")
                or meta.get("global_env_steps") != report.get("global_env_steps")):
            raise ValueError("resume checkpoint counter mismatch")
        if not meta.get("state_file_sha256") or meta.get("state_file_sha256") != _sha256_file(str(state_file)) or report.get("checkpoint_state_sha256") != meta.get("state_file_sha256"):
            raise ValueError("resume checkpoint state hash mismatch")
        try:
            with state_file.open("rb") as state_handle:
                persisted_state = pickle.load(state_handle)
        except Exception as exc:
            raise ValueError("resume checkpoint state unreadable") from exc
        if not _numeric_leaves_finite(persisted_state):
            raise ValueError("resume checkpoint contains NaN/Inf")
        metadata_without_hash = {k: v for k, v in meta.items() if k != "checkpoint_hash"}
        expected_checkpoint_hash = hashlib.sha256(
            json.dumps(metadata_without_hash, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()
        if meta.get("checkpoint_hash") != expected_checkpoint_hash:
            raise ValueError("resume checkpoint metadata hash mismatch")
        if report.get("checkpoint_hash") != meta.get("checkpoint_hash"):
            raise ValueError("resume report checkpoint hash mismatch")
        if not report.get("checkpoint_content_hash") or not report.get("fresh_process_restore_equivalent"):
            raise ValueError("resume checkpoint content/restore evidence missing")
        reports.append(report)
    runstate_dir = Path(run_dir) / "runstate"
    state_stems = {p.name[:-len(".state.pkl")] for p in runstate_dir.glob("*.state.pkl")}
    meta_stems = {p.name[:-len(".meta.json")] for p in runstate_dir.glob("*.meta.json")}
    if state_stems != meta_stems or state_stems != expected_stems:
        raise ValueError("resume runstate orphan, missing half, or unreported checkpoint")
    return reports, len(reports) + 1, reports[-1].get("checkpoint_path") if reports else None


def _write_initial_boundary(run_dir: str, session_idx: int, *,
                            train_state, training_rng, source_commit: str,
                            start_global_update: int,
                            start_global_env_steps: int,
                            previous_checkpoint: str | None) -> dict:
    """Persist checkpoint-authoritative session input before training.

    Under boundary semantics B, environment and recurrent memory are both
    fresh and therefore intentionally absent from the carried state.  Params,
    optimizer, train step and training RNG are the complete carried state and
    are serialized atomically for an independent verifier.
    """
    boundary_dir = os.path.join(run_dir, "boundaries")
    os.makedirs(boundary_dir, exist_ok=True)
    stem = os.path.join(boundary_dir, f"session_{session_idx:03d}_initial")
    state_path = stem + ".state.pkl"
    meta_path = stem + ".meta.json"
    payload = {
        "schema": "simulator_frontier.e3_session_initial_boundary/v1",
        "session_idx": int(session_idx),
        "source_commit": str(source_commit),
        "previous_checkpoint": str(previous_checkpoint or ""),
        "params": train_state.params,
        "opt_state": train_state.opt_state,
        "train_step": int(train_state.step),
        "training_rng": training_rng,
        "global_update_step": int(start_global_update),
        "global_env_steps": int(start_global_env_steps),
        "session_boundary_semantics": "B_NEW_SESSION_ENV_AND_MEMORY_RESET",
        "environment_restore_input": None,
        "architecture_memory_restore_input": None,
    }
    state_tmp = state_path + ".tmp"
    with open(state_tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    os.replace(state_tmp, state_path)
    meta = {
        "schema": "simulator_frontier.e3_session_initial_boundary_meta/v1",
        "session_idx": int(session_idx),
        "state_file": os.path.basename(state_path),
        "state_file_sha256": _sha256_file(state_path),
        "source_commit": str(source_commit),
        "boundary_semantics": "B_NEW_SESSION_ENV_AND_MEMORY_RESET",
    }
    _write_json(meta_path, meta)
    return {"state_path": state_path, "meta_path": meta_path,
            "state_file_sha256": meta["state_file_sha256"]}


def _mount_student(candidate_id: str, checkpoint_params=None) -> dict:
    """Mount the selected student via the production adapter."""
    import run_e3_real_smoke as prod
    mount = prod.mount_student(candidate_id)
    if checkpoint_params is not None:
        # Session k>0: mount the adapter identity (identity gates still run on
        # the canonical checkpoint) but inject the UPDATED params — the
        # resumed student — so the frontier/actual-N use the trained student.
        mount["params"] = checkpoint_params
        mount["params_sha256"] = _params_hash(checkpoint_params)
    return mount


def _resolved_config_hash() -> str:
    """Canonical hash of the frozen E3 formal resolved config (16x128)."""
    payload = {
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "max_updates_per_session": UPDATES_PER_SESSION,
        "curriculum_slot_count": CURRICULUM_SLOT_COUNT_CONSERVATIVE,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, default=str))


def run_one_session(*, candidate_id: str, session_idx: int, run_dir: str,
                    prev_runstate: str | None, source_commit: str,
                    trusted_signer: str, formal_asset_registry_hash: str,
                    authorization_manifest_hash: str = "") -> dict:
    """Run ONE production E3 session (100 outer updates) — delegates to the
    production chain.  frontier/actual-N/LLM/curriculum run ONCE per session.

    P0-6: trusted_signer comes from the controller-signed authorization
    manifest (never hardcoded).  P0-7: formal_asset_registry_hash is the real
    registry SHA from the manifest (never zeros)."""
    import jax
    from dicode.dreaming.gen_manager import GenManager
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        DiCodeOneUpdateContext,
        mint_canonical_dicode_session_runtime,
        execute_session,
        callable_source_sha256,
    )
    from dicode.simulator_frontier.runstate_codec import (
        RunStateCheckpointManager,
        build_full_run_state,
        fresh_process_restore,
        runstate_content_hash,
    )
    import run_e3_real_smoke as prod

    started = time.time()

    # ---- 0. Resolve starting state -------------------------------------------
    start_global_update = 0
    start_global_env_steps = 0
    current_session_idx = 1
    training_rng = jax.random.PRNGKey(SEED)
    student_params = None
    architecture_memory_serialized = None

    if prev_runstate is not None:
        manager = RunStateCheckpointManager()
        restored = manager.restore(prev_runstate)
        prev_state = restored["run_state"]
        # Resume is never allowed to carry a numerically poisoned state into
        # a new session; reject before mount/training or any new evidence.
        if (prev_state.get("params") is None
                or prev_state.get("opt_state") is None
                or not _jax_tree_finite(prev_state.get("params"))
                or not _jax_tree_finite(prev_state.get("opt_state"))
                or not _numeric_leaves_finite(prev_state.get("architecture_memory"))):
            raise RuntimeError("FINITE_GATE_RESUMED_RUNSTATE_NAN_OR_INF")
        start_global_update = int(prev_state["global_update_step"])
        start_global_env_steps = int(prev_state["global_env_steps"])
        current_session_idx = int(prev_state["current_session_idx"]) + 1
        training_rng = prev_state["training_rng"]
        student_params = prev_state["params"]
        # Session boundary semantics (sole-controller audit 2026-08-10): B —
        # NEW SESSION.  The environment and the recurrent policy memory are
        # RESET together at the start of each session; only params / optimizer
        # / training RNG / global counters continue.  The previous session's
        # final architecture memory is recorded in its RunState for evidence
        # but is NOT injected into the new session (never a half-restored
        # mixed state: old memory with a fresh env).
        architecture_memory_serialized = prev_state.get("architecture_memory")
        _log(f"session {session_idx}: resume from {prev_runstate} "
             f"global_update={start_global_update} "
             f"global_env={start_global_env_steps} session={current_session_idx} "
             f"boundary=B_NEW_SESSION_ENV_AND_MEMORY_RESET "
             f"prev_arch_memory_recorded={architecture_memory_serialized is not None}")

    # ---- 1. mount student (real) ---------------------------------------------
    mount = _mount_student(candidate_id, checkpoint_params=student_params)
    _log(f"session {session_idx}: student mounted "
         f"params_sha={mount['params_sha256'][:16]}... "
         f"arch={mount['architecture_family']}")

    # ---- 2. real frontier capsule + same-state actual-N (once per session) ----
    # P0-1/2/3: capture ONE real frontier capsule from a real Student rollout,
    # then run N branches from that SAME capsule (only branch RNG differs).
    # success is decided by a TASK-BASED predicate (all relevant achievements
    # done); death/timeout is never auto-success; state_id is the real encoded
    # payload hash.
    import e3_capsule_actualn as capsule_mod
    # Session-boundary semantics B: the capture rollout starts from the
    # Student's fresh initial memory (never the previous session's recurrent
    # memory) — the env and the memory reset together.  The task-based
    # success/progress predicate is built inside capture and bound to the
    # capsule; actual-N reuses it.
    capsule = capsule_mod.capture_frontier_capsule(
        student=mount["adapter"], student_params=mount["params"],
        run_id=f"e3-longrun-s{session_idx}",
        reset_seed=SEED, capture_at_step=SEARCH_HORIZON,
        max_timesteps=SEARCH_HORIZON + 8, success_threshold=0.50,
        memory_mode="SAVED_POLICY_MEMORY",
        initial_memory=None,
    )
    capsule["student"] = mount["adapter"]
    capsule["student_params"] = mount["params"]
    actual_n = capsule_mod.run_same_state_actual_n(
        capsule=capsule, n=ACTUAL_N, horizon=SEARCH_HORIZON,
        seed_base=SEED, memory_mode="SAVED_POLICY_MEMORY",
    )
    est = actual_n["estimate"]
    state_id = capsule["state_id"]
    _log(f"session {session_idx}: capsule={state_id[:12]}... actual-N="
         f"{est.total_actual_branches} successes={est.successes} "
         f"sr={est.success_rate:.3f}")

    evidence = {
        "feasibility": {
            "state_id": state_id,
            "total_actual_branches": int(est.total_actual_branches),
            "actual_branches_by_source": dict(est.actual_branches_by_source),
            "successes": int(est.successes),
            "success_rate": float(est.success_rate),
            "confidence_interval": [float(est.confidence_interval[0]),
                                    float(est.confidence_interval[1])],
            "mean_progress": float(est.mean_progress),
            "max_progress": float(est.max_progress),
            "transition_cost": int(est.transition_cost),
            "uncertainty": float(est.uncertainty),
            "estimate_version": est.estimate_version,
        },
        "archive_summary": {
            "entry_count": len(capsule["archive"]),
            "bucket_diversity": len(capsule["archive"].list()),
            "evidence_ids": [f"e3-capture-{state_id[:12]}"],
            "bucket_id": state_id[:16],
            "capture_source_timestep": capsule["steps_executed"],
            "capture_source_checkpoint": capsule["params_sha"][:16],
            "capture_student_id": capsule["capture_student_id"],
        },
        "data_source": "TRAINING_FRONTIER_CAPTURE",
    }

    # ---- 3. two REAL LLM roles (once per session) ----------------------------
    two_llm = prod.build_two_llm_runtime(
        max_output_tokens_per_call=4096,
        max_total_tokens_per_call=20000,
        retry_cap=0,
        provider="dashscope")
    os.environ["E3_FRONTIER_STATE_ID"] = state_id
    os.environ["E3_FRONTIER_BUCKET_ID"] = state_id[:16]
    os.environ["E3_ACTUAL_N"] = str(int(est.total_actual_branches))
    os.environ["E3_HORIZON"] = str(SEARCH_HORIZON)
    llm_result = prod.run_two_real_llm_roles(two_llm, evidence)
    plan = llm_result["planner"]
    _log(f"session {session_idx}: two-LLM {llm_result['llm_calls']} calls "
         f"plan_id={plan.plan_id}")

    # ---- 4. hydra config (100 updates/session) + GenManager ------------------
    work_dir = os.path.join(run_dir, "canonical_update", f"s{session_idx:03d}")
    os.makedirs(work_dir, exist_ok=True)
    config = prod.build_hydra_config(
        work_dir, max_updates_per_session=UPDATES_PER_SESSION)
    gen_manager = GenManager(config)

    # ---- 5. train_state (session 0: from checkpoint; k>0: resumed) ----------
    if prev_runstate is None:
        selected = prod.build_train_state_from_selected_student(
            config, mount, candidate_id)
        train_state = selected["train_state"]
        backend = selected["backend"]
        checkpoint_params = selected["checkpoint_params"]
        initial_params_sha = selected["checkpoint_params_sha256"]
    else:
        # Rebuild the backend for the candidate, then reattach the resumed
        # params + opt_state so the optimizer continues (never a fresh reset).
        selected = prod.build_train_state_from_selected_student(
            config, mount, candidate_id)
        backend = selected["backend"]
        checkpoint_params = student_params
        # Recreate TrainState from restored params + opt_state via the backend.
        tx = selected["train_state"].tx
        apply_fn = selected["train_state"].apply_fn
        from flax.training.train_state import TrainState
        opt_state = prev_state["opt_state"]
        train_step = int(prev_state["train_step"])
        train_state = TrainState(
            apply_fn=apply_fn, params=student_params, tx=tx,
            opt_state=opt_state, step=train_step)
        initial_params_sha = _params_hash(student_params)
        _log(f"session {session_idx}: resumed TrainState params="
             f"{initial_params_sha[:16]}... opt_step={train_step}")

    boundary_report = _write_initial_boundary(
        run_dir, session_idx, train_state=train_state,
        training_rng=training_rng, source_commit=source_commit,
        start_global_update=start_global_update,
        start_global_env_steps=start_global_env_steps,
        previous_checkpoint=prev_runstate)

    # ---- 6. canonical 15+1 plan + real TaskArchive ---------------------------
    register_result = prod.compile_and_register(
        {"planner": plan, "evidence_hash": llm_result["evidence_hash"]},
        run_id=f"e3-longrun-s{session_idx}",
        state_id=str(est.state_id), memory_mode="SAVED_POLICY_MEMORY",
        gen_manager=gen_manager, session_idx=current_session_idx)
    canonical_plan = register_result["canonical_plan"]
    _log(f"session {session_idx}: canonical plan "
         f"{len(canonical_plan.curriculum_slots)} slots; "
         f"{len(register_result['registered_ids'])} registered")

    # ---- 7. canonical SESSION (100 updates, threaded counters) ---------------
    # One invocation == one native run_session_training == 100 outer updates.
    # NEVER a for-loop of execute_one_update calls.
    session_runtime = mint_canonical_dicode_session_runtime(
        runtime_id=f"e3-formal-session-{candidate_id[:12]}-{session_idx}",
        selected_candidate_id=candidate_id,
        run_session_training_entrypoint="dicode.training:run_session_training",
        run_session_implementation_hash=callable_source_sha256(
            "run_session_training",
            __import__("dicode.training", fromlist=["run_session_training"])
            .run_session_training),
        run_training_session_entrypoint="dicode.ppo_tr:run_training_session",
        run_training_implementation_hash=callable_source_sha256(
            "run_training_session",
            __import__("dicode.ppo_tr", fromlist=["run_training_session"])
            .run_training_session),
        # P0-6: trusted_signer from the controller-signed authorization
        # manifest — never a hardcoded value.
        trusted_signer=trusted_signer,
    )
    context = DiCodeOneUpdateContext(
        config=config,
        rng=training_rng,
        rl_train_state=train_state,
        gen_manager=gen_manager,
        global_update_step=start_global_update,
        global_env_steps=start_global_env_steps,
        current_session_idx=current_session_idx,
        original_return_prev_session=0.0,
        selected_candidate_id=candidate_id,
        runtime_bundle_hash=session_runtime.runtime_hash,
        # P0-7: the real formal asset registry hash from the authorization
        # manifest — never a zero-filled placeholder.
        formal_asset_registry_hash=formal_asset_registry_hash,
    )
    receipt = execute_session(
        session_runtime, context=context,
        plan=register_result["canonical_plan"],
        adapter=register_result["env_adapter"],
        backend=backend,
        checkpoint_params=checkpoint_params,
        # Session-boundary semantics B: training starts from the backend's
        # fresh initial memory (the env is reset this session too) — no
        # cross-session recurrent-memory injection.
        initial_memory_dict=None)
    if int(receipt["num_updates_in_session"]) != UPDATES_PER_SESSION:
        raise RuntimeError(
            f"session {session_idx}: expected {UPDATES_PER_SESSION} updates, "
            f"got {int(receipt['num_updates_in_session'])} (fail closed)")
    _log(f"session {session_idx}: canonical session "
         f"{receipt['num_updates_in_session']} updates -> global_update="
         f"{int(receipt['global_update_step'])} env="
         f"{int(receipt['global_env_steps'])}")

    # ---- 8. full RunState + fresh-process restore -----------------------------
    new_state = receipt["rl_train_state"]
    _assert_finite_training_artifacts(train_state=new_state, receipt=receipt)
    env_rng = jax.random.split(receipt["rng"])[1]
    archive_parts = []
    for tid in sorted(register_result["registered_ids"]):
        archive_parts.append(tid)
    archive_identity = _sha256_text("|".join(archive_parts))
    extra = {}
    if backend is not None:
        arch_memory = receipt.get("architecture_memory")
        if arch_memory is None:
            raise RuntimeError(
                f"session {session_idx}: backend bound but no architecture "
                "memory in receipt (fail closed)")
        extra["architecture_memory"] = backend.serialize_memory_state(arch_memory)
    run_state = build_full_run_state(
        rl_train_state=new_state,
        training_rng=receipt["rng"],
        env_rng=env_rng,
        global_update_step=int(receipt["global_update_step"]),
        global_env_steps=int(receipt["global_env_steps"]),
        # P0-5: store the CURRENT completed session idx (NOT +1).  Restore
        # adds +1 to obtain the NEXT session idx -> strict 1 -> 2 -> 3.
        current_session_idx=current_session_idx,
        task_archive_identity=archive_identity,
        plan_hash=canonical_plan.plan_hash,
        runtime_bundle_hash=session_runtime.runtime_hash,
        config_hash=_resolved_config_hash(),
        source_commit=source_commit,
        candidate_id=candidate_id,
        architecture_family=mount["architecture_family"],
        extra=extra,
    )
    ckpt_dir = os.path.join(run_dir, "runstate")
    os.makedirs(ckpt_dir, exist_ok=True)
    manager = RunStateCheckpointManager()
    ckpt_path = os.path.join(ckpt_dir,
                             f"e3_canonical_runstate_s{session_idx:03d}")
    save_report = manager.save(run_state, ckpt_path,
                               idempotency_token=f"e3-longrun-s{session_idx}")
    local_content_hash = runstate_content_hash(run_state)
    restored = fresh_process_restore(ckpt_path, extra_pythonpath=SRC_DIR)
    equivalent = bool(restored.get("content_hash") == local_content_hash)
    if not equivalent:
        raise RuntimeError(
            f"session {session_idx}: FRESH_PROCESS_RESTORE mismatch "
            f"(parent={local_content_hash[:16]} child="
            f"{restored.get('content_hash', '')[:16]}) (fail closed)")
    _log(f"session {session_idx}: RunState saved + fresh-restore OK "
         f"sha={save_report['state_file_sha256'][:16]}...")

    # ---- 9. evidence ----------------------------------------------------------
    durable_refs = list(llm_result.get("audit_events", []))
    paid_new_calls = sum(1 for e in durable_refs if e.get("paid_new"))
    reused_successes = sum(1 for e in durable_refs if e.get("reused"))
    journal_path = os.environ.get("E3_LLM_JOURNAL_PATH", "")
    session_report = {
        "schema": "simulator_frontier.e3_formal_longrun_session/v1",
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "session_idx": session_idx,
        "run_id": f"e3-longrun-s{session_idx}",
        "candidate_id": candidate_id,
        "source_commit": source_commit,
        "authorization_manifest_hash": authorization_manifest_hash,
        "previous_checkpoint": prev_runstate,
        "architecture_family": mount["architecture_family"],
        "params_sha256": mount["params_sha256"],
        "initial_trainstate_params_sha256": initial_params_sha,
        "checkpoint_params_sha256": checkpoint_params and _params_hash(checkpoint_params),
        "initial_equals_checkpoint": bool(initial_params_sha == (checkpoint_params and _params_hash(checkpoint_params))),
        "start_global_update": start_global_update,
        "start_global_env_steps": start_global_env_steps,
        "current_session_idx": current_session_idx,
        "global_update_step": int(receipt["global_update_step"]),
        "global_env_steps": int(receipt["global_env_steps"]),
        "num_updates_in_session": int(receipt["num_updates_in_session"]),
        "expected_updates_per_session": UPDATES_PER_SESSION,
        "task_class_count": CURRICULUM_SLOT_COUNT_CONSERVATIVE + 1,
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "optimizer_semantics": ("NEW_OPTIMIZER_PHASE_FROM_SESSION0_THEN_CONTINUOUS"
                                if prev_runstate is None
                                else "RESUME_PREVIOUS_SESSION_OPT_STATE"),
        # Session-boundary semantics B: env + recurrent memory reset together;
        # only params/optimizer/RNG/global counters continue.
        "session_boundary_semantics": "B_NEW_SESSION_ENV_AND_MEMORY_RESET",
        "initial_boundary_state_sha256": boundary_report["state_file_sha256"],
        "initial_boundary_state_path": boundary_report["state_path"],
        "success_predicate": capsule.get("predicate_meta"),
        "capture_success_basis": capsule.get("success_basis"),
        "predicate_applicability": (capsule.get("facts", {}).get("predicate_applicability")),
        "actual_n_predicate_meta": actual_n.get("predicate_meta"),
        "actual_n_branch_seeds": actual_n.get("branch_seeds"),
        "curriculum_slots": len(canonical_plan.curriculum_slots),
        "registered_ids": list(register_result["registered_ids"]),
        "plan_hash": canonical_plan.plan_hash,
        "llm_calls": int(llm_result["llm_calls"]),
        "durable_role_journal_refs": durable_refs,
        "paid_new_calls": paid_new_calls,
        "reused_successes": reused_successes,
        "evidence_hash": llm_result["evidence_hash"],
        "actual_n_branches": int(est.total_actual_branches),
        "actual_n_successes": int(est.successes),
        "fresh_process_restore_equivalent": equivalent,
        "checkpoint_path": ckpt_path,
        "checkpoint_state_sha256": save_report["state_file_sha256"],
        "checkpoint_hash": save_report["checkpoint_hash"],
        "checkpoint_content_hash": local_content_hash,
        "elapsed_s": round(time.time() - started, 2),
    }
    _write_json(os.path.join(run_dir, "evidence",
                             f"session_{session_idx:03d}.json"), session_report)
    return session_report


def _wandb_offline_init(run_id: str) -> None:
    """The canonical DiCode PPO loop (ppo_tr) logs through a jax.debug
    callback that calls wandb.log() unconditionally (even with
    use_wandb=false).  Initialize wandb OFFLINE so those calls succeed
    (never a network run, never a fake) — mirrors run_e3_real_smoke."""
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(mode="offline", project="e3_formal_longrun",
                   entity="e3", name=run_id, reinit=True)
    except Exception as exc:
        _log(f"wandb offline init warning: {exc!r}")


def main(argv=None, *, _lease_held: bool = False) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    candidate_id = None
    sessions = 2  # default = integration (2 sessions x 100); full-budget passes 151
    run_dir = None
    auth_manifest = None
    preseed_journal = None
    resume = False
    for arg in argv:
        if arg.startswith("--student="):
            candidate_id = arg.split("=", 1)[1]
        elif arg.startswith("--sessions="):
            sessions = int(arg.split("=", 1)[1])
        elif arg.startswith("--out="):
            run_dir = arg.split("=", 1)[1]
        elif arg.startswith("--auth-manifest="):
            auth_manifest = arg.split("=", 1)[1]
        elif arg.startswith("--preseed-journal="):
            preseed_journal = arg.split("=", 1)[1]
        elif arg == "--resume":
            resume = True
        else:
            print(f"[e3-longrun-ctrl] unknown argument {arg!r}", flush=True)
            return FAIL
    if candidate_id not in (PERSISTENT, RESET128, SLOWGRU):
        print(f"[e3-longrun-ctrl] invalid --student {candidate_id!r}", flush=True)
        return FAIL
    if not run_dir:
        print("[e3-longrun-ctrl] --out=<RUN_DIR> required", flush=True)
        return FAIL
    import e3_authorization as auth_mod
    # Mechanical budget gate occurs before any authorization, output claim,
    # transport construction, model mount, or GPU/JAX import.
    budget_scope = "formal" if int(sessions) > VERIFICATION_SESSIONS_MAX else "verification"
    try:
        budget = auth_mod.resolve_e3_budget(candidate=candidate_id,
                                             sessions=int(sessions),
                                             scope=budget_scope)
    except ValueError as exc:
        print(f"[e3-longrun-ctrl] BUDGET_BLOCKED: {exc}", flush=True)
        return FAIL
    # P0-6/7/8: require a controller-signed authorization manifest.  Without
    # it, the formal launch is BLOCKED before ANY output dir / LLM / GPU.
    # PRE-GPU AUTHORIZATION: e3_authorization is dependency-free and loads its
    # Ed25519 verifier directly by file.  Do not import run_e3_real_smoke,
    # mount a Student, import JAX or query a GPU until this block passes.
    source_commit = _git_head()
    if budget_scope == "formal" and not _git_worktree_clean():
        print("[e3-longrun-ctrl] BLOCKED_RUNTIME_WORKTREE_DIRTY", flush=True)
        return BLOCKED
    # Full-budget gate: until the sole controller signs a full-budget
    # authorization AND closes the audit, E3_FORMAL_LONGRUN_AUTHORIZED is
    # False and launches above the verification cap are blocked.
    if auth_manifest is None and int(sessions) > VERIFICATION_SESSIONS_MAX:
        print(f"[e3-longrun-ctrl] BLOCKED_CONTROLLER_SIGNATURE_REQUIRED: "
              f"E3_FORMAL_LONGRUN_AUTHORIZED=False — {sessions} sessions "
              f"(cap {VERIFICATION_SESSIONS_MAX}) is a full-budget launch and "
              f"is NOT authorized until the audit is closed and the sole "
              f"controller signs the full-budget manifest (fail closed)",
              flush=True)
        return BLOCKED
    try:
        if not auth_manifest:
            raise ValueError(
                "--auth-manifest=<path> is required: the sole controller must "
                "sign an E3 authorization manifest (audit fail closed)")
        # Bind only standard-library-readable immutable artifacts first.
        runner_sha256 = _sha256_file(
            os.path.join(SCRIPT_DIR, "run_e3_formal_longrun.py"))
        auth = auth_mod.load_authorization(
            auth_manifest,
            public_key_path=AUTH_PUBLIC_KEY,
            registry_path=AUTH_REGISTRY,
        )
        static_assets = auth_mod.resolve_candidate_static_assets(
            AUTH_REGISTRY, candidate_id)
        auth_mod.verify_runtime_authorization(
            auth, source_commit, candidate_id,
            runner_sha256, static_assets["checkpoint_sha256"],
            auth.student_profile_sha256, AUTH_REGISTRY,
            task_asset_manifest_sha256=
                static_assets["task_asset_manifest_sha256"],
            executable_anchor_manifest_sha256=
                static_assets.get("executable_anchor_manifest_sha256", ""))
        anchor_path = static_assets.get("executable_anchor_manifest_path", "")
        anchor_sha = static_assets.get("executable_anchor_manifest_sha256", "")
        client_hash = ""
        if budget_scope == "formal":
            if not anchor_path or not anchor_sha or _sha256_file(anchor_path) != anchor_sha:
                raise ValueError("formal executable anchor manifest asset missing or drifted")
            override = os.environ.get("E3_EXECUTABLE_ANCHOR_MANIFEST", "")
            if override and os.path.realpath(override) != os.path.realpath(anchor_path):
                raise ValueError("executable anchor manifest override is not registry path")
            os.environ["E3_EXECUTABLE_ANCHOR_MANIFEST"] = anchor_path
        if budget_scope == "formal":
            if auth.scope != "formal":
                raise ValueError("verification authorization cannot authorize formal scope")
            requested_model = os.environ.get("QWEN_MODEL", "qwen-plus")
            # The client factory hash is part of the signed contract; resolve
            # it dependency-free from the production module only after auth.
            from dicode.simulator_frontier.canonical_dicode_runtime import callable_source_sha256
            clients_mod = __import__("dicode.simulator_frontier._e3_real_llm_clients", fromlist=["client_factory"])
            from dicode.simulator_frontier.e3_durable_llm_journal import implementation_hash
            client_hash = implementation_hash(
                clients_mod.__file__,
                os.path.join(SRC_DIR, "dicode", "simulator_frontier", "e3_durable_llm_journal.py"))
            auth_mod.verify_formal_authorization_budget(
                auth, candidate=candidate_id, sessions=int(sessions),
                provider="dashscope", requested_model=requested_model,
                client_factory_hash=client_hash)
        elif auth.scope != "verification":
            raise ValueError("formal authorization cannot be used for verification scope")
        if bool(preseed_journal) != bool(getattr(auth, "preseed_journal_sha256", "")):
            raise ValueError("preseed journal argument/signature mismatch")
    except (ValueError, OSError, KeyError) as exc:
        print(f"[e3-longrun-ctrl] AUTHORIZATION_BLOCKED: {exc}", flush=True)
        return BLOCKED
    try:
        import run_e3_real_smoke as _formal_probe
        _formal_probe.assert_formal_curriculum_adapter_ready()
        # The signed executable anchor manifest is a production prerequisite;
        # legacy identity-only anchor manifests and test fixtures never pass.
        _formal_probe.load_controller_executable_anchor_manifest()
    except RuntimeError as exc:
        print(f"[e3-longrun-ctrl] {exc}", flush=True)
        return BLOCKED
    if budget_scope == "formal":
        try:
            actual_gpu_uuid = _assert_formal_gpu_binding(
                expected_uuid=auth.expected_physical_gpu_uuid or "")
        except RuntimeError as exc:
            print(f"[e3-longrun-ctrl] {exc}", flush=True)
            return BLOCKED
    else:
        actual_gpu_uuid = ""
    if budget_scope == "formal":
        try:
            disk_free_bytes, disk_required_bytes = _assert_formal_disk_capacity(run_dir)
        except RuntimeError as exc:
            print(f"[e3-longrun-ctrl] {exc}", flush=True)
            return BLOCKED
    else:
        disk_free_bytes = disk_required_bytes = 0
    if not _lease_held:
        with run_lease(run_dir, metadata={"source_commit": source_commit,
            "authorization_id": auth.authorization_id, "authorization_manifest_hash": auth.manifest_hash}):
            return main(argv, _lease_held=True)
    resume_meta = None
    results = []
    start_session = 1
    prev_runstate = None
    if resume:
        try:
            resume_meta = verify_run_metadata(run_dir, {"source_commit": source_commit,
                "authorization_manifest_hash": auth.manifest_hash, "candidate_id": candidate_id,
                "sessions": int(sessions), "resolved_budget_env_steps": budget.resolved_env_steps,
                "runner_sha256": runner_sha256})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[e3-longrun-ctrl] RESUME_METADATA_BLOCKED: {exc}", flush=True)
            return BLOCKED
        try:
            results, start_session, prev_runstate = _load_and_validate_completed_sessions(
                run_dir, candidate=candidate_id, source_commit=source_commit, sessions=int(sessions),
                authorization_manifest_hash=auth.manifest_hash)
        except Exception as exc:
            print(f"[e3-longrun-ctrl] RESUME_EVIDENCE_BLOCKED: {exc}", flush=True)
            return BLOCKED
    # P0-8: atomic unique directory claim (rejects duplicate run ids).
    run_id = f"e3-{candidate_id[:12]}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    preseed_info = None
    try:
        if not resume:
            auth_mod.claim_output_dir(run_dir, run_id)
    except ValueError as exc:
        print(f"[e3-longrun-ctrl] DIR_CLAIM_BLOCKED: {exc}", flush=True)
        return BLOCKED
    if preseed_journal:
        try:
            preseed_info = _install_preseed_journal(
                preseed_path=preseed_journal, run_dir=run_dir, auth=auth,
                source_commit=source_commit, candidate_id=candidate_id,
                client_hash=client_hash)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[e3-longrun-ctrl] PRESEED_BLOCKED: {exc}", flush=True)
            return BLOCKED
    # Authorization and the atomic output claim have passed.  Production/JAX
    # imports and model mounting are now permitted, followed by a second
    # identity check against the already-authorized checkpoint file.
    import run_e3_real_smoke as prod
    probe = prod.mount_student(candidate_id)
    mounted_file_sha = str(probe.get("loaded", {}).get("file_sha256", ""))
    mounted_params_sha = str(probe.get("params_sha256", ""))
    if (mounted_file_sha != static_assets["checkpoint_sha256"]
            or mounted_params_sha != auth.student_profile_sha256):
        print("[e3-longrun-ctrl] POST_MOUNT_AUTHORIZATION_BLOCKED: mounted "
              "checkpoint file or params SHA differs from signed authorization",
              flush=True)
        return BLOCKED
    # The authorization is the Ed25519 signature (independently verifiable).
    # trusted_signer / formal_asset_registry_hash now come from the signed
    # manifest that was verified against the running artifacts.
    trusted_signer = auth.authorization_id
    formal_asset_registry_hash = auth.formal_asset_registry_hash
    _wandb_offline_init(f"e3-formal-{candidate_id[:12]}")

    metadata_payload = _metadata_payload(
        source_commit=source_commit, runner_sha256=runner_sha256, auth=auth,
        candidate_id=candidate_id, sessions=sessions, budget=budget,
        mounted_file_sha=mounted_file_sha, mounted_params_sha=mounted_params_sha,
        gpu=(actual_gpu_uuid or _gpu_uuid()), pid=os.getpid(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        client_combo_hash=getattr(auth, "client_factory_implementation_hash", "") or "",
        checkpoint_sha256=static_assets["checkpoint_sha256"],
        task_asset_manifest_sha256=static_assets["task_asset_manifest_sha256"],
        disk_free_bytes=disk_free_bytes, disk_required_bytes=disk_required_bytes,
        preseed_info=preseed_info)
    if resume:
        if resume_meta.get("mounted_file_sha256") != mounted_file_sha or resume_meta.get("mounted_params_sha256") != mounted_params_sha:
            return BLOCKED
        before = Path(run_dir, "RUN_METADATA.json").read_bytes()
        append_resume_event(run_dir, {"pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if Path(run_dir, "RUN_METADATA.json").read_bytes() != before:
            return BLOCKED
    else:
        write_run_metadata_once(run_dir, metadata_payload)
    _write_json(os.path.join(run_dir, "RESOLVED_CONFIG.json"), {
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "task_class_count": CURRICULUM_SLOT_COUNT_CONSERVATIVE + 1,
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "env_steps_per_update_changed": False,
        "max_updates_per_session_native": MAX_UPDATES_PER_SESSION_NATIVE,
        "updates_per_session": UPDATES_PER_SESSION,
        "curriculum_slot_count": CURRICULUM_SLOT_COUNT_CONSERVATIVE,
        "optimizer_semantics": "NEW_OPTIMIZER_PHASE_FROM_SESSION0_THEN_CONTINUOUS",
        "optimizer_semantics_note": ("P0-9: source Student checkpoint is params-only "
            "(no opt_state).  Session 0 initializes a fresh optimizer (Adam, "
            "step=0); session k>0 resumes the previous session's opt_state + step "
            "continuously.  No silent rebuild between sessions; optimizer reset "
            "only happens once at session 0 from the canonical checkpoint."),
        "total_timesteps_native": TOTAL_TIMESTEPS,
        "native_total_updates": NATIVE_TOTAL_UPDATES,
        "reference_experiment_sessions": REFERENCE_EXPERIMENT_SESSIONS,
        "reference_experiment_updates": REFERENCE_EXPERIMENT_UPDATES,
        "reference_experiment_env_steps": REFERENCE_EXPERIMENT_ENV_STEPS,
        "seed": SEED,
        "actual_n": ACTUAL_N,
        "search_horizon": SEARCH_HORIZON,
        "requested_n_per_session": REQUESTED_N_PER_SESSION,
        "budget_semantics": budget.budget_semantics,
        "resolved_budget_env_steps": budget.resolved_env_steps,
        "api_paid_new_calls": 0,
        "api_reused_successes": 0,
        "api_success_key_ceiling": 302,
    })
    _write_json(os.path.join(run_dir, "GIT_BINDING.json"), {
        "branch": FORMAL_BRANCH,
        "head": source_commit,
        "source_commit_authority": "SIGNED_AUTHORIZATION_MANIFEST",
        "head_matches_signed_manifest": source_commit == auth.source_commit,
    })

    _log(f"START candidate={candidate_id} sessions={sessions} run_dir={run_dir} "
         f"head={source_commit[:12]} num_envs={NUM_ENVS} num_steps={NUM_STEPS} "
         f"task_classes={CURRICULUM_SLOT_COUNT_CONSERVATIVE + 1} "
         f"updates_per_session={UPDATES_PER_SESSION}")
    for s in range(start_session, sessions + 1):
        os.environ["E3_SOURCE_COMMIT"] = source_commit
        os.environ["E3_CANDIDATE_ID"] = candidate_id
        os.environ["E3_SESSION_IDX"] = str(s)
        os.environ["E3_LLM_PROVIDER"] = "dashscope"
        os.environ["E3_LLM_JOURNAL_PATH"] = os.path.join(run_dir, "LLM_PAID_CALL_JOURNAL.json")
        os.environ["E3_CLIENT_FACTORY_IMPLEMENTATION_HASH"] = getattr(auth, "client_factory_implementation_hash", "") or ""
        report = run_one_session(
            candidate_id=candidate_id, session_idx=s, run_dir=run_dir,
            prev_runstate=prev_runstate, source_commit=source_commit,
            trusted_signer=trusted_signer,
            formal_asset_registry_hash=formal_asset_registry_hash,
            authorization_manifest_hash=auth.manifest_hash)
        results.append(report)
        prev_runstate = report["checkpoint_path"]

    final = {
        "schema": "simulator_frontier.e3_formal_longrun_final/v2",
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "candidate_id": candidate_id,
        "sessions_completed": len(results),
        "updates_per_session": UPDATES_PER_SESSION,
        "final_global_update": results[-1]["global_update_step"] if results else 0,
        "final_global_env_steps": results[-1]["global_env_steps"] if results else 0,
        "latest_checkpoint": prev_runstate,
        "all_fresh_restore_ok": all(r["fresh_process_restore_equivalent"]
                                    for r in results),
        "experiment_updates": (results[-1]["global_update_step"] - 0) if results else 0,
        "experiment_env_steps": (results[-1]["global_env_steps"] - 0) if results else 0,
        "source_commit": source_commit,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget_semantics": budget.budget_semantics,
        "resolved_budget_env_steps": budget.resolved_env_steps,
    }
    expected_updates = int(sessions) * UPDATES_PER_SESSION
    expected_env_steps = expected_updates * ENV_STEPS_PER_UPDATE
    finite_reports = all(_numeric_leaves_finite(r) for r in results)
    logical_role_events = sum(
        len(r.get("durable_role_journal_refs", [])) for r in results)
    paid_new_total = sum(int(r.get("paid_new_calls", 0)) for r in results)
    reused_total = sum(int(r.get("reused_successes", 0)) for r in results)
    final.update({
        "status": "PENDING_VALIDATION",
        "finite_metrics": finite_reports,
        "logical_role_events": logical_role_events,
        "attempt_paid_new_calls": paid_new_total,
        "attempt_reused_successes": reused_total,
        "api_unique_paid_successes": paid_new_total,
    })
    formal_journal_ok = True
    if budget_scope == "formal":
        journal_path = os.path.join(run_dir, "LLM_PAID_CALL_JOURNAL.json")
        try:
            from dicode.simulator_frontier.e3_durable_llm_journal import DurablePaidCallJournal
            entries = DurablePaidCallJournal(journal_path)._load().get("entries", {})
            formal_journal_ok = len(entries) == 302 and logical_role_events == 302
        except Exception:
            formal_journal_ok = False
    final["journal_complete"] = formal_journal_ok
    if (final["final_global_update"] != expected_updates
            or final["final_global_env_steps"] != expected_env_steps
            or not finite_reports
            or not formal_journal_ok
            or (budget_scope == "formal" and (expected_updates != 15100
                                                or expected_env_steps != 1_979_187_200))):
        final["status"] = "FAIL"
        _write_json(os.path.join(run_dir, "FINAL_STATUS.json"), final)
        return FAIL
    final["status"] = "PASS"
    _write_json(os.path.join(run_dir, "FINAL_STATUS.json"), final)
    _log(f"DONE candidate={candidate_id} sessions={len(results)} "
         f"final_global_update={final['final_global_update']} "
         f"final_global_env={final['final_global_env_steps']}")
    return PASS


if __name__ == "__main__":
    raise SystemExit(main())
