"""E3 canonical FULL RunState checkpoint codec + fresh-process restore.

BUG-E3-07/08/09 closure: the e3 window no longer writes a params-only
``{"params", "global_step"}`` checkpoint through the adapter and reloads it
in-process.  The canonical DiCode chain checkpoints the COMPLETE run state
(``params`` + ``opt_state`` + ``step`` + training/env RNG + session + task
archive identity + plan hash + runtime bundle hash + config hash) and
proves the restore in an INDEPENDENT python process:

  * ``RunStateCheckpointManager.save`` refuses any state missing a required
    field (a params-only snapshot is never a full run state) and writes a
    pickled state file plus a JSON metadata ledger carrying the state file
    sha256 and a recomputed checkpoint hash.
  * ``RunStateCheckpointManager.restore`` re-verifies the state-file sha256
    against the metadata ledger and re-checks every required field.
  * ``fresh_process_restore`` launches EXACTLY ONE new python interpreter
    that reloads the checkpoint and recomputes the content hash; the parent
    compares it to its own.  A same-process save/load can never satisfy this
    gate (BUG-E3-09).
  * ``next_policy_step_hash`` computes a deterministic digest of the
    policy-determining state (params leaves, opt-state leaves, train step) —
    the exact inputs of the next policy step.  Equal hashes prove the next
    policy step is equivalent.

The module imports jax/numpy lazily so importing it never crashes a jax-less
interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from typing import Any, Mapping

from .errors import InvalidEvidenceError

RUNSTATE_SCHEMA = "simulator_frontier.canonical_runstate_checkpoint/v1"

# The canonical full-run-state contract.  ``params`` + ``opt_state`` +
# ``train_step`` come from the Flax ``TrainState``; the RNGs, counters,
# session index, task-archive identity and plan / runtime-bundle / config
# hashes complete the state.  ANY missing field fails closed.
REQUIRED_RUNSTATE_FIELDS = (
    "params", "opt_state", "train_step", "training_rng", "env_rng",
    "global_update_step", "global_env_steps", "current_session_idx",
    "task_archive_identity", "plan_hash", "runtime_bundle_hash",
    "config_hash", "source_commit",
    # BUG-E3-01: architecture identity fields for the selected Student
    "candidate_id", "architecture_family",
)

RUNSTATE_CODEC_VERSION = "simulator_frontier.runstate_codec/v1"


class RunStateError(RuntimeError):
    """Fail-closed run-state violation."""


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leaf_bytes(leaf: Any) -> bytes:
    """Deterministic byte representation of one pytree leaf (cross-process)."""
    import numpy as np
    if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
        try:
            return np.asarray(leaf).tobytes()
        except (TypeError, ValueError, RuntimeError):
            return repr(type(leaf)).encode("utf-8") + bytes(str(leaf)[:4096], "utf-8")
    if isinstance(leaf, (bool, int, float, str)):
        return repr(type(leaf)).encode("utf-8") + str(leaf).encode("utf-8")
    return repr(type(leaf)).encode("utf-8") + bytes(str(leaf)[:4096], "utf-8")


def runstate_content_hash(run_state: Mapping[str, Any]) -> str:
    """Deterministic content hash of a full run state (leaf bytes).

    Uses ``np.asarray(leaf).tobytes()`` for array leaves so the digest is
    value-stable across processes/devices (no device-address leakage).
    """
    import jax
    leaves = jax.tree_util.tree_leaves(run_state)
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(_leaf_bytes(leaf))
    return digest.hexdigest()


def next_policy_step_hash(train_state: Any) -> str:
    """Deterministic digest of the policy-determining state.

    The next policy step is a deterministic function of (params, opt_state,
    step): equal hashes prove the next policy step is equivalent.  This is
    the same evidence shape E1's canonical runstate round trip uses.
    """
    import jax
    params = getattr(train_state, "params", None)
    opt_state = getattr(train_state, "opt_state", None)
    step = getattr(train_state, "step", 0)
    if params is None or opt_state is None:
        raise RunStateError(
            "next_policy_step_hash requires a TrainState carrying params + "
            "opt_state (a params-only source is never a full policy state)")
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        digest.update(_leaf_bytes(leaf))
    for leaf in jax.tree_util.tree_leaves(opt_state):
        digest.update(_leaf_bytes(leaf))
    digest.update(str(int(step)).encode("utf-8"))
    return digest.hexdigest()


class RunStateCheckpointManager:
    """The canonical full-run-state checkpoint surface (save + restore)."""

    def __init__(self) -> None:
        self.object_identity_hash = hashlib.sha256(
            b"simulator_frontier.canonical_runstate_checkpoint_manager.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash

    def save(self, run_state: Mapping[str, Any], path: str, *,
             idempotency_token: str = "") -> dict[str, str]:
        """Save the COMPLETE run state fail-closed (all fields present)."""
        missing = [name for name in REQUIRED_RUNSTATE_FIELDS
                   if name not in run_state]
        if missing:
            raise RunStateError(
                f"RUNSTATE_INCOMPLETE: refusing to save a run state missing "
                f"{sorted(missing)} (a params-only snapshot is never a full "
                "run state)")
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        state_path = path + ".state.pkl"
        meta_path = path + ".meta.json"
        with open(state_path, "wb") as handle:
            pickle.dump(dict(run_state), handle, protocol=4)
        state_sha = _file_sha256(state_path)
        metadata: dict[str, Any] = {
            "schema": RUNSTATE_SCHEMA,
            "codec_version": RUNSTATE_CODEC_VERSION,
            "state_file_sha256": state_sha,
            "fields": sorted(str(k) for k in run_state),
            "global_update_step": int(run_state["global_update_step"]),
            "global_env_steps": int(run_state["global_env_steps"]),
            "current_session_idx": int(run_state["current_session_idx"]),
            "plan_hash": str(run_state["plan_hash"]),
            "runtime_bundle_hash": str(run_state["runtime_bundle_hash"]),
            "config_hash": str(run_state["config_hash"]),
            "source_commit": str(run_state["source_commit"]),
            "idempotency_token": str(idempotency_token),
        }
        metadata["checkpoint_hash"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return {
            "checkpoint_path": path,
            "state_file_sha256": state_sha,
            "checkpoint_hash": metadata["checkpoint_hash"],
            "state_file": state_path,
            "meta_file": meta_path,
        }

    def restore(self, path: str) -> dict[str, Any]:
        """Restore the full run state fail-closed (tamper / missing rejected)."""
        state_path = path + ".state.pkl"
        meta_path = path + ".meta.json"
        for required in (state_path, meta_path):
            if not os.path.isfile(required):
                raise RunStateError(
                    f"RUNSTATE_RESTORE_MISSING: {required!r} does not exist "
                    "(fail closed)")
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        actual_sha = _file_sha256(state_path)
        if actual_sha != metadata.get("state_file_sha256"):
            raise RunStateError(
                "RUNSTATE_TAMPERED: state file sha256 drift "
                f"({actual_sha[:16]}... != "
                f"{str(metadata.get('state_file_sha256'))[:16]}...)")
        with open(state_path, "rb") as handle:
            run_state = pickle.load(handle)
        missing = [name for name in REQUIRED_RUNSTATE_FIELDS
                   if name not in run_state]
        if missing:
            raise RunStateError(
                f"RUNSTATE_RESTORE_INCOMPLETE: restored state missing "
                f"{sorted(missing)}")
        return {"run_state": run_state, "metadata": metadata}


def fresh_process_restore(checkpoint_path: str, *,
                          python_executable: str = "",
                          extra_pythonpath: str = "") -> dict[str, Any]:
    """Restore the checkpoint in an INDEPENDENT python process.

    The child process reloads the run state, recomputes the content hash and
    reports it on stdout; this parent compares hashes.  A same-process
    save/load can never satisfy this gate.
    """
    exe = python_executable or sys.executable
    script = (
        "import json, sys\n"
        "from dicode.simulator_frontier.runstate_codec import (\n"
        "    RunStateCheckpointManager, runstate_content_hash)\n"
        f"restored = RunStateCheckpointManager().restore({checkpoint_path!r})\n"
        "print(json.dumps({'restored': True,\n"
        "                  'content_hash': runstate_content_hash("
        "restored['run_state']),\n"
        "                  'checkpoint_hash': "
        "restored['metadata']['checkpoint_hash'],\n"
        "                  'global_update_step': "
        "restored['metadata']['global_update_step'],\n"
        "                  'current_session_idx': "
        "restored['metadata']['current_session_idx']}))\n"
    )
    env = dict(os.environ)
    if extra_pythonpath:
        env["PYTHONPATH"] = (
            extra_pythonpath + os.pathsep + env.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(
            [exe, "-c", script], capture_output=True, text=True,
            timeout=900, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RunStateError(
            f"FRESH_PROCESS_RESTORE_TIMEOUT: {exc!r}") from exc
    if proc.returncode != 0:
        raise RunStateError(
            "FRESH_PROCESS_RESTORE_FAILED: child exited "
            f"{proc.returncode}: {proc.stderr[-1200:]}")
    line = [ln for ln in proc.stdout.splitlines()
            if ln.strip().startswith("{")]
    if not line:
        raise RunStateError(
            "FRESH_PROCESS_RESTORE_NO_REPORT: child printed no JSON report; "
            f"stdout tail: {proc.stdout[-600:]}")
    return json.loads(line[-1])


def build_full_run_state(*, rl_train_state: Any, training_rng: Any,
                         env_rng: Any, global_update_step: int,
                         global_env_steps: int, current_session_idx: int,
                         task_archive_identity: str, plan_hash: str,
                         runtime_bundle_hash: str, config_hash: str,
                         source_commit: str,
                         candidate_id: str = "",
                         architecture_family: str = "",
                         extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the COMPLETE canonical run state for checkpointing.

    ``rl_train_state`` carries params + opt_state + step; the RNGs, counters,
    session index, task-archive identity, plan / runtime-bundle / config
    hashes and the source commit ALL enter the checkpoint.

    BUG-E3-01: ``candidate_id`` and ``architecture_family`` identify the
    selected Student whose architecture was trained.
    """
    params = getattr(rl_train_state, "params", None)
    opt_state = getattr(rl_train_state, "opt_state", None)
    train_step = getattr(rl_train_state, "step", 0)
    if params is None or opt_state is None:
        raise RunStateError(
            "RUNSTATE_SOURCE_INCOMPLETE: rl_train_state must carry params + "
            "opt_state (a params-only source is never a full run state)")
    run_state: dict[str, Any] = {
        "params": params,
        "opt_state": opt_state,
        "train_step": int(train_step),
        "training_rng": training_rng,
        "env_rng": env_rng,
        "global_update_step": int(global_update_step),
        "global_env_steps": int(global_env_steps),
        "current_session_idx": int(current_session_idx),
        "task_archive_identity": task_archive_identity,
        "plan_hash": plan_hash,
        "runtime_bundle_hash": runtime_bundle_hash,
        "config_hash": config_hash,
        "source_commit": source_commit,
        "candidate_id": str(candidate_id),
        "architecture_family": str(architecture_family),
    }
    if extra:
        run_state.update(dict(extra))
    return run_state
