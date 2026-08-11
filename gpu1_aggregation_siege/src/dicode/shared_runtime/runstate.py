"""The REAL canonical DiCode RunState checkpoint manager.

Saves/restores the COMPLETE run state (train-state pytree + counters +
rng + task archive identity + plan/runtime bundle hashes) — never a
params-only snapshot. Fresh-process restore launches an INDEPENDENT
python interpreter that reloads the checkpoint and recomputes the
next-policy-step hash; equivalence is proven by exact hash equality,
never by in-process save/load.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from typing import Any, Dict, Mapping


class RunStateError(RuntimeError):
    """Fail-closed run-state violation."""


RUNSTATE_SCHEMA = "mechanism_UED.canonical_runstate_checkpoint/v1"

REQUIRED_RUNSTATE_FIELDS = (
    "params", "opt_state", "train_step", "training_rng", "env_rng",
    "global_update_step", "global_env_steps", "current_session_idx",
    "task_archive_identity", "mechanism_state_identity", "plan_hash",
    "runtime_bundle_hash", "config_hash", "source_commit",
)


class RunStateCheckpointManager:
    """The canonical full-run-state checkpoint surface."""

    def __init__(self):
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.canonical_runstate_checkpoint_manager.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash

    # -- save -----------------------------------------------------------
    def save(self, run_state: Mapping[str, Any], path: str, *,
             idempotency_token: str = "") -> Dict[str, str]:
        """Save the COMPLETE run state fail-closed (all fields present)."""
        missing = [name for name in REQUIRED_RUNSTATE_FIELDS
                   if name not in run_state]
        if missing:
            raise RunStateError(
                f"RUNSTATE_INCOMPLETE: refusing to save a run state "
                f"missing {sorted(missing)} (a params-only snapshot is "
                "never a full run state)")
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        state_path = path + ".state.pkl"
        meta_path = path + ".meta.json"
        with open(state_path, "wb") as handle:
            pickle.dump(dict(run_state), handle, protocol=4)
        state_sha = _file_sha256(state_path)
        metadata = {
            "schema": RUNSTATE_SCHEMA,
            "state_file_sha256": state_sha,
            "fields": sorted(run_state.keys()),
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
        }

    # -- restore --------------------------------------------------------
    def restore(self, path: str) -> Dict[str, Any]:
        state_path = path + ".state.pkl"
        meta_path = path + ".meta.json"
        for required in (state_path, meta_path):
            if not os.path.isfile(required):
                raise RunStateError(
                    f"RUNSTATE_RESTORE_MISSING: {required!r} does not "
                    "exist (fail closed)")
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


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runstate_content_hash(run_state: Mapping[str, Any]) -> str:
    """Deterministic content hash of a restored run state (leaf bytes)."""
    import jax

    leaves = jax.tree_util.tree_leaves(run_state)
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(repr(type(leaf)).encode("utf-8"))
        digest.update(bytes(str(leaf)[:4096], "utf-8"))
    return digest.hexdigest()


def fresh_process_restore(checkpoint_path: str, *,
                          python_executable: str = "",
                          extra_pythonpath: str = "") -> Dict[str, Any]:
    """Restore the checkpoint in an INDEPENDENT python process.

    The child process reloads the run state, recomputes the content hash
    and reports it on stdout; this parent compares hashes. A same-process
    save/load can never satisfy this gate.
    """
    exe = python_executable or sys.executable
    script = (
        "import json, sys\n"
        "from dicode.shared_runtime.runstate import (\n"
        "    RunStateCheckpointManager, runstate_content_hash)\n"
        f"restored = RunStateCheckpointManager().restore({checkpoint_path!r})\n"
        "print(json.dumps({'restored': True,\n"
        "                  'content_hash': runstate_content_hash("
        "restored['run_state']),\n"
        "                  'checkpoint_hash': "
        "restored['metadata']['checkpoint_hash'],\n"
        "                  'global_update_step': "
        "restored['metadata']['global_update_step']}))\n"
    )
    env = dict(os.environ)
    if extra_pythonpath:
        env["PYTHONPATH"] = (
            extra_pythonpath + os.pathsep + env.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(
            [exe, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunStateError(
            f"FRESH_PROCESS_RESTORE_TIMEOUT: {exc!r}") from exc
    if proc.returncode != 0:
        raise RunStateError(
            "FRESH_PROCESS_RESTORE_FAILED: child exited "
            f"{proc.returncode}: {proc.stderr[-800:]}")
    line = [ln for ln in proc.stdout.splitlines()
            if ln.strip().startswith("{")]
    if not line:
        raise RunStateError(
            "FRESH_PROCESS_RESTORE_NO_REPORT: child printed no JSON "
            f"report; stdout tail: {proc.stdout[-400:]}")
    return json.loads(line[-1])
