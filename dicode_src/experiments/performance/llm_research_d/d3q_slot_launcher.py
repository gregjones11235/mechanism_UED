#!/usr/bin/env python3
"""D3Q slot executor -- local launcher (Phase 1).

Orchestrates one or more matrix slots on the remote training server:

1. verifies the frozen inputs locally (FROZEN_MANIFEST / repair template /
   budget module / executor module hashes) and that the remote exec root
   ``/tmp/d3q_exec_<UTC>`` does not already exist;
2. creates the exec root, deploys the runner + CPU validator + frozen budget
   module + frozen manifests, and verifies remote sha256 == local sha256;
3. captures GPU and Ollama evidence before the run (GPU2 UUID gate, no external
   compute PIDs, Ollama model digests and PIDs);
4. for every requested slot: double-checks the shared budget from the remote
   ledger (slot <= 3, provider <= 108) *before* dispatching, runs the remote
   runner, collects the slot result, and re-checks the ledger afterwards;
5. captures GPU and Ollama evidence after the run, verifies nothing changed;
6. removes the exec root and verifies it is absent (fail-closed on failure);
7. stages the full artifact set privately, verifies sha256 / closed schema /
   no-secret scan, then atomically publishes to ``d3q_artifacts/<run_id>/``.

The launcher never reads or echoes credential values; only the environment
variable name ``EXP_DEEPSEEK_API_KEY`` is referenced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3q_slot_runner as runner_mod  # noqa: E402

from d3q_budget import BudgetExceededError, D3QLedger  # noqa: E402

SSH_KEY = "D:/Projects/dicode-codex-director/orchestration/control/ssh_oseasy_172_25_14_221_ed25519"
SSH_TARGET = "oseasy@172.25.14.221"
REMOTE_PYTHON = "/home/oseasy/venvs/skill_preflight_e0e1/bin/python"
REMOTE_ENV_FILE = "/home/oseasy/.config/dicode/experiment_llm.env"
MASON_WORKTREE = "/home/oseasy/git_work/wt_d3q_mason_91a75e5"
MASON_WORKTREE_COMMIT = "91a75e5a1d3bfca5114caf776a710a0339f692d8"
EXPECTED_GPU2_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"
EXPECTED_OLLAMA_QWEN_DIGEST_PREFIX = "9ec8897f747e"

EXEC_ROOT_PREFIX = "/tmp/d3q_exec_"
_EXEC_ROOT_RE = re.compile(r"^/tmp/d3q_exec_[0-9]{8}T[0-9]{6}Z$")

PHASE0_DIR_NAME = "d3q_phase0_reconciliation_20260815T011126Z"

DEPLOY_FILES = (
    "d3q_slot_runner.py",
    "d3q_cpu_validate_remote.py",
    "d3q_budget.py",
    "FROZEN_MANIFEST.json",
    "D3Q_FROZEN_REPAIR_TEMPLATE.json",
)


def _local_file(name: str) -> Path:
    if name == "D3Q_FROZEN_REPAIR_TEMPLATE.json":
        return HERE / PHASE0_DIR_NAME / name
    return HERE / name

SLOT_RESULT_REQUIRED_FIELDS = (
    "classification", "schema_version", "run_id", "slot_id", "arm", "repeat",
    "prompt_index", "prompt_slot", "provider", "model", "initial_valid",
    "final_valid", "attempts", "repair_requests", "repair_success",
    "empty_response", "timeout", "connection_error", "http_4xx", "http_5xx",
    "invalid_json", "extract_error", "syntax_error", "api_enum_error",
    "cpu_jax_error", "duplicate_code", "prompt_tokens", "completion_tokens",
    "cached_tokens", "generation_wall_s", "repair_wall_s",
    "cpu_validation_wall_s", "final_code_sha256",
)


class LauncherError(RuntimeError):
    """Sanitized failure carrying only a fixed classification."""

    ALLOWED = {
        "remote_command_failed", "secret_like_output", "unexpected_remote_output",
        "local_output_exists", "artifact_tamper", "remote_root_exists",
        "cleanup_failed", "gpu_unavailable", "gpu2_uuid_mismatch",
        "gpu2_insufficient_free", "gpu2_external_app", "ollama_digest_changed",
        "ollama_pid_changed", "budget_exceeded", "frozen_hash_mismatch",
        "model_mismatch", "credential_missing", "schema_invalid",
        "no_secret_scan_failed", "slot_failed", "fatal_api_blocked",
        "worktree_mismatch", "run_id_invalid",
    }

    def __init__(self, reason: str):
        self.reason = reason if reason in self.ALLOWED else "remote_command_failed"
        self.detail: Any | None = None
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    argv: tuple[str, ...]
    shell: bool = False


CommandRunner = Callable[[CommandInvocation], CommandResult]


def _command_failure(argv: Sequence[str], result: CommandResult) -> dict[str, Any]:
    """Sanitized snapshot of a failed remote command (never secret-bearing)."""
    return {
        "command": list(argv),
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-4000:],
    }


def _default_runner(invocation: CommandInvocation) -> CommandResult:
    if invocation.shell is not False:
        raise LauncherError("remote_command_failed")
    completed = subprocess.run(
        list(invocation.argv),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        check=False,
        shell=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        raise LauncherError("artifact_tamper") from None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contains_sensitive_output(value: str) -> bool:
    return runner_mod.contains_secret(value)


def _strip_ssh_informational(stderr: str) -> str:
    if not stderr:
        return stderr
    return "\n".join(
        line for line in stderr.splitlines() if not line.lstrip().startswith("** ")
    )


def _validate_ssh_target(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.@\-]+", value):
        raise LauncherError("remote_command_failed")
    return value


def _validate_remote_path(value: str) -> str:
    path = str(value)
    if not path.startswith("/") or ".." in path.split("/"):
        raise LauncherError("remote_command_failed")
    return path


def _run_command(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> CommandResult:
    if not isinstance(argv, (list, tuple)) or not argv or not all(
        isinstance(item, str) for item in argv
    ):
        raise LauncherError("remote_command_failed")
    try:
        invocation = CommandInvocation(tuple(argv), shell=False)
        result = runner(invocation)
    except Exception:
        raise LauncherError("remote_command_failed") from None
    if _contains_sensitive_output(result.stdout) or _contains_sensitive_output(result.stderr):
        raise LauncherError("secret_like_output")
    result = CommandResult(
        result.returncode,
        result.stdout,
        _strip_ssh_informational(result.stderr),
    )
    if result.returncode not in allowed_returncodes:
        exc = LauncherError("remote_command_failed")
        exc.detail = _command_failure(argv, result)
        raise exc
    return result


def _require_exact_identity_selection(argv: Sequence[str]) -> list[str]:
    identity_options = [
        argv[index + 1]
        for index, item in enumerate(argv[:-1])
        if item == "-o" and argv[index + 1].startswith("IdentitiesOnly=")
    ]
    if identity_options != ["IdentitiesOnly=yes"]:
        raise LauncherError("remote_command_failed")
    return list(argv)


def _ssh_base(ssh_target: str, ssh_key: str | Path) -> list[str]:
    return _require_exact_identity_selection(
        [
            "ssh",
            "-i",
            str(ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "--",
            ssh_target,
        ]
    )


def _scp_base(ssh_key: str | Path) -> list[str]:
    return _require_exact_identity_selection(
        [
            "scp",
            "-q",
            "-i",
            str(ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "--",
        ]
    )


def _ssh_argv(
    ssh_target: str, ssh_key: str | Path, remote_command: Sequence[str]
) -> list[str]:
    return _ssh_base(ssh_target, ssh_key) + [shlex.join(list(remote_command))]


def _remote_python_argv(
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    script: str,
    *args: str,
) -> list[str]:
    return _ssh_argv(ssh_target, ssh_key, [remote_python, "-c", script, *args])


_REMOTE_SCRIPT_CREATE = (
    "import sys; from pathlib import Path; "
    "p = Path(sys.argv[1]); p.mkdir(parents=False, exist_ok=False); print('CREATED')"
)
_REMOTE_SCRIPT_EXISTS = (
    "import sys; from pathlib import Path; "
    "print('EXISTS' if Path(sys.argv[1]).exists() else 'ABSENT')"
)
_REMOTE_SCRIPT_REMOVE = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "import d3q_slot_runner as r; r.fs_remove_tree(sys.argv[2]); print('REMOVED')"
)
_REMOTE_SCRIPT_HASH = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "import d3q_slot_runner as r; print(r.fs_sha256_file(sys.argv[2]))"
)
_REMOTE_SCRIPT_LEDGER = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "import d3q_slot_runner as r; print(r.canonical_json(r.fs_ledger_summary(sys.argv[2])))"
)

_REMOTE_SCRIPT_OLLAMA_TAGS = (
    "import sys, json, urllib.request; "
    "body = urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=10).read(); "
    "data = json.loads(body.decode('utf-8')); "
    "print(json.dumps({m['name']: m['digest'] for m in data.get('models', [])}, sort_keys=True))"
)


def _expect_exact(result: CommandResult, expected: str) -> None:
    if result.stderr or result.stdout.strip() != expected:
        exc = LauncherError("unexpected_remote_output")
        exc.detail = _command_failure((), result)
        raise exc


def _parse_json_object(result: CommandResult) -> dict[str, Any]:
    if result.stderr:
        exc = LauncherError("unexpected_remote_output")
        exc.detail = _command_failure((), result)
        raise exc
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        exc = LauncherError("unexpected_remote_output")
        exc.detail = _command_failure((), result)
        raise exc from None
    if not isinstance(decoded, dict):
        exc = LauncherError("unexpected_remote_output")
        exc.detail = _command_failure((), result)
        raise exc
    return decoded

# ---------------------------------------------------------------------------
# Remote GPU / Ollama evidence.
# ---------------------------------------------------------------------------


def _gpu_snapshot(
    runner: CommandRunner, ssh_target: str, ssh_key: str | Path
) -> dict[str, Any]:
    try:
        gpu = _run_command(
            runner,
            _ssh_argv(
                ssh_target, ssh_key,
                ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
            ),
        )
        apps = _run_command(
            runner,
            _ssh_argv(
                ssh_target, ssh_key,
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                 "--format=csv,noheader,nounits"],
            ),
        )
    except LauncherError as exc:
        if exc.reason == "secret_like_output":
            raise
        raise LauncherError("gpu_unavailable") from None

    rows: list[dict[str, Any]] = []
    for raw in gpu.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 7:
            raise LauncherError("gpu_unavailable")
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mib": int(float(parts[3])),
                    "memory_used_mib": int(float(parts[4])),
                    "memory_free_mib": int(float(parts[5])),
                    "utilization_gpu_pct": int(float(parts[6])),
                }
            )
        except ValueError:
            raise LauncherError("gpu_unavailable") from None

    apps_by_uuid: dict[str, list[dict[str, Any]]] = {}
    for raw in apps.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise LauncherError("gpu_unavailable")
        uuid, pid, name = parts[0], parts[1], parts[2]
        if pid in {"", "N/A", "[N/A]"}:
            continue
        try:
            apps_by_uuid.setdefault(uuid, []).append({"pid": int(pid), "process_name": name})
        except ValueError:
            raise LauncherError("gpu_unavailable") from None

    gpu2 = next((row for row in rows if row["index"] == 2), None)
    if gpu2 is None:
        raise LauncherError("gpu_unavailable")
    if gpu2["uuid"] != EXPECTED_GPU2_UUID:
        raise LauncherError("gpu2_uuid_mismatch")
    gpu2_external = [app for app in apps_by_uuid.get(gpu2["uuid"], [])]
    return {
        "gpus": rows,
        "gpu2": {
            "gpu_index": 2,
            "uuid": gpu2["uuid"],
            "memory_free_mib": gpu2["memory_free_mib"],
            "memory_used_mib": gpu2["memory_used_mib"],
            "external_compute_pids": sorted(app["pid"] for app in gpu2_external),
            "external_compute_processes": gpu2_external,
        },
        "apps_by_uuid": apps_by_uuid,
    }


def _ollama_snapshot(
    runner: CommandRunner, ssh_target: str, ssh_key: str | Path, remote_python: str
) -> dict[str, Any]:
    try:
        tags = _run_command(
            runner,
            _remote_python_argv(
                ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_OLLAMA_TAGS
            ),
        )
        procs = _run_command(
            runner,
            _ssh_argv(ssh_target, ssh_key, ["pgrep", "-af", "ollama"]),
        )
    except LauncherError as exc:
        if exc.reason == "secret_like_output":
            raise
        raise LauncherError("gpu_unavailable") from None

    decoded = _parse_json_object(tags)
    if not isinstance(decoded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in decoded.items()
    ):
        raise LauncherError("gpu_unavailable")
    pids: list[str] = []
    for raw in procs.stdout.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("pgrep"):
            continue
        first = stripped.split()[0]
        if first.isdigit():
            pids.append(first)
    return {"models": dict(sorted(decoded.items())), "pids": sorted(pids)}


def _enforce_ollama_unchanged(pre: dict[str, Any], post: dict[str, Any]) -> None:
    qwen_pre = pre["models"].get(runner_mod.SMALL_MODEL, "")
    qwen_post = post["models"].get(runner_mod.SMALL_MODEL, "")
    if not qwen_pre.startswith(EXPECTED_OLLAMA_QWEN_DIGEST_PREFIX):
        raise LauncherError("ollama_digest_changed")
    if qwen_pre != qwen_post:
        raise LauncherError("ollama_digest_changed")
    if pre["pids"] != post["pids"]:
        raise LauncherError("ollama_pid_changed")


# ---------------------------------------------------------------------------
# Exec root lifecycle.
# ---------------------------------------------------------------------------


def _validate_exec_root(path: str) -> str:
    if not _EXEC_ROOT_RE.fullmatch(path):
        raise LauncherError("remote_command_failed")
    return path


def _exec_root_for(now: datetime, run_id: str) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise LauncherError("remote_command_failed")
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _validate_exec_root(EXEC_ROOT_PREFIX + stamp)


def _remote_exec_root(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    exec_root: str,
) -> None:
    exists = _run_command(
        runner,
        _remote_python_argv(ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_EXISTS, exec_root),
    )
    if not exists.stderr and exists.stdout.strip() == "EXISTS":
        raise LauncherError("remote_root_exists")
    _expect_exact(exists, "ABSENT")
    created = _run_command(
        runner,
        _remote_python_argv(ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_CREATE, exec_root),
    )
    _expect_exact(created, "CREATED")


def _deploy(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    exec_root: str,
    local_files: Mapping[str, Path],
) -> None:
    for name, local_path in local_files.items():
        deployed = _run_command(
            runner,
            _scp_base(ssh_key) + [str(local_path), f"{ssh_target}:{exec_root}/{name}"],
        )
        if deployed.stdout or deployed.stderr:
            raise LauncherError("unexpected_remote_output")


def _verify_remote_hashes(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    exec_root: str,
    local_files: Mapping[str, Path],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, local_path in local_files.items():
        remote_hash = _run_command(
            runner,
            _remote_python_argv(
                ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_HASH, exec_root,
                f"{exec_root}/{name}",
            ),
        )
        expected = _sha256_file(local_path)
        if remote_hash.stderr or remote_hash.stdout.strip() != expected:
            raise LauncherError("artifact_tamper")
        hashes[name] = expected
    return hashes


def _remote_ledger_summary(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    exec_root: str,
) -> dict[str, Any]:
    result = _run_command(
        runner,
        _remote_python_argv(
            ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_LEDGER, exec_root,
            f"{exec_root}/ledger.jsonl",
        ),
    )
    return _parse_json_object(result)


def _enforce_budget_before_slot(
    summary: dict[str, Any],
    slot_id: str,
    provider: str,
) -> None:
    slot_counts = summary.get("slot_counts") or {}
    provider_counts = summary.get("provider_counts") or {}
    if slot_counts.get(slot_id, 0) >= runner_mod.MAX_POSTS_PER_SLOT:
        raise LauncherError("budget_exceeded")
    if provider_counts.get(provider, 0) >= runner_mod.MAX_PROVIDER_POSTS:
        raise LauncherError("budget_exceeded")


def _enforce_budget_after_slot(
    summary: dict[str, Any],
    slot_id: str,
    provider: str,
    expected_slot_posts: int,
) -> None:
    slot_counts = summary.get("slot_counts") or {}
    provider_counts = summary.get("provider_counts") or {}
    actual_slot = slot_counts.get(slot_id, 0)
    actual_provider = provider_counts.get(provider, 0)
    if actual_slot > runner_mod.MAX_POSTS_PER_SLOT:
        raise LauncherError("budget_exceeded")
    if actual_provider > runner_mod.MAX_PROVIDER_POSTS:
        raise LauncherError("budget_exceeded")
    if actual_slot != expected_slot_posts:
        raise LauncherError("budget_exceeded")


def _cleanup_exec_root(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    exec_root: str,
) -> None:
    removed = _run_command(
        runner,
        _remote_python_argv(
            ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_REMOVE, exec_root, exec_root
        ),
    )
    _expect_exact(removed, "REMOVED")
    exists = _run_command(
        runner,
        _remote_python_argv(ssh_target, ssh_key, remote_python, _REMOTE_SCRIPT_EXISTS, exec_root),
    )
    _expect_exact(exists, "ABSENT")

# ---------------------------------------------------------------------------
# Slot dispatch.
# ---------------------------------------------------------------------------


def _dispatch_slot(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    *,
    remote_python: str,
    remote_env_file: str,
    exec_root: str,
    run_id: str,
    slot_id: str,
    mason_src: str,
) -> dict[str, Any]:
    repeat, arm, prompt_index = runner_mod.parse_slot_id(slot_id)
    provider, model, _base_url = runner_mod.arm_to_provider_model(arm)
    args = [
        remote_python,
        f"{exec_root}/d3q_slot_runner.py",
        "--exec-root", exec_root,
        "--slot-id", slot_id,
        "--run-id", run_id,
        "--manifest", f"{exec_root}/FROZEN_MANIFEST.json",
        "--repair-template", f"{exec_root}/D3Q_FROZEN_REPAIR_TEMPLATE.json",
        "--ledger", f"{exec_root}/ledger.jsonl",
        "--env-file", remote_env_file,
        "--remote-python", remote_python,
        "--cpu-validate-script", f"{exec_root}/d3q_cpu_validate_remote.py",
        "--mason-src", mason_src,
    ]
    dispatched = _run_command(runner, _ssh_argv(ssh_target, ssh_key, args))
    summary = _parse_json_object(dispatched)
    status = summary.get("status")
    if status != "DONE":
        raise LauncherError("slot_failed")
    if summary.get("slot_id") != slot_id:
        raise LauncherError("slot_failed")
    return {
        "slot_id": slot_id,
        "arm": arm,
        "repeat": repeat,
        "provider": provider,
        "model": model,
        "summary": summary,
    }


def _collect_slot_dir(
    runner: CommandRunner,
    ssh_target: str,
    ssh_key: str | Path,
    exec_root: str,
    slot_id: str,
    local_slots_dir: Path,
) -> None:
    # Windows OpenSSH scp cannot handle the POSIX ``dir/.`` source form, so
    # the slot directory is archived remotely with tar, fetched as one file,
    # and extracted locally.
    local_slots_dir.mkdir(parents=True, exist_ok=True)
    local_slot_dir = local_slots_dir / slot_id
    tar_name = f"{slot_id}.tgz"
    tar_remote = f"{exec_root}/{tar_name}"
    archive = _run_command(
        runner,
        _ssh_argv(
            ssh_target, ssh_key,
            ["tar", "-C", f"{exec_root}/slots", "-czf", tar_remote, slot_id],
        ),
    )
    if archive.stdout or archive.stderr:
        raise LauncherError("unexpected_remote_output")
    local_tar = local_slots_dir / tar_name
    try:
        downloaded = _run_command(
            runner,
            _scp_base(ssh_key) + [f"{ssh_target}:{tar_remote}", str(local_tar)],
        )
        if downloaded.stdout or downloaded.stderr:
            raise LauncherError("unexpected_remote_output")
        with tarfile.open(local_tar, "r:gz") as handle:
            try:
                handle.extractall(local_slots_dir, filter="data")
            except TypeError:
                handle.extractall(local_slots_dir)
    finally:
        try:
            local_tar.unlink()
        except OSError:
            pass
    if not local_slot_dir.is_dir():
        raise LauncherError("artifact_tamper")


def _load_slot_result(local_slot_dir: Path, slot_id: str) -> dict[str, Any]:
    path = local_slot_dir / f"{slot_id}.result.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise LauncherError("schema_invalid") from None
    if not isinstance(data, dict) or data.get("slot_id") != slot_id:
        raise LauncherError("schema_invalid")
    verify_slot_result_schema(data)
    return data


def verify_slot_result_schema(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("classification") != "D3Q_SLOT_RESULT":
        raise LauncherError("schema_invalid")
    for field in SLOT_RESULT_REQUIRED_FIELDS:
        if field not in result:
            raise LauncherError("schema_invalid")
    if type(result["attempts"]) is not int or not 0 <= result["attempts"] <= 3:
        raise LauncherError("schema_invalid")
    if type(result["repair_requests"]) is not int or not 0 <= result["repair_requests"] <= 2:
        raise LauncherError("schema_invalid")
    for flag in (
        "initial_valid", "final_valid", "fatal_api_blocked",
    ):
        if type(result[flag]) is not bool:
            raise LauncherError("schema_invalid")
    for field in (
        "prompt_tokens", "completion_tokens", "cached_tokens",
        "empty_response", "timeout", "connection_error", "http_4xx", "http_5xx",
        "invalid_json", "extract_error", "syntax_error", "api_enum_error",
        "cpu_jax_error", "duplicate_code",
    ):
        if type(result[field]) is not int or result[field] < 0:
            raise LauncherError("schema_invalid")
    for field in (
        "generation_wall_s", "repair_wall_s", "cpu_validation_wall_s",
    ):
        if type(result[field]) not in (int, float) or result[field] < 0:
            raise LauncherError("schema_invalid")
    final_sha = result.get("final_code_sha256")
    if final_sha is not None and not _is_sha256(final_sha):
        raise LauncherError("schema_invalid")
    if result["final_valid"] and not _is_sha256(final_sha):
        raise LauncherError("schema_invalid")
    return dict(result)


# ---------------------------------------------------------------------------
# Artifact staging + publishing.
# ---------------------------------------------------------------------------


def _scan_text_for_secrets(text: str) -> list[str]:
    return [pattern.pattern for pattern in runner_mod.SECRET_PATTERNS if pattern.search(text)]


def _no_secret_scan_directory(directory: Path) -> dict[str, Any]:
    scanned = 0
    violations: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        scanned += 1
        try:
            data = path.read_bytes()
        except OSError:
            violations.append(f"{path.name}:unreadable")
            continue
        if b"\x00" in data[:4096]:
            continue  # binary artifact (e.g. none expected here)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        matches = _scan_text_for_secrets(text)
        if matches:
            violations.append(f"{path.name}:{','.join(matches)}")
    return {"files_scanned": scanned, "violations": violations, "passed": not violations}


def _write_sha256sums(directory: Path) -> Path:
    lines = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(directory).as_posix()
        lines.append(f"{_sha256_file(path)}  {relative}\n")
    sums_path = directory / "SHA256SUMS"
    if sums_path.exists():
        raise LauncherError("local_output_exists")
    sums_path.write_text("".join(lines), encoding="utf-8")
    return sums_path


def _publish_staging(staging: Path, final: Path) -> None:
    if final.exists():
        raise LauncherError("local_output_exists")
    try:
        os.rename(str(staging), str(final))
    except OSError:
        raise LauncherError("artifact_tamper") from None
    if not final.is_dir():
        raise LauncherError("artifact_tamper")


def _create_private_staging_dir(parent: Path, run_id: str) -> Path:
    token = secrets.token_hex(8)
    staging = parent / f".staging_{run_id}_{token}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise LauncherError("local_output_exists") from None
    return staging


# ---------------------------------------------------------------------------
# Run manifest + launcher result.
# ---------------------------------------------------------------------------


def _local_hash_binding() -> dict[str, str]:
    return {
        "runner": _sha256_file(HERE / "d3q_slot_runner.py"),
        "cpu_validate": _sha256_file(HERE / "d3q_cpu_validate_remote.py"),
        "launcher": _sha256_file(HERE / "d3q_slot_launcher.py"),
        "budget": _sha256_file(HERE / "d3q_budget.py"),
        "manifest": _sha256_file(HERE / "FROZEN_MANIFEST.json"),
        "repair_template": _sha256_file(_local_file("D3Q_FROZEN_REPAIR_TEMPLATE.json")),
    }


def _observed_utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_git_object_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_head() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False
        )
        head = completed.stdout.strip()
    except Exception:
        head = ""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        branch = completed.stdout.strip()
    except Exception:
        branch = ""
    return {"commit": head if _is_git_object_id(head) else "", "branch": branch}

# ---------------------------------------------------------------------------
# Main launcher flow.
# ---------------------------------------------------------------------------

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def run_launcher(
    *,
    run_id: str,
    slots: Sequence[str],
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    remote_env_file: str,
    mason_worktree: str,
    mason_src: str,
    artifacts_dir: str | Path,
    runner: CommandRunner | None = None,
    now: datetime | None = None,
    deploy_check_only: bool = False,
) -> dict[str, Any]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise LauncherError("run_id_invalid")
    command_runner = runner or _default_runner
    target = _validate_ssh_target(ssh_target)
    python_path = _validate_remote_path(remote_python)
    env_path = _validate_remote_path(remote_env_file)
    worktree_path = _validate_remote_path(mason_worktree)
    src_path = _validate_remote_path(mason_src)
    observed = now or datetime.now(timezone.utc)
    exec_root = _exec_root_for(observed, run_id)
    artifacts = Path(artifacts_dir).resolve()

    binding = _local_hash_binding()
    manifest_raw_sha = binding["manifest"]
    if manifest_raw_sha != runner_mod.FROZEN_MANIFEST_RAW_SHA256:
        raise LauncherError("frozen_hash_mismatch")
    repair = json.loads(_local_file("D3Q_FROZEN_REPAIR_TEMPLATE.json").read_text(encoding="utf-8"))
    if repair.get("template_sha256") != runner_mod.FROZEN_REPAIR_TEMPLATE_SHA256:
        raise LauncherError("frozen_hash_mismatch")

    slot_specs = []
    for slot_id in slots:
        repeat, arm, prompt_index = runner_mod.parse_slot_id(slot_id)
        provider, model, _base = runner_mod.arm_to_provider_model(arm)
        slot_specs.append(
            {
                "slot_id": slot_id, "repeat": repeat, "arm": arm,
                "prompt_index": prompt_index, "provider": provider, "model": model,
            }
        )

    local_files = {
        name: _local_file(name)
        for name in DEPLOY_FILES
    }
    for name, path in local_files.items():
        if not path.is_file():
            raise LauncherError("artifact_tamper")

    staging = None
    published = False
    result: dict[str, Any] = {
        "classification": "D3Q_SLOT_LAUNCHER",
        "schema_version": 1,
        "run_id": run_id,
        "status": "BLOCKED",
        "reason": "remote_command_failed",
        "exec_root": exec_root,
        "deploy_check_only": bool(deploy_check_only),
        "slots": [spec["slot_id"] for spec in slot_specs],
        "remote_python": python_path,
        "remote_env_file": env_path,
        "credential_variable": "EXP_DEEPSEEK_API_KEY",
        "mason_worktree": worktree_path,
        "mason_worktree_commit": MASON_WORKTREE_COMMIT,
        "mason_src": src_path,
        "observed_utc": _observed_utc(observed),
        "git_head": _git_head(),
        "local_hash_binding": binding,
        "exec_root_absent_pre": None,
        "exec_root_absent_post": None,
        "deployed_hashes_verified": False,
        "deployed_hashes_post_verified": False,
        "gpu_pre": None,
        "gpu_post": None,
        "ollama_pre": None,
        "ollama_post": None,
        "ledger_pre": None,
        "ledger_post": None,
        "slot_results": {},
        "slot_summaries": {},
        "artifact_dir": None,
        "no_secret_scan": None,
        "artifact_internal_sha256": None,
        "cleanup_verified": False,
    }

    failure: LauncherError | None = None
    root_created = False
    deployed_pre: dict[str, str] | None = None
    collected_slots: list[dict[str, Any]] = []

    try:
        if (artifacts / run_id).exists():
            raise LauncherError("local_output_exists")

        gpu_pre = _gpu_snapshot(command_runner, target, ssh_key)
        result["gpu_pre"] = gpu_pre
        if gpu_pre["gpu2"]["uuid"] != EXPECTED_GPU2_UUID:
            raise LauncherError("gpu2_uuid_mismatch")
        if gpu_pre["gpu2"]["external_compute_pids"]:
            raise LauncherError("gpu2_external_app")
        result["ollama_pre"] = _ollama_snapshot(command_runner, target, ssh_key, python_path)
        qwen_pre = result["ollama_pre"]["models"].get(runner_mod.SMALL_MODEL, "")
        if not qwen_pre.startswith(EXPECTED_OLLAMA_QWEN_DIGEST_PREFIX):
            raise LauncherError("ollama_digest_changed")

        _remote_exec_root(command_runner, target, ssh_key, python_path, exec_root)
        root_created = True
        result["exec_root_absent_pre"] = True
        _deploy(command_runner, target, ssh_key, exec_root, local_files)
        deployed_pre = _verify_remote_hashes(
            command_runner, target, ssh_key, python_path, exec_root, local_files
        )
        result["deployed_hashes_verified"] = True

        if not deploy_check_only:
            local_slots_dir = artifacts / f".slots_{run_id}"
            for spec in slot_specs:
                summary = _remote_ledger_summary(
                    command_runner, target, ssh_key, python_path, exec_root
                )
                if spec == slot_specs[0]:
                    result["ledger_pre"] = summary
                _enforce_budget_before_slot(summary, spec["slot_id"], spec["provider"])
                _dispatch_slot(
                    command_runner, target, ssh_key,
                    remote_python=python_path,
                    remote_env_file=env_path,
                    exec_root=exec_root,
                    run_id=run_id,
                    slot_id=spec["slot_id"],
                    mason_src=src_path,
                )
                after = _remote_ledger_summary(
                    command_runner, target, ssh_key, python_path, exec_root
                )
                result["ledger_post"] = after
                _enforce_budget_after_slot(
                    after, spec["slot_id"], spec["provider"],
                    expected_slot_posts=(after.get("slot_counts") or {}).get(
                        spec["slot_id"], 0
                    ),
                )
                # Collect this slot's evidence immediately after completion so
                # a later mid-run failure cannot lose already-finished slot
                # artifacts (incident D3Q_PHASE2_INCIDENT_01).
                _collect_slot_dir(
                    command_runner, target, ssh_key, exec_root,
                    spec["slot_id"], local_slots_dir,
                )
                slot_result = _load_slot_result(
                    local_slots_dir / spec["slot_id"], spec["slot_id"]
                )
                collected_slots.append(
                    {"slot_id": spec["slot_id"], "result": slot_result}
                )
                # NOTE: deliberately no post-slot _enforce_budget_before_slot()
                # here.  A slot that legally exhausts its 3-POST budget is a
                # valid experimental outcome, not a violation; real overshoot
                # (> limit) is already caught by _enforce_budget_after_slot,
                # and the next iteration re-reads a fresh ledger summary for
                # the next slot's pre-check.

        # verify deployed hashes are unchanged post-run
        deployed_post = _verify_remote_hashes(
            command_runner, target, ssh_key, python_path, exec_root, local_files
        )
        if deployed_post != deployed_pre:
            raise LauncherError("artifact_tamper")
        result["deployed_hashes_post_verified"] = True

        # GPU + ollama unchanged
        gpu_post = _gpu_snapshot(command_runner, target, ssh_key)
        result["gpu_post"] = gpu_post
        if gpu_post["gpu2"]["external_compute_pids"]:
            raise LauncherError("gpu2_external_app")
        result["ollama_post"] = _ollama_snapshot(command_runner, target, ssh_key, python_path)
        _enforce_ollama_unchanged(result["ollama_pre"], result["ollama_post"])

        # cleanup
        _cleanup_exec_root(command_runner, target, ssh_key, python_path, exec_root)
        root_created = False
        result["exec_root_absent_post"] = True
        result["cleanup_verified"] = True

        for spec in slot_specs:
            for collected in collected_slots:
                if collected["slot_id"] == spec["slot_id"]:
                    spec["result"] = collected["result"]
                    break
        result["slot_summaries"] = {
            spec["slot_id"]: {
                "initial_valid": spec.get("result", {}).get("initial_valid"),
                "final_valid": spec.get("result", {}).get("final_valid"),
                "attempts": spec.get("result", {}).get("attempts"),
                "repair_requests": spec.get("result", {}).get("repair_requests"),
            }
            for spec in slot_specs
        }
        result["slot_results"] = {
            spec["slot_id"]: spec.get("result") for spec in slot_specs
        }
        result["status"] = "PASS"
        result["reason"] = None
    except LauncherError as exc:
        failure = exc
        result["status"] = "BLOCKED"
        result["reason"] = exc.reason
        if getattr(exc, "detail", None) is not None:
            result["remote_failure"] = exc.detail
    except Exception as exc:  # unexpected local fault: still fail closed
        failure = LauncherError("remote_command_failed")
        result["status"] = "BLOCKED"
        result["reason"] = failure.reason
        result["remote_failure"] = {
            "error_type": type(exc).__name__,
            "error_str": str(exc)[-2000:],
        }

    # Always attempt cleanup and preserve evidence.
    if root_created:
        try:
            _cleanup_exec_root(command_runner, target, ssh_key, python_path, exec_root)
            result["exec_root_absent_post"] = True
            result["cleanup_verified"] = True
        except LauncherError:
            result["exec_root_absent_post"] = False
            result["cleanup_verified"] = False

    # Publish artifacts (evidence even on failure).
    try:
        final_dir = artifacts / run_id
        staging = _create_private_staging_dir(artifacts, run_id)
        if deployed_pre is not None:
            result["deployed_hashes_pre"] = deployed_pre
        if collected_slots and not result["slot_results"]:
            result["slot_results"] = {
                item["slot_id"]: item["result"] for item in collected_slots
            }

        run_manifest = {
            "classification": "D3Q_RUN_MANIFEST",
            "schema_version": 1,
            "run_id": run_id,
            "slots": [spec["slot_id"] for spec in slot_specs],
            "git_head": result["git_head"],
            "frozen_manifest": {
                "file": "FROZEN_MANIFEST.json",
                "raw_sha256": binding["manifest"],
                "manifest_sha256": runner_mod.FROZEN_MANIFEST_SHA256,
            },
            "repair_template": {
                "file": "D3Q_FROZEN_REPAIR_TEMPLATE.json",
                "template_sha256": runner_mod.FROZEN_REPAIR_TEMPLATE_SHA256,
            },
            "hash_binding": binding,
            "mason_worktree": {
                "path": worktree_path,
                "commit": MASON_WORKTREE_COMMIT,
            },
            "remote_env_file": env_path,
            "credential_variable": "EXP_DEEPSEEK_API_KEY",
            "remote_python": python_path,
            "timeouts_frozen": {
                "connect_s": runner_mod.CONNECT_TIMEOUT_S,
                "read_s": runner_mod.READ_TIMEOUT_S,
            },
            "post_limits": {
                "per_slot": runner_mod.MAX_POSTS_PER_SLOT,
                "per_provider": runner_mod.MAX_PROVIDER_POSTS,
            },
            "max_repairs_per_slot": runner_mod.MAX_REPAIRS_PER_SLOT,
            "observed_utc": result["observed_utc"],
        }

        # copy collected slot dirs into staging
        if collected_slots:
            slots_parent = staging / "slots"
            slots_parent.mkdir(exist_ok=False)
            local_slots_dir = artifacts / f".slots_{run_id}"
            for item in collected_slots:
                source = local_slots_dir / item["slot_id"]
                shutil.copytree(source, slots_parent / item["slot_id"])

        _write_json(staging / "D3Q_RUN_MANIFEST.json", run_manifest)
        _write_json(staging / "D3Q_SLOT_LAUNCHER_RESULT.json", result)

        scan = _no_secret_scan_directory(staging)
        result["no_secret_scan"] = scan
        if not scan["passed"]:
            raise LauncherError("no_secret_scan_failed")
        # rewrite launcher result with the scan evidence included
        _write_json(staging / "D3Q_SLOT_LAUNCHER_RESULT.json", result)

        result["artifact_dir"] = f"d3q_artifacts/{run_id}"
        to_publish = {k: v for k, v in result.items() if k != "artifact_internal_sha256"}
        result["artifact_internal_sha256"] = runner_mod.canonical_json_sha256(to_publish)
        _write_json(staging / "D3Q_SLOT_LAUNCHER_RESULT.json", result)
        sums_path = _write_sha256sums(staging)
        _publish_staging(staging, final_dir)
        published = True
        staging = None
    except LauncherError as exc:
        if failure is None:
            failure = exc
            result["status"] = "BLOCKED"
            result["reason"] = exc.reason
        if staging is not None:
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
    finally:
        slots_parent = artifacts / f".slots_{run_id}"
        if slots_parent.exists():
            try:
                shutil.rmtree(slots_parent)
            except OSError:
                pass

    if published:
        result["status"] = "PASS" if result["status"] == "PASS" else "BLOCKED"
        sealed = {k: v for k, v in result.items() if k != "artifact_internal_sha256"}
        result["artifact_internal_sha256"] = runner_mod.canonical_json_sha256(sealed)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--slots", default="slot_r1_small_p00,slot_r1_large_p00",
        help="comma-separated slot ids to execute",
    )
    parser.add_argument("--ssh-key", default=SSH_KEY)
    parser.add_argument("--ssh-target", default=SSH_TARGET)
    parser.add_argument("--remote-python", default=REMOTE_PYTHON)
    parser.add_argument("--remote-env-file", default=REMOTE_ENV_FILE)
    parser.add_argument("--mason-worktree", default=MASON_WORKTREE)
    parser.add_argument(
        "--artifacts-dir",
        default=str(HERE / "d3q_artifacts"),
    )
    parser.add_argument("--deploy-check-only", action="store_true")
    args = parser.parse_args(argv)

    run_id = args.run_id
    if not run_id:
        run_id = "d3q_p1_smoke_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    slots = [slot.strip() for slot in args.slots.split(",") if slot.strip()]
    mason_src = args.mason_worktree.rstrip("/") + "/dicode_src/src"

    try:
        result = run_launcher(
            run_id=run_id,
            slots=slots,
            ssh_target=args.ssh_target,
            ssh_key=args.ssh_key,
            remote_python=args.remote_python,
            remote_env_file=args.remote_env_file,
            mason_worktree=args.mason_worktree,
            mason_src=mason_src,
            artifacts_dir=args.artifacts_dir,
            deploy_check_only=args.deploy_check_only,
        )
    except LauncherError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.reason}, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "run_id": result["run_id"],
        "artifact_dir": result["artifact_dir"],
        "cleanup_verified": result["cleanup_verified"],
    }
    if result.get("reason"):
        summary["reason"] = result["reason"]
    print(json.dumps(summary, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
