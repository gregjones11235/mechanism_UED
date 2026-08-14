#!/usr/bin/env python3
"""External integrity launcher for the D3 DeepSeek Flash metadata gate.

All process calls use argv arrays with ``shell=False``.  The launcher never
reads a credential and never places one in argv or retained output.  It writes
only beneath a validated unique remote ``/tmp`` root and a caller-declared
local output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import d3_deepseek_metadata_gate_remote as gate


CLASSIFICATION = "D3_DEEPSEEK_GATE_LAUNCHER"
SCHEMA_VERSION = 1
CANONICAL_ALGORITHM = "canonical_json_sha256"
CANONICAL_SCOPE = "ALL_FIELDS_EXCLUDING_ARTIFACT_SHA256"
REMOTE_ROOT_PREFIX = "/tmp/d3_deepseek_flash_gate_"
EXPECTED_GPU2_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"
MINIMUM_GPU2_FREE_MIB = 4096
LOCAL_ARTIFACT_NAME = "deepseek_metadata_gate_artifact.json"
LOCAL_RESULT_NAME = "deepseek_gate_launcher_result.json"
REMOTE_PROVIDER_NAME = "d3_deepseek_provider.py"
REMOTE_GATE_NAME = "d3_deepseek_metadata_gate_remote.py"
REMOTE_ARTIFACT_NAME = "deepseek_metadata_gate_artifact.json"

HASH_KEYS = frozenset({"tool", "provider"})
GPU_KEYS = frozenset(
    {"gpu_index", "uuid", "memory_free_mib", "external_compute_pids"}
)
RESULT_KEYS = frozenset(
    {
        "artifact_internal_sha256",
        "artifact_path",
        "artifact_request_count",
        "artifact_sha256",
        "artifact_sha256_algorithm",
        "artifact_sha256_scope",
        "artifact_status",
        "classification",
        "cleanup_verified",
        "completion_requests",
        "embedding_requests",
        "external_artifact_hash_verified",
        "external_execution_hashes_verified",
        "gpu_post",
        "gpu_pre",
        "local_artifact_sha256",
        "manifest_sha256",
        "model",
        "base_url",
        "credential_variable",
        "observed_utc",
        "post_execution_sha256",
        "pre_execution_sha256",
        "reason",
        "remote_artifact_sha256",
        "remote_root",
        "schema_version",
        "status",
    }
)
ALLOWED_REASONS = frozenset(
    {
        "artifact_gate_blocked",
        "artifact_request_count_invalid",
        "artifact_tamper",
        "cleanup_failed",
        "gpu2_external_app",
        "gpu2_insufficient_free",
        "gpu2_uuid_mismatch",
        "gpu_unavailable",
        "hash_mismatch",
        "local_output_exists",
        "remote_command_failed",
        "remote_root_exists",
        "secret_like_output",
        "unexpected_remote_output",
    }
)

_REMOTE_ROOT_RE = re.compile(
    r"^/tmp/d3_deepseek_flash_gate_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}$"
)
_SSH_TARGET_RE = re.compile(r"^(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9_.-]+$")
_SAFE_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
_SENSITIVE_OUTPUT_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)\bAuthorization\s*:\s*\S+"),
    re.compile(r"(?i)EXP_DEEPSEEK_API_KEY\s*="),
    re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{16,}"),
)

_EXISTENCE_SCRIPT = (
    "import pathlib,sys;"
    "print('EXISTS' if pathlib.Path(sys.argv[1]).exists() else 'ABSENT')"
)
_CREATE_SCRIPT = (
    "import pathlib,sys;"
    "pathlib.Path(sys.argv[1]).mkdir(mode=0o700);"
    "print('CREATED')"
)
_HASH_FILES_SCRIPT = (
    "import hashlib,json,pathlib,sys;"
    "h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();"
    "print(json.dumps({'tool':h(sys.argv[1]),'provider':h(sys.argv[2])},"
    "sort_keys=True,separators=(',',':')))"
)
_HASH_ONE_SCRIPT = (
    "import hashlib,pathlib,sys;"
    "print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())"
)
_CLEANUP_SCRIPT = """import pathlib,re,shutil,sys
p=sys.argv[1]
if not re.fullmatch(r'/tmp/d3_deepseek_flash_gate_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}',p):
    raise SystemExit(3)
path=pathlib.Path(p)
if path.exists():
    shutil.rmtree(path)
print('REMOVED')
"""


class LauncherError(RuntimeError):
    """Sanitized failure carrying only a fixed classification."""

    def __init__(self, reason: str):
        self.reason = reason if reason in ALLOWED_REASONS else "remote_command_failed"
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[Sequence[str]], CommandResult]


def _default_runner(argv: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=60,
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
        raise LauncherError("hash_mismatch") from None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_remote_root(value: str) -> str:
    if not _REMOTE_ROOT_RE.fullmatch(value) or not value.startswith(REMOTE_ROOT_PREFIX):
        raise LauncherError("cleanup_failed")
    return value


def _validate_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not _SAFE_REMOTE_PATH_RE.fullmatch(value)
        or not path.is_absolute()
        or ".." in path.parts
    ):
        raise LauncherError("remote_command_failed")
    return value


def _validate_ssh_target(value: str) -> str:
    if not _SSH_TARGET_RE.fullmatch(value):
        raise LauncherError("remote_command_failed")
    return value


def _remote_root(now: datetime, token: str) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise LauncherError("remote_command_failed")
    if not re.fullmatch(r"[0-9a-f]{12}", token):
        raise LauncherError("remote_command_failed")
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _validate_remote_root(REMOTE_ROOT_PREFIX + stamp + "_" + token)


def _contains_sensitive_output(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SENSITIVE_OUTPUT_PATTERNS)


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
        result = runner(list(argv))
    except Exception:
        raise LauncherError("remote_command_failed") from None
    if _contains_sensitive_output(result.stdout) or _contains_sensitive_output(result.stderr):
        raise LauncherError("secret_like_output")
    if result.returncode not in allowed_returncodes:
        raise LauncherError("remote_command_failed")
    return result


def _ssh_base(ssh_target: str, ssh_key: str | Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "--",
        ssh_target,
    ]


def _scp_base(ssh_key: str | Path) -> list[str]:
    return [
        "scp",
        "-q",
        "-i",
        str(ssh_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "--",
    ]


def _ssh_argv(
    ssh_target: str,
    ssh_key: str | Path,
    remote_command: Sequence[str],
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


def _expect_exact(result: CommandResult, expected: str) -> None:
    if result.stderr or result.stdout.strip() != expected:
        raise LauncherError("unexpected_remote_output")


def _parse_json_object(result: CommandResult) -> dict[str, Any]:
    if result.stderr:
        raise LauncherError("unexpected_remote_output")
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise LauncherError("unexpected_remote_output") from None
    if not isinstance(decoded, dict):
        raise LauncherError("unexpected_remote_output")
    return decoded


def _parse_hashes(result: CommandResult) -> dict[str, str]:
    decoded = _parse_json_object(result)
    if set(decoded) != HASH_KEYS or not all(_is_sha256(decoded[key]) for key in HASH_KEYS):
        raise LauncherError("unexpected_remote_output")
    return {key: decoded[key] for key in sorted(HASH_KEYS)}


def _parse_gpu_snapshot(gpu_result: CommandResult, apps_result: CommandResult) -> dict[str, Any]:
    if gpu_result.stderr or apps_result.stderr:
        raise LauncherError("gpu_unavailable")
    rows: list[tuple[int, str, int]] = []
    for raw in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise LauncherError("gpu_unavailable")
        try:
            rows.append((int(parts[0]), parts[1], int(float(parts[2]))))
        except ValueError:
            raise LauncherError("gpu_unavailable") from None
    gpu2 = next((row for row in rows if row[0] == 2), None)
    if gpu2 is None:
        raise LauncherError("gpu_unavailable")
    pids: list[int] = []
    for raw in apps_result.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise LauncherError("gpu_unavailable")
        if parts[0] == gpu2[1] and parts[1] not in {"", "N/A", "[N/A]"}:
            try:
                pids.append(int(parts[1]))
            except ValueError:
                raise LauncherError("gpu_unavailable") from None
    snapshot = {
        "gpu_index": 2,
        "uuid": gpu2[1],
        "memory_free_mib": gpu2[2],
        "external_compute_pids": sorted(set(pids)),
    }
    return snapshot


def _enforce_gpu_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot["uuid"] != EXPECTED_GPU2_UUID:
        raise LauncherError("gpu2_uuid_mismatch")
    if snapshot["memory_free_mib"] < MINIMUM_GPU2_FREE_MIB:
        raise LauncherError("gpu2_insufficient_free")
    if snapshot["external_compute_pids"]:
        raise LauncherError("gpu2_external_app")


def _gpu_snapshot(
    runner: CommandRunner, ssh_target: str, ssh_key: str | Path
) -> dict[str, Any]:
    try:
        gpu = _run_command(
            runner,
            _ssh_argv(
                ssh_target,
                ssh_key,
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.free",
                    "--format=csv,noheader,nounits",
                ],
            ),
        )
        apps = _run_command(
            runner,
            _ssh_argv(
                ssh_target,
                ssh_key,
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
            ),
        )
    except LauncherError as exc:
        if exc.reason == "secret_like_output":
            raise
        raise LauncherError("gpu_unavailable") from None
    return _parse_gpu_snapshot(gpu, apps)


def _canonical_sha256(value: Any) -> str:
    return gate.canonical_json_sha256(value)


def _base_result(
    *, remote_root: str, manifest: Mapping[str, str], observed_utc: str
) -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "schema_version": SCHEMA_VERSION,
        "model": gate.provider.DEEPSEEK_MODEL_ID,
        "base_url": gate.provider.DEEPSEEK_BASE_URL,
        "credential_variable": gate.CREDENTIAL_VARIABLE,
        "status": "BLOCKED",
        "reason": "remote_command_failed",
        "remote_root": remote_root,
        "manifest_sha256": dict(manifest),
        "pre_execution_sha256": None,
        "post_execution_sha256": None,
        "remote_artifact_sha256": None,
        "local_artifact_sha256": None,
        "artifact_path": None,
        "artifact_internal_sha256": None,
        "artifact_status": None,
        "artifact_request_count": None,
        "external_execution_hashes_verified": False,
        "external_artifact_hash_verified": False,
        "gpu_pre": None,
        "gpu_post": None,
        "completion_requests": 0,
        "embedding_requests": 0,
        "cleanup_verified": False,
        "observed_utc": observed_utc,
        "artifact_sha256_algorithm": CANONICAL_ALGORITHM,
        "artifact_sha256_scope": CANONICAL_SCOPE,
    }


def _require_exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LauncherError("artifact_tamper")
    return value


def verify_launcher_result(value: Any) -> dict[str, Any]:
    result = _require_exact_mapping(value, RESULT_KEYS, "launcher result")
    if result["classification"] != CLASSIFICATION or result["schema_version"] != SCHEMA_VERSION:
        raise LauncherError("artifact_tamper")
    if (
        result["model"] != gate.provider.DEEPSEEK_MODEL_ID
        or result["base_url"] != gate.provider.DEEPSEEK_BASE_URL
        or result["credential_variable"] != gate.CREDENTIAL_VARIABLE
    ):
        raise LauncherError("artifact_tamper")
    if type(result["schema_version"]) is not int:
        raise LauncherError("artifact_tamper")
    if result["artifact_sha256_algorithm"] != CANONICAL_ALGORITHM:
        raise LauncherError("artifact_tamper")
    if result["artifact_sha256_scope"] != CANONICAL_SCOPE:
        raise LauncherError("artifact_tamper")
    if not _is_sha256(result["artifact_sha256"]):
        raise LauncherError("artifact_tamper")
    payload = {key: item for key, item in result.items() if key != "artifact_sha256"}
    if _canonical_sha256(payload) != result["artifact_sha256"]:
        raise LauncherError("artifact_tamper")
    manifest = _require_exact_mapping(result["manifest_sha256"], HASH_KEYS, "manifest")
    if not all(_is_sha256(manifest[key]) for key in HASH_KEYS):
        raise LauncherError("artifact_tamper")
    for key in ("pre_execution_sha256", "post_execution_sha256"):
        if result[key] is not None:
            hashes = _require_exact_mapping(result[key], HASH_KEYS, key)
            if not all(_is_sha256(hashes[name]) for name in HASH_KEYS):
                raise LauncherError("artifact_tamper")
    for key in ("gpu_pre", "gpu_post"):
        if result[key] is not None:
            snapshot = _require_exact_mapping(result[key], GPU_KEYS, key)
            if (
                type(snapshot["gpu_index"]) is not int
                or type(snapshot["memory_free_mib"]) is not int
                or not isinstance(snapshot["uuid"], str)
                or not isinstance(snapshot["external_compute_pids"], list)
                or not all(type(pid) is int for pid in snapshot["external_compute_pids"])
            ):
                raise LauncherError("artifact_tamper")
    if result["status"] not in {"PASS", "BLOCKED"}:
        raise LauncherError("artifact_tamper")
    if result["status"] == "PASS" and result["reason"] is not None:
        raise LauncherError("artifact_tamper")
    if result["status"] == "BLOCKED" and result["reason"] not in ALLOWED_REASONS:
        raise LauncherError("artifact_tamper")
    for flag in (
        "external_execution_hashes_verified",
        "external_artifact_hash_verified",
        "cleanup_verified",
    ):
        if type(result[flag]) is not bool:
            raise LauncherError("artifact_tamper")
    if result["completion_requests"] != 0 or result["embedding_requests"] != 0:
        raise LauncherError("artifact_tamper")
    if (
        type(result["completion_requests"]) is not int
        or type(result["embedding_requests"]) is not int
    ):
        raise LauncherError("artifact_tamper")
    _validate_remote_root(result["remote_root"])
    if not isinstance(result["observed_utc"], str) or not result["observed_utc"].endswith("Z"):
        raise LauncherError("artifact_tamper")
    for key in ("remote_artifact_sha256", "local_artifact_sha256"):
        if result[key] is not None and not _is_sha256(result[key]):
            raise LauncherError("artifact_tamper")
    if result["artifact_internal_sha256"] is not None and not _is_sha256(
        result["artifact_internal_sha256"]
    ):
        raise LauncherError("artifact_tamper")
    if result["artifact_request_count"] is not None and type(
        result["artifact_request_count"]
    ) is not int:
        raise LauncherError("artifact_tamper")
    if result["artifact_path"] is not None and result["artifact_path"] != LOCAL_ARTIFACT_NAME:
        raise LauncherError("artifact_tamper")
    if result["artifact_status"] is not None and result["artifact_status"] not in {
        "PASS",
        "BLOCKED",
    }:
        raise LauncherError("artifact_tamper")
    if result["status"] == "PASS":
        if not all(
            result[key] is True
            for key in (
                "external_execution_hashes_verified",
                "external_artifact_hash_verified",
                "cleanup_verified",
            )
        ):
            raise LauncherError("artifact_tamper")
        if result["artifact_status"] != "PASS" or result["artifact_request_count"] != 1:
            raise LauncherError("artifact_tamper")
    return result


def _seal_result(result: dict[str, Any]) -> dict[str, Any]:
    sealed = gate.canonical(result)
    sealed["artifact_sha256"] = _canonical_sha256(sealed)
    return verify_launcher_result(sealed)


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    verify_launcher_result(dict(result))
    text = json.dumps(gate.canonical(result), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise LauncherError("local_output_exists") from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _observed_utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_launcher(
    *,
    ssh_target: str,
    ssh_key: str | Path,
    remote_python: str,
    remote_env_file: str,
    local_output_dir: str | Path,
    runner: CommandRunner | None = None,
    now: datetime | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    command_runner = runner or _default_runner
    target = _validate_ssh_target(ssh_target)
    python_path = _validate_remote_path(remote_python)
    env_path = _validate_remote_path(remote_env_file)
    observed = now or datetime.now(timezone.utc)
    root = _remote_root(observed, token or secrets.token_hex(6))
    output_dir = Path(local_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / LOCAL_ARTIFACT_NAME
    result_path = output_dir / LOCAL_RESULT_NAME

    source_dir = Path(__file__).resolve().parent
    provider_source = source_dir / REMOTE_PROVIDER_NAME
    gate_source = source_dir / REMOTE_GATE_NAME
    manifest = {
        "tool": _sha256_file(gate_source),
        "provider": _sha256_file(provider_source),
    }
    result = _base_result(
        remote_root=root,
        manifest=manifest,
        observed_utc=_observed_utc(observed),
    )
    root_created = False
    failure: LauncherError | None = None
    remote_gate = root + "/" + REMOTE_GATE_NAME
    remote_provider = root + "/" + REMOTE_PROVIDER_NAME
    remote_artifact = root + "/" + REMOTE_ARTIFACT_NAME

    try:
        if artifact_path.exists() or result_path.exists():
            raise LauncherError("local_output_exists")
        exists = _run_command(
            command_runner,
            _remote_python_argv(target, ssh_key, python_path, _EXISTENCE_SCRIPT, root),
        )
        if not exists.stderr and exists.stdout.strip() == "EXISTS":
            raise LauncherError("remote_root_exists")
        _expect_exact(exists, "ABSENT")
        created = _run_command(
            command_runner,
            _remote_python_argv(target, ssh_key, python_path, _CREATE_SCRIPT, root),
        )
        _expect_exact(created, "CREATED")
        root_created = True

        for local_source, remote_name in (
            (provider_source, REMOTE_PROVIDER_NAME),
            (gate_source, REMOTE_GATE_NAME),
        ):
            deployed = _run_command(
                command_runner,
                _scp_base(ssh_key)
                + [str(local_source), f"{target}:{root}/{remote_name}"],
            )
            if deployed.stdout or deployed.stderr:
                raise LauncherError("unexpected_remote_output")

        pre_hash_result = _run_command(
            command_runner,
            _remote_python_argv(
                target,
                ssh_key,
                python_path,
                _HASH_FILES_SCRIPT,
                remote_gate,
                remote_provider,
            ),
        )
        result["pre_execution_sha256"] = _parse_hashes(pre_hash_result)
        try:
            gate.verify_external_execution_hashes(
                result["pre_execution_sha256"],
                result["pre_execution_sha256"],
                manifest,
            )
        except gate.GateArtifactError:
            raise LauncherError("hash_mismatch") from None

        result["gpu_pre"] = _gpu_snapshot(command_runner, target, ssh_key)
        _enforce_gpu_snapshot(result["gpu_pre"])

        gate_run = _run_command(
            command_runner,
            _ssh_argv(
                target,
                ssh_key,
                [
                    python_path,
                    remote_gate,
                    "--env-file",
                    env_path,
                    "--output",
                    remote_artifact,
                ],
            ),
            allowed_returncodes=frozenset({0, 2}),
        )
        stdout_artifact = _parse_json_object(gate_run)
        gate.verify_artifact(stdout_artifact)
        result["artifact_internal_sha256"] = stdout_artifact["artifact_sha256"]
        result["artifact_status"] = stdout_artifact["status"]
        result["artifact_request_count"] = stdout_artifact["request_count"]
        if stdout_artifact["request_count"] != 1:
            raise LauncherError("artifact_request_count_invalid")
        if stdout_artifact["completion_requests"] != 0 or stdout_artifact[
            "embedding_requests"
        ] != 0:
            raise LauncherError("artifact_tamper")

        result["gpu_post"] = _gpu_snapshot(command_runner, target, ssh_key)
        _enforce_gpu_snapshot(result["gpu_post"])

        post_hash_result = _run_command(
            command_runner,
            _remote_python_argv(
                target,
                ssh_key,
                python_path,
                _HASH_FILES_SCRIPT,
                remote_gate,
                remote_provider,
            ),
        )
        result["post_execution_sha256"] = _parse_hashes(post_hash_result)
        try:
            gate.verify_external_execution_hashes(
                result["pre_execution_sha256"],
                result["post_execution_sha256"],
                manifest,
            )
        except gate.GateArtifactError:
            raise LauncherError("hash_mismatch") from None
        result["external_execution_hashes_verified"] = True

        artifact_hash_result = _run_command(
            command_runner,
            _remote_python_argv(
                target,
                ssh_key,
                python_path,
                _HASH_ONE_SCRIPT,
                remote_artifact,
            ),
        )
        if artifact_hash_result.stderr or not _is_sha256(artifact_hash_result.stdout.strip()):
            raise LauncherError("unexpected_remote_output")
        result["remote_artifact_sha256"] = artifact_hash_result.stdout.strip()

        downloaded = _run_command(
            command_runner,
            _scp_base(ssh_key) + [f"{target}:{remote_artifact}", str(artifact_path)],
        )
        if downloaded.stdout or downloaded.stderr:
            raise LauncherError("unexpected_remote_output")
        result["local_artifact_sha256"] = _sha256_file(artifact_path)
        if result["local_artifact_sha256"] != result["remote_artifact_sha256"]:
            raise LauncherError("hash_mismatch")
        gate.verify_external_artifact_hash(
            artifact_path, result["remote_artifact_sha256"]
        )
        result["external_artifact_hash_verified"] = True
        local_artifact = gate.load_artifact(artifact_path)
        if local_artifact != stdout_artifact:
            raise LauncherError("artifact_tamper")
        result["artifact_path"] = LOCAL_ARTIFACT_NAME
        if local_artifact["request_count"] != 1:
            raise LauncherError("artifact_request_count_invalid")
        if local_artifact["completion_requests"] != 0 or local_artifact["embedding_requests"] != 0:
            raise LauncherError("artifact_tamper")
        if local_artifact["status"] != "PASS":
            raise LauncherError("artifact_gate_blocked")
    except LauncherError as exc:
        failure = exc
    except (gate.GateArtifactError, OSError, ValueError, TypeError):
        failure = LauncherError("artifact_tamper")
    finally:
        if root_created:
            try:
                _validate_remote_root(root)
                removed = _run_command(
                    command_runner,
                    _remote_python_argv(
                        target, ssh_key, python_path, _CLEANUP_SCRIPT, root
                    ),
                )
                _expect_exact(removed, "REMOVED")
                absent = _run_command(
                    command_runner,
                    _remote_python_argv(
                        target, ssh_key, python_path, _EXISTENCE_SCRIPT, root
                    ),
                )
                _expect_exact(absent, "ABSENT")
                result["cleanup_verified"] = True
            except LauncherError:
                failure = LauncherError("cleanup_failed")

    if failure is None:
        result["status"] = "PASS"
        result["reason"] = None
    else:
        result["status"] = "BLOCKED"
        result["reason"] = failure.reason
    sealed = _seal_result(result)
    _write_result(result_path, sealed)
    return sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the external DeepSeek metadata gate")
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-key", required=True, type=Path)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--remote-env-file", required=True)
    parser.add_argument("--local-output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_launcher(
        ssh_target=args.ssh_target,
        ssh_key=args.ssh_key,
        remote_python=args.remote_python,
        remote_env_file=args.remote_env_file,
        local_output_dir=args.local_output_dir,
    )
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
