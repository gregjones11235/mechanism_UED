#!/usr/bin/env python3
"""CPU-only replay of production worker-to-main candidate validation.

The public process verifies the frozen perf48 manifest and starts cache-off and
cache-on arms in separate child processes.  Each child binds the real
``EnvGenerator._check_compilation_uncached`` method, validates every frozen
candidate once in worker order and once again in main order, and records only
code hashes and validation evidence (never source code).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping


CLASSIFICATION = "VALIDATION_CACHE_REPLAY"
ARMS = ("off", "on")
PHASES = ("worker", "main")


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_manifest = _load_sibling(
    "perf48_combo_manifest_validation_replay", "perf48_combo_manifest.py"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return _sha256_bytes(payload)


def _hashed_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("result_sha256", None)
    result["result_sha256"] = _fingerprint(result)
    return result


def _atomic_json(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = _hashed_document(document)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_canonical(result), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return result


def _load_hashed_json(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = document.get("result_sha256")
    if not expected or _hashed_document(document)["result_sha256"] != expected:
        raise ValueError(f"result hash mismatch: {path}")
    return document


def load_replay_inputs(
    manifest_path: str | Path, stage_name: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Strictly load frozen material and return ordered in-memory candidates."""
    manifest = _manifest.load_manifest(manifest_path)
    try:
        stage = next(item for item in manifest["stages"] if item["name"] == stage_name)
    except StopIteration as exc:
        raise ValueError(f"stage {stage_name!r} absent from manifest") from exc
    task_ids = [str(value) for value in stage["task_ids"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("candidate task order contains duplicates")
    task_rows = {str(row["id"]): row for row in stage["tasks"]}
    candidate_entries = stage["candidate_codes"]
    candidates: list[dict[str, Any]] = []
    for task_id in task_ids:
        if task_id not in candidate_entries or task_id not in task_rows:
            raise ValueError(f"candidate {task_id!r} missing from frozen archive")
        entry = candidate_entries[task_id]
        path = Path(entry["path"])
        code_bytes = path.read_bytes()
        file_hash = _sha256_bytes(code_bytes)
        if file_hash != entry["sha256"]:
            raise ValueError(f"candidate code hash changed for {task_id}")
        try:
            code = code_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"candidate code is not UTF-8 for {task_id}") from exc
        if not code.strip():
            raise ValueError(f"empty candidate code for {task_id}")
        # Candidate mirrors written on Windows may contain CRLF while GraphML
        # stores LF.  Production's validation key has the same normalization.
        normalized = code.replace("\r\n", "\n").replace("\r", "\n")
        code_hash = _sha256_bytes(normalized.encode())
        if code_hash != task_rows[task_id]["code_sha256"]:
            raise ValueError(f"candidate/archive code hash mismatch for {task_id}")
        candidates.append({
            "id": task_id,
            "code_sha256": code_hash,
            "file_sha256": file_hash,
            "code": code,
        })
    hashes = [item["code_sha256"] for item in candidates]
    if len(hashes) != len(set(hashes)):
        raise ValueError("validation replay requires unique candidate code hashes")
    graph_path = Path(stage["graph"]["path"])
    if _file_sha256(graph_path) != stage["graph"]["sha256"]:
        raise ValueError("frozen archive graph hash changed")
    for section in manifest["source_config"].values():
        for entry in section.values():
            if _file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("frozen source/config hash changed")
    return manifest, stage, candidates


def _assert_cpu_environment() -> None:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("JAX_PLATFORMS must be exactly cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")


def _block_network() -> None:
    """Fail closed if an imported dependency attempts outbound networking."""
    class GuardedSocket(socket.socket):
        def connect(self, *args, **kwargs):  # pragma: no cover - only on violation
            raise RuntimeError("network disabled for validation replay")

        def connect_ex(self, *args, **kwargs):  # pragma: no cover - only on violation
            raise RuntimeError("network disabled for validation replay")

    def blocked(*args, **kwargs):  # pragma: no cover - only on violation
        raise RuntimeError("network disabled for validation replay")

    socket.socket = GuardedSocket
    socket.create_connection = blocked


def _validator_source_entry(
    manifest: Mapping[str, Any], env_generator_class: type
) -> tuple[Path, str]:
    source = inspect.getsourcefile(env_generator_class)
    if not source:
        raise RuntimeError("EnvGenerator source is not inspectable")
    source_path = Path(source).resolve()
    matches = [
        entry
        for entry in manifest["source_config"]["source"].values()
        if Path(entry["path"]).resolve() == source_path
    ]
    if not matches:
        raise RuntimeError("EnvGenerator source is not bound by frozen manifest")
    actual = _file_sha256(source_path)
    if any(entry["sha256"] != actual for entry in matches):
        raise RuntimeError("EnvGenerator frozen source hash mismatch")
    return source_path, actual


def _new_real_env_generator(
    cache_enabled: bool, max_entries: int, source_sha: str
) -> tuple[Any, type, dict[str, Any]]:
    _assert_cpu_environment()
    from dicode.dreaming.gen_manager import EnvGenerator, VALIDATOR_CACHE_VERSION
    from dicode.runtime_analysis import tracker

    tracker.configure(enabled=False, reset=True)
    instance = EnvGenerator.__new__(EnvGenerator)
    instance.performance = {
        "validation_cache": bool(cache_enabled),
        "validation_cache_max_entries": max(1, int(max_entries)),
        "validation_static_lint": False,
    }
    instance._validation_cache = OrderedDict()
    instance._validation_cache_lock = threading.RLock()
    instance._validation_inflight = {}
    instance._validation_source_sha = str(source_sha)
    bound = instance._check_compilation_uncached
    if (
        getattr(bound, "__self__", None) is not instance
        or getattr(bound, "__func__", None) is not EnvGenerator._check_compilation_uncached
    ):
        raise RuntimeError("uncached validator is not the real bound production method")
    evidence = {
        "real_method_bound": True,
        "method_module": EnvGenerator._check_compilation_uncached.__module__,
        "method_qualname": EnvGenerator._check_compilation_uncached.__qualname__,
        "method_source_sha256": _sha256_bytes(
            inspect.getsource(EnvGenerator._check_compilation_uncached).encode()
        ),
        "validator_cache_version": VALIDATOR_CACHE_VERSION,
    }
    return instance, EnvGenerator, evidence


class _RealMethodCounter:
    def __init__(self, method):
        self._code = method.__code__
        self.count = 0
        self._previous = None

    def _profile(self, frame, event, arg):
        if event == "call" and frame.f_code is self._code:
            self.count += 1

    def __enter__(self):
        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, traceback):
        sys.setprofile(self._previous)


def _error_class(success: bool, error: str) -> str | None:
    if success:
        return None
    if str(error).startswith("Compilation error:"):
        return "CompilationError"
    return "ValidationError"


def _run_phase(
    env_generator: Any,
    candidates: list[dict[str, Any]],
    phase: str,
    counter: _RealMethodCounter,
    request_offset: int = 0,
) -> list[dict[str, Any]]:
    if phase not in PHASES:
        raise ValueError(f"invalid validation phase {phase!r}")
    rows = []
    for index, candidate in enumerate(candidates):
        code = candidate["code"]
        key = env_generator._validation_key(code)
        if key[0] != candidate["code_sha256"]:
            raise RuntimeError("validator key/code hash mismatch")
        before = counter.count
        started = time.perf_counter_ns()
        success, error = env_generator.check_compilation(code)
        elapsed = max(0, time.perf_counter_ns() - started) / 1e9
        delta = counter.count - before
        if delta not in (0, 1):
            raise RuntimeError("unexpected uncached validator call delta")
        rows.append({
            "request_index": request_offset + index,
            "candidate_id": candidate["id"],
            "code_sha256": candidate["code_sha256"],
            "phase": phase,
            "success": bool(success),
            "error_class": _error_class(bool(success), str(error)),
            "wall_s": elapsed,
            "uncached_call_delta": delta,
            "uncached_call_count": counter.count,
            "cache_hit": delta == 0,
            "validator_key": {
                "code_sha256": key[0],
                "validator_cache_version": str(key[1]),
                "jax_version": str(key[2]),
                "source_sha": str(key[3]),
                "fingerprint": _fingerprint(list(key)),
            },
        })
    return rows


def _run_arm(
    manifest_path: str | Path, stage_name: str, arm: str, output: str | Path
) -> dict[str, Any]:
    _assert_cpu_environment()
    _block_network()
    manifest, stage, candidates = load_replay_inputs(manifest_path, stage_name)
    import jax
    from dicode.dreaming.gen_manager import EnvGenerator

    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise RuntimeError("validation replay acquired a non-CPU JAX device")
    source_path, source_sha = _validator_source_entry(manifest, EnvGenerator)
    cache_enabled = arm == "on"
    env_generator, env_class, method_evidence = _new_real_env_generator(
        cache_enabled, len(candidates), source_sha
    )
    with _RealMethodCounter(env_class._check_compilation_uncached) as counter:
        worker = _run_phase(env_generator, candidates, "worker", counter)
        main = _run_phase(
            env_generator, candidates, "main", counter, request_offset=len(worker)
        )
    expected_uncached = len(candidates) if cache_enabled else 2 * len(candidates)
    if counter.count != expected_uncached:
        raise RuntimeError(
            f"{arm} uncached count {counter.count} != {expected_uncached}"
        )
    if any(row["cache_hit"] for row in worker):
        raise RuntimeError("worker validation unexpectedly hit cache")
    expected_main_hit = cache_enabled
    if any(row["cache_hit"] is not expected_main_hit for row in main):
        raise RuntimeError("main validation cache-hit contract mismatch")
    requests = worker + main
    wall = _wall_summary(requests)
    result = {
        "classification": CLASSIFICATION,
        "arm": arm,
        "cache_enabled": cache_enabled,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "manifest_sha256": manifest["manifest_sha256"],
        "stage": stage_name,
        "archive": {
            "path": stage["graph"]["path"],
            "sha256": stage["graph"]["sha256"],
        },
        "candidate_count": len(candidates),
        "candidate_order_sha256": _fingerprint([
            {"id": item["id"], "code_sha256": item["code_sha256"]}
            for item in candidates
        ]),
        "requests": requests,
        **wall,
        "uncached_call_count": counter.count,
        "expected_uncached_call_count": expected_uncached,
        "success_count": sum(row["success"] for row in worker + main),
        "failure_count": sum(not row["success"] for row in worker + main),
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "platforms": sorted({device.platform for device in jax.devices()}),
            "device_count": len(jax.devices()),
            "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "source": {"path": str(source_path), "sha256": source_sha},
        "validator": method_evidence,
        "network_guard": True,
        "llm_api_calls": 0,
        "full_code_recorded": False,
    }
    return _atomic_json(output, result)


def _child_environment(source_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
    })
    source = str(source_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return env


def _request_semantics(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{
        "request_index": row["request_index"],
        "candidate_id": row["candidate_id"],
        "code_sha256": row["code_sha256"],
        "phase": row["phase"],
        "success": row["success"],
        "error_class": row["error_class"],
        "validator_key": row["validator_key"],
    } for row in document["requests"]]


def _wall_summary(requests: list[Mapping[str, Any]]) -> dict[str, float]:
    worker_wall_s = sum(
        float(row["wall_s"]) for row in requests if row["phase"] == "worker"
    )
    main_wall_s = sum(
        float(row["wall_s"]) for row in requests if row["phase"] == "main"
    )
    return {
        "worker_wall_s": worker_wall_s,
        "main_wall_s": main_wall_s,
        "total_wall_s": worker_wall_s + main_wall_s,
    }


def _speedup(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    manifest, stage, candidates = load_replay_inputs(args.manifest, args.stage)
    root = Path(args.out)
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    root.mkdir(parents=True)
    commands = []
    arms: dict[str, dict[str, Any]] = {}
    script = Path(__file__).resolve()
    source_root = script.parents[2]
    env = _child_environment(source_root)
    try:
        for arm in ARMS:
            arm_path = root / f"{arm}.json"
            command = [
                str(args.python), str(script), "--manifest", str(args.manifest),
                "--stage", args.stage, "--out", str(arm_path),
                "--internal-arm", arm,
            ]
            commands.append(command)
            completed = subprocess.run(
                command, env=env, text=False, capture_output=True, check=False
            )
            command_evidence = {
                "arm": arm,
                "argv": command,
                "returncode": completed.returncode,
                "stdout_bytes": len(completed.stdout),
                "stderr_bytes": len(completed.stderr),
                "stdout_sha256": _sha256_bytes(completed.stdout),
                "stderr_sha256": _sha256_bytes(completed.stderr),
            }
            if completed.returncode != 0:
                raise RuntimeError(
                    f"validation replay arm {arm} failed: {command_evidence}"
                )
            document = _load_hashed_json(arm_path)
            if document.get("arm") != arm:
                raise RuntimeError("validation arm result mismatch")
            document["subprocess"] = command_evidence
            arms[arm] = document
        # Re-validate immutable frozen inputs after both child processes.
        post_manifest, post_stage, post_candidates = load_replay_inputs(
            args.manifest, args.stage
        )
        if post_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError("manifest changed during replay")
        if post_stage["graph"] != stage["graph"] or [
            (item["id"], item["code_sha256"]) for item in post_candidates
        ] != [(item["id"], item["code_sha256"]) for item in candidates]:
            raise RuntimeError("archive/candidates changed during replay")
        if _request_semantics(arms["off"]) != _request_semantics(arms["on"]):
            raise RuntimeError("cache off/on validation semantics or order differ")
        candidate_count = len(candidates)
        if arms["off"]["uncached_call_count"] != 2 * candidate_count:
            raise RuntimeError("cache-off uncached count mismatch")
        if arms["on"]["uncached_call_count"] != candidate_count:
            raise RuntimeError("cache-on uncached count mismatch")
        off_wall = _wall_summary(arms["off"]["requests"])
        on_wall = _wall_summary(arms["on"]["requests"])
        for arm, wall in (("off", off_wall), ("on", on_wall)):
            if any(arms[arm][key] != value for key, value in wall.items()):
                raise RuntimeError(f"cache-{arm} wall summary mismatch")
        main_wall_s_avoided = off_wall["main_wall_s"] - on_wall["main_wall_s"]
        total_wall_s_avoided = off_wall["total_wall_s"] - on_wall["total_wall_s"]
        result = {
            "classification": CLASSIFICATION,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(Path(args.manifest).resolve()),
            "stage": args.stage,
            "candidate_count": candidate_count,
            "phase_order": list(PHASES),
            "arms": {arm: {
                "result_path": str(root / f"{arm}.json"),
                "result_sha256": arms[arm]["result_sha256"],
                "pid": arms[arm]["pid"],
                "ppid": arms[arm]["ppid"],
                "uncached_call_count": arms[arm]["uncached_call_count"],
                "success_count": arms[arm]["success_count"],
                "failure_count": arms[arm]["failure_count"],
                "worker_wall_s": arms[arm]["worker_wall_s"],
                "main_wall_s": arms[arm]["main_wall_s"],
                "total_wall_s": arms[arm]["total_wall_s"],
                "subprocess": arms[arm]["subprocess"],
            } for arm in ARMS},
            "separate_processes": (
                arms["off"]["pid"] != arms["on"]["pid"]
                and all(arms[arm]["pid"] != os.getpid() for arm in ARMS)
            ),
            "semantic_order_equal": True,
            "off_expected_uncached": 2 * candidate_count,
            "on_expected_uncached": candidate_count,
            "post_run_materials_verified": True,
            "validation_cache_effect": {
                "uncached_calls_avoided": candidate_count,
                "request_count_equal": True,
                "off": off_wall,
                "on": on_wall,
                "main_wall_s_avoided": main_wall_s_avoided,
                "main_speedup": _speedup(
                    off_wall["main_wall_s"], on_wall["main_wall_s"]
                ),
                "total_wall_s_avoided": total_wall_s_avoided,
                "total_speedup": _speedup(
                    off_wall["total_wall_s"], on_wall["total_wall_s"]
                ),
            },
            "jax_platform": "cpu",
            "cuda_visible_devices": "",
            "network_guard": True,
            "llm_api_calls": 0,
            "full_code_recorded": False,
        }
        if not result["separate_processes"]:
            raise RuntimeError("validation arms did not run in separate processes")
        return _atomic_json(root / "RESULT.json", result)
    except Exception as exc:
        _atomic_json(root / "failure.json", {
            "classification": CLASSIFICATION,
            "error_class": type(exc).__name__,
            "error": str(exc),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "stage": args.stage,
            "commands": commands,
        })
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("early", "mid", "late"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--internal-arm", choices=ARMS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.internal_arm:
        _run_arm(args.manifest, args.stage, args.internal_arm, args.out)
    else:
        run_replay(args)


if __name__ == "__main__":
    main()
