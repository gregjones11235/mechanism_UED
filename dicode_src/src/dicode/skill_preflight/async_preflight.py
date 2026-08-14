"""Default-off, one-session-delayed asynchronous learnability preflight.

The worker evaluates an immutable checkpoint/archive snapshot and only writes a
score receipt.  Archive routing remains a main-process operation.  This is an
explicit research scheduling change, not the historical semantic mainline.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


CLASSIFICATION = "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE"

SOURCE_RELATIVES = {
    "async_preflight": "src/dicode/skill_preflight/async_preflight.py",
    "run_dicode": "experiments/training/run_dicode.py",
    "evaluate_new_tasks": "src/dicode/evaluation/online_evaluation.py",
    "load_agent_state": "src/dicode/setup.py",
    "load_tasks_from_env_codes": "src/dicode/task_utils.py",
    "learnability_summary": "src/dicode/skill_preflight/learnability_summary.py",
    "preflight_route": "src/dicode/skill_preflight/preflight_route.py",
    "route": "src/dicode/skill_preflight/preflight.py",
}


class AsyncPreflightError(RuntimeError):
    """Fail-closed asynchronous preflight contract error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if not root.exists():
        raise AsyncPreflightError(f"checkpoint path does not exist: {root}")
    if root.is_file():
        return _fingerprint({".": _file_sha256(root)})
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise AsyncPreflightError(f"checkpoint path contains no files: {root}")
    return _fingerprint(
        {item.relative_to(root).as_posix(): _file_sha256(item) for item in files}
    )


def _hashed_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("result_sha256", None)
    result["result_sha256"] = _fingerprint(result)
    return result


def atomic_json(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    require_absent: bool = False,
) -> dict[str, Any]:
    """Atomically write a canonical, self-hashed JSON receipt."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = _hashed_document(document)
    payload = _canonical_bytes(result) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if require_absent:
            # Hard-linking the fully fsynced temporary file is an atomic
            # create-if-absent operation on the job filesystem.  It prevents a
            # duplicate worker/apply attempt from replacing durable evidence.
            os.link(temporary, target)
            os.unlink(temporary)
        else:
            os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


def load_hashed_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    document = json.loads(target.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or not document.get("result_sha256")
        or _hashed_document(document)["result_sha256"] != document["result_sha256"]
    ):
        raise AsyncPreflightError(f"self-hash mismatch: {target}")
    return document


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _nested_get(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if hasattr(current, "get"):
            current = current.get(key, default)
        elif isinstance(current, Mapping):
            current = current.get(key, default)
        else:
            return default
        if current is default:
            return default
    return current


def async_pipeline_enabled(config: Any) -> bool:
    """The only check required on the historical/default-off path."""
    return bool(_nested_get(config, "performance", "async_preflight_pipeline", default=False))


def validate_async_contract(config: Any) -> None:
    """Reject unsupported modes before any async rollout or subprocess launch."""
    requirements = {
        "skill_preflight.use_preflight": bool(
            _nested_get(config, "skill_preflight", "use_preflight", default=False)
        ),
        "training.conditioning_type=one_hot": (
            _nested_get(config, "training", "conditioning_type") == "one_hot"
        ),
        "dicode_manager.score_function=learnability": (
            _nested_get(config, "dicode_manager", "score_function") == "learnability"
        ),
        "performance.learnability_fused_preflight_summary=true": bool(
            _nested_get(
                config,
                "performance",
                "learnability_fused_preflight_summary",
                default=False,
            )
        ),
        "performance.preflight_reuse_loaded_tasks=true": bool(
            _nested_get(
                config, "performance", "preflight_reuse_loaded_tasks", default=False
            )
        ),
        "performance.async_preflight_gpu_uuid": bool(
            _nested_get(config, "performance", "async_preflight_gpu_uuid")
        ),
    }
    failed = [name for name, ok in requirements.items() if not ok]
    if failed:
        raise AsyncPreflightError(
            "async preflight contract rejected before launch: " + ", ".join(failed)
        )


def plan_async_session(
    *, async_enabled: bool, delayed_ids: Sequence[str], fresh_ids: Sequence[str], pending: bool
) -> dict[str, list[str]]:
    """Pure scheduling helper used by the production loop and CPU tests."""
    delayed = list(delayed_ids)
    fresh = list(fresh_ids)
    if not async_enabled:
        return {"training_new_ids": fresh, "launch_ids": []}
    if pending and fresh:
        raise AsyncPreflightError(
            "async preflight worker is still pending while a fresh batch arrived"
        )
    return {"training_new_ids": delayed, "launch_ids": fresh}


def _resolved_config(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.to_container(config, resolve=True)
    except Exception:
        value = copy.deepcopy(config)
    if not isinstance(value, dict):
        raise AsyncPreflightError("resolved config must be a mapping")
    return value


def _json_safe_graph_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "null"
    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, (str, int, float, bool)):
                return scalar
        except Exception:
            pass
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _snapshot_graph(archive: Any, path: Path) -> None:
    import networkx as nx

    lock = getattr(archive, "_lock", None)
    if lock is None:
        graph = copy.deepcopy(archive.graph)
    else:
        with lock:
            graph = copy.deepcopy(archive.graph)
    for _, attrs in graph.nodes(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = _json_safe_graph_value(value)
    for _, _, attrs in graph.edges(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = _json_safe_graph_value(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    nx.write_graphml(graph, temporary)
    os.replace(temporary, path)


def _candidate_code_hashes(archive: Any, task_ids: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lock = getattr(archive, "_lock", None)

    def collect() -> None:
        for task_id in task_ids:
            if not archive.graph.has_node(task_id):
                raise AsyncPreflightError(f"candidate missing from archive: {task_id}")
            code = archive.graph.nodes[task_id].get("code")
            if not isinstance(code, str) or not code:
                raise AsyncPreflightError(f"candidate has no code: {task_id}")
            rows.append(
                {"task_id": str(task_id), "code_sha256": hashlib.sha256(code.encode()).hexdigest()}
            )

    if lock is None:
        collect()
    else:
        with lock:
            collect()
    return rows


def _rng_receipt(rng: Any) -> dict[str, Any]:
    try:
        import numpy as np

        array = np.asarray(rng)
        values = array.tolist()
        dtype = str(array.dtype)
        shape = list(array.shape)
    except Exception:
        values = list(rng) if isinstance(rng, (list, tuple)) else rng
        dtype = type(rng).__name__
        shape = [len(values)] if isinstance(values, list) else []
    receipt = {"values": values, "dtype": dtype, "shape": shape}
    receipt["sha256"] = _fingerprint(receipt)
    return receipt


def _source_evidence(source_root: Path) -> dict[str, Any]:
    paths = {name: source_root / relative for name, relative in SOURCE_RELATIVES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AsyncPreflightError(f"async source evidence missing: {missing}")
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    return {
        "source_root": str(source_root.resolve()),
        "relatives": dict(SOURCE_RELATIVES),
        "hashes": hashes,
        "sha256": _fingerprint({"relatives": SOURCE_RELATIVES, "hashes": hashes}),
    }


def _validate_source_evidence(evidence: Mapping[str, Any]) -> None:
    source_root = Path(str(evidence.get("source_root", "")))
    if evidence.get("relatives") != SOURCE_RELATIVES:
        raise AsyncPreflightError("async source mapping mismatch")
    current = _source_evidence(source_root)
    if current["hashes"] != evidence.get("hashes") or current["sha256"] != evidence.get("sha256"):
        raise AsyncPreflightError("async source hash mismatch")


class AsyncPreflightManager:
    """Own one asynchronous preflight subprocess and its durable receipts."""

    def __init__(
        self,
        config: Any,
        *,
        source_root: str | Path | None = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        register_atexit: Callable[[Callable[[], None]], Any] = atexit.register,
    ) -> None:
        validate_async_contract(config)
        self.config = config
        self.gpu_uuid = str(
            _nested_get(config, "performance", "async_preflight_gpu_uuid")
        )
        self.root = Path(
            str(
                _nested_get(
                    config,
                    "performance",
                    "async_preflight_root",
                    default="async_preflight",
                )
            )
        ).resolve()
        self.result_timeout_s = float(
            _nested_get(
                config, "performance", "async_preflight_result_timeout_s", default=0
            )
        )
        self.shutdown_timeout_s = float(
            _nested_get(
                config,
                "performance",
                "async_preflight_shutdown_timeout_s",
                default=120,
            )
        )
        if self.result_timeout_s < 0 or self.shutdown_timeout_s < 0:
            raise AsyncPreflightError("async preflight timeouts must be non-negative")
        self.source_root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self.process_factory = process_factory
        self._process: Any | None = None
        self._streams: tuple[Any, Any] | None = None
        self._pending_job_dir: Path | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover()
        register_atexit(self.shutdown)

    @property
    def pending(self) -> bool:
        return self._pending_job_dir is not None

    def _recover(self) -> None:
        outstanding: list[Path] = []
        for job_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            job_path = job_dir / "JOB.json"
            if not job_path.exists():
                raise AsyncPreflightError(f"unrecognized async job directory: {job_dir}")
            load_hashed_json(job_path)
            applied = job_dir / "APPLIED.json"
            applying = job_dir / "APPLYING.json"
            result = job_dir / "RESULT.json"
            failure = job_dir / "FAILURE.json"
            if applied.exists():
                if result.exists():
                    load_hashed_json(result)
                load_hashed_json(applied)
                continue
            if applying.exists():
                load_hashed_json(applying)
                raise AsyncPreflightError(
                    f"unrecoverable async apply in progress (route may be partial): {job_dir}"
                )
            if failure.exists():
                failure_doc = load_hashed_json(failure)
                raise AsyncPreflightError(
                    f"unapplied async worker failure {failure_doc.get('error_class')}: "
                    f"{failure_doc.get('error')}"
                )
            if result.exists():
                load_hashed_json(result)
                outstanding.append(job_dir)
                continue
            if (job_dir / "RUNNING.json").exists():
                load_hashed_json(job_dir / "RUNNING.json")
                raise AsyncPreflightError(
                    f"unrecoverable running async job is not owned by this process: {job_dir}"
                )
            raise AsyncPreflightError(f"incomplete async job directory: {job_dir}")
        if len(outstanding) > 1:
            raise AsyncPreflightError("multiple completed async jobs await application")
        self._pending_job_dir = outstanding[0] if outstanding else None

    def launch(
        self,
        *,
        session_idx: int,
        global_update_step: int,
        task_ids: Sequence[str],
        pf_rng: Any,
        archive: Any,
        rl_ckpt_manager: Any,
        rl_ckpt_path: str | Path,
    ) -> Path:
        validate_async_contract(self.config)
        if self.pending:
            raise AsyncPreflightError("async preflight already has a pending job")
        ordered_ids = [str(value) for value in task_ids]
        if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
            raise AsyncPreflightError("async preflight task IDs must be nonempty and unique")
        if not hasattr(rl_ckpt_manager, "wait_until_finished"):
            raise AsyncPreflightError("checkpoint manager cannot wait for async saves")
        rl_ckpt_manager.wait_until_finished()
        if hasattr(rl_ckpt_manager, "check_for_errors"):
            rl_ckpt_manager.check_for_errors()
        if not hasattr(rl_ckpt_manager, "item_metadata"):
            raise AsyncPreflightError(
                "checkpoint manager cannot verify exact-step load metadata"
            )
        try:
            checkpoint_metadata = rl_ckpt_manager.item_metadata(
                int(global_update_step)
            )
        except Exception as exc:
            raise AsyncPreflightError(
                f"exact checkpoint metadata is not loadable at step "
                f"{int(global_update_step)}"
            ) from exc
        if checkpoint_metadata is None:
            raise AsyncPreflightError(
                f"exact checkpoint metadata missing at step {int(global_update_step)}"
            )
        checkpoint_path = Path(rl_ckpt_path).resolve() / str(int(global_update_step))
        checkpoint_sha = _tree_sha256(checkpoint_path)
        codes = _candidate_code_hashes(archive, ordered_ids)
        rng_receipt = _rng_receipt(pf_rng)
        source = _source_evidence(self.source_root)
        identity = {
            "session_idx": int(session_idx),
            "global_update_step": int(global_update_step),
            "task_ids": ordered_ids,
            "task_code_hashes": codes,
            "pf_rng_sha256": rng_receipt["sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "source_sha256": source["sha256"],
        }
        job_dir = self.root / (
            f"session_{int(session_idx):06d}_step_{int(global_update_step):09d}_"
            f"{_fingerprint(identity)[:12]}"
        )
        job_dir.mkdir(parents=False, exist_ok=False)
        config_path = job_dir / "config.resolved.json"
        graph_path = job_dir / "archive.graphml"
        resolved = _resolved_config(self.config)
        _write_atomic_bytes(config_path, _canonical_bytes(resolved) + b"\n")
        _snapshot_graph(archive, graph_path)
        job = atomic_json(
            job_dir / "JOB.json",
            {
                "classification": CLASSIFICATION,
                "not_semantic_mainline": True,
                "llm_api_calls": 0,
                "session_idx": int(session_idx),
                "global_update_step": int(global_update_step),
                "task_ids": ordered_ids,
                "task_code_hashes": codes,
                "pf_rng": rng_receipt,
                "config_path": str(config_path.resolve()),
                "config_sha256": _file_sha256(config_path),
                "graph_path": str(graph_path.resolve()),
                "graph_sha256": _file_sha256(graph_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_metadata_verified": True,
                "source_evidence": source,
                "gpu_uuid": self.gpu_uuid,
                "worker_contract": {
                    "conditioning_type": "one_hot",
                    "score_function": "learnability",
                    "fused_summary": True,
                    "reuse_loaded_tasks": True,
                    "route_in_worker": False,
                },
            },
            require_absent=True,
        )
        stdout = (job_dir / "worker.stdout").open("w", encoding="utf-8")
        stderr = (job_dir / "worker.stderr").open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "dicode.skill_preflight.async_preflight",
            "--job",
            str(job_dir / "JOB.json"),
        ]
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=self.gpu_uuid,
            WANDB_MODE="offline",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            DICODE_ASYNC_PREFLIGHT_NO_NETWORK="1",
        )
        try:
            process = self.process_factory(
                command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        self._process = process
        self._streams = (stdout, stderr)
        self._pending_job_dir = job_dir
        atomic_json(
            job_dir / "RUNNING.json",
            {
                "classification": CLASSIFICATION,
                "job_sha256": job["result_sha256"],
                "pid": int(process.pid),
                "owned_by_pid": os.getpid(),
                "llm_api_calls": 0,
            },
            require_absent=True,
        )
        return job_dir

    def _finish_streams(self) -> None:
        if self._streams is not None:
            for stream in self._streams:
                stream.close()
        self._streams = None

    def poll_and_apply(
        self,
        *,
        archive: Any,
        current_session_idx: int,
        route_apply_fn: Callable[..., Any],
        route_fn: Callable[..., Any],
    ) -> list[str] | None:
        if not self.pending:
            return []
        job_dir = self._pending_job_dir
        assert job_dir is not None
        process = self._process
        if process is not None and process.poll() is None:
            if self.result_timeout_s <= 0:
                return None
            try:
                process.wait(timeout=self.result_timeout_s)
            except subprocess.TimeoutExpired:
                return None
        if process is not None:
            returncode = process.poll()
            self._finish_streams()
            if returncode != 0:
                failure_path = job_dir / "FAILURE.json"
                detail = load_hashed_json(failure_path) if failure_path.exists() else {}
                raise AsyncPreflightError(
                    f"async worker exited {returncode}: {detail.get('error_class')} "
                    f"{detail.get('error')}"
                )
        failure_path = job_dir / "FAILURE.json"
        if failure_path.exists():
            failure = load_hashed_json(failure_path)
            raise AsyncPreflightError(
                f"async worker failure {failure.get('error_class')}: {failure.get('error')}"
            )
        result_path = job_dir / "RESULT.json"
        if not result_path.exists():
            if process is None:
                raise AsyncPreflightError("recovered async job has no result")
            raise AsyncPreflightError("async worker exited successfully without RESULT.json")
        job = load_hashed_json(job_dir / "JOB.json")
        result = load_hashed_json(result_path)
        self._validate_result(job, result, archive, current_session_idx)
        applying_path = job_dir / "APPLYING.json"
        applied_path = job_dir / "APPLIED.json"
        if applied_path.exists() or applying_path.exists():
            raise AsyncPreflightError("async preflight result has already begun application")
        applying = atomic_json(
            applying_path,
            {
                "classification": CLASSIFICATION,
                "job_sha256": job["result_sha256"],
                "result_receipt_sha256": result["result_sha256"],
                "session_applied": int(current_session_idx),
                "llm_api_calls": 0,
            },
            require_absent=True,
        )
        scores = {
            str(row["task_index"]): {
                "sr": float(row["sr"]),
                "priority_score": float(row["priority_score"]),
            }
            for row in result["score_projection"]
        }
        kept: list[str] = []
        route_apply_fn(scores, list(job["task_ids"]), kept, archive, route_fn)
        applied = atomic_json(
            applied_path,
            {
                "classification": CLASSIFICATION,
                "job_sha256": job["result_sha256"],
                "result_receipt_sha256": result["result_sha256"],
                "applying_sha256": applying["result_sha256"],
                "session_applied": int(current_session_idx),
                "kept_ids": list(kept),
                "route_calls": 1,
                "llm_api_calls": 0,
            },
            require_absent=True,
        )
        if applied["route_calls"] != 1:
            raise AssertionError("unreachable route count mismatch")
        self._pending_job_dir = None
        self._process = None
        return kept

    def _validate_result(
        self,
        job: Mapping[str, Any],
        result: Mapping[str, Any],
        archive: Any,
        current_session_idx: int,
    ) -> None:
        if int(current_session_idx) <= int(job["session_idx"]):
            raise AsyncPreflightError("async result cannot be applied in its launch session")
        expected = {
            "classification": CLASSIFICATION,
            "not_semantic_mainline": True,
            "llm_api_calls": 0,
            "job_sha256": job["result_sha256"],
            "session_idx": job["session_idx"],
            "global_update_step": job["global_update_step"],
            "task_ids": job["task_ids"],
            "task_code_hashes": job["task_code_hashes"],
            "pf_rng_sha256": job["pf_rng"]["sha256"],
            "config_sha256": job["config_sha256"],
            "graph_sha256": job["graph_sha256"],
            "checkpoint_path": job["checkpoint_path"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "checkpoint_metadata_verified": True,
            "source_sha256": job["source_evidence"]["sha256"],
            "gpu_uuid": job["gpu_uuid"],
        }
        mismatches = {
            key: {"expected": value, "actual": result.get(key)}
            for key, value in expected.items()
            if result.get(key) != value
        }
        if mismatches:
            raise AsyncPreflightError(f"async result receipt mismatch: {mismatches}")
        if _file_sha256(job["config_path"]) != job["config_sha256"]:
            raise AsyncPreflightError("async resolved config changed")
        if _file_sha256(job["graph_path"]) != job["graph_sha256"]:
            raise AsyncPreflightError("async archive snapshot changed")
        if _tree_sha256(job["checkpoint_path"]) != job["checkpoint_sha256"]:
            raise AsyncPreflightError("async checkpoint changed")
        _validate_source_evidence(job["source_evidence"])
        if _candidate_code_hashes(archive, job["task_ids"]) != job["task_code_hashes"]:
            raise AsyncPreflightError("live archive candidate order/code changed")
        projection = result.get("score_projection")
        if not isinstance(projection, list) or len(projection) != len(job["task_ids"]):
            raise AsyncPreflightError("async score projection length mismatch")
        for index, (task_id, row) in enumerate(zip(job["task_ids"], projection)):
            if row.get("task_index") != index or row.get("task_id") != task_id:
                raise AsyncPreflightError("async score projection order mismatch")
        if result.get("score_fingerprint") != _fingerprint(projection):
            raise AsyncPreflightError("async score fingerprint mismatch")
        if (
            result.get("jax_backend") != "gpu"
            or result.get("jax_device_count") != 1
            or result.get("route_calls") != 0
            or result.get("archive_mutations") != 0
            or result.get("embedding_model_used") is not False
        ):
            raise AsyncPreflightError("async worker mechanism receipt mismatch")
        for row in projection:
            sr = float(row.get("sr"))
            priority = float(row.get("priority_score"))
            if not (-1.0 <= sr <= 1.0):
                raise AsyncPreflightError("async score SR is out of range")
            clipped = min(max(sr, 0.0), 1.0) if sr >= 0 else 0.0
            if abs(priority - clipped * (1.0 - clipped)) > 1e-12:
                raise AsyncPreflightError("async score priority formula mismatch")

    def shutdown(self) -> None:
        """Stop only the Popen object created and owned by this manager."""
        process = self._process
        if process is None:
            self._finish_streams()
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=max(1.0, self.shutdown_timeout_s))
        self._finish_streams()
        self._process = None


def _install_network_guard() -> None:
    if os.environ.get("DICODE_ASYNC_PREFLIGHT_NO_NETWORK") != "1":
        raise AsyncPreflightError("async worker network guard was not requested")

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise AsyncPreflightError("network access is forbidden in async preflight worker")

    socket.create_connection = blocked  # type: ignore[assignment]
    original_socket = socket.socket

    class GuardedSocket(original_socket):
        def connect(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

        def connect_ex(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

    socket.socket = GuardedSocket  # type: ignore[assignment]


def _scores_from_counts(
    task_ids: Sequence[str], finished_counts: Sequence[Any], success_counts: Sequence[Any]
) -> list[dict[str, Any]]:
    finished = [int(value) for value in finished_counts]
    successes = [int(value) for value in success_counts]
    if len(finished) != len(task_ids) or len(successes) != len(task_ids):
        raise AsyncPreflightError("fused counter length mismatch")
    projection = []
    for index, (task_id, num_finished, num_successes) in enumerate(
        zip(task_ids, finished, successes)
    ):
        if num_finished < 0 or num_successes < 0 or num_successes > num_finished:
            raise AsyncPreflightError("invalid fused learnability counters")
        sr = -1.0 if num_finished == 0 else num_successes / num_finished
        clipped = min(max(sr, 0.0), 1.0) if sr >= 0 else 0.0
        projection.append(
            {
                "task_index": index,
                "task_id": str(task_id),
                "sr": float(sr),
                "priority_score": float(clipped * (1.0 - clipped)),
            }
        )
    return projection


def run_worker_job(
    job_path: str | Path,
    *,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one immutable worker job. ``runtime`` is CPU-test injection only."""
    job_path = Path(job_path).resolve()
    job_dir = job_path.parent
    job = load_hashed_json(job_path)
    if job.get("classification") != CLASSIFICATION or job.get("llm_api_calls") != 0:
        raise AsyncPreflightError("invalid async worker job classification")
    if job.get("checkpoint_metadata_verified") is not True or job.get(
        "worker_contract"
    ) != {
        "conditioning_type": "one_hot",
        "score_function": "learnability",
        "fused_summary": True,
        "reuse_loaded_tasks": True,
        "route_in_worker": False,
    }:
        raise AsyncPreflightError("invalid async worker mechanism contract")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job.get("gpu_uuid"):
        raise AsyncPreflightError("async worker CUDA_VISIBLE_DEVICES mismatch")
    if _file_sha256(job["config_path"]) != job["config_sha256"]:
        raise AsyncPreflightError("async worker config hash mismatch")
    if _file_sha256(job["graph_path"]) != job["graph_sha256"]:
        raise AsyncPreflightError("async worker graph hash mismatch")
    if _tree_sha256(job["checkpoint_path"]) != job["checkpoint_sha256"]:
        raise AsyncPreflightError("async worker checkpoint hash mismatch")
    _validate_source_evidence(job["source_evidence"])
    if runtime is None:
        _install_network_guard()
        import jax
        import jax.numpy as jnp
        from omegaconf import OmegaConf
        from dicode.dreaming.gen_manager import TaskArchive
        from dicode.evaluation import evaluate_new_tasks
        from dicode.setup import _load_agent_state
        from dicode.task_utils import load_tasks_from_env_codes

        devices = jax.devices()
        if jax.default_backend() != "gpu" or len(devices) != 1:
            raise AsyncPreflightError("async worker requires JAX gpu with one visible device")
        runtime = {
            "jax": jax,
            "jnp": jnp,
            "OmegaConf": OmegaConf,
            "TaskArchive": TaskArchive,
            "evaluate_new_tasks": evaluate_new_tasks,
            "load_agent_state": _load_agent_state,
            "load_tasks": load_tasks_from_env_codes,
            "backend": jax.default_backend(),
            "device_count": len(devices),
        }
    if runtime.get("backend") != "gpu" or int(runtime.get("device_count", 0)) != 1:
        raise AsyncPreflightError("async worker runtime did not prove gpu/device1")
    resolved = json.loads(Path(job["config_path"]).read_text(encoding="utf-8"))
    config = runtime["OmegaConf"].create(resolved)
    validate_async_contract(config)
    archive = runtime["TaskArchive"](SimpleNamespace(graph_path=job["graph_path"]))
    if _candidate_code_hashes(archive, job["task_ids"]) != job["task_code_hashes"]:
        raise AsyncPreflightError("worker archive candidate code mismatch")
    train_state = runtime["load_agent_state"](config, job["checkpoint_path"])
    classes, ok_ids = runtime["load_tasks"](archive, list(job["task_ids"]))
    if list(ok_ids) != list(job["task_ids"]) or len(classes) != len(job["task_ids"]):
        raise AsyncPreflightError("worker did not load the exact ordered candidate set")
    rng = runtime["jnp"].asarray(
        job["pf_rng"]["values"], dtype=job["pf_rng"]["dtype"]
    )
    raw = runtime["evaluate_new_tasks"](
        config,
        rng,
        train_state,
        list(job["task_ids"]),
        archive,
        None,
        preloaded_task_classes=classes,
        preloaded_task_ids=list(job["task_ids"]),
    )
    summary = raw.get("learnability_summary") if isinstance(raw, Mapping) else None
    if not isinstance(summary, Mapping):
        raise AsyncPreflightError("worker fused evaluation returned no summary")
    finished, successes = runtime["jax"].device_get(
        (summary.get("finished_counts"), summary.get("success_counts"))
    )
    projection = _scores_from_counts(job["task_ids"], finished, successes)
    return atomic_json(
        job_dir / "RESULT.json",
        {
            "classification": CLASSIFICATION,
            "not_semantic_mainline": True,
            "llm_api_calls": 0,
            "job_sha256": job["result_sha256"],
            "session_idx": job["session_idx"],
            "global_update_step": job["global_update_step"],
            "task_ids": job["task_ids"],
            "task_code_hashes": job["task_code_hashes"],
            "pf_rng_sha256": job["pf_rng"]["sha256"],
            "config_sha256": job["config_sha256"],
            "graph_sha256": job["graph_sha256"],
            "checkpoint_path": job["checkpoint_path"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "checkpoint_metadata_verified": job["checkpoint_metadata_verified"],
            "source_sha256": job["source_evidence"]["sha256"],
            "gpu_uuid": job["gpu_uuid"],
            "jax_backend": runtime["backend"],
            "jax_device_count": int(runtime["device_count"]),
            "score_projection": projection,
            "score_fingerprint": _fingerprint(projection),
            "route_calls": 0,
            "archive_mutations": 0,
            "embedding_model_used": False,
        },
        require_absent=True,
    )


def worker_entry(job_path: str | Path) -> int:
    job_path = Path(job_path).resolve()
    try:
        run_worker_job(job_path)
    except Exception as exc:
        atomic_json(
            job_path.parent / "FAILURE.json",
            {
                "classification": CLASSIFICATION,
                "llm_api_calls": 0,
                "error_class": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            require_absent=True,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    return worker_entry(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
