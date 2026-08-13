#!/usr/bin/env python3
"""D3 metadata-only provider and GPU gate.

This module is deliberately limited to read-only metadata calls.  The Ollama
arm uses ``/api/tags`` and the official DeepSeek arm uses ``/models``.  No
completion, chat, or embedding endpoint is present in the implementation.
The gate is run before any D3 generation and fails closed when a required
provider/model or the dedicated GPU2 is unavailable.

The output is an independent research artifact.  It never serializes API key
values (or URLs containing query credentials), and it refuses to overwrite an
existing artifact directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

CLASSIFICATION = "D3_METADATA_GATE"
SCHEMA_VERSION = 1
OLLAMA_MODEL = "qwen2.5-coder:14b"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com/v1"
CANONICAL_ALGORITHM = "canonical_json_sha256"
HASH_SCOPE = "D3_METADATA_GATE_FIELDS_EXCLUDING_ARTIFACT_SHA256"


def canonical(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(x) for x in value]
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_base_url(value: str) -> str:
    """Normalize a base URL and discard query/fragment credentials."""
    raw = str(value or "").strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid provider base URL: {value!r}")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _error_class(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return "unauthorized"
        if exc.code == 403:
            return "forbidden"
        if exc.code == 404:
            return "not_found"
        if 500 <= exc.code < 600:
            return "server_error"
        return "http_error"
    if isinstance(exc, urllib.error.URLError):
        reason = str(getattr(exc, "reason", exc)).lower()
        if "timed out" in reason or "timeout" in reason:
            return "timeout"
        if any(x in reason for x in ("refused", "unreachable", "connection", "name resolution")):
            return "connection_error"
        return "url_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (ConnectionError, ConnectionRefusedError)):
        return "connection_error"
    return type(exc).__name__.lower()


def _request_json(url: str, *, headers: Mapping[str, str] | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json", **dict(headers or {})}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("metadata response must be a JSON object")
        return {"http_status": int(getattr(resp, "status", 200)), "data": data}


def check_ollama(base_url: str = DEFAULT_OLLAMA_BASE, expected_model: str = OLLAMA_MODEL,
                 timeout_s: float = 5.0) -> dict[str, Any]:
    base = safe_base_url(base_url)
    endpoint = base + "/api/tags"
    result: dict[str, Any] = {
        "provider": "ollama",
        "base_url": base,
        "endpoint": endpoint,
        "expected_model": expected_model,
        "metadata_endpoint_only": True,
        "metadata_requests": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "reachable": False,
        "http_status": None,
        "model_names": [],
        "exact_model_available": False,
        "error_class": None,
    }
    try:
        response = _request_json(endpoint, timeout_s=timeout_s)
        result["metadata_requests"] = 1
        result["reachable"] = True
        result["http_status"] = response["http_status"]
        models = response["data"].get("models", [])
        if not isinstance(models, list):
            raise ValueError("Ollama models field must be a list")
        result["model_names"] = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
        result["exact_model_available"] = expected_model in result["model_names"]
        if not result["exact_model_available"]:
            result["error_class"] = "model_not_listed"
    except Exception as exc:
        result["metadata_requests"] = 1
        result["error_class"] = _error_class(exc)
    return result


def check_deepseek(base_url: str = DEFAULT_DEEPSEEK_BASE, expected_model: str = DEEPSEEK_MODEL,
                   api_key_env: str = "DEEPSEEK_API_KEY", timeout_s: float = 5.0) -> dict[str, Any]:
    base = safe_base_url(base_url)
    endpoint = base + "/models"
    key = os.environ.get(api_key_env, "")
    result: dict[str, Any] = {
        "provider": "deepseek_official",
        "base_url": base,
        "endpoint": endpoint,
        "expected_model": expected_model,
        "credential_env_name": api_key_env,
        "credential_present": bool(key),
        "credential_value_serialized": False,
        "metadata_endpoint_only": True,
        "metadata_requests": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "reachable": False,
        "http_status": None,
        "model_ids": [],
        "exact_model_available": False,
        "error_class": None,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = _request_json(endpoint, headers=headers, timeout_s=timeout_s)
        result["metadata_requests"] = 1
        result["reachable"] = True
        result["http_status"] = response["http_status"]
        models = response["data"].get("data", [])
        if not isinstance(models, list):
            raise ValueError("DeepSeek models field must be a list")
        result["model_ids"] = [str(item.get("id", "")) for item in models if isinstance(item, dict)]
        result["exact_model_available"] = expected_model in result["model_ids"]
        if not result["exact_model_available"]:
            result["error_class"] = "model_not_listed"
    except Exception as exc:
        result["metadata_requests"] = 1
        result["http_status"] = getattr(exc, "code", None)
        result["error_class"] = _error_class(exc)
    return result


def gpu_metadata() -> dict[str, Any]:
    query = [
        "index,name,uuid,memory.free,memory.total,utilization.gpu,compute_pid"
    ]
    result: dict[str, Any] = {"nvidia_smi_present": False, "query_ok": False, "gpus": [], "gpu2_present": False}
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query[0]}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        result["nvidia_smi_present"] = True
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            result["stderr"] = completed.stderr[-500:]
            return result
        fields = ["index", "name", "uuid", "memory_free_mib", "memory_total_mib", "utilization_gpu", "compute_pid"]
        for line in completed.stdout.splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) != len(fields):
                continue
            row = dict(zip(fields, values))
            try:
                row["index"] = int(row["index"])
                row["memory_free_mib"] = int(float(row["memory_free_mib"]))
                row["memory_total_mib"] = int(float(row["memory_total_mib"]))
                row["utilization_gpu"] = int(float(row["utilization_gpu"]))
            except ValueError:
                pass
            result["gpus"].append(row)
        result["query_ok"] = True
        result["gpu2_present"] = any(row.get("index") == 2 for row in result["gpus"])
    except Exception as exc:
        result["error_class"] = _error_class(exc)
    return result


def build_gate(*, source_commit: str, source_branch: str, ollama_base_url: str = DEFAULT_OLLAMA_BASE,
               deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE, large_model: str = DEEPSEEK_MODEL,
               timeout_s: float = 5.0) -> dict[str, Any]:
    ollama = check_ollama(ollama_base_url, OLLAMA_MODEL, timeout_s)
    deepseek = check_deepseek(deepseek_base_url, large_model, timeout_s=timeout_s)
    gpu = gpu_metadata()
    reasons: list[str] = []
    if not ollama["exact_model_available"]:
        reasons.append("Ollama exact qwen2.5-coder:14b model is not listed")
    if not deepseek["credential_present"]:
        reasons.append("DEEPSEEK_API_KEY is not present in this shell")
    if not deepseek["exact_model_available"]:
        reasons.append(f"DeepSeek exact model {large_model!r} is not listed by /models")
    if not gpu["gpu2_present"]:
        reasons.append("dedicated GPU2 is not present")
    gate_open = not reasons
    result: dict[str, Any] = {
        "classification": CLASSIFICATION,
        "schema_version": SCHEMA_VERSION,
        "stage": "D3",
        "source_commit": source_commit,
        "source_branch": source_branch,
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "small_arm": {"model": OLLAMA_MODEL, "provider": "ollama", "gpu": 0},
        "large_arm": {"model": large_model, "provider": "deepseek_official", "model_size": "UNKNOWN", "gpu": "provider-managed"},
        "ollama": ollama,
        "deepseek": deepseek,
        "gpu": gpu,
        "completion_requests_total": 0,
        "embedding_requests_total": 0,
        "metadata_requests_total": ollama["metadata_requests"] + deepseek["metadata_requests"],
        "gpu2_smoke_allowed": gate_open,
        "gate_status": "PASS" if gate_open else "BLOCKED_METADATA_GATE",
        "blocked_reasons": reasons,
        "artifact_sha256_algorithm": CANONICAL_ALGORITHM,
        "artifact_sha256_scope": HASH_SCOPE,
    }
    result["artifact_sha256"] = canonical_json_sha256({k: v for k, v in result.items() if k != "artifact_sha256"})
    return result


def _write_no_clobber(path: Path, data: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_gate_artifacts(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write non-empty artifact directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "D3_METADATA_GATE.json"
    _write_no_clobber(result_path, json.dumps(canonical(result), indent=2, ensure_ascii=False) + "\n")
    report = [
        "# D3 metadata gate",
        "",
        f"- status: `{result['gate_status']}`",
        f"- metadata requests: `{result['metadata_requests_total']}`",
        f"- completion requests: `{result['completion_requests_total']}`",
        f"- embedding requests: `{result['embedding_requests_total']}`",
        f"- GPU2 smoke allowed: `{result['gpu2_smoke_allowed']}`",
        "",
        "## Blocked reasons",
        "",
    ]
    report.extend(f"- {reason}" for reason in result.get("blocked_reasons", []))
    report_path = out / "D3_METADATA_GATE_REPORT.md"
    _write_no_clobber(report_path, "\n".join(report) + "\n")
    sums = f"{file_sha256(result_path)}  {result_path.name}\n{file_sha256(report_path)}  {report_path.name}\n"
    sums_path = out / "SHA256SUMS"
    _write_no_clobber(sums_path, sums)
    return {"result": str(result_path), "report": str(report_path), "sums": str(sums_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--source-branch", default="unknown")
    parser.add_argument("--ollama-base-url", default=os.environ.get("D3_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE))
    parser.add_argument("--deepseek-base-url", default=os.environ.get("D3_DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE))
    parser.add_argument("--large-model", default=os.environ.get("D3_LARGE_MODEL", DEEPSEEK_MODEL))
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    result = build_gate(
        source_commit=args.source_commit, source_branch=args.source_branch,
        ollama_base_url=args.ollama_base_url, deepseek_base_url=args.deepseek_base_url,
        large_model=args.large_model, timeout_s=args.timeout_s,
    )
    paths = write_gate_artifacts(result, args.output_dir)
    print(json.dumps({"gate_status": result["gate_status"], "gpu2_smoke_allowed": result["gpu2_smoke_allowed"], **paths}, indent=2))
    return 0 if result["gate_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
