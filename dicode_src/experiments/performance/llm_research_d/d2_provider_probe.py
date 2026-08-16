#!/usr/bin/env python3
"""D2 provider availability probe (evidence repair, read-only).

Checks whether the 235B provider (DeepInfra) is actually available in the
specified environment WITHOUT calling any model inference, WITHOUT touching the
external DeepInfra endpoint (unless a credential is present — which it is not
in this repair), and WITHOUT using any GPU.

Only a single read-only metadata check is performed against localhost:5000 and
the local Ollama models endpoint. Results are persisted atomically with a
canonical JSON self-hash and refuse-overwrite semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
CLASSIFICATION = "D2_PROVIDER_AVAILABILITY_PROBE"
CANONICAL_ALGORITHM = "canonical_json_sha256"
CANONICAL_SCOPE = "PROBE_FIELDS_EXCLUDING_ARTIFACT_SHA256"
FINAL_EVIDENCE_SCOPE = "D2_EVIDENCE_FINAL_FIELDS_EXCLUDING_ARTIFACT_SHA256"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(x) for x in value]
    return value


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        canonical(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8"))


def read_config(path: Path) -> dict[str, str]:
    """Read a YAML-ish config file; record only provider/base_url/model + SHA."""
    text = path.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        fields[k.strip()] = v.strip()
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "provider": fields.get("provider", ""),
        "base_url": fields.get("base_url", ""),
        "model": fields.get("model", ""),
    }


def credential_present() -> bool:
    return bool(os.environ.get("DEEPINFRA_API_KEY"))


def check_localhost_5000(timeout_s: float = 3.0) -> dict[str, Any]:
    """One metadata health check to localhost:5000/v1/models (no completions)."""
    url = "http://127.0.0.1:5000/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("models response must be a JSON object")
                model_ids = []
                for m in data.get("data", []):
                    if isinstance(m, dict) and m.get("id"):
                        model_ids.append(m["id"])
            except (json.JSONDecodeError, ValueError, TypeError):
                return {"reachable": True, "http_status": resp.status,
                        "model_ids": [], "local_model_available": False,
                        "error_class": "invalid_json"}
            return {"reachable": True, "http_status": resp.status,
                    "model_ids": model_ids,
                    "local_model_available": False, "error_class": None}
    except Exception as e:
        return {"reachable": False, "http_status": None, "model_ids": [],
                "local_model_available": False,
                "error_class": _normalize_error(e)}


def list_ollama_models(timeout_s: float = 3.0) -> dict[str, Any]:
    """Read only the local Ollama model names (no model invocation)."""
    url = "http://127.0.0.1:11434/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Ollama response must be a JSON object")
                names = [m.get("name", "") for m in data.get("models", [])
                         if isinstance(m, dict)]
            except (json.JSONDecodeError, ValueError, TypeError):
                return {"reachable": True, "http_status": resp.status,
                        "model_names": [], "error_class": "invalid_json"}
            return {"reachable": True, "http_status": resp.status, "model_names": names,
                    "error_class": None}
    except Exception as e:
        return {"reachable": False, "http_status": None, "model_names": [],
                "error_class": _normalize_error(e)}


def _normalize_error(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return "http_error"
    if isinstance(e, urllib.error.URLError):
        reason = e.reason
        if isinstance(reason, BaseException):
            return _normalize_error(reason)
        reason_text = str(reason).lower()
        if "timed out" in reason_text or "timeout" in reason_text:
            return "timeout"
        if any(token in reason_text for token in
               ("connection", "refused", "unreachable", "name resolution")):
            return "connection_error"
        return "unknown_error"
    if isinstance(e, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(e, (ConnectionError, ConnectionRefusedError)):
        return "connection_error"
    name = type(e).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "connection" in name or "connect" in name:
        return "connection_error"
    return "unknown_error"


def build_probe(config_root: str, source_branch: str, source_head: str,
                source_commit: str, probe_utc: str) -> dict[str, Any]:
    root = Path(config_root)
    deepinfra = read_config(root / "conf" / "gen_manager" / "llm" / "deepinfra.yaml")
    local_gen = read_config(root / "conf" / "gen_manager" / "llm" / "local_gen.yaml")
    cred_present = credential_present()
    # external discipline: never touch DeepInfra when no credential is present
    external_request = False
    local = check_localhost_5000()
    ollama = list_ollama_models()

    # Provider targets are intentionally independent. DeepInfra and the local
    # OpenAI-compatible server use different configured IDs for the same model
    # family (the local target includes the FP8 suffix).
    model_id = deepinfra["model"]
    local_model_id = local_gen["model"]
    local_model_available = any(
        m == local_model_id for m in local.get("model_ids", []))
    local["local_model_available"] = local_model_available

    blocked_reasons = []
    if not cred_present:
        blocked_reasons.append("DEEPINFRA_API_KEY not exported in the specified non-interactive shell")
    if not local_model_available:
        blocked_reasons.append("no exact 235B model served on localhost:5000")
    blocked_reasons.append("no explicit budget authorization evidence")

    probe = {
        "classification": CLASSIFICATION,
        "schema_version": SCHEMA_VERSION,
        "repair_probe": True,
        "probe_utc": probe_utc,
        "source_branch": source_branch,
        "source_head": source_head,
        "source_commit": source_commit,
        "probe_tool_sha256": file_sha256(Path(__file__).resolve()),
        "config_evidence": [deepinfra, local_gen],
        "provider": deepinfra["provider"],
        "base_url": deepinfra["base_url"],
        "model": model_id,
        "model_id_explicit": bool(model_id),
        "local_model": local_model_id,
        "local_model_id_explicit": bool(local_model_id),
        "credential_env_name": "DEEPINFRA_API_KEY",
        "credential_present": cred_present,
        "credential_value_serialized": False,
        "external_provider_request_performed": external_request,
        "local_endpoint_probe": local,
        "ollama_models": ollama,
        "budget_authorization": "NOT_OBSERVED",
        "llm_api_calls": 0,
        "gpu_used": False,
        "decision_inputs": {
            "credential_present": cred_present,
            "local_model_available": local_model_available,
            "local_target_model": local_model_id,
            "budget_authorization": "NOT_OBSERVED",
        },
        "conclusion": "D2_BLOCKED_EXTERNAL_PROVIDER",
        "artifact_sha256_algorithm": CANONICAL_ALGORITHM,
        "artifact_sha256_scope": CANONICAL_SCOPE,
    }
    probe["artifact_sha256"] = canonical_json_sha256(
        {k: v for k, v in probe.items() if k != "artifact_sha256"})
    return probe


def atomic_write_refusing_overwrite(path: Path, data: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(data), f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_probe(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recomputed = canonical_json_sha256(
        {k: v for k, v in raw.items() if k != "artifact_sha256"})
    if recomputed != raw.get("artifact_sha256"):
        raise ValueError("probe artifact_sha256 mismatch (tampered)")
    return raw


D2_RESULT_CLASSIFICATION = "RESEARCH_NON_SEMANTIC_MODEL_COMPARISON"


def build_d2_result(probe: dict, probe_path: str,
                    original_probe_path: str | None = None,
                    original_probe_path_cleaned: bool | None = None) -> dict:
    """Build the D2 result artifact. Status is BLOCKED (no benchmark ran)."""
    blocked_reasons = []
    if not probe.get("credential_present"):
        blocked_reasons.append("DEEPINFRA_API_KEY not exported in the specified non-interactive shell")
    if not probe.get("decision_inputs", {}).get("local_model_available"):
        blocked_reasons.append("no exact 235B model served on localhost:5000")
    blocked_reasons.append("no explicit budget authorization evidence")
    result = {
        "classification": D2_RESULT_CLASSIFICATION,
        "stage": "D2",
        "status": "BLOCKED",
        "conclusion": "D2_BLOCKED_EXTERNAL_PROVIDER",
        "provider_probe_sha256": probe.get("artifact_sha256"),
        "provider_probe_path": probe_path,
        "provider_probe_provenance": {
            "original_probe_runtime_path": original_probe_path,
            "sandbox_cleaned": original_probe_path_cleaned,
            "original_probe_runtime_path_present_after_cleanup": (
                False if original_probe_path_cleaned else None),
        },
        "arms_planned": ["Ollama 14B", "Qwen3 235B"],
        "arms_executed": 0,
        "chat_requests": 0,
        "embedding_requests": 0,
        "llm_api_calls": 0,
        "performance_comparison_available": False,
        "quality_comparison_available": False,
        "no_semantic_equivalence_claim": True,
        "blocked_reasons": blocked_reasons,
        "limitations": [
            "D2 provider availability gate blocked; no benchmark executed",
            "no 235B-vs-14B speed or quality conclusion is available",
        ],
        "artifact_sha256_algorithm": CANONICAL_ALGORITHM,
        "artifact_sha256_scope": "D2_RESULT_FIELDS_EXCLUDING_ARTIFACT_SHA256",
    }
    result["artifact_sha256"] = canonical_json_sha256(
        {k: v for k, v in result.items() if k != "artifact_sha256"})
    return result


def load_result(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recomputed = canonical_json_sha256(
        {k: v for k, v in raw.items() if k != "artifact_sha256"})
    if recomputed != raw.get("artifact_sha256"):
        raise ValueError("D2 result artifact_sha256 mismatch (tampered)")
    return raw


def load_final_evidence(path: Path) -> dict:
    """Load final D2 evidence and reject any canonical-content tampering."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    recomputed = canonical_json_sha256(
        {k: v for k, v in raw.items() if k != "artifact_sha256"})
    if recomputed != raw.get("artifact_sha256"):
        raise ValueError("D2 final evidence artifact_sha256 mismatch (tampered)")
    if raw.get("artifact_sha256_algorithm") != CANONICAL_ALGORITHM:
        raise ValueError("D2 final evidence hash algorithm mismatch")
    if raw.get("artifact_sha256_scope") != FINAL_EVIDENCE_SCOPE:
        raise ValueError("D2 final evidence hash scope mismatch")
    return raw


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        raise SystemExit("usage: d2_provider_probe.py <config_root> <output_json>")
    config_root, output = args[0], args[1]
    # read git branch/head/commit (best-effort, offline)
    import subprocess
    def _git(cmd):
        try:
            return subprocess.check_output(["git"] + cmd, text=True,
                                           cwd=config_root).strip()
        except Exception:
            return "unknown"
    branch = _git(["branch", "--show-current"])
    head = _git(["rev-parse", "HEAD"])
    commit = _git(["rev-parse", "--short", "HEAD"])
    probe = build_probe(config_root, branch, head, commit,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_write_refusing_overwrite(Path(output), probe)
    summary = {k: probe[k] for k in
               ("classification", "conclusion", "credential_present",
                "external_provider_request_performed", "llm_api_calls",
                "gpu_used", "artifact_sha256")}
    summary["local_model_available"] = probe["decision_inputs"][
        "local_model_available"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
