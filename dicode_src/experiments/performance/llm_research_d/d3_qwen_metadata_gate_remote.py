#!/usr/bin/env python3
"""Read-only remote D3 Qwen metadata gate.

This script is sent over SSH on stdin and never written on the target host.
It reads only the two Qwen variables from the declared experiment env file and
the Qwen-bearing lines of the existing launcher.  It performs one Ollama
``/api/tags`` GET and, when all local safety checks pass, one Qwen ``/models``
GET.  No completion, embedding, shell evaluation, DeepSeek value, or GPU
process is started.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ENV_PATH = Path("/home/oseasy/.config/dicode/experiment_llm.env")
LAUNCHER_PATH = Path("/home/oseasy/.config/dicode/run_four_api_smoke.sh")
OLLAMA_MODEL = "qwen2.5-coder:14b"
GPU2_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"
QWEN_BASE_VAR = "EXP_QWEN_BASE_URL"
QWEN_CRED_VAR = "EXP_QWEN_API_KEY"


def _reject_tokens(value: str) -> None:
    if "$(" in value or "`" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("unsafe_value")


def _parse_qwen_value(raw: str) -> str:
    value = raw.strip()
    _reject_tokens(value)
    if not value:
        return ""
    if value[0] in "'\"":
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid_quoted_value") from exc
        if not isinstance(decoded, str):
            raise ValueError("invalid_quoted_value")
        _reject_tokens(decoded)
        return decoded
    return value


def read_qwen_env() -> tuple[dict[str, str], dict[str, bool]]:
    values: dict[str, str] = {}
    duplicate = False
    invalid = False
    if not ENV_PATH.exists():
        return values, {"env_exists": False, "invalid": False, "duplicate": False}
    for raw in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            invalid = True
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        # Do not inspect or retain values for any non-Qwen variable.
        if key not in {QWEN_BASE_VAR, QWEN_CRED_VAR}:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            invalid = True
            continue
        if key in values:
            duplicate = True
            continue
        try:
            values[key] = _parse_qwen_value(value)
        except ValueError:
            invalid = True
    return values, {"env_exists": True, "invalid": invalid, "duplicate": duplicate}


def launcher_qwen_models() -> tuple[list[str], bool]:
    if not LAUNCHER_PATH.exists():
        return [], False
    candidates: set[str] = set()
    # Read only Qwen-bearing lines; non-Qwen lines are never retained.
    for raw in LAUNCHER_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        low = line.lower()
        if "qwen" not in low:
            continue
        # The launcher declares the model in a Qwen-bearing ``--arg`` line;
        # collect only qwen-* tokens so unrelated provider lines are ignored.
        for token in re.findall(r"(?i)\bqwen[-A-Za-z0-9_./:+]*", line):
            token = token.strip("\"' ,)")
            if token.lower().startswith("qwen-") and token.lower() not in {"qwen.json"}:
                candidates.add(token)
    return sorted(candidates), True


def safe_base_url(raw: str) -> str:
    parsed = urllib.parse.urlsplit(raw.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid_base_url")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("unsafe_base_url")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def gpu_snapshot() -> dict[str, object]:
    result: dict[str, object] = {
        "query_ok": False,
        "compute_query_ok": False,
        "gpus": [],
        "gpu2": None,
        "gpu2_uuid_ok": False,
        "gpu2_free_ge_4gib": False,
        "gpu2_external_pid_clear": False,
    }
    try:
        smi = shutil.which("nvidia-smi")
        if smi is None and Path("/usr/bin/nvidia-smi").exists():
            smi = "/usr/bin/nvidia-smi"
        if smi is None:
            return result
        completed = subprocess.run(
            [
                smi,
                "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return result
    if completed.returncode != 0:
        return result
    fields = ("index", "uuid", "memory_free_mib", "memory_total_mib", "utilization_gpu")
    compute_pids: dict[str, list[str]] = {}
    try:
        apps = subprocess.run(
            [smi, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if apps.returncode == 0:
            result["compute_query_ok"] = True
            for raw in apps.stdout.splitlines():
                parts = [item.strip() for item in raw.split(",")]
                if len(parts) == 2 and parts[0] and parts[1] and parts[1] not in {"[N/A]", "N/A"}:
                    compute_pids.setdefault(parts[0], []).append(parts[1])
    except Exception:
        pass
    rows = []
    for raw in completed.stdout.splitlines():
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != len(fields):
            continue
        row = dict(zip(fields, parts))
        try:
            row["index"] = int(row["index"])
            row["memory_free_mib"] = int(float(row["memory_free_mib"]))
            row["memory_total_mib"] = int(float(row["memory_total_mib"]))
            row["utilization_gpu"] = int(float(row["utilization_gpu"]))
        except (TypeError, ValueError):
            continue
        row["compute_pid"] = ",".join(compute_pids.get(row["uuid"], []))
        row["external_pid_present"] = bool(row["compute_pid"])
        rows.append(row)
    result["query_ok"] = True
    result["gpus"] = rows
    gpu2 = next((row for row in rows if row.get("index") == 2), None)
    result["gpu2"] = gpu2
    result["gpu2_uuid_ok"] = bool(gpu2 and gpu2.get("uuid") == GPU2_UUID)
    result["gpu2_free_ge_4gib"] = bool(gpu2 and gpu2.get("memory_free_mib", 0) >= 4096)
    result["gpu2_external_pid_clear"] = bool(
        result["compute_query_ok"] and gpu2 and not gpu2.get("external_pid_present", True)
    )
    return result


def check_models(url: str, expected_model: str, headers: dict[str, str], secret: str | None) -> dict[str, object]:
    body = b""
    try:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", **headers},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=10.0) as response:
            status = int(getattr(response, "status", 200))
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            body = exc.read()
        except Exception:
            body = b""
        if secret and secret.encode("utf-8") in body:
            return {
                "http_status": status,
                "model_ids": [],
                "exact_model_available": False,
                "credential_echo": True,
                "error_class": "credential_echo_detected",
            }
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "unauthorized" if status == 401 else "http_error",
        }
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        return {
            "http_status": None,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "transport_error",
        }
    if secret and secret.encode("utf-8") in body:
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": True,
            "error_class": "credential_echo_detected",
        }
    if status == 401:
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "unauthorized",
        }
    if status < 200 or status >= 300:
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "http_error",
        }
    try:
        data = json.loads(body.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "invalid_json",
        }
    if not isinstance(data, dict):
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "invalid_model_list",
        }
    raw_models = data.get("models") if url.endswith("/api/tags") else data.get("data")
    if not isinstance(raw_models, list):
        return {
            "http_status": status,
            "model_ids": [],
            "exact_model_available": False,
            "credential_echo": False,
            "error_class": "invalid_model_list",
        }
    field = "name" if url.endswith("/api/tags") else "id"
    model_ids = [str(item[field]) for item in raw_models if isinstance(item, dict) and item.get(field) is not None]
    if not url.endswith("/api/tags"):
        # The provider's catalog can contain other vendors.  Never persist or
        # print those IDs; D3 only needs the Qwen namespace and exact match.
        model_ids = [model_id for model_id in model_ids if "qwen" in model_id.lower()]
    return {
        "http_status": status,
        "model_ids": model_ids,
        "exact_model_available": expected_model in model_ids,
        "credential_echo": False,
        "error_class": None,
    }


def main() -> int:
    values, env_meta = read_qwen_env()
    launcher_models, launcher_exists = launcher_qwen_models()
    before = gpu_snapshot()
    result: dict[str, object] = {
        "classification": "D3_QWEN_METADATA_GATE",
        "schema_version": 1,
        "status": "BLOCKED",
        "conclusion": "BLOCKED",
        "provider": "qwen_api",
        "small_model": OLLAMA_MODEL,
        "large_model": launcher_models[0] if len(launcher_models) == 1 else None,
        "large_model_source": "run_four_api_smoke.sh:qwen_model",
        "env_variable_names": [QWEN_BASE_VAR, QWEN_CRED_VAR],
        "credential_variable": QWEN_CRED_VAR,
        "base_url_variable": QWEN_BASE_VAR,
        "credential_present": bool(values.get(QWEN_CRED_VAR)),
        "credential_value_serialized": False,
        "launcher_exists": launcher_exists,
        "launcher_model_ids": launcher_models,
        "env": env_meta,
        "ollama": {"request_count": 0},
        "qwen": {"request_count": 0},
        "completion_requests": 0,
        "embedding_requests": 0,
        "gpu_before": before,
        "gpu_after": None,
        "gpu2_uuid_expected": GPU2_UUID,
        "blocked_reasons": [],
        "secret_values_serialized": False,
    }
    reasons: list[str] = []
    if not env_meta["env_exists"]:
        reasons.append("qwen_env_missing")
    elif env_meta["invalid"]:
        reasons.append("qwen_env_invalid")
    elif env_meta["duplicate"]:
        reasons.append("qwen_env_duplicate")
    elif not values.get(QWEN_BASE_VAR):
        reasons.append("qwen_base_url_missing")
    elif not values.get(QWEN_CRED_VAR):
        reasons.append("qwen_credential_missing")
    elif not launcher_exists or len(launcher_models) != 1:
        reasons.append("qwen_launcher_model_uncertain")
    elif not before["query_ok"] or not before["gpu2_uuid_ok"] or not before["gpu2_free_ge_4gib"] or not before["gpu2_external_pid_clear"]:
        reasons.append("gpu2_preflight_gate_failed")
    else:
        try:
            qwen_base = safe_base_url(values[QWEN_BASE_VAR])
        except ValueError as exc:
            reasons.append(str(exc))
        else:
            ollama = check_models("http://127.0.0.1:11434/api/tags", OLLAMA_MODEL, {}, None)
            result["ollama"] = {"request_count": 1, **ollama}
            if not ollama["exact_model_available"]:
                reasons.append("ollama_model_missing_or_unavailable")
            else:
                qwen = check_models(
                    qwen_base + "/models",
                    result["large_model"],
                    {"Authorization": "Bearer " + values[QWEN_CRED_VAR]},
                    values[QWEN_CRED_VAR],
                )
                result["qwen"] = {"request_count": 1, **qwen}
                if qwen["credential_echo"]:
                    reasons.append("qwen_credential_echo_detected")
                elif qwen["error_class"] == "unauthorized":
                    reasons.append("qwen_unauthorized")
                elif not qwen["exact_model_available"]:
                    reasons.append("qwen_model_missing_or_unavailable")
    after = gpu_snapshot()
    result["gpu_after"] = after
    before_gpu2 = before.get("gpu2")
    after_gpu2 = after.get("gpu2")
    if before_gpu2 and after_gpu2:
        if after_gpu2.get("uuid") != GPU2_UUID or after_gpu2.get("memory_free_mib", 0) < 4096 or after_gpu2.get("external_pid_present"):
            reasons.append("gpu2_post_gate_failed")
    result["blocked_reasons"] = sorted(set(reasons))
    result["status"] = "PASS" if not reasons else "BLOCKED"
    result["conclusion"] = result["status"]
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
