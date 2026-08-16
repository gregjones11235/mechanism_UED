#!/usr/bin/env python3
"""Materialize the read-only server D3 gate and credential-safety disposition.

This is a report generator only: it has no network code, reads no environment
variables, and never handles provider credentials.  The observations were
collected by the separately controlled SSH metadata gate and are recorded here
without a command transcript or sensitive value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def make_out_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"server_security_gate_{stamp}"
    serial = 1
    while out.exists():
        out = root / f"server_security_gate_{stamp}_{serial}"
        serial += 1
    out.mkdir(parents=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=HERE / ".." / ".." / ".." / ".." / "d3_artifacts")
    args = ap.parse_args()

    artifact_root = HERE / ".." / ".." / ".." / ".." / "d3_artifacts" / "metadata_gate_20260814_025305"
    contract_path = artifact_root / "D3_CONTRACT.json"
    local_gate_path = artifact_root / "D3_METADATA_GATE.json"
    facts = {
        "server": {"host": "172.25.14.221", "hostname": "i-00000226", "observation_mode": "read_only_ssh"},
        "ollama": {
            "tags_http_status": 200,
            "required_model": "qwen2.5-coder:14b",
            "required_model_present": True,
            "model_parameter_size": "14.8B",
            "model_quantization": "Q4_K_M",
            "service_mutated": False,
        },
        "gpu": {
            "gpu2_index": 2,
            "gpu2_uuid": "GPU-8df11537-ab79-722d-606f-411966196c4c",
            "gpu2_memory_used_mib": 1,
            "gpu2_memory_free_mib": 45619,
            "gpu2_compute_processes": [],
            "gpu2_gate": "PASS",
            "gpu3_touched": False,
        },
        "deepseek": {
            "models_endpoint": "/v1/models",
            "first_metadata_http_status": 401,
            "metadata_authenticated": False,
            "model_metadata_available": False,
            "completion_requests": 0,
            "embedding_requests": 0,
            "credential_safety_event": "A malformed shell retry exposed a credential fragment in operator-visible output; value is intentionally absent from this artifact.",
            "credential_value_recorded": False,
            "credential_rotation_required": True,
            "further_external_calls": "STOPPED",
        },
        "remote_safety": {
            "remote_processes_started": 0,
            "remote_files_modified": 0,
            "completion_or_embedding_sent": False,
            "gpu2_smoke_started": False,
        },
    }
    result = {
        "schema_version": 1,
        "classification": "D3_SERVER_METADATA_SECURITY_GATE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_EXTERNAL_PROVIDER_CREDENTIAL_EXPOSURE",
        "conclusion": "BLOCKED_EXTERNAL_PROVIDER_CREDENTIAL_EXPOSURE",
        "eligible_for_gpu2_smoke": False,
        "network_calls_after_safety_stop": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "gpu_processes_started": 0,
        "executed_count": 0,
        "input_paths": [str(contract_path), str(local_gate_path)],
        "source": {
            "contract_sha256": sha256(contract_path),
            "local_gate_sha256": sha256(local_gate_path) if local_gate_path.exists() else None,
            "remote_observation_sha256": canonical_sha256(facts),
            "transcript_persisted": False,
        },
        "facts": facts,
        "guardrails": [
            "No secret value, token fragment, or authorization header is serialized.",
            "No environment file is read by this script.",
            "No further external provider request is permitted without credential rotation and explicit re-approval.",
            "The local Windows gate is not used as a server conclusion; server Ollama/GPU2 facts are separate.",
        ],
    }
    out = make_out_dir(args.output_root.resolve())
    result_path = out / "D3_SERVER_SECURITY_RESULT.json"
    report_path = out / "D3_SERVER_SECURITY_REPORT.md"
    result["script_sha256"] = sha256(Path(__file__).resolve())
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = """# D3 server metadata and credential-safety gate

- Server Ollama `qwen2.5-coder:14b`: **PASS** (`/api/tags` HTTP 200; exact model present)
- Dedicated GPU2: **PASS** (UUID recorded; 1 MiB used, no compute PID)
- DeepSeek `/v1/models`: **BLOCKED** (HTTP 401 in the initial metadata gate)
- Safety event: a malformed metadata-only shell retry exposed a credential fragment in operator-visible output; no value is retained here
- Completion requests: `0`; embedding requests: `0`; post-stop external calls: `0`
- Final disposition: **BLOCKED_EXTERNAL_PROVIDER_CREDENTIAL_EXPOSURE**

The server gate is independent of the earlier local-Windows blocked gate. No remote process,
file, GPU, completion, or embedding was started or modified. Further external-provider work is
stopped pending credential rotation and explicit re-approval.
"""
    report_path.write_text(report, encoding="utf-8")
    result["report_sha256"] = sha256(report_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = "\n".join(f"{sha256(p)}  {p.name}" for p in (result_path, report_path)) + "\n"
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
