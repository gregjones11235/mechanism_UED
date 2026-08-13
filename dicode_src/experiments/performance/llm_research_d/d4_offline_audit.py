#!/usr/bin/env python3
"""D4 offline audit for the Mason attempt-06 embedding request shape.

The attempt-06 evidence contains a frozen workload/shape manifest and a
summary of an earlier D1C replay, but no byte-level request/response capture.
This script intentionally performs no network or Ollama call.  It emits an
independent result that records why the planned max-in-flight replay cannot be
claimed from the available evidence.
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_out_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"production_shape_audit_{stamp}"
    serial = 1
    while out.exists():
        out = root / f"production_shape_audit_{stamp}_{serial}"
        serial += 1
    out.mkdir(parents=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=HERE / ".." / ".." / ".." / ".." / "d4_artifacts")
    args = ap.parse_args()

    manifest_path = HERE / "FROZEN_PRODUCTION_EMBEDDING_MANIFEST.json"
    d1c_path = HERE / "D1C_ALL_RESULTS.json"
    mason_path = HERE / "MASON_LLM_BASELINE_AUDIT.json"
    manifest = read_json(manifest_path)
    d1c = read_json(d1c_path)
    mason = read_json(mason_path)

    d1c_results = d1c.get("results", [])
    d1c_ok = sum(
        1
        for item in d1c_results
        for req in item.get("request_results", [])
        if req.get("status") == "ok"
    )
    d1c_errors = sum(
        1
        for item in d1c_results
        for req in item.get("request_results", [])
        if req.get("status") != "ok"
    )
    d1c_retries = sum(int(item.get("total_sdk_retries", 0)) for item in d1c_results)
    audit_embedding_size = mason.get("provider_config", {}).get("embedding_model", {}).get("embedding_size")
    manifest_embedding_size = manifest.get("embedding_size")

    source = {
        "manifest": {"path": manifest_path.name, "sha256": sha256(manifest_path)},
        "d1c_summary": {"path": d1c_path.name, "sha256": sha256(d1c_path)},
        "mason_audit": {"path": mason_path.name, "sha256": sha256(mason_path)},
    }
    result = {
        "schema_version": 1,
        "classification": "D4_EMBEDDING_SHAPE_OFFLINE_AUDIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NO_REPLAY_PAYLOAD",
        "conclusion": "NO_REPLAY_PAYLOAD",
        "eligible_for_mainline": False,
        "network_calls": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "gpu_processes_started": 0,
        "executed_count": 0,
        "planned_matrix": {"max_in_flight": [1, 2, 4], "repeats": 3},
        "executed_matrix": [],
        "payload_availability": {
            "request_bytes": False,
            "response_bytes": False,
            "request_text_bytes": False,
            "task_text_pool_in_manifest": bool(manifest.get("text_pool")),
            "prompt_message_bytes": False,
            "embedding_values": False,
            "shape_metadata": True,
            "batch_and_task_order": True,
        },
        "frozen_shape": {
            "embedding_provider": manifest.get("embedding_provider"),
            "embedding_model": manifest.get("embedding_model"),
            "embedding_size": manifest.get("embedding_size"),
            "text_total": manifest.get("text_total"),
            "batch_size_sequence": manifest.get("batch_size_sequence"),
            "ordered_task_ids": manifest.get("ordered_task_ids"),
            "run_order": manifest.get("run_order"),
            "max_retries": manifest.get("max_retries"),
        },
        "shape_consistency": {
            "manifest_embedding_size": manifest_embedding_size,
            "mason_provider_config_embedding_size": audit_embedding_size,
            "consistent": manifest_embedding_size == audit_embedding_size,
            "note": "The archived manifest and provider-config summary disagree; neither contains byte-level values to resolve the discrepancy.",
        },
        "historical_d1c_context": {
            "arms": len(d1c_results),
            "successful_requests": d1c_ok,
            "failed_requests": d1c_errors,
            "sdk_retries": d1c_retries,
            "shape_observed": manifest.get("embedding_size"),
            "not_a_d4_replay": True,
        },
        "attempt06_context": {
            "mason_root": mason.get("mason_root"),
            "embedding_requests_in_attempt06": mason.get("request_volume", {}).get("embeddings"),
            "attempt06_response_payload_logged": False,
            "attempt06_request_timing_logged": False,
        },
        "source": source,
        "guardrails": [
            "No provider endpoint was contacted.",
            "No Ollama process or GPU was started.",
            "D1C results are cited only as prior context, not counted as D4 arms.",
            "No byte-level request/response payload is available for faithful replay.",
        ],
    }

    out = make_out_dir(args.output_root.resolve())
    result_path = out / "D4_RESULT.json"
    report_path = out / "D4_REPORT.md"
    result["input_paths"] = [str(p) for p in (manifest_path, d1c_path, mason_path)]
    result["script_sha256"] = sha256(Path(__file__).resolve())
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# D4 embedding request-shape audit

- Status: **NO_REPLAY_PAYLOAD**
- Mainline eligible: **no**
- Network/completion/embedding calls: `0/0/0`
- Planned matrix: `max_in_flight={{1,2,4}}`, 3 repeats each
- Executed matrix: none

The frozen manifest preserves model, embedding dimension (`{manifest.get('embedding_size')}`),
16 task order, batch-size sequence, lifecycle labels, and a task-text pool. Mason attempt 06 does
not preserve the exact serialized request message bytes, request/response bytes, returned embedding
values, or request-level timings. Therefore a max-in-flight replay would not be a faithful experiment.
Existing D1C summary rows are retained as historical context only and are not counted as D4 observations. The
manifest reports dimension `{manifest_embedding_size}`, while the archived provider-config summary
reports `{audit_embedding_size}`; this unresolved shape conflict is another replay gate.

No external provider, Ollama endpoint, GPU, or original evidence file was modified.
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
