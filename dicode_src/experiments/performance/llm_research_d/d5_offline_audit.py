#!/usr/bin/env python3
"""D5 offline retry/error classification for Mason attempt 06.

Only archived JSON/Markdown summaries are read.  The script does not contact
any model endpoint and does not infer a transport root cause that the archived
logs did not record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    out = root / f"retry_diagnostic_{stamp}"
    serial = 1
    while out.exists():
        out = root / f"retry_diagnostic_{stamp}_{serial}"
        serial += 1
    out.mkdir(parents=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=HERE / ".." / ".." / ".." / ".." / "d5_artifacts")
    args = ap.parse_args()

    mason_path = HERE / "MASON_LLM_BASELINE_AUDIT.json"
    d1c_report_path = HERE / "D1C_FINAL_REPORT.md"
    mason = read_json(mason_path)
    volume = mason.get("request_volume", {})
    failures = mason.get("failure_classification", {})
    direct = mason.get("direct_evidence_vs_inference", {})
    retry_by_endpoint = volume.get("retry_by_endpoint", {})

    # The audit explicitly reports endpoint-level retry counts and HTTP status
    # outcomes.  It does not expose connection/timeout/session identifiers.
    categories = {
        "retry_events_total": volume.get("retry_events"),
        "retry_events_chat_completions": retry_by_endpoint.get("chat_completions"),
        "retry_events_embeddings": retry_by_endpoint.get("embeddings"),
        "http_200": volume.get("total_http_200_ok"),
        "http_non_200": volume.get("non_200_http"),
        "parse_failures": failures.get("parse_failures"),
        "empty_responses": failures.get("empty_responses"),
        "compilation_failures": failures.get("compilation_failures"),
        "requeue_events": failures.get("requeue_events"),
        "reflection_events": failures.get("reflection_events"),
        "api_repair_events": failures.get("api_repair_events"),
    }
    unavailable = {
        "connection_reset_vs_pool_contention": None,
        "timeout_events": None,
        "event_loop_identity": None,
        "http_client_identity": None,
        "session_boundary": None,
        "request_id": None,
        "request_start_end_timestamps": None,
        "prompt_or_response_payload": None,
    }
    result = {
        "schema_version": 1,
        "classification": "D5_RETRY_DIAGNOSTIC_OFFLINE_AUDIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INSUFFICIENT_REQUEST_TRACE",
        "conclusion": "NOT_REPRODUCED",
        "root_cause": None,
        "network_calls": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "gpu_processes_started": 0,
        "executed_count": 0,
        "observed_categories": categories,
        "unavailable_request_level_fields": unavailable,
        "transport_evidence": {
            "all_attempt06_requests_eventually_http_200": volume.get("non_200_http") == 0,
            "rate_limit_or_server_error_observed": False,
            "retry_description": volume.get("retry_nature"),
            "direct_evidence": direct.get("direct", []),
            "inference_not_allowed": direct.get("inference", []),
        },
        "source": {
            "mason_audit": {"path": mason_path.name, "sha256": sha256(mason_path)},
            "d1c_report": {"path": d1c_report_path.name, "sha256": sha256(d1c_report_path)},
        },
        "guardrails": [
            "No completion, embedding, or metadata provider request was made.",
            "No credential was read, sourced, or serialized.",
            "No root cause is asserted without request-level trace fields.",
            "No D1/D2 evidence file was modified.",
        ],
    }

    out = make_out_dir(args.output_root.resolve())
    result_path = out / "D5_RESULT.json"
    report_path = out / "D5_REPORT.md"
    result["input_paths"] = [str(p) for p in (mason_path, d1c_report_path)]
    result["script_sha256"] = sha256(Path(__file__).resolve())
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# D5 retry diagnosis (offline)

- Status: **INSUFFICIENT_REQUEST_TRACE**
- Conclusion: **NOT_REPRODUCED**
- Archived request events replayed: `0`
- Network/provider calls: `0`

Attempt 06 reports `{categories['retry_events_total']}` transport-level retry events
(`{categories['retry_events_chat_completions']}` chat and `{categories['retry_events_embeddings']}` embedding),
with `{categories['http_200']}` eventual HTTP 200 responses and `{categories['http_non_200']}` non-200 responses.
Parse, empty-response, compilation, requeue, reflection, and API-repair counters are all zero in the
summary. The evidence does not retain request IDs, connection/timeout subtype, event-loop or client
identity, session boundaries, or per-request timestamps. Consequently the trigger cannot be separated
into connection reset, pool contention, timeout, or server close, and a retry fix would be speculation.

No external endpoint, GPU, credential, or source evidence was touched.
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
