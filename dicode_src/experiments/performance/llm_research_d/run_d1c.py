#!/usr/bin/env python3
"""D1c production-shape embedding retry replay orchestrator.

Arms:
  A  PERSISTENT_CONTIGUOUS  — one long-lived client, contiguous batched requests
  B  FRESH_CLIENT_PER_REQUEST — a new client per batched request
  C  PERSISTENT_IDLE_GAP   — one long-lived client, fixed idle gap (30s / 120s)
  D  NO_SDK_RETRY_DIAGNOSTIC — only if A/B/C shows retry; max_retries=0 capture
  E  (skipped: see NOT_APPLICABLE if embedding/chat share no real idle pattern)

Each arm replays the frozen production batch sequence (one batched request per
batch) with 2 repeats. Continuous 2s GPU0 sampling covers every arm wall clock.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = "/tmp/llm_research_d1c/tools"
PERSIST = "/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1c"
MANIFEST = os.path.join(PERSIST, "FROZEN_PRODUCTION_EMBEDDING_MANIFEST.json")
sys.path.insert(0, TOOLS)

from d1c_harness import D1CEmbeddingClient, EventWriter  # noqa: E402
from llm_replay_harness import enable_sdk_retry_counting  # noqa: E402
from llm_replay_manifest import fingerprint  # noqa: E402
import llm_replay_benchmark as b  # noqa: E402
import llm_replay_gpu as gpu  # noqa: E402


def load_production_manifest(path):
    raw = json.load(open(path, encoding="utf-8"))
    recomputed = fingerprint({k: v for k, v in raw.items() if k != "manifest_sha256"})
    if recomputed != raw.get("manifest_sha256"):
        raise ValueError("production embedding manifest tamper check failed")
    return raw

# reuse b91e50b hash contract
HASH_LEGACY = "legacy_default_json_sha256"
HASH_CANONICAL = "canonical_json_sha256"


def ollama_pids():
    out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout
    return sorted(p for p in out.split() if p.strip())


async def run_arm(manifest, *, arm, client_lifecycle, idle_gap_s, repeat, out_dir, sdk_counter):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"d1c-{arm}-{repeat}-{int(time.time_ns())}"
    writer = EventWriter(str(out / "events.jsonl"))
    client = D1CEmbeddingClient(
        base_url=manifest["base_url"], model=manifest["embedding_model"],
        api_key="token-", timeout_s=manifest["timeout_s"],
        max_retries=manifest["max_retries"], embedding_size=manifest["embedding_size"])

    text_pool = manifest["text_pool"]
    sizes = manifest["batch_size_sequence"]
    results = []
    try:
        for bi, size in enumerate(sizes):
            if client_lifecycle == "fresh":
                # fresh: a brand-new client per request (diagnostic arm)
                await client.aclose()
                client.reset()
            texts = text_pool[:size]
            if idle_gap_s and bi > 0:
                await asyncio.sleep(idle_gap_s)
            pid_before = ollama_pids()
            r = await client.embed(
                texts, manifest["instruction"], sdk_counter, writer,
                run_id=run_id, arm=arm, repeat=repeat, request_id=f"{run_id[:8]}-{bi:02d}",
                batch_index=bi, batch_size=size, client_lifecycle=client_lifecycle,
                idle_gap_s=idle_gap_s, ollama_pid_before=pid_before,
                ollama_pid_after_getter=ollama_pids)
            results.append(r)
    finally:
        await client.aclose()

    retry_requests = [r for r in results if r.get("sdk_retry_count", 0) > 0]
    error_requests = [r for r in results if r.get("error_class")]
    result = {
        "classification": "D1C_EMBEDDING_ARM_RESULT",
        "not_end_to_end_ued": True,
        "run_id": run_id, "arm": arm, "repeat": repeat,
        "client_lifecycle": client_lifecycle, "idle_gap_s": idle_gap_s,
        "manifest_sha256": manifest["manifest_sha256"],
        "batch_count": len(sizes), "batch_size_sequence": sizes,
        "request_results": results,
        "sdk_retry_request_count": len(retry_requests),
        "total_sdk_retries": sum(r.get("sdk_retry_count", 0) for r in results),
        "error_request_count": len(error_requests),
        "error_classes": {r["error_class"]: sum(1 for x in results if x.get("error_class") == r["error_class"])
                          for r in error_requests},
        "result_sha256_scope": "RESULT_FIELDS_EXCLUDING_RESULT_SHA256_AND_ARTIFACT_INVENTORY",
        "result_sha256_algorithm": HASH_LEGACY,
        "legacy_result_hash_algorithm": HASH_LEGACY,
        "canonical_summary_hash_algorithm": HASH_CANONICAL,
    }
    result["result_sha256"] = b.legacy_json_sha256(
        {k: v for k, v in result.items() if k != "result_sha256"})
    b._atomic_json(out / "RESULT.json", result)
    _write_d1c_csv(out / "events.csv", (out / "events.jsonl"))
    b._atomic_json(out / "critical_path.json", {"note": "D1c embedding arm; see RESULT.json"})
    # artifact inventory + enriched hash
    inv = b._write_artifact_inventory(out)
    result["artifact_inventory"] = inv
    result["artifact_inventory_sha256"] = b.canonical_json_sha256(inv)
    result["enriched_summary_sha256_scope"] = "ENRICHED_SUMMARY_FIELDS_EXCLUDING_ENRICHED_SUMMARY_SHA256"
    result["enriched_summary_sha256_algorithm"] = HASH_CANONICAL
    result["enriched_summary_sha256"] = b.canonical_json_sha256(
        {k: v for k, v in result.items() if k != "enriched_summary_sha256"})
    return result


ARMS = [
    ("A", "persistent", 0.0),
    ("B", "fresh", 0.0),
    ("C30", "persistent", 30.0),
    ("C120", "persistent", 120.0),
]


def _write_d1c_csv(csv_path, jsonl_path):
    import csv
    from d1c_harness import EVENT_FIELDS
    rows = [json.loads(l) for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r.get(k) for k in EVENT_FIELDS} for r in rows)


def main():
    manifest = load_production_manifest(MANIFEST)
    sdk_counter = enable_sdk_retry_counting()
    baseline_pids = ollama_pids()
    print("baseline ollama pids:", baseline_pids)
    stop, thread = gpu.start_sampler(os.path.join(PERSIST, "gpu0_memory_2s.csv"), interval_s=2.0)
    all_results = []
    try:
        for arm, lifecycle, gap in ARMS:
            for repeat in ("r1", "r2"):
                print(f"=== arm {arm} {repeat} (lifecycle={lifecycle}, gap={gap}s) ===", flush=True)
                out = os.path.join(PERSIST, "out", f"{arm}_{repeat}")
                r = asyncio.run(run_arm(manifest, arm=arm, client_lifecycle=lifecycle,
                                        idle_gap_s=gap, repeat=repeat, out_dir=out,
                                        sdk_counter=sdk_counter))
                all_results.append(r)
                print(json.dumps({k: r[k] for k in
                                  ("arm", "repeat", "batch_count", "sdk_retry_request_count",
                                   "total_sdk_retries", "error_request_count", "error_classes")},
                                 sort_keys=True), flush=True)
                if ollama_pids() != baseline_pids:
                    print("SAFETY ABORT: Ollama PID changed", flush=True)
                    break
    finally:
        stop.set()
        thread.join(5)

    gpu_stats = gpu.compute_gpu_stats(os.path.join(PERSIST, "gpu0_memory_2s.csv"))
    summary = {"results": all_results, "gpu_stats": gpu_stats,
               "baseline_ollama_pids": baseline_pids, "final_ollama_pids": ollama_pids()}
    with open(os.path.join(PERSIST, "D1C_ALL_RESULTS.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("=== D1C DONE ===")
    print(json.dumps(gpu_stats, sort_keys=True))
    print("final ollama pids:", ollama_pids())


if __name__ == "__main__":
    main()
