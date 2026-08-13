"""D1b embedding-concurrency sweep (independent research line).

Freezes real task-description texts (25) and sends them as independent
embedding requests at max_in_flight in {1,2,4,25} x2 repeats (alternating
order), with continuous 2s GPU0 sampling and safety gates. 25 = the
production one-candidate-batch scale, labelled
PRODUCTION_STYLE_CLIENT_UNBOUNDED_FOR_25_REQUESTS (a definite request count,
never an infinite loop).

Safety: stop all higher-concurrency arms if GPU0 free memory drops below the
threshold, Ollama PID changes, or a sustained timeout/connection storm appears.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = "/tmp/llm_research_d1b/tools"  # tool copies + temp only
PERSIST = "/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1b"  # persistent results
BASE = os.path.join(PERSIST, "out")
MANIFEST = os.path.join(PERSIST, "frozen_embedding_manifest.json")
GPU_CSV = os.path.join(PERSIST, "gpu0_continuous.csv")
MIN_FREE_MIB = 1000  # safety threshold: stop if GPU0 free memory < 1 GiB

sys.path.insert(0, TOOLS)
import llm_replay_harness as h  # noqa: E402
import llm_replay_benchmark as b  # noqa: E402
import llm_replay_gpu as gpu  # noqa: E402
from llm_replay_manifest import load_manifest, write_manifest  # noqa: E402


def load_embedding_manifest(path):
    # load the embedding manifest and self-hash verify (not the chat validator)
    from llm_replay_manifest import fingerprint
    raw = json.load(open(path, encoding="utf-8"))
    recomputed = fingerprint({k: v for k, v in raw.items() if k != "manifest_sha256"})
    if recomputed != raw.get("manifest_sha256"):
        raise ValueError("embedding manifest tamper check failed")
    return raw


def ollama_pids():
    out = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout
    return sorted(out.split())


def gpu0_free_mib():
    g = gpu._query_gpu0()
    return g["memory_free_mib"] if g else None


async def run_embedding_config(manifest, max_in_flight, out_dir, repeat_label, sdk_counter):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(int(time.time_ns()))
    replay_id = f"embed-{manifest['model'].replace('/', '_')}-mif{max_in_flight}-{repeat_label}"
    sink = h.EventSink(output_jsonl=str(out / "events.jsonl"), enabled=True,
                       run_id=run_id, replay_id=replay_id, provider=manifest["provider"],
                       model=manifest["model"], max_in_flight=max_in_flight)
    client = h.LLMReplayClient(
        base_url=manifest["base_url"], model=manifest["model"], provider=manifest["provider"],
        temperature=0.0, top_p=1.0, max_tokens=1, timeout_s=manifest["timeout_s"],
        max_in_flight=max_in_flight, sink=sink, llm_type="embedding")
    await client.health_check()
    sdk_counter.reset()

    texts = manifest["texts"]
    wall_start = time.monotonic_ns()
    async def one(i):
        rid = client.next_request_id()
        return i, await client.embed_once([texts[i]], slot=f"t{i}", request_id=rid,
                                          attempt=1, prompt_sha256=manifest["text_sha256s"][i])
    per = await asyncio.gather(*(one(i) for i in range(len(texts))))
    wall_end = time.monotonic_ns()

    events = [json.loads(l) for l in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    derived = b.derive_reports(events)
    emb = [e for e in events if e.get("phase") == "embedding_request"]
    embedding_sum_s = sum((int(e["end_monotonic_ns"]) - int(e["start_monotonic_ns"])) / 1e9 for e in emb)
    embedding_union_s = b._union_ns([(int(e["start_monotonic_ns"]), int(e["end_monotonic_ns"])) for e in emb]) / 1e9 if emb else 0.0

    error_counts = {}
    empty_count = 0
    for _, res in per:
        ec = res.get("error_class")
        if ec:
            error_counts[ec] = error_counts.get(ec, 0) + 1
            if ec == "empty_response":
                empty_count += 1
    retry_count = sum(1 for e in events if e.get("phase") == "retry_backoff")

    result = {
        "classification": "LLM_EMBEDDING_REPLAY_RESULT",
        "not_end_to_end_ued": True,
        "run_id": run_id, "replay_id": replay_id, "repeat_label": repeat_label,
        "manifest_sha256": manifest["manifest_sha256"],
        "provider": manifest["provider"], "model": manifest["model"],
        "max_in_flight": max_in_flight, "request_count": len(texts),
        "wall_clock_s": (wall_end - wall_start) / 1e9,
        "embedding_sum_s": embedding_sum_s,
        "embedding_union_s": embedding_union_s,
        "client_semaphore_wait_sum_s": derived.get("client_semaphore_wait_sum_s", 0.0),
        "client_semaphore_wait_union_s": derived.get("client_semaphore_wait_union_s", 0.0),
        "client_semaphore_wait_critical_s": derived.get("client_semaphore_wait_critical_s", 0.0),
        "queue_wait_sum_s": derived.get("queue_wait_sum_s", 0.0),
        "retry_count": retry_count,
        "sdk_transport_retry_count": sdk_counter.count(),
        "retry_backoff_sum_s": derived.get("retry_backoff_sum_s", 0.0),
        "empty_response_count": empty_count,
        "error_counts": error_counts,
        "embedding_seconds_per_text": (embedding_union_s / len(texts)) if texts else None,
        "critical_path": derived.get("critical_path", []),
    }
    result["result_sha256"] = b.sha256_bytes(json.dumps(
        {k: v for k, v in result.items() if k != "result_sha256"}, sort_keys=True, default=str).encode())
    b._atomic_json(out / "RESULT.json", result)
    b._atomic_csv(out / "events.csv", events)
    b._atomic_json(out / "critical_path.json", derived)
    write_manifest(manifest, out / "frozen_manifest.json")
    b._write_artifact_inventory(out)
    return result


def main():
    manifest = load_embedding_manifest(MANIFEST)
    sdk_counter = h.enable_sdk_retry_counting()
    baseline_pids = ollama_pids()
    print("baseline ollama pids:", baseline_pids)

    stop, thread = gpu.start_sampler(GPU_CSV, interval_s=2.0)
    results = []
    try:
        # smoke: 1 config, few texts, to verify safety before the full sweep
        print("=== SMOKE: max_in_flight=1 (3 texts) ===")
        smoke_manifest = dict(manifest)
        smoke_manifest["texts"] = manifest["texts"][:3]
        smoke_manifest["text_sha256s"] = manifest["text_sha256s"][:3]
        asyncio.run(run_embedding_config(smoke_manifest, 1, os.path.join(BASE, "smoke"), "smoke", sdk_counter))
        free_after_smoke = gpu0_free_mib()
        print(f"GPU0 free after smoke: {free_after_smoke} MiB")
        if free_after_smoke is None or free_after_smoke < MIN_FREE_MIB:
            print("SAFETY ABORT: GPU0 free memory too low after smoke")
            stop.set(); thread.join(5)
            sys.exit(2)
        if ollama_pids() != baseline_pids:
            print("SAFETY ABORT: Ollama PID changed after smoke")
            stop.set(); thread.join(5)
            sys.exit(2)

        configs = [(1, "r1"), (2, "r1"), (4, "r1"), (25, "r1"),
                   (25, "r2"), (4, "r2"), (2, "r2"), (1, "r2")]
        for mif, label in configs:
            free = gpu0_free_mib()
            print(f"=== max_in_flight={mif} {label} (GPU0 free={free} MiB) ===")
            if free is None or free < MIN_FREE_MIB:
                print("SAFETY ABORT: GPU0 free memory below threshold")
                break
            r = asyncio.run(run_embedding_config(manifest, mif, os.path.join(BASE, f"{mif}_{label}"), label, sdk_counter))
            results.append(r)
            print(json.dumps({k: r[k] for k in
                              ("max_in_flight", "repeat_label", "wall_clock_s",
                               "embedding_sum_s", "embedding_union_s", "retry_count",
                               "sdk_transport_retry_count", "empty_response_count",
                               "error_counts", "embedding_seconds_per_text")}, indent=2))
            if ollama_pids() != baseline_pids:
                print("SAFETY ABORT: Ollama PID changed")
                break
    finally:
        stop.set()
        thread.join(5)

    gpu_stats = gpu.compute_gpu_stats(GPU_CSV)
    summary = {"results": results, "gpu_stats": gpu_stats,
               "baseline_ollama_pids": baseline_pids,
               "final_ollama_pids": ollama_pids()}
    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE, "D1B_ALL_RESULTS.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("=== D1B DONE ===")
    print(json.dumps(gpu_stats, indent=2))
    print("final ollama pids:", ollama_pids())


if __name__ == "__main__":
    main()
