"""D1b batch-mode embedding test (16-text batches, the production embedding shape).

The production ``get_embedding`` sends a LIST of texts as ONE batched
/embeddings request (Mason's "one-hot embeddings for N tasks", N up to 16). This
sends 16-text batches (and a 25-text full batch) at max_in_flight 1/4/25 to test
whether the BATCHED shape — as opposed to the unbatched shape in run_d1b.py —
triggers the SDK transport retries Mason observed (~100%).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

TOOLS = "/tmp/llm_research_d1b/tools"
PERSIST = "/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1b"
MANIFEST = os.path.join(PERSIST, "frozen_embedding_manifest.json")
sys.path.insert(0, TOOLS)

import llm_replay_harness as h  # noqa: E402
import llm_replay_benchmark as b  # noqa: E402
from llm_replay_manifest import fingerprint  # noqa: E402


def load_manifest(path):
    raw = json.load(open(path, encoding="utf-8"))
    assert fingerprint({k: v for k, v in raw.items() if k != "manifest_sha256"}) == raw["manifest_sha256"]
    return raw


async def run_batches(manifest, batches, max_in_flight, out_dir, label, sdk_counter):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(int(time.time_ns()))
    replay_id = f"embed-batch-mif{max_in_flight}-{label}"
    sink = h.EventSink(output_jsonl=str(out / "events.jsonl"), enabled=True,
                       run_id=run_id, replay_id=replay_id, provider=manifest["provider"],
                       model=manifest["model"], max_in_flight=max_in_flight)
    client = h.LLMReplayClient(base_url=manifest["base_url"], model=manifest["model"],
                               provider=manifest["provider"], temperature=0.0, top_p=1.0,
                               max_tokens=1, timeout_s=manifest["timeout_s"],
                               max_in_flight=max_in_flight, sink=sink, llm_type="embedding")
    await client.health_check()
    sdk_counter.reset()

    wall_start = time.monotonic_ns()
    async def one(i):
        rid = client.next_request_id()
        return i, await client.embed_once(batches[i], slot=f"b{i}", request_id=rid,
                                          attempt=1, prompt_sha256=manifest["texts_sha256"])
    per = await asyncio.gather(*(one(i) for i in range(len(batches))))
    wall_end = time.monotonic_ns()

    events = [json.loads(l) for l in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    error_counts = {}
    for _, res in per:
        ec = res.get("error_class")
        if ec:
            error_counts[ec] = error_counts.get(ec, 0) + 1
    result = {
        "classification": "LLM_EMBEDDING_BATCH_RESULT",
        "max_in_flight": max_in_flight, "label": label,
        "batch_sizes": [len(x) for x in batches], "num_batches": len(batches),
        "wall_clock_s": (wall_end - wall_start) / 1e9,
        "sdk_transport_retry_count": sdk_counter.count(),
        "error_counts": error_counts,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    result["result_sha256"] = b.sha256_bytes(json.dumps(
        {k: v for k, v in result.items() if k != "result_sha256"}, sort_keys=True, default=str).encode())
    b._atomic_json(out / "RESULT.json", result)
    b._atomic_csv(out / "events.csv", events)
    return result


def main():
    manifest = load_manifest(MANIFEST)
    sdk_counter = h.enable_sdk_retry_counting()
    texts = manifest["texts"]
    # production shape: 16-text batch (late-stage scale) + a 9-text remainder
    batches = [texts[:16], texts[16:25]]  # [16, 9]
    # for concurrency testing, replicate the 16-text batch to 25 requests
    batch16 = texts[:16]
    concurrency_batches = [batch16 for _ in range(25)]

    results = []
    results.append(asyncio.run(run_batches(manifest, batches, 1,
                                           os.path.join(PERSIST, "out_batch", "1_single"), "single", sdk_counter)))
    for mif in (1, 4, 25):
        r = asyncio.run(run_batches(manifest, concurrency_batches, mif,
                                    os.path.join(PERSIST, "out_batch", f"{mif}_25x16"), f"{mif}x25", sdk_counter))
        results.append(r)
        print(json.dumps({k: r[k] for k in ("max_in_flight", "label", "batch_sizes",
                                             "num_batches", "wall_clock_s",
                                             "sdk_transport_retry_count", "error_counts")}))
    with open(os.path.join(PERSIST, "out_batch", "BATCH_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("=== BATCH DONE ===")


if __name__ == "__main__":
    main()
