"""Chat unbounded-concurrency comparison (D1 follow-up).

Uses the SAME 12 synthetic code-generation prompts as D1, adds max_in_flight=12
(labelled PRODUCTION_STYLE_CLIENT_UNBOUNDED_FOR_12_REQUESTS — a definite request
count, not the full production scale), alternating 1 -> 12 -> 12 -> 1. Purpose is
only: does unbounded (12) increase retry / affect union wall / affect valid rate.
It is NOT a full production unbounded control (production batches may exceed 12).
"""
import asyncio
import json
import os
import sys
import time

TOOLS = "/tmp/llm_research_d1b/tools"
PERSIST = "/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1b"
sys.path.insert(0, TOOLS)

from llm_replay_manifest import load_manifest  # noqa: E402
from llm_replay_benchmark import run_replay  # noqa: E402

manifest = load_manifest(os.path.join(PERSIST, "frozen_chat_manifest.json"))

configs = [(1, "c1"), (12, "c1"), (12, "c2"), (1, "c2")]
results = []
for mif, label in configs:
    out = os.path.join(PERSIST, "out_chat_unbounded", f"{mif}_{label}")
    print(f"=== max_in_flight={mif} {label} ===", flush=True)
    r = asyncio.run(run_replay(manifest, max_in_flight=mif, out_dir=out,
                               repeat_label=label, do_cpu_jax=True, enabled_events=True))
    results.append(r)
    print(json.dumps({k: r[k] for k in
                      ("max_in_flight", "repeat_label", "wall_clock_s",
                       "llm_union_s", "retry_count", "sdk_transport_retry_count",
                       "empty_response_count", "valid_tasks", "valid_task_rate",
                       "llm_seconds_per_valid_task")}, indent=2), flush=True)

with open(os.path.join(PERSIST, "out_chat_unbounded", "CHAT_UNBOUNDED_RESULTS.json"), "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)
print("=== CHAT UNBOUNDED DONE ===", flush=True)
