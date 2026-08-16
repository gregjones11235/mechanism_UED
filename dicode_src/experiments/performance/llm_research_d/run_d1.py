"""D1 experiment: same-model scheduling sweep max_in_flight in {1,2,4} x2 repeats.

Runs all configs in ONE process (amortizes the ~130s one-time craftax import),
in alternating order (1,2,4,4,2,1) to avoid fixed first-round bias. Samples GPU0
memory before/after each config. The 235B arm of D2 is BLOCKED_EXTERNAL_PROVIDER
(no local 235B service, no DEEPINFRA_API_KEY) and is NOT run here.
"""
import asyncio
import json
import os
import subprocess
import sys

TOOLS = "/tmp/llm_research_d_BrRxKL/tools"
BASE = "/tmp/llm_research_d_BrRxKL/d1_out"
sys.path.insert(0, TOOLS)

from llm_replay_manifest import load_manifest  # noqa: E402
from llm_replay_benchmark import run_replay  # noqa: E402

manifest = load_manifest("/tmp/llm_research_d_BrRxKL/frozen_manifest.json")


def gpu0_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("0,"):
            parts = [p.strip() for p in line.split(",")]
            # parts = [index, uuid, util%, usedMiB, totalMiB]
            used = int(parts[3].split()[0])
            total = int(parts[4].split()[0])
            return {"util": parts[2], "used_mib": used, "total_mib": total}
    return {}


# Warm up craftax (amortize ~130s one-time import + texture load) before timing.
from llm_replay_harness import static_lint  # noqa: E402
print("warming up craftax...", flush=True)
static_lint("from craftax.craftax.constants import BlockType\nx = BlockType.COAL")
print("warm-up done", flush=True)

# Alternating order to avoid fixed first-round bias.
configs = [
    (1, "r1"), (2, "r1"), (4, "r1"),
    (4, "r2"), (2, "r2"), (1, "r2"),
]

results = []
gpu_log = {}
for mif, label in configs:
    gpu_log[f"before_{mif}_{label}"] = gpu0_mem()
    out = os.path.join(BASE, f"{mif}_{label}")
    print(f"=== max_in_flight={mif} {label} ===", flush=True)
    r = asyncio.run(run_replay(
        manifest, max_in_flight=mif, out_dir=out, repeat_label=label,
        do_cpu_jax=True, enabled_events=True))
    gpu_log[f"after_{mif}_{label}"] = gpu0_mem()
    results.append(r)
    print(json.dumps({k: r[k] for k in
                      ("max_in_flight", "repeat_label", "wall_clock_s",
                       "llm_union_s", "llm_sum_s", "queue_wait_sum_s",
                       "retry_count", "empty_response_count", "valid_tasks",
                       "valid_task_rate", "llm_seconds_per_valid_task",
                       "unique_code_hashes", "static_invalid", "jax_failed")},
                     indent=2), flush=True)

with open(os.path.join(BASE, "ALL_RESULTS.json"), "w") as f:
    json.dump({"results": results, "gpu_log": gpu_log}, f, indent=2, sort_keys=True)
print("=== D1 DONE ===", flush=True)
