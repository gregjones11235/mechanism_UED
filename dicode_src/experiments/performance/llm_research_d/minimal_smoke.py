"""Minimal prompt smoke against the real Ollama 14B service (GPU0).

Sends two small code prompts through the harness (max_in_flight=1, no CPU-JAX)
to verify the full pipeline: health-check -> generation -> events -> RESULT.
Records GPU0 memory before/after to prove the smoke does not disturb Ollama.
"""
import asyncio
import json
import os
import subprocess
import sys

TOOLS = "/tmp/llm_research_d_BrRxKL/tools"
OUT = "/tmp/llm_research_d_BrRxKL/smoke_out"
sys.path.insert(0, TOOLS)

import llm_replay_manifest as m
from llm_replay_benchmark import run_replay


def gpu0():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("0,"):
            return line.strip()
    return "gpu0 not found"


print("=== GPU0 before ===")
print(gpu0())

spec = {
    "classification": "LLM_REPLAY_MANIFEST",
    "source_commit": "91a75e5",
    "provider": "local", "model": "qwen2.5-coder:14b",
    "base_url": "http://127.0.0.1:11434/v1",
    "temperature": 0.6, "top_p": 0.95, "max_tokens": 256,
    "timeout_s": 300, "max_retries": 2, "repair_limit": 0,
    "system_prompt": "You are a helpful coding assistant. Return only Python code.",
    "user_prompts": [
        "Write a Python function that adds two numbers.",
        "Write a Python function that returns the squares of a list.",
    ],
    "candidate_slots": ["smoke_0", "smoke_1"],
    "request_order": [{"index": 0, "slot": "smoke_0", "kind": "code"},
                      {"index": 1, "slot": "smoke_1", "kind": "code"}],
    "validation": {"static_lint": True, "cpu_jax": False, "dedup_by_code_hash": True},
}
man = m.build_replay_manifest(spec)
man = m.write_manifest(man, os.path.join(TOOLS, "smoke_manifest.json"))

result = asyncio.run(run_replay(
    man, max_in_flight=1, out_dir=OUT, repeat_label="smoke",
    do_cpu_jax=False, enabled_events=True))

print("=== RESULT ===")
print(json.dumps({k: result[k] for k in
                  ("run_id", "replay_id", "max_in_flight", "wall_clock_s",
                   "llm_union_s", "llm_sum_s", "queue_wait_sum_s", "retry_count",
                   "empty_response_count", "valid_tasks", "error_counts",
                   "llm_seconds_per_valid_task")}, indent=2))

print("=== GPU0 after ===")
print(gpu0())
print("=== events.jsonl lines ===")
print(sum(1 for _ in open(os.path.join(OUT, "events.jsonl"))))
print("=== SMOKE DONE ===")
