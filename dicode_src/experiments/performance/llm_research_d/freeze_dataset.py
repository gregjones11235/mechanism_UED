"""Freeze a representative evolution-input replay dataset from Mason attempt_06.

Reads the REAL gen_env system prompt (from the prompt module at source commit
91a75e5) and REAL task descriptions (from the run's task_graph.graphml), then
builds a frozen manifest. The system prompt and user prompts are SHA256-pinned;
the manifest is atomically written and reload-verified.

This is a RECONSTRUCTED freeze (the Mason run did not log prompt bytes): the
prompt CONTENT is rebuilt from the exact source commit + archive, and this is
declared explicitly — it is NOT a byte-for-byte capture of any logged prompt.
"""
import importlib
import json
import os
import re
import sys

S = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/source/dicode_src"
GRAPHML = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/attempt_06/run/task_graph.graphml"
sys.path.insert(0, os.path.join(S, "src"))

import networkx as nx  # noqa: E402

from llm_replay_manifest import build_replay_manifest, write_manifest, load_manifest  # noqa: E402

# --- prompt module + contexts (mirror EnvGenerator.__init__ at 91a75e5) ---
gen_env = importlib.import_module("dicode.dreaming.prompts.cl_.gen_env")
_kbd = importlib.import_module("dicode.dreaming.prompts.cl_.knowledge_base_designer")
craftax_code = getattr(_kbd, "context", None) or _kbd.knowledge_base_designer
wrapper_code = importlib.import_module("dicode.dreaming.prompts.cl_.minicraftax_coder").context
mobs = importlib.import_module("dicode.dreaming.prompts.dicode.mobs").context

system_prompt = gen_env.system_prompt.format(
    CRAFTAX_CODE=craftax_code, MINICRAFTAX_CODE=wrapper_code, MOBS=mobs)

# --- representative task descriptions (early/mid/late by task number) ---
graph = nx.read_graphml(GRAPHML)
tasks = []
for node, attrs in graph.nodes(data=True):
    desc = (attrs.get("d2") or attrs.get("description") or "").strip()
    if not desc:
        continue
    m = re.search(r"task_(\d+)", str(node))
    num = int(m.group(1)) if m else 0
    tasks.append((num, str(node), desc))
tasks.sort(key=lambda t: t[0])
print(f"total tasks with description: {len(tasks)}")

n = len(tasks)
# 4 per stage (early/mid/late); skip the very first seed rows if n large
selected = []
for (num, node, desc) in tasks[:4]:
    selected.append(("early", num, node, desc))
mid_start = max(4, n // 2 - 2)
for (num, node, desc) in tasks[mid_start:mid_start + 4]:
    selected.append(("mid", num, node, desc))
for (num, node, desc) in tasks[-4:]:
    selected.append(("late", num, node, desc))

seed_code = open(os.path.join(S, "src/minicraftax/tasks/seed_tasks/collecting.py")).read()

user_prompts, slots, stages, order = [], [], {}, []
for i, (stage, num, node, desc) in enumerate(selected):
    up = gen_env.user_prompt.format(TASK_DESCRIPTION=desc, CODE_EXAMPLES=seed_code)
    user_prompts.append(up)
    slots.append(node)
    stages.setdefault(stage, []).append(i)
    order.append({"index": i, "slot": node, "kind": "code"})
    print(f"  [{stage}] {node}: {desc[:60]}...")

spec = {
    "classification": "LLM_REPLAY_MANIFEST",
    "source_commit": "91a75e5a1d3bfca5114caf776a710a0339f692d8",
    "provider": "local",
    "model": "qwen2.5-coder:14b",
    "base_url": "http://127.0.0.1:11434/v1",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "timeout_s": 600,
    "max_retries": 2,
    "repair_limit": 0,
    "system_prompt": system_prompt,
    "user_prompts": user_prompts,
    "candidate_slots": slots,
    "request_order": order,
    "prompt_stages": stages,
    "validation": {"static_lint": True, "cpu_jax": True, "dedup_by_code_hash": True},
}
manifest = build_replay_manifest(spec)
out = "/tmp/llm_research_d_BrRxKL/frozen_manifest.json"
written = write_manifest(manifest, out)
reloaded = load_manifest(out)
assert reloaded["manifest_sha256"] == written["manifest_sha256"]

# freeze evidence (read-only proof of provenance, no credentials)
evidence = {
    "freeze_kind": "RECONSTRUCTED_FROM_SOURCE_AND_ARCHIVE",
    "source_commit": "91a75e5a1d3bfca5114caf776a710a0339f692d8",
    "graphml_path": GRAPHML,
    "manifest_sha256": written["manifest_sha256"],
    "system_prompt_sha256": manifest["system_prompt_sha256"],
    "num_user_prompts": len(user_prompts),
    "prompt_stages": stages,
    "selected_task_ids": slots,
    "note": "prompts reconstructed from source commit + archive; not a byte capture of logged prompts",
}
with open("/tmp/llm_research_d_BrRxKL/freeze_evidence.json", "w") as f:
    json.dump(evidence, f, indent=2, sort_keys=True)
print("=== FREEZE DONE ===")
print("manifest_sha256:", written["manifest_sha256"])
print("system_prompt_sha256:", manifest["system_prompt_sha256"])
print("num_user_prompts:", len(user_prompts))
print("stages:", {k: len(v) for k, v in stages.items()})
