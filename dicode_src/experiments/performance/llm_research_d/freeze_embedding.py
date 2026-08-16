"""Freeze a real task-description text set for the D1b embedding replay.

Reads REAL task descriptions from Mason attempt_06's task_graph.graphml, selects
25 texts spread across the run (matching the production one-candidate-batch
scale), and writes a frozen embedding manifest (atomic + self-hash). The same
manifest is used for every D1b concurrency config.
"""
import json
import os
import re
import sys

S = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/source/dicode_src"
GRAPHML = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/attempt_06/run/task_graph.graphml"
OUT = "/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1b/frozen_embedding_manifest.json"
sys.path.insert(0, os.path.join(S, "src"))

import networkx as nx  # noqa: E402
from llm_replay_manifest import sha256_text, sha256_bytes, write_manifest  # noqa: E402

graph = nx.read_graphml(GRAPHML)
texts = []
for node, attrs in graph.nodes(data=True):
    desc = (attrs.get("description") or "").strip()
    if not desc:
        continue
    m = re.search(r"task_(\d+)", str(node))
    num = int(m.group(1)) if m else 0
    texts.append((num, desc))
texts.sort(key=lambda t: t[0])
print(f"total tasks with description: {len(texts)}")

N = 25
if len(texts) >= N:
    idxs = [int(round(i * (len(texts) - 1) / (N - 1))) for i in range(N)]
    selected = [texts[i] for i in idxs]
else:
    selected = texts[:N]
frozen_texts = [desc for _, desc in selected]
print(f"frozen {len(frozen_texts)} texts (of {len(texts)} available)")

manifest = {
    "classification": "LLM_EMBEDDING_REPLAY_MANIFEST",
    "not_end_to_end_ued": True,
    "workload_label": "FROZEN_EMBEDDING_TEXTS_FROM_MASON_ARCHIVE",
    "source_commit": "91a75e5a1d3bfca5114caf776a710a0339f692d8",
    "provider": "local",
    "model": "nomic-embed-text",
    "base_url": "http://127.0.0.1:11434/v1",
    "embedding_size": 768,
    "instruction": None,
    "timeout_s": 60,
    "max_retries": 2,
    "texts": frozen_texts,
    "text_sha256s": [sha256_text(t) for t in frozen_texts],
    "texts_sha256": sha256_bytes("\n".join(frozen_texts).encode()),
}
written = write_manifest(manifest, OUT)

# reload + verify tamper-check (self-hash only; not the chat validator)
from llm_replay_manifest import fingerprint  # noqa: E402
reloaded = json.load(open(OUT, encoding="utf-8"))
recomputed = fingerprint({k: v for k, v in reloaded.items() if k != "manifest_sha256"})
assert recomputed == reloaded["manifest_sha256"], "embedding manifest tamper check failed"

print("=== FREEZE EMBEDDING DONE ===")
print("manifest_sha256:", written["manifest_sha256"])
print("texts_sha256:", manifest["texts_sha256"])
print("num_texts:", len(frozen_texts))
print("first_text_head:", frozen_texts[0][:60])
