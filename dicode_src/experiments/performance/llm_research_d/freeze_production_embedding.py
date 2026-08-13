"""Freeze the production-shape embedding dataset for D1c.

Reads REAL task descriptions from Mason attempt_06's task_graph.graphml and
freezes a deterministic text pool + the production batch-size sequence (5,10,
12,13,16,10,15,16,9,16,16,10). Each batch is pool[0:size] in deterministic order.
The manifest is atomically written with a self-hash and reload-verified.
"""
import hashlib
import json
import os
import re
import sys
import tempfile

S = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/source/dicode_src"
GRAPHML = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/attempt_06/run/task_graph.graphml"
OUT = "/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1c/FROZEN_PRODUCTION_EMBEDDING_MANIFEST.json"
sys.path.insert(0, os.path.join(S, "src"))

import networkx as nx  # noqa: E402
from llm_replay_manifest import sha256_text, sha256_bytes, canonical, fingerprint  # noqa: E402

EMBEDDING_INSTRUCTION = (
    "Generate an embedding for this Craftax task description to evaluate its "
    "conceptual similarity to other tasks. The embedding should capture the core "
    "gameplay loop, the primary skills the agent must use (e.g., navigation, "
    "crafting, combat), the overall strategic objective, and how the world is built."
)

BATCH_SEQUENCE = [5, 10, 12, 13, 16, 10, 15, 16, 9, 16, 16, 10]


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    graph = nx.read_graphml(GRAPHML)
    tasks = []
    for node, attrs in graph.nodes(data=True):
        desc = (attrs.get("description") or "").strip()
        if not desc:
            continue
        m = re.search(r"task_(\d+)", str(node))
        num = int(m.group(1)) if m else 0
        tasks.append((num, str(node), desc))
    tasks.sort(key=lambda t: t[0])

    # 16 representative texts spread across the run (late-stage batch scale)
    N = 16
    idxs = [int(round(i * (len(tasks) - 1) / (N - 1))) for i in range(N)]
    pool = [tasks[i] for i in idxs]
    text_pool = [desc for _, _, desc in pool]
    task_ids = [node for _, node, _ in pool]

    # deterministic batch composition: batch i = pool[0:size_i]
    batch_composition = {}
    for bi, size in enumerate(BATCH_SEQUENCE):
        batch_composition[str(bi)] = list(range(size))

    manifest = {
        "classification": "LLM_PRODUCTION_EMBEDDING_MANIFEST",
        "not_end_to_end_ued": True,
        "source_commit": "91a75e5a1d3bfca5114caf776a710a0339f692d8",
        "mason_artifact_root": "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/attempt_06",
        "task_graph_sha256": _file_sha256(GRAPHML),
        "archive_sha256": _file_sha256(GRAPHML),
        "ordered_task_ids": task_ids,
        "description_sha256s": {tid: sha256_text(d) for tid, d in zip(task_ids, text_pool)},
        "text_total": len(text_pool),
        "embedding_provider": "local",
        "embedding_model": "nomic-embed-text",
        "embedding_size": 768,
        "instruction": EMBEDDING_INSTRUCTION,
        "base_url": "http://127.0.0.1:11434/v1",
        "timeout_s": 60,
        "max_retries": 2,
        "batch_size_sequence": BATCH_SEQUENCE,
        "client_lifecycle_mode": "per_arm",  # persistent / fresh / idle_gap
        "run_order": "A1,A2,B1,B2,C30_1,C30_2,C120_1,C120_2",
        "repeat": 2,
        "text_pool": text_pool,
        "text_pool_sha256s": [sha256_text(t) for t in text_pool],
        "texts_sha256": sha256_bytes("\n".join(text_pool).encode()),
        "batch_composition": batch_composition,
    }
    manifest["manifest_sha256"] = fingerprint(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", dir=os.path.dirname(OUT))
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        json.dump(canonical(manifest), f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(tmp, OUT)

    reloaded = json.load(open(OUT, encoding="utf-8"))
    assert fingerprint({k: v for k, v in reloaded.items() if k != "manifest_sha256"}) == reloaded["manifest_sha256"]

    print("=== FREEZE PRODUCTION EMBEDDING DONE ===")
    print("manifest_sha256:", manifest["manifest_sha256"])
    print("text_total:", len(text_pool))
    print("batch_sequence:", BATCH_SEQUENCE)
    print("task_graph_sha256:", manifest["task_graph_sha256"])


if __name__ == "__main__":
    main()
