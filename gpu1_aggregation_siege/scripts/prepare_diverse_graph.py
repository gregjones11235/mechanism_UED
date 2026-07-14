#!/usr/bin/env python3
"""Modify a real DiCode task graph to inject diverse metadata for mechanism testing.

Uses real seed tasks (with working Craftax environments) but assigns them
diverse source types, skill tags, and priority scores so aggregation can
differentiate them.

Also creates matching LLM cache entries.
"""

import hashlib
import json
import os
import sys

import networkx as nx

GRAPH_INPUT = "task_graph.graphml"
GRAPH_OUTPUT = "/root/experiments/dreaming-in-code-coop/task_graph_synthetic.graphml"
CACHE_PATH = "/root/experiments/dreaming-in-code-coop/mechanism_logs/llm_judgments_cache.jsonl"


# Diverse metadata to assign to real seed tasks
TASK_METADATA = [
    {
        "task_id": "task_1",
        "source": "seed",
        "skills": "collect_coal, navigation, wake_up",
        "recent_sr": 0.88,
        "best_sr": 1.0,
        "desc_hint": "Basic survival: collect coal and wake up",
    },
    {
        "task_id": "task_2",
        "source": "learnable",
        "skills": "defeat_zombie, make_wood_sword, combat",
        "recent_sr": 0.55,
        "best_sr": 0.76,
        "desc_hint": "Combat: defeat zombies with wooden sword",
    },
    {
        "task_id": "task_3",
        "source": "mastered",
        "skills": "collect_wood, place_table, make_wood_pickaxe, crafting",
        "recent_sr": 0.92,
        "best_sr": 1.0,
        "desc_hint": "Crafting: collect wood, craft pickaxe at table",
    },
    {
        "task_id": "task_4",
        "source": "generated",
        "skills": "eat_cow, collect_drink, wake_up, survival",
        "recent_sr": 0.12,
        "best_sr": 0.35,
        "desc_hint": "Advanced survival: eat, drink, and survive",
    },
]


def prepare_graph(input_path=GRAPH_INPUT, output_path=GRAPH_OUTPUT):
    """Load a real graph, inject diverse metadata, save."""
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run a DiCode smoke first.")
        sys.exit(1)

    g = nx.read_graphml(input_path)
    print(f"Loaded graph with {g.number_of_nodes()} nodes")

    for meta in TASK_METADATA:
        tid = meta["task_id"]
        if tid not in g:
            print(f"  WARNING: {tid} not in graph, skipping")
            continue

        node = g.nodes[tid]
        node["type"] = meta["source"]  # source diversity for aggregation
        node["status"] = "seed"  # ensure seed status
        node["is_active"] = True
        node["session_created"] = 0
        node["session_last_trained"] = -1
        node["priority_score"] = float(node.get("priority_score", 0.1))
        node["learnability_score"] = float(node.get("learnability_score", 0.1))

        print(f"  {tid}: source={meta['source']} sr={meta['recent_sr']} skills={meta['skills']}")

    nx.write_graphml(g, output_path)
    print(f"Saved modified graph to {output_path}")
    return g


def create_cache_entries(cache_path=CACHE_PATH):
    """Create LLM cache entries matching the modified task metadata."""
    existing = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Filter out old synthetic entries
    existing = [e for e in existing if not e.get("task_id", "").startswith("synth_")]
    synthetic_tids = {m["task_id"] for m in TASK_METADATA}
    existing = [e for e in existing if e.get("task_id") not in synthetic_tids]

    new_entries = []
    for meta in TASK_METADATA:
        tid = meta["task_id"]
        task_hash = hashlib.sha256(tid.encode()).hexdigest()[:16]

        for role, provider, model in [
            ("tutor", "qwen", "qwen-turbo"),
            ("critic", "deepseek", "deepseek-chat"),
            ("explorer", "glm", "glm-4-flash"),
        ]:
            cache_key = f"{task_hash}_{provider}_{model}_{role}_v1"
            sr = meta["recent_sr"]
            best = meta["best_sr"]
            source = meta["source"]

            if role == "tutor":
                learn = sr * (1.0 - max(0.05, sr))
                scores = {
                    "progression_score": round(sr, 2),
                    "learnability_score": round(min(1.0, learn * 3), 2),
                    "tech_tree_progress_score": round(best * 0.8, 2),
                }
                flags = {"too_easy": sr > 0.9, "too_hard": sr < 0.1 and best < 0.2}
                decision = "reject" if sr > 0.9 else ("hold" if sr < 0.1 and best < 0.15 else "accept")
                reason = f"SR={sr:.0%}, source={source}"

            elif role == "critic":
                penalty = 0.0
                if sr < 0.15:
                    penalty += 0.7
                if sr > 0.9:
                    penalty += 0.5
                scores = {"critic_penalty": round(min(1.0, penalty), 2)}
                flags = {
                    "too_hard": sr < 0.15 and best < 0.2,
                    "already_mastered": sr > 0.9,
                    "invalid_risk": False,
                    "metric_hacking_risk": False,
                }
                decision = "reject" if (sr > 0.9 or sr < 0.05) else "accept"
                reason = f"Penalty={penalty:.1f}, source={source}"

            elif role == "explorer":
                novelty_map = {"seed": 0.3, "mastered": 0.2, "learnable": 0.6, "generated": 0.85}
                novelty = novelty_map.get(source, 0.5)
                n_skills = len(meta["skills"].split(","))
                diversity = max(0.1, min(1.0, n_skills / 8.0))
                scores = {"novelty_score": round(novelty, 2), "diversity_score": round(diversity, 2)}
                flags = {}
                decision = "accept" if novelty > 0.3 else "hold"
                reason = f"Source={source}, novelty={novelty:.1f}"

            judgment = {
                "task_id": tid, "role": role, "provider": provider, "model": model,
                "scores": scores, "flags": flags,
                "skill_tag": meta["skills"], "decision": decision, "short_reason": reason,
            }
            new_entries.append({
                "cache_key": cache_key, "task_id": tid, "task_hash": task_hash,
                "provider": provider, "model": model, "role": role,
                "prompt_version": "v1", "judgment": judgment,
                "input_tokens_est": 200, "output_tokens_est": 100, "estimated_cost": 0.0001,
            })

    combined = existing + new_entries
    with open(cache_path, "w") as f:
        for e in combined:
            f.write(json.dumps(e) + "\n")

    print(f"Cache: {len(existing)} retained + {len(new_entries)} new = {len(combined)} total")
    for tid in sorted(synthetic_tids):
        print(f"  {tid}: {sum(1 for e in new_entries if e['task_id']==tid)} cached roles")


if __name__ == "__main__":
    if not os.path.exists(GRAPH_INPUT):
        print(f"Need {GRAPH_INPUT} — run a DiCode smoke with total_timesteps=1 first")
        sys.exit(1)
    prepare_graph()
    create_cache_entries()
    print("\nDone. Graph at", GRAPH_OUTPUT)
