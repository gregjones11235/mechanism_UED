#!/usr/bin/env python3
"""Create a synthetic diverse candidate pool for mechanism validation.

Generates a task_graph.graphml with 12 tasks across 4 sources and 6 skill groups,
plus matching LLM cache entries (rule-based fallback scores).

This is explicitly synthetic — for mechanism testing, not real curriculum data.
"""

import json
import os
import sys
import hashlib
import time

import networkx as nx

OUTPUT_GRAPH = "task_graph.graphml"
CACHE_PATH = "/root/experiments/dreaming-in-code-coop/mechanism_logs/llm_judgments_cache.jsonl"
PENDING_PATH = "mechanism_logs/pending_llm_tasks.jsonl"

# Synthetic task definitions
SYNTHETIC_TASKS = [
    # (task_id, source, status, description, skills, recent_sr, best_sr, priority)
    ("synth_collect_wood_basic", "seed", "A",
     "Collect 20 wood from nearby trees. Basic navigation and tool use.",
     "collect_wood, navigation", 0.92, 0.95, 0.07),
    ("synth_craft_stone_pickaxe", "seed", "A",
     "Craft a stone pickaxe using collected wood and mined stone.",
     "make_stone_pickaxe, collect_stone, place_table", 0.88, 0.91, 0.10),
    ("synth_defeat_single_zombie", "seed", "B",
     "Craft a wooden sword and defeat one zombie in forest biome.",
     "defeat_zombie, make_wood_sword, combat", 0.65, 0.72, 0.23),
    ("synth_craft_iron_armour", "mastered", "A",
     "Mine iron ore, smelt in furnace, craft full iron armour set.",
     "collect_iron, make_iron_armour, place_furnace, collect_coal", 0.94, 0.97, 0.06),
    ("synth_enter_dungeon_level1", "mastered", "B",
     "Navigate to dungeon entrance and explore first level. Fight basic mobs.",
     "enter_dungeon, navigation, combat, defeat_skeleton", 0.60, 0.78, 0.24),
    ("synth_collect_ruby_cave", "mastered", "C",
     "Mine ruby ore from deep caves. Requires iron pickaxe, torch placement.",
     "collect_ruby, make_iron_pickaxe, place_torch, collect_stone", 0.35, 0.55, 0.23),
    ("synth_defeat_troll_mines", "learnable", "B",
     "Enter troll mines and defeat a troll using iron sword and shield.",
     "enter_troll_mines, defeat_troll, make_iron_sword, combat", 0.45, 0.52, 0.25),
    ("synth_craft_diamond_sword", "learnable", "C",
     "Collect diamonds from deep mines and craft a diamond sword.",
     "collect_diamond, make_diamond_sword, make_iron_pickaxe, place_table", 0.28, 0.40, 0.20),
    ("synth_enchant_armour_magic", "learnable", "D",
     "Find enchantment materials in dungeon and enchant iron armour.",
     "enchant_armour, enter_dungeon, make_iron_armour, collect_diamond", 0.12, 0.18, 0.10),
    ("synth_defeat_necromancer", "generated", "C",
     "Enter graveyard, fight past skeleton guards, defeat necromancer boss.",
     "defeat_necromancer, enter_graveyard, defeat_skeleton, combat, make_iron_sword", 0.08, 0.15, 0.07),
    ("synth_enter_fire_realm", "generated", "D",
     "Navigate to fire realm, survive elemental damage, collect unique loot.",
     "enter_fire_realm, defeat_fire_elemental, collect_sapphire, navigation", 0.05, 0.10, 0.05),
    ("synth_collect_all_ores", "generated", "D",
     "Collect at least one of each ore type: coal, iron, diamond, ruby, sapphire.",
     "collect_coal, collect_iron, collect_diamond, collect_ruby, collect_sapphire, make_iron_pickaxe", 0.02, 0.05, 0.02),
]


def create_synthetic_graph(output_path=OUTPUT_GRAPH):
    """Create a synthetic task graph with diverse sources and skills."""
    g = nx.DiGraph()

    for i, (tid, source, status, desc, skills, recent_sr, best_sr, priority) in enumerate(SYNTHETIC_TASKS):
        skill_list = [s.strip() for s in skills.split(",")]
        g.add_node(
            tid,
            status="seed",  # mark all as seed so seed training activates them
            type=source,  # source is stored in 'type' field for diversity tracking
            description=desc,
            code=(
                f'""" {desc} """\n'
                f'class Env:\n'
                f'    def __init__(self, static_params=None, params=None):\n'
                f'        self.label = "{desc[:50]}"\n'
                f'        self.relevant_achievements = []\n'
                f'        self.completed_achievements = []\n'
            ),
            performance_history=json.dumps([
                {"sr": recent_sr, "session": 0},
            ]),
            session_created=0,  # Must be 0 so seed training runs
            is_active=True,
            priority_score=float(priority),
            learnability_score=float(priority),
            session_last_trained=i % 6,
        )

    nx.write_graphml(g, output_path)
    print(f"Created synthetic graph: {g.number_of_nodes()} nodes -> {output_path}")

    # Print distribution
    sources = {}
    for _, _, source, _, _, _, _, _ in SYNTHETIC_TASKS:
        sources[source] = sources.get(source, 0) + 1
    print(f"  Source distribution: {sources}")
    return g


def create_synthetic_cache_entries(cache_path=CACHE_PATH):
    """Create rule-based LLM cache entries for the synthetic tasks.

    These are synthetic judgments — not real LLM calls.
    They simulate what cached judgments would look like for mechanism testing.
    """
    import copy

    # Load existing cache
    existing_entries = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Keep existing entries for real tasks
    new_entries = []

    for tid, source, status, desc, skills, recent_sr, best_sr, priority in SYNTHETIC_TASKS:
        task_hash = hashlib.sha256(tid.encode()).hexdigest()[:16]
        skill_list = [s.strip() for s in skills.split(",")]

        for role, provider, model in [
            ("tutor", "qwen", "qwen-turbo"),
            ("critic", "deepseek", "deepseek-chat"),
            ("explorer", "glm", "glm-4-flash"),
        ]:
            cache_key = f"{task_hash}_{provider}_{model}_{role}_v1"

            # Rule-based scores (synthetic, not real LLM)
            if role == "tutor":
                learnability = recent_sr * (1.0 - max(0.05, recent_sr))
                scores = {
                    "progression_score": round(recent_sr, 2),
                    "learnability_score": round(min(1.0, learnability * 3), 2),
                    "tech_tree_progress_score": round(best_sr * 0.8, 2),
                }
                flags = {
                    "too_easy": recent_sr > 0.9,
                    "too_hard": recent_sr < 0.1 and best_sr < 0.2,
                }
                decision = "accept"
                if recent_sr > 0.9:
                    decision = "reject"
                elif recent_sr < 0.1 and best_sr < 0.15:
                    decision = "hold"
                reason = f"SR={recent_sr:.0%}, best={best_sr:.0%}, source={source}"

            elif role == "critic":
                penalty = 0.0
                if recent_sr < 0.1:
                    penalty += 0.7
                if recent_sr > 0.9:
                    penalty += 0.5
                if status == "D" and recent_sr < 0.2:
                    penalty += 0.3
                scores = {"critic_penalty": round(min(1.0, penalty), 2)}
                flags = {
                    "too_hard": recent_sr < 0.1 and best_sr < 0.15,
                    "already_mastered": recent_sr > 0.9,
                    "invalid_risk": False,
                    "metric_hacking_risk": False,
                }
                decision = "reject" if (recent_sr > 0.9 or (recent_sr < 0.05)) else "accept"
                reason = f"Penalty={penalty:.1f}, too_hard={flags['too_hard']}, mastered={flags['already_mastered']}"

            elif role == "explorer":
                # Novelty based on source — generated > learnable > mastered > seed
                novelty_map = {"seed": 0.3, "mastered": 0.2, "learnable": 0.6, "generated": 0.85}
                novelty = novelty_map.get(source, 0.5)
                diversity = len(skill_list) / 8.0  # more skills = more diverse
                scores = {
                    "novelty_score": round(novelty, 2),
                    "diversity_score": round(min(1.0, diversity), 2),
                }
                flags = {}
                decision = "accept" if novelty > 0.3 else "hold"
                reason = f"Source={source}, skills={len(skill_list)}, novelty={novelty:.1f}"

            judgment = {
                "task_id": tid,
                "role": role,
                "provider": provider,
                "model": model,
                "scores": scores,
                "flags": flags,
                "skill_tag": skills,
                "decision": decision,
                "short_reason": reason,
            }

            entry = {
                "cache_key": cache_key,
                "task_id": tid,
                "task_hash": task_hash,
                "provider": provider,
                "model": model,
                "role": role,
                "prompt_version": "v1",
                "judgment": judgment,
                "input_tokens_est": 200,
                "output_tokens_est": 100,
                "estimated_cost": 0.0001,
            }
            new_entries.append(entry)

    # Merge with existing: keep real entries, add synthetic
    existing_task_ids = {e.get("task_id") for e in existing_entries}
    filtered_existing = [e for e in existing_entries if not e.get("task_id", "").startswith("synth_")]

    combined = filtered_existing + new_entries
    with open(cache_path, "w") as f:
        for entry in combined:
            f.write(json.dumps(entry) + "\n")

    print(f"Cache entries: {len(filtered_existing)} existing + {len(new_entries)} synthetic = {len(combined)} total")
    print(f"  Synthetic task_ids: {sorted(set(e['task_id'] for e in new_entries))}")

    # Verify
    from dicode.mechanisms.llm_cache import load_cache, get_cached_judgments_by_task_id
    cache = load_cache(cache_path)
    for tid in [t[0] for t in SYNTHETIC_TASKS[:3]]:
        result = get_cached_judgments_by_task_id(cache, tid)
        print(f"  {tid}: {len(result)} cached roles")


def create_pending_tasks(output_path=PENDING_PATH):
    """Export synthetic task summaries for reference."""
    tasks = []
    for tid, source, status, desc, skills, recent_sr, best_sr, priority in SYNTHETIC_TASKS:
        tasks.append({
            "task_id": tid,
            "source": source,
            "status": status,
            "description": desc,
            "skills": [s.strip() for s in skills.split(",")],
            "recent_success": recent_sr,
            "best_success": best_sr,
            "priority_score": priority,
            "learnability_score": priority,
            "synthetic": True,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Exported {len(tasks)} pending tasks to {output_path}")


if __name__ == "__main__":
    create_synthetic_graph()
    create_synthetic_cache_entries()
    create_pending_tasks()
    print("\nSynthetic pool ready. Run experiments with this graph and cache.")
