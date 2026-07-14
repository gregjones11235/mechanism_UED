#!/usr/bin/env python3
"""Stage L1: Fake cached role test for multi-LLM curriculum collaboration.

Tests:
1. Provider connectivity (one request per provider)
2. Fake Craftax task judgment with 3 roles
3. Cache write/read/fallback
4. JSON validation (scores in [0,1], decision in accept/hold/reject)
5. No API keys printed anywhere

Max: 9 total API calls (3 providers × 3 roles, or 3 connectivity + 6 role calls).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dicode.mechanisms.llm_providers import (
    PROVIDER_CONFIGS,
    ROLE_PROVIDER_MAP,
    call_llm_api,
    get_api_key,
)
from dicode.mechanisms.llm_roles import (
    ROLE_DEFINITIONS,
    build_role_prompt,
    call_role_judge,
    parse_role_response,
)
from dicode.mechanisms.llm_cache import (
    compute_cache_key,
    compute_task_hash,
    get_cached_judgment,
    get_cache_stats,
    load_cache,
    write_cache_entry,
)
from dicode.mechanisms.llm_costs import LLMCostTracker


# ==============================================================================
# Fake candidate tasks (Craftax-like descriptions)
# ==============================================================================

FAKE_TASKS = [
    {
        "task_id": "fake_task_collect_wood_advanced",
        "source": "seed",
        "description": "Collect 50 wood from trees while avoiding zombies. Agent must navigate forest terrain, identify trees, craft a wooden axe, and defend against occasional zombie attacks.",
        "skills": "collect_wood, make_wood_pickaxe, defeat_zombie, navigation",
        "recent_success": 0.35,
        "best_success": 0.42,
    },
    {
        "task_id": "fake_task_craft_stone_tools",
        "source": "learnable",
        "description": "Craft stone pickaxe and stone sword. Requires mining stone, collecting wood for handles, and using a crafting table. Moderate difficulty for early-game agents.",
        "skills": "make_stone_pickaxe, make_stone_sword, collect_stone, place_table",
        "recent_success": 0.55,
        "best_success": 0.68,
    },
    {
        "task_id": "fake_task_defeat_dungeon_skeleton",
        "source": "mastered",
        "description": "Enter the dungeon, find a skeleton, and defeat it using iron sword. Agent must have already crafted iron tools and armor. High difficulty, requires multiple prerequisite skills.",
        "skills": "enter_dungeon, defeat_skeleton, make_iron_sword, make_iron_armour",
        "recent_success": 0.12,
        "best_success": 0.25,
    },
]


# ==============================================================================
# Test functions
# ==============================================================================


def test_provider_connectivity():
    """Test one simple API call per provider to verify connectivity."""
    print("=" * 60)
    print("TEST 1: Provider Connectivity")
    print("=" * 60)

    results = {}
    for provider_name in ["qwen", "deepseek", "glm"]:
        config = PROVIDER_CONFIGS[provider_name]
        api_key = get_api_key(provider_name)

        if not api_key:
            print(f"  {provider_name}: SKIP (no API key)")
            results[provider_name] = {"status": "skip", "reason": "no API key"}
            continue

        response = call_llm_api(
            provider_name=provider_name,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0.0,
            timeout=30,
        )

        if response["success"]:
            print(f"  {provider_name}: OK (cost=${response['estimated_cost']:.6f}, model={response['model']})")
            results[provider_name] = {"status": "ok", "cost": response["estimated_cost"]}
        else:
            print(f"  {provider_name}: FAIL ({response.get('error', 'unknown')[:100]})")
            results[provider_name] = {"status": "fail", "error": response.get("error", "unknown")[:200]}

    return results


def test_fake_role_judgments(connectivity_results: dict):
    """Test role judgments on 3 fake tasks with 3 roles each."""
    print("\n" + "=" * 60)
    print("TEST 2: Fake Role Judgments (3 tasks × 3 roles = 9 max calls)")
    print("=" * 60)

    cost_tracker = LLMCostTracker(
        max_total_cost=5.0,
        max_api_calls=9,
        log_path="/root/experiments/dicode_runs/aggregation/llm_cost_log.jsonl",
    )

    cache_path = "mechanism_logs/llm_judgments_cache.jsonl"
    # Clear previous fake test cache
    if os.path.exists(cache_path):
        os.remove(cache_path)

    all_results = []

    for task in FAKE_TASKS:
        task_id = task["task_id"]
        print(f"\n  Task: {task_id}")

        for role in ["tutor", "critic", "explorer"]:
            provider = ROLE_PROVIDER_MAP.get(role, "unknown")

            # Check if provider is available
            if connectivity_results.get(provider, {}).get("status") != "ok":
                print(f"    {role} ({provider}): SKIP (provider unavailable)")
                all_results.append({
                    "task_id": task_id,
                    "role": role,
                    "provider": provider,
                    "success": False,
                    "error": f"Provider {provider} unavailable",
                })
                continue

            # Check budget
            if not cost_tracker.can_call():
                print(f"    {role} ({provider}): SKIP (budget cap reached)")
                continue

            # --- Cache check ---
            cache = load_cache(cache_path)
            config = PROVIDER_CONFIGS[provider]
            model = config["default_model"]
            cached = get_cached_judgment(cache, task, provider, model, role)

            if cached:
                print(f"    {role} ({provider}): CACHED (cost=$0)")
                all_results.append({
                    "task_id": task_id,
                    "role": role,
                    "provider": provider,
                    "success": True,
                    "cached": True,
                    "judgment": cached,
                })
                continue

            # --- API call ---
            print(f"    {role} ({provider}): Calling API...")
            result = call_role_judge(
                role=role,
                task_summary=task,
                max_tokens=256,
            )

            cost_tracker.record_call(result)

            if result["success"]:
                judgment = result["judgment"]
                # Write cache
                write_cache_entry(
                    cache_path, task, provider, model, role,
                    judgment, result,
                )

                # Validate
                scores = judgment.get("scores", {})
                flags = judgment.get("flags", {})
                decision = judgment.get("decision", "")
                all_scores_valid = all(0.0 <= float(v) <= 1.0 for v in scores.values())
                decision_valid = decision in ("accept", "hold", "reject")

                status = "OK" if (all_scores_valid and decision_valid) else "VALIDATION_FAIL"
                print(f"    {role} ({provider}): {status} "
                      f"decision={decision} cost=${result['estimated_cost']:.6f} "
                      f"scores={json.dumps({k: round(v, 2) for k, v in scores.items()})}")

                if not all_scores_valid:
                    print(f"      WARNING: scores out of [0,1] range: {scores}")
                if not decision_valid:
                    print(f"      WARNING: invalid decision: {decision}")

                all_results.append({
                    "task_id": task_id,
                    "role": role,
                    "provider": provider,
                    "success": True,
                    "cached": False,
                    "judgment": judgment,
                    "cost": result["estimated_cost"],
                })
            else:
                print(f"    {role} ({provider}): FAIL ({result.get('error', 'unknown')[:100]})")
                all_results.append({
                    "task_id": task_id,
                    "role": role,
                    "provider": provider,
                    "success": False,
                    "error": result.get("error", "unknown")[:200],
                })

    # --- Cache re-read test ---
    print("\n" + "=" * 60)
    print("TEST 3: Cache Re-read (should be 0 API calls)")
    print("=" * 60)

    cache = load_cache(cache_path)
    stats = get_cache_stats(cache_path)
    print(f"  Cache entries: {stats['total_entries']}")
    print(f"  By provider: {stats['by_provider']}")
    print(f"  By role: {stats['by_role']}")
    print(f"  Total cost: ${stats['total_cost']:.6f}")

    # Verify all 3 tasks × 3 roles are cached
    for task in FAKE_TASKS:
        for role in ["tutor", "critic", "explorer"]:
            provider = ROLE_PROVIDER_MAP.get(role, "unknown")
            if connectivity_results.get(provider, {}).get("status") != "ok":
                continue
            config = PROVIDER_CONFIGS[provider]
            model = config["default_model"]
            cached = get_cached_judgment(cache, task, provider, model, role)
            status = "CACHED" if cached else "MISSING"
            print(f"  {task['task_id'][:30]}... {role}: {status}")

    # --- Rule-based fallback test ---
    print("\n" + "=" * 60)
    print("TEST 4: Rule-based Fallback (missing cache)")
    print("=" * 60)

    # A fake task that was never cached
    unknown_task = {
        "task_id": "fake_task_unknown",
        "source": "unknown",
        "description": "An uncached task that should trigger rule-based fallback.",
        "skills": "unknown",
        "recent_success": 0.0,
        "best_success": 0.0,
    }

    for role in ["tutor", "critic", "explorer"]:
        provider = ROLE_PROVIDER_MAP.get(role, "unknown")
        config = PROVIDER_CONFIGS[provider]
        model = config["default_model"]
        cached = get_cached_judgment(cache, unknown_task, provider, model, role)
        if cached:
            print(f"  {role}: CACHED (unexpected)")
        else:
            # Apply rule-based fallback
            fallback = _rule_based_fallback(unknown_task, role)
            print(f"  {role}: FALLBACK decision={fallback['decision']}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    cost_summary = cost_tracker.get_summary()
    print(f"  Total API calls: {cost_summary['total_calls']}")
    print(f"  Total cost: ${cost_summary['total_cost']:.6f}")
    print(f"  By provider: {cost_summary['by_provider']}")
    print(f"  Cache entries: {stats['total_entries']}")
    print(f"  Successful judgments: {sum(1 for r in all_results if r.get('success'))}")
    print(f"  Failed judgments: {sum(1 for r in all_results if not r.get('success'))}")

    return all_results, cost_summary


def _rule_based_fallback(task_summary: dict, role: str) -> dict:
    """Simple rule-based fallback when cache is missing and API is unavailable."""
    recent_sr = float(task_summary.get("recent_success", 0.5))
    best_sr = float(task_summary.get("best_success", 0.5))
    source = task_summary.get("source", "unknown")

    scores = {}
    flags = {}

    if role == "tutor":
        learnability = recent_sr * (1.0 - recent_sr)  # peaks at 0.5
        scores = {
            "progression_score": max(0.0, min(1.0, recent_sr)),
            "learnability_score": max(0.0, min(1.0, learnability * 4)),  # scale up
            "tech_tree_progress_score": 0.5,
        }
        flags = {"too_easy": recent_sr > 0.9, "too_hard": recent_sr < 0.1 and best_sr < 0.2}
        decision = "hold"
        if recent_sr > 0.9:
            decision = "reject"
        elif recent_sr < 0.1:
            decision = "hold"
        else:
            decision = "accept"

    elif role == "critic":
        critic_penalty = 0.0
        too_hard = recent_sr < 0.1
        already_mastered = recent_sr > 0.9
        invalid_risk = source == "unknown"
        scores = {"critic_penalty": max(0.0, min(1.0, (0.3 if too_hard else 0.0) + (0.5 if already_mastered else 0.0)))}
        flags = {"too_hard": too_hard, "already_mastered": already_mastered, "invalid_risk": invalid_risk, "metric_hacking_risk": False}
        decision = "reject" if (too_hard or already_mastered) else "accept"

    elif role == "explorer":
        novelty = 0.8 if source == "unknown" else (0.3 if source == "seed" else 0.5)
        diversity = 0.5
        scores = {"novelty_score": novelty, "diversity_score": diversity}
        flags = {}
        decision = "accept" if novelty > 0.4 else "hold"

    else:
        decision = "hold"

    return {
        "task_id": task_summary.get("task_id", "unknown"),
        "role": role,
        "provider": "rule_based",
        "model": "none",
        "scores": scores,
        "flags": flags,
        "skill_tag": str(task_summary.get("skills", "")),
        "decision": decision,
        "short_reason": f"Rule-based fallback: sr={recent_sr:.2f}, source={source}",
    }


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 60)
    print("STAGE L1: Multi-LLM Role Collaboration Test")
    print("=" * 60)
    print(f"API keys available: ", end="")
    for p in ["qwen", "deepseek", "glm"]:
        key = get_api_key(p)
        print(f"{p}={bool(key)} ", end="")
    print()

    # Test 1: Connectivity
    connectivity = test_provider_connectivity()

    # Test 2-4: Role judgments + cache + fallback
    results, cost_summary = test_fake_role_judgments(connectivity)

    # Final report
    print("\n" + "=" * 60)
    print("STAGE L1 COMPLETE")
    print("=" * 60)
    conn_status = ", ".join(f"{k}={v.get('status', '?')}" for k, v in connectivity.items())
    print(f"Connectivity: {conn_status}")
    print(f"Total API calls: {cost_summary['total_calls']}")
    print(f"Total cost: ${cost_summary['total_cost']:.6f}")
    print(f"Cache: mechanism_logs/llm_judgments_cache.jsonl")
    print(f"Cost log: /root/experiments/dicode_runs/aggregation/llm_cost_log.jsonl")

    exit_code = 0 if cost_summary["total_calls"] <= 9 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
