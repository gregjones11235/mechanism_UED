#!/usr/bin/env python3
"""Stage L3: Generate cached LLM role judgments for real candidate tasks.

Reads pending_llm_tasks.jsonl, calls each role judge once per task,
writes results to llm_judgments_cache.jsonl.

Configurable caps:
  - max_candidates (default 8)
  - max_api_calls (default 24)
  - max_output_tokens (default 256)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dicode.mechanisms.llm_providers import PROVIDER_CONFIGS, ROLE_PROVIDER_MAP, get_api_key
from dicode.mechanisms.llm_roles import call_role_judge
from dicode.mechanisms.llm_cache import load_cache, write_cache_entry, get_cached_judgment, get_cache_stats
from dicode.mechanisms.llm_costs import LLMCostTracker


def generate_judgments(
    tasks_path="mechanism_logs/pending_llm_tasks.jsonl",
    cache_path="mechanism_logs/llm_judgments_cache.jsonl",
    max_candidates=8,
    max_api_calls=24,
    max_output_tokens=256,
    output_dir="/root/experiments/dicode_runs/aggregation/llm_pilot",
):
    """Generate LLM role judgments for real candidate tasks."""
    os.makedirs(output_dir, exist_ok=True)

    # Load tasks
    if not os.path.exists(tasks_path):
        print(f"No tasks file: {tasks_path}")
        return

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    tasks = tasks[:max_candidates]
    print(f"Loaded {len(tasks)} candidate tasks from {tasks_path}")

    # Load existing cache
    cache = load_cache(cache_path)
    print(f"Existing cache entries: {len(cache)}")

    # Cost tracker
    cost_tracker = LLMCostTracker(
        max_total_cost=1.0,
        max_api_calls=max_api_calls,
        log_path=os.path.join(output_dir, "llm_cost_log.jsonl"),
    )

    roles = ["tutor", "critic", "explorer"]
    results = []
    cache_hits = 0
    api_calls = 0
    failures = 0

    for task in tasks:
        task_id = task["task_id"]
        print(f"\n  Task: {task_id}")

        for role in roles:
            provider = ROLE_PROVIDER_MAP.get(role, "unknown")

            # Check provider availability
            if not get_api_key(provider):
                print(f"    {role} ({provider}): SKIP (no API key)")
                results.append({
                    "task_id": task_id, "role": role, "provider": provider,
                    "success": False, "error": "No API key",
                })
                continue

            # Check budget
            if not cost_tracker.can_call():
                print(f"    {role} ({provider}): SKIP (budget cap)")
                continue

            # Check cache
            config = PROVIDER_CONFIGS[provider]
            model = config["default_model"]
            cached = get_cached_judgment(cache, task, provider, model, role)

            if cached:
                print(f"    {role} ({provider}): CACHED")
                cache_hits += 1
                results.append({
                    "task_id": task_id, "role": role, "provider": provider,
                    "success": True, "cached": True, "judgment": cached,
                })
                continue

            # API call
            print(f"    {role} ({provider}): Calling API...")
            result = call_role_judge(
                role=role,
                task_summary=task,
                max_tokens=max_output_tokens,
            )

            cost_tracker.record_call(result)
            api_calls += 1

            if result["success"]:
                judgment = result["judgment"]
                write_cache_entry(cache_path, task, provider, model, role, judgment, result)
                cache = load_cache(cache_path)  # Refresh cache
                print(f"    {role} ({provider}): OK "
                      f"decision={judgment.get('decision', '?')} "
                      f"cost=${result['estimated_cost']:.6f}")
                results.append({
                    "task_id": task_id, "role": role, "provider": provider,
                    "success": True, "cached": False, "judgment": judgment,
                    "cost": result["estimated_cost"],
                })
            else:
                failures += 1
                print(f"    {role} ({provider}): FAIL ({result.get('error', 'unknown')[:100]})")
                results.append({
                    "task_id": task_id, "role": role, "provider": provider,
                    "success": False, "error": result.get("error", "unknown")[:200],
                })

    # Write pilot report
    report = {
        "total_tasks": len(tasks),
        "total_roles": len(roles),
        "total_possible_calls": len(tasks) * len(roles),
        "actual_api_calls": api_calls,
        "cache_hits": cache_hits,
        "failures": failures,
        "cost_summary": cost_tracker.get_summary(),
        "cache_stats": get_cache_stats(cache_path),
        "results": [
            {k: v for k, v in r.items() if k != "judgment"}
            for r in results
        ],
    }

    report_path = os.path.join(output_dir, "pilot_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nPilot report: {report_path}")

    # Print summary
    print(f"\n=== L3 Summary ===")
    print(f"Tasks evaluated: {len(tasks)}")
    print(f"API calls: {api_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"Failures: {failures}")
    print(f"Total cost: ${cost_tracker.total_cost:.6f}")
    print(f"Cache entries: {get_cache_stats(cache_path)['total_entries']}")

    # Show decisions by task
    print(f"\nDecisions by task:")
    for task in tasks:
        tid = task["task_id"]
        task_results = [r for r in results if r["task_id"] == tid and r.get("success")]
        decisions = []
        for r in task_results:
            j = r.get("judgment", {})
            decisions.append(f"{r['role']}={j.get('decision', '?')}")
        print(f"  {tid}: {', '.join(decisions)}")

    return results, report


if __name__ == "__main__":
    generate_judgments()
