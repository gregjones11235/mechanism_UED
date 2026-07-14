#!/usr/bin/env python3
"""Complete L5 comparison: B0-B6 once all data is collected.

Compares:
  B0: rule-based robust_weighted
  B1: rule-based soft_copeland
  B2: llm tutor-only (Qwen progression only)
  B3: llm tutor+critic (Qwen + DeepSeek)
  B4: llm full 3-role (Qwen + DeepSeek + GLM)
  B5: llm budgeted_soft_copeland
  B6: llm entropy_regularized
"""

import csv
import json
import os
import sys
from collections import defaultdict
import statistics


def load(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except: pass
    return rows


def main():
    rows = load("/root/experiments/dicode_runs/aggregation/mechanism_logs/aggregation_selector.jsonl")
    out = "/root/experiments/dicode_runs/aggregation/l5_full_comparison"

    # Classify rows by variant
    variants = defaultdict(list)
    for r in rows:
        mode = r.get("aggregation_mode", "?")
        llm = r.get("llm_enabled", False)
        hits = r.get("llm_cache_hits", 0)
        misses = r.get("llm_cache_misses", 0)

        # Determine variant
        if mode == "robust_weighted" and not llm:
            v = "B0_rule_robust_weighted"
        elif mode == "soft_copeland" and not llm:
            v = "B1_rule_soft_copeland"
        elif mode == "soft_copeland" and llm and hits > 0:
            # Check which roles are active by cache hit pattern
            v = "B4_llm_3role_soft_copeland"
        elif mode == "soft_copeland" and llm and hits == 0:
            v = "B4_llm_3role_soft_copeland"  # Old L5 with 0% hit
        elif mode == "budgeted_soft_copeland" and llm:
            v = "B5_llm_budgeted_soft_copeland"
        elif mode == "entropy_regularized" and llm:
            v = "B6_llm_entropy_regularized"
        elif mode == "budgeted_retention_trigger":
            v = "Bx_budgeted_retention_trigger"
        else:
            v = f"other_{mode}"

        variants[v].append(r)

    # Stats per variant
    print("=== L5 Full Comparison ===\n")
    print(f"{'Variant':<35} {'Rows':>6} {'Entropy':>8} {'CacheHit':>8} {'Forgetting':>10}")
    print("-" * 70)

    results = []
    for v in sorted(variants.keys()):
        vr = variants[v]
        e = [r.get("curriculum_entropy", 0) for r in vr]
        h = [r.get("llm_cache_hit_rate", 0) for r in vr if r.get("llm_enabled")]
        f = [r.get("forgetting_index", 0) for r in vr]
        results.append({
            "variant": v,
            "rows": len(vr),
            "mean_entropy": round(statistics.mean(e), 4) if e else 0,
            "cache_hit_rate": round(statistics.mean(h), 4) if h else 0,
            "mean_forgetting": round(statistics.mean(f), 4) if f else 0,
        })
        hit_str = f"{results[-1]['cache_hit_rate']:.2%}" if h else "N/A"
        print(f"{v:<35} {len(vr):>6} {results[-1]['mean_entropy']:>8.4f} {hit_str:>8} {results[-1]['mean_forgetting']:>10.4f}")

    # CSV
    csv_path = out + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "rows", "mean_entropy", "cache_hit_rate", "mean_forgetting"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV: {csv_path}")

    # Best by entropy
    best = max(results, key=lambda r: r["mean_entropy"])
    print(f"\nBest entropy: {best['variant']} ({best['mean_entropy']:.4f})")
    if best["cache_hit_rate"] > 0:
        print("LLM cache was active and contributing to this variant.")


if __name__ == "__main__":
    main()
