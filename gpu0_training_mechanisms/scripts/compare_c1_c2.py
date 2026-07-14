#!/usr/bin/env python3
"""Post-hoc comparison of C1 (soft_copeland) vs C2 (budgeted_soft_copeland).

Produces:
  /root/experiments/dicode_runs/aggregation/soft_vs_budgeted_soft_copeland.csv
  /root/experiments/dicode_runs/aggregation/soft_vs_budgeted_soft_copeland.md
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def compute_entropy(counts):
    """Compute entropy from a dict of {key: count}."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    import math
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())


def analyze(agg_rows, output_dir):
    """Analyze C1 vs C2."""
    os.makedirs(output_dir, exist_ok=True)

    c1_rows = [r for r in agg_rows if r.get('aggregation_mode') == 'soft_copeland']
    c2_rows = [r for r in agg_rows if r.get('aggregation_mode') == 'budgeted_soft_copeland']

    def stats(rows, label):
        if not rows:
            return {"label": label, "count": 0}

        entropies = [r.get('curriculum_entropy', 0) for r in rows]
        source_shares = [r.get('source_share', {}) for r in rows]
        forgetting = [r.get('forgetting_index', 0) for r in rows]
        hit_rates = [r.get('llm_cache_hit_rate', 0) for r in rows if r.get('llm_enabled')]
        n_candidates = [r.get('num_candidates', 0) for r in rows]
        n_selected = [len(r.get('selected_task_ids', [])) for r in rows]

        # Source entropy over all selections
        source_counts = Counter()
        for share in source_shares:
            for src, frac in share.items():
                source_counts[src] += frac

        # Count unique sources per selection
        unique_sources_per = []
        for r in rows:
            sources = r.get('selected_task_sources', [])
            unique_sources_per.append(len(set(sources)))

        import statistics
        return {
            "label": label,
            "count": len(rows),
            "mean_entropy": statistics.mean(entropies) if entropies else 0,
            "min_entropy": min(entropies) if entropies else 0,
            "max_entropy": max(entropies) if entropies else 0,
            "source_entropy": compute_entropy(source_counts),
            "source_distribution": dict(source_counts.most_common(10)),
            "mean_unique_sources": statistics.mean(unique_sources_per) if unique_sources_per else 0,
            "max_unique_sources": max(unique_sources_per) if unique_sources_per else 0,
            "mean_forgetting": statistics.mean(forgetting) if forgetting else 0,
            "mean_cache_hit_rate": statistics.mean(hit_rates) if hit_rates else 0,
            "mean_candidates": statistics.mean(n_candidates) if n_candidates else 0,
            "mean_selected": statistics.mean(n_selected) if n_selected else 0,
        }

    c1_stats = stats(c1_rows, "C1_soft_copeland")
    c2_stats = stats(c2_rows, "C2_budgeted_soft_copeland")

    # CSV
    csv_path = os.path.join(output_dir, "soft_vs_budgeted_soft_copeland.csv")
    fieldnames = ["label", "count", "mean_entropy", "min_entropy", "max_entropy",
                   "source_entropy", "mean_unique_sources", "max_unique_sources",
                   "mean_forgetting", "mean_cache_hit_rate",
                   "mean_candidates", "mean_selected"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        if c1_stats["count"] > 0:
            writer.writerow(c1_stats)
        if c2_stats["count"] > 0:
            writer.writerow(c2_stats)

    # Markdown
    md_path = os.path.join(output_dir, "soft_vs_budgeted_soft_copeland.md")
    lines = []
    lines.append("# soft_copeland vs budgeted_soft_copeland Comparison\n")
    lines.append("Candidates: 4 real Craftax tasks with diverse source metadata (seed, learnable, mastered, generated).\n")

    lines.append("## Results\n")
    lines.append("| Metric | C1 soft_copeland | C2 budgeted_soft_copeland | Winner |")
    lines.append("|--------|-----------------|--------------------------|--------|")

    def compare_metric(name, key, fmt="{:.4f}", lower_is_better=False):
        v1 = c1_stats.get(key, 0)
        v2 = c2_stats.get(key, 0)
        if lower_is_better:
            winner = "C1" if v1 < v2 else ("C2" if v2 < v1 else "tie")
        else:
            winner = "C1" if v1 > v2 else ("C2" if v2 > v1 else "tie")
        return f"| {name} | {fmt.format(v1)} | {fmt.format(v2)} | {winner} |"

    lines.append(compare_metric("Selections", "count", "{}", False))
    lines.append(compare_metric("Curriculum Entropy (mean)", "mean_entropy"))
    lines.append(compare_metric("Curriculum Entropy (min)", "min_entropy"))
    lines.append(compare_metric("Curriculum Entropy (max)", "max_entropy"))
    lines.append(compare_metric("Source Entropy", "source_entropy"))
    lines.append(compare_metric("Mean Unique Sources/Selection", "mean_unique_sources"))
    lines.append(compare_metric("Forgetting Index (mean)", "mean_forgetting", lower_is_better=True))
    lines.append(compare_metric("LLM Cache Hit Rate", "mean_cache_hit_rate"))
    lines.append(compare_metric("Candidates", "mean_candidates"))
    lines.append(compare_metric("Selected", "mean_selected"))

    lines.append("")
    lines.append("## Source Distribution\n")
    lines.append("| Source | C1 Share | C2 Share |")
    lines.append("|--------|---------|---------|")
    all_sources = set()
    if c1_stats.get("source_distribution"):
        all_sources.update(c1_stats["source_distribution"].keys())
    if c2_stats.get("source_distribution"):
        all_sources.update(c2_stats["source_distribution"].keys())
    for src in sorted(all_sources):
        s1 = c1_stats.get("source_distribution", {}).get(src, 0)
        s2 = c2_stats.get("source_distribution", {}).get(src, 0)
        lines.append(f"| {src} | {s1:.4f} | {s2:.4f} |")

    lines.append("")
    lines.append("## Analysis\n")

    # Q1: Does budgeted reduce collapse?
    lines.append("### 1. Does budgeted_soft_copeland reduce source collapse?")
    if c1_stats["count"] > 0 and c2_stats["count"] > 0:
        se1, se2 = c1_stats["source_entropy"], c2_stats["source_entropy"]
        us1, us2 = c1_stats["mean_unique_sources"], c2_stats["mean_unique_sources"]
        if se2 > se1 or us2 > us1:
            lines.append(f"Yes — budgeted_soft_copeland has higher source entropy ({se2:.4f} vs {se1:.4f})")
            lines.append(f"and more unique sources per selection ({us2:.2f} vs {us1:.2f}).")
        else:
            lines.append(f"No significant improvement (source entropy: {se2:.4f} vs {se1:.4f}).")
            lines.append("With only 4 candidates from 4 different sources, source caps have limited effect.")
    else:
        lines.append("Insufficient data.")
    lines.append("")

    # Q2: Does it hurt performance?
    lines.append("### 2. Does it hurt or improve mean_return?")
    lines.append("mean_return/eval_return not available in current diagnostics.")
    lines.append("Check training logs for evaluation metrics.")
    lines.append("")

    # Q3: Artificial diversity?
    lines.append("### 3. Does it improve curriculum entropy or enforce artificial diversity?")
    if c1_stats["count"] > 0 and c2_stats["count"] > 0:
        e1, e2 = c1_stats["mean_entropy"], c2_stats["mean_entropy"]
        if e2 > e1:
            lines.append(f"budgeted_soft_copeland has higher mean entropy ({e2:.4f} vs {e1:.4f}),")
            lines.append("suggesting genuine diversity improvement, not just artificial enforcement.")
        else:
            lines.append(f"soft_copeland has higher entropy ({e1:.4f} vs {e2:.4f}).")
            lines.append("Budget caps may reduce entropy if the pool is already balanced.")
    lines.append("")

    # Q4: Worth it?
    lines.append("### 4. Is budgeted_soft_copeland worth using?")
    if c1_stats["count"] > 0 and c2_stats["count"] > 0:
        improvements = 0
        if c2_stats["source_entropy"] > c1_stats["source_entropy"]:
            improvements += 1
        if c2_stats["mean_unique_sources"] > c1_stats["mean_unique_sources"]:
            improvements += 1
        if c2_stats["mean_entropy"] > c1_stats["mean_entropy"]:
            improvements += 1
        if c2_stats["mean_forgetting"] < c1_stats["mean_forgetting"]:
            improvements += 1

        if improvements >= 2:
            lines.append("**Yes** — budgeted_soft_copeland shows improvement on multiple metrics.")
        elif improvements >= 1:
            lines.append("**Maybe** — budgeted_soft_copeland shows marginal improvement.")
        else:
            lines.append("**Not clearly** — no significant advantage over soft_copeland with the current pool size.")
            lines.append("Re-test with a larger (>8) candidate pool for definitive conclusion.")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Comparison written to {md_path}")
    print(f"CSV written to {csv_path}")

    # Print summary
    print(f"\nC1 (soft_copeland): {c1_stats['count']} selections, entropy={c1_stats['mean_entropy']:.4f}")
    print(f"C2 (budgeted):       {c2_stats['count']} selections, entropy={c2_stats['mean_entropy']:.4f}")


if __name__ == "__main__":
    jsonl = "/root/experiments/dicode_runs/aggregation/mechanism_logs/aggregation_selector.jsonl"
    out = "/root/experiments/dicode_runs/aggregation"
    analyze(load_jsonl(jsonl), out)
