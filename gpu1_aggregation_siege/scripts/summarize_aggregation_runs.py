#!/usr/bin/env python3
"""Summarize aggregation experiment runs.

Reads:
  - mechanism_logs/aggregation_selector*.jsonl
  - Training log files

Produces:
  /root/experiments/dicode_runs/aggregation/summary.csv
  /root/experiments/dicode_runs/aggregation/summary.md

Usage:
    cd /root/experiments/dreaming-in-code-coop
    PYTHONPATH=src:$PYTHONPATH python scripts/summarize_aggregation_runs.py \
        --log-dir /root/experiments/dicode_runs/aggregation/logs \
        --output-dir /root/experiments/dicode_runs/aggregation
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dicode.mechanisms.diagnostics import load_diagnostics_log, summarize_diagnostics


def find_jsonl_files(log_dir: str) -> list[str]:
    """Find all aggregation selector JSONL files in the log directory tree
    and also in Hydra output directories."""
    jsonl_files = []

    # Search in the specified log directory
    pattern = os.path.join(log_dir, "**", "aggregation_selector*.jsonl")
    jsonl_files.extend(glob.glob(pattern, recursive=True))

    # Also search in Hydra output directories (where experiments write diagnostics)
    hydra_outputs = os.path.join(
        os.path.dirname(log_dir) if "dicode_runs" in log_dir else ".",
        "..", "dreaming-in-code-coop", "outputs"
    )
    # Resolve relative to the repo root
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    hydra_path = os.path.normpath(os.path.join(repo_root, "outputs"))
    if os.path.isdir(hydra_path):
        pattern = os.path.join(hydra_path, "**", "aggregation_selector*.jsonl")
        jsonl_files.extend(glob.glob(pattern, recursive=True))

    # Deduplicate
    return sorted(set(jsonl_files))


def find_run_logs(log_dir: str) -> dict[str, str]:
    """Find run.log files and map them to run names."""
    run_logs = {}
    pattern = os.path.join(log_dir, "**", "run.log")
    for path in sorted(glob.glob(pattern, recursive=True)):
        # Try to extract run name from the directory path
        dir_name = os.path.basename(os.path.dirname(path))
        run_logs[dir_name] = path
    return run_logs


def extract_run_metrics_from_log(log_path: str) -> dict:
    """Extract key metrics from a run.log file.

    Returns dict with any extractable metrics.
    """
    metrics = {
        "exit_code": None,
        "final_global_step": None,
        "final_env_steps": None,
        "mean_return_last": None,
        "sr_last": None,
        "error_count": 0,
        "warnings": [],
    }

    if not os.path.exists(log_path):
        return metrics

    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return metrics

    # Check exit code file
    exit_code_path = os.path.join(os.path.dirname(log_path), "exit_code.txt")
    if os.path.exists(exit_code_path):
        try:
            with open(exit_code_path) as f:
                metrics["exit_code"] = int(f.read().strip())
        except Exception:
            pass

    # Count errors
    metrics["error_count"] = content.lower().count("error")

    # Search for patterns
    import re

    # Global steps
    m = re.search(r"Global:\s+(\d+)\s+updates", content)
    if m:
        metrics["final_global_step"] = int(m.group(1))

    # Env steps
    m = re.search(r"Global:\s+\d+\s+updates,\s+(\d+)\s+env steps", content)
    if m:
        metrics["final_env_steps"] = int(m.group(1))

    # Mean return (last)
    returns = re.findall(r"mean_return['\"]?\s*:\s*([\d.]+)", content)
    if returns:
        metrics["mean_return_last"] = float(returns[-1])

    # Success rate (last)
    srs = re.findall(r"'sr'\s*:\s*([\d.-]+)", content)
    if srs:
        metrics["sr_last"] = float(srs[-1])

    return metrics


def produce_summary(
    jsonl_files: list[str],
    run_logs: dict[str, str],
    output_dir: str,
) -> None:
    """Produce summary CSV and Markdown files."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all rows from all JSONL files
    all_rows = []
    for jf in jsonl_files:
        rows = load_diagnostics_log(jf)
        # Tag each row with its source
        for row in rows:
            row["_source"] = jf
        all_rows.extend(rows)

    if not all_rows:
        print("No diagnostics data found. Creating empty summary.")
        _write_empty_summary(output_dir)
        return

    # Group by mode
    by_mode = defaultdict(list)
    for row in all_rows:
        mode = row.get("aggregation_mode", "unknown")
        by_mode[mode].append(row)

    # Also extract per-run metrics
    run_metrics = {}
    for run_name, log_path in run_logs.items():
        run_metrics[run_name] = extract_run_metrics_from_log(log_path)

    # --- Build summary rows ---
    summary_rows = []
    for mode, mode_rows in sorted(by_mode.items()):
        n = len(mode_rows)

        forgetting_indices = [r.get("forgetting_index", 0.0) for r in mode_rows]
        entropies = [r.get("curriculum_entropy", 0.0) for r in mode_rows]
        anti_forgetting_count = sum(1 for r in mode_rows if r.get("anti_forgetting_mode"))
        scores = [r.get("signal_share", {}).get("max_score", 0.0) for r in mode_rows]
        num_selected = [len(r.get("selected_task_ids", [])) for r in mode_rows]
        sources_per_selection = [
            len(set(r.get("selected_task_sources", []))) for r in mode_rows
        ]

        # Find matching run metrics
        matching_runs = [
            v for k, v in run_metrics.items()
            if mode.replace("_", "") in k.replace("_", "") or mode in k
        ]
        error_count = sum(r.get("error_count", 0) for r in matching_runs)
        exit_codes = [r.get("exit_code") for r in matching_runs if r.get("exit_code") is not None]
        failures = sum(1 for ec in exit_codes if ec != 0)

        summary_rows.append({
            "mode": mode,
            "num_selections": n,
            "mean_forgetting_index": round(np.mean(forgetting_indices), 4) if forgetting_indices else 0,
            "max_forgetting_index": round(np.max(forgetting_indices), 4) if forgetting_indices else 0,
            "anti_forgetting_trigger_rate": round(anti_forgetting_count / max(1, n), 4),
            "mean_curriculum_entropy": round(np.mean(entropies), 4) if entropies else 0,
            "mean_selected_tasks": round(np.mean(num_selected), 1) if num_selected else 0,
            "mean_unique_sources": round(np.mean(sources_per_selection), 1) if sources_per_selection else 0,
            "error_count": error_count,
            "failure_count": failures,
            "stability_score": round(1.0 - (np.mean(forgetting_indices) if forgetting_indices else 0), 4),
        })

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV written to: {csv_path}")

    # --- Write Markdown ---
    md_path = os.path.join(output_dir, "summary.md")
    _write_markdown_summary(md_path, summary_rows, by_mode, all_rows)
    print(f"Summary Markdown written to: {md_path}")


def _write_empty_summary(output_dir: str) -> None:
    """Write empty summary files when no data exists."""
    csv_path = os.path.join(output_dir, "summary.csv")
    md_path = os.path.join(output_dir, "summary.md")
    with open(csv_path, "w") as f:
        f.write("mode,num_selections,mean_forgetting_index\n")
    with open(md_path, "w") as f:
        f.write("# Aggregation Sweep Summary\n\n**No data available.**\n")


def _write_markdown_summary(
    md_path: str,
    summary_rows: list[dict],
    by_mode: dict,
    all_rows: list[dict],
) -> None:
    """Write detailed Markdown summary with analysis."""
    lines = []
    lines.append("# Performance-Aware Aggregation Sweep Summary\n")
    lines.append(f"Research goal: Evaluate whether robust aggregation among heterogeneous")
    lines.append(f"curriculum signals can improve training efficiency, final performance,")
    lines.append(f"curriculum stability, generalization, or reduce forgetting.\n")

    # Overview table — ranked by composite score (stability + entropy - forgetting)
    lines.append("## Overview\n")
    lines.append("| Mode | Selections | Mean Forgetting ↓ | Max Forgetting | AF Trigger Rate | Mean Entropy | Failures | Composite ↑ |")
    lines.append("|------|-----------|-------------------|---------------|-----------------|-------------|----------|-------------|")

    for row in summary_rows:
        # Composite: stability (1-forgetting) + entropy, penalized by failures
        composite = row["stability_score"] + row["mean_curriculum_entropy"] - 0.5 * row["failure_count"]
        row["composite_score"] = round(composite, 4)

    sorted_by_composite = sorted(summary_rows, key=lambda r: -r.get("composite_score", 0))

    for row in sorted_by_composite:
        lines.append(
            f"| {row['mode']} | {row['num_selections']} | "
            f"{row['mean_forgetting_index']:.4f} | {row['max_forgetting_index']:.4f} | "
            f"{row['anti_forgetting_trigger_rate']:.2%} | {row['mean_curriculum_entropy']:.4f} | "
            f"{row['failure_count']} | {row.get('composite_score', 0):.4f} |"
        )

    # Analysis
    lines.append("\n## Analysis\n")

    # Q1: Training efficiency / performance (best composite)
    if sorted_by_composite:
        best = sorted_by_composite[0]
        lines.append(f"### 1. Best Overall (Composite: Stability + Entropy)")
        lines.append(f"**{best['mode']}** has the highest composite score ({best.get('composite_score', 0):.4f})")
        lines.append(f"- Forgetting index: {best['mean_forgetting_index']:.4f}")
        lines.append(f"- Curriculum entropy: {best['mean_curriculum_entropy']:.4f}")
        lines.append(f"- Anti-forgetting trigger rate: {best['anti_forgetting_trigger_rate']:.2%}\n")

    # Q2: Curriculum diversity
    if sorted_by_composite:
        most_diverse = max(sorted_by_composite, key=lambda r: r["mean_curriculum_entropy"])
        lines.append(f"### 2. Curriculum Diversity (Highest Entropy)")
        lines.append(f"**{most_diverse['mode']}** has the highest curriculum entropy ({most_diverse['mean_curriculum_entropy']:.4f}) — most diverse task selections.\n")

    # Q3: Stability (lowest forgetting)
    if summary_rows:
        most_stable = min(summary_rows, key=lambda r: r["mean_forgetting_index"])
        lines.append(f"### 3. Most Stable (Lowest Forgetting)")
        lines.append(f"**{most_stable['mode']}** has the lowest mean forgetting index ({most_stable['mean_forgetting_index']:.4f}).\n")

    # Q4: Collapse risk
    lines.append("### 4. Curriculum Collapse Risk (Source/Signal Concentration)")
    for row in sorted_by_composite:
        if row["mean_curriculum_entropy"] < 0.5:
            lines.append(f"- ⚠️ **{row['mode']}**: Low entropy ({row['mean_curriculum_entropy']:.4f}) — risk of collapse to single source/skill")
        elif row["mean_curriculum_entropy"] < 1.5:
            lines.append(f"- ⚡ **{row['mode']}**: Moderate entropy ({row['mean_curriculum_entropy']:.4f})")
        else:
            lines.append(f"- ✅ **{row['mode']}**: High entropy ({row['mean_curriculum_entropy']:.4f}) — healthy task diversity")
    lines.append("")

    # Q5: Recommendation
    lines.append("### 5. Recommendation for Longer Runs")
    if sorted_by_composite:
        top_pick = sorted_by_composite[0]
        runner_up = sorted_by_composite[1] if len(sorted_by_composite) > 1 else None

        lines.append(f"**Primary recommendation: `{top_pick['mode']}`**")
        lines.append(f"- Composite score: {top_pick.get('composite_score', 0):.4f}")
        lines.append(f"- Forgetting index: {top_pick['mean_forgetting_index']:.4f}")
        lines.append(f"- Curriculum entropy: {top_pick['mean_curriculum_entropy']:.4f}")
        lines.append(f"- Anti-forgetting trigger rate: {top_pick['anti_forgetting_trigger_rate']:.2%}")

        if runner_up:
            lines.append(f"\n**Secondary recommendation: `{runner_up['mode']}`**")
            lines.append(f"- Composite score: {runner_up.get('composite_score', 0):.4f}")
            lines.append(f"- Forgetting index: {runner_up['mean_forgetting_index']:.4f}")
            lines.append(f"- Curriculum entropy: {runner_up['mean_curriculum_entropy']:.4f}")
    lines.append("")

    # Q6: Failures
    lines.append("### 6. Failures and Robustness")
    failures_found = False
    for row in sorted_by_composite:
        if row["failure_count"] > 0:
            lines.append(f"- ❌ **{row['mode']}**: {row['failure_count']} failures, {row['error_count']} errors")
            failures_found = True
    if not failures_found:
        lines.append("No failures detected across all runs — all modes are robust.")
    lines.append("")

    # Data stats
    lines.append("## Data Statistics\n")
    lines.append(f"- Total selections logged: {len(all_rows)}")
    lines.append(f"- Unique modes: {len(by_mode)}")
    lines.append(f"- Modes present: {', '.join(sorted(by_mode.keys()))}")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Summarize aggregation experiment runs.")
    parser.add_argument(
        "--log-dir",
        default="/root/experiments/dicode_runs/aggregation/logs",
        help="Directory containing run logs and JSONL diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/experiments/dicode_runs/aggregation",
        help="Directory for output summary files.",
    )
    args = parser.parse_args()

    print(f"Scanning for JSONL files in: {args.log_dir}")
    jsonl_files = find_jsonl_files(args.log_dir)
    print(f"  Found {len(jsonl_files)} JSONL files.")

    print(f"Scanning for run logs in: {args.log_dir}")
    run_logs = find_run_logs(args.log_dir)
    print(f"  Found {len(run_logs)} run logs.")

    produce_summary(jsonl_files, run_logs, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
