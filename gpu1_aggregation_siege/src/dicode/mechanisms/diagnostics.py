"""Diagnostics utilities for aggregation mechanisms.

Provides functions for reading, parsing, and summarizing the JSONL
diagnostics logs produced by the aggregation selector.
"""

import json
import os
from collections import defaultdict
from typing import Any, Optional


def load_diagnostics_log(log_path: str) -> list[dict]:
    """Load all rows from an aggregation diagnostics JSONL file.

    Args:
        log_path: Path to the JSONL file.

    Returns:
        List of dicts, one per line.
    """
    rows = []
    if not os.path.exists(log_path):
        return rows

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows


def summarize_diagnostics(rows: list[dict]) -> dict[str, Any]:
    """Compute summary statistics from diagnostics rows.

    Args:
        rows: List of diagnostics dicts from load_diagnostics_log().

    Returns:
        Dict with summary statistics keyed by aggregation mode.
    """
    if not rows:
        return {}

    # Group by mode
    by_mode = defaultdict(list)
    for row in rows:
        mode = row.get("aggregation_mode", "unknown")
        by_mode[mode].append(row)

    summary = {}
    for mode, mode_rows in by_mode.items():
        n = len(mode_rows)

        forgetting_indices = [r.get("forgetting_index", 0.0) for r in mode_rows]
        anti_forgetting_count = sum(1 for r in mode_rows if r.get("anti_forgetting_mode"))
        entropies = [r.get("curriculum_entropy", 0.0) for r in mode_rows]
        num_candidates = [r.get("num_candidates", 0) for r in mode_rows]
        num_selected = [len(r.get("selected_task_ids", [])) for r in mode_rows]

        summary[mode] = {
            "num_selections": n,
            "mean_forgetting_index": sum(forgetting_indices) / max(1, n),
            "max_forgetting_index": max(forgetting_indices) if forgetting_indices else 0.0,
            "anti_forgetting_trigger_rate": anti_forgetting_count / max(1, n),
            "mean_entropy": sum(entropies) / max(1, n),
            "mean_candidates": sum(num_candidates) / max(1, n),
            "mean_selected": sum(num_selected) / max(1, n),
            "first_timestamp": mode_rows[0].get("timestamp") if mode_rows else None,
            "last_timestamp": mode_rows[-1].get("timestamp") if mode_rows else None,
        }

    return summary


def print_diagnostics_summary(summary: dict[str, Any]) -> None:
    """Pretty-print a diagnostics summary.

    Args:
        summary: Dict from summarize_diagnostics().
    """
    print("\n" + "=" * 80)
    print("AGGREGATION DIAGNOSTICS SUMMARY")
    print("=" * 80)

    for mode, stats in sorted(summary.items()):
        print(f"\n--- {mode} ---")
        print(f"  Selections:           {stats['num_selections']}")
        print(f"  Mean forgetting idx:  {stats['mean_forgetting_index']:.4f}")
        print(f"  Max forgetting idx:   {stats['max_forgetting_index']:.4f}")
        print(f"  Anti-forgetting rate: {stats['anti_forgetting_trigger_rate']:.2%}")
        print(f"  Mean entropy:         {stats['mean_entropy']:.4f}")
        print(f"  Mean candidates:      {stats['mean_candidates']:.1f}")
        print(f"  Mean selected:        {stats['mean_selected']:.1f}")

    print("\n" + "=" * 80)
