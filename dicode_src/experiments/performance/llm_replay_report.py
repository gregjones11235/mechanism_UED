#!/usr/bin/env python3
"""LLM replay report generator (stage D research line).

Aggregates one or more ``RESULT.json`` files produced by the benchmark into a
``LLM_REPLAY_REPORT.md`` that compares ``max_in_flight ∈ {1,2,4}`` (and, for D2,
models) on median wall clock, valid-task rate, empty-response rate, retry count,
and LLM-seconds-per-valid-task. It never sums concurrent durations naively — it
reports the already-derived union/critical-path metrics from each RESULT.

Independent research tool; does not import production orchestration.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from llm_replay_benchmark import CLASSIFICATION as RESULT_CLASSIFICATION

FINAL_CONCLUSIONS = (
    "LLM_RESEARCH_REPLAY_PASS",
    "LLM_RESEARCH_PASS_WITH_CONCERNS",
    "LLM_RESEARCH_NO_IMPROVEMENT",
    "LLM_RESEARCH_BLOCKED_EXTERNAL",
    "LLM_RESEARCH_REJECTED",
)


def collect_results(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    results = []
    for p in sorted(root.rglob("RESULT.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("classification") == RESULT_CLASSIFICATION:
            data["_path"] = str(p)
            results.append(data)
    return results


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate(results: list[dict]) -> dict[str, Any]:
    """Group results by (provider, model, max_in_flight) and take medians."""
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for r in results:
        key = (r.get("provider", "?"), r.get("model", "?"), int(r.get("max_in_flight", -1)))
        groups.setdefault(key, []).append(r)

    rows = []
    for (provider, model, mif), rs in sorted(groups.items()):
        rows.append({
            "provider": provider, "model": model, "max_in_flight": mif,
            "repeats": len(rs),
            "wall_clock_s_median": _median([r.get("wall_clock_s") or 0.0 for r in rs]),
            "llm_union_s_median": _median([r.get("llm_union_s") or 0.0 for r in rs]),
            "llm_sum_s_median": _median([r.get("llm_sum_s") or 0.0 for r in rs]),
            "queue_wait_s_median": _median([r.get("queue_wait_sum_s") or 0.0 for r in rs]),
            "retry_count_median": _median([float(r.get("retry_count") or 0) for r in rs]),
            "empty_response_median": _median([float(r.get("empty_response_count") or 0) for r in rs]),
            "valid_tasks_median": _median([float(r.get("valid_tasks") or 0) for r in rs]),
            "valid_task_rate_median": _median([r.get("valid_task_rate") or 0.0 for r in rs]),
            "unique_code_hashes_median": _median([float(r.get("unique_code_hashes") or 0) for r in rs]),
            "llm_s_per_valid_median": _median([r.get("llm_seconds_per_valid_task") or 0.0 for r in rs if r.get("llm_seconds_per_valid_task") is not None]),
            "request_count": rs[0].get("request_count"),
            "candidate_slots": rs[0].get("candidate_slots"),
            "manifest_sha256": rs[0].get("manifest_sha256"),
            "source_commit": rs[0].get("source_commit"),
            "run_ids": [r.get("run_id") for r in rs],
        })
    return {"rows": rows, "group_count": len(rows)}


def _fmt(v: float | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def render_report(agg: Mapping[str, Any], *, conclusion: str,
                  extra: str = "") -> str:
    rows = agg["rows"]
    lines: list[str] = []
    lines.append("# LLM 独立性能研究线 — 固定 replay 报告（阶段 D）")
    lines.append("")
    lines.append("分类：`LLM_RESEARCH`（允许比较模型与调度行为；**非**语义一致优化，不与 B/C 主线合并）")
    lines.append("")
    lines.append(f"## 0. 结论\n\n**{conclusion}**\n")
    if extra:
        lines.append(extra)
        lines.append("")
    lines.append("## 1. 配置矩阵汇总（中位数）\n")
    lines.append("| provider | model | max_in_flight | repeats | 墙钟(s) | LLM union(s) | LLM sum(s) | queue wait(s) | retry | 空响应 | 有效任务 | 有效率 | 唯一代码 | LLM s/有效 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['provider']} | {r['model']} | {r['max_in_flight']} | {r['repeats']} | "
            f"{_fmt(r['wall_clock_s_median'])} | {_fmt(r['llm_union_s_median'])} | "
            f"{_fmt(r['llm_sum_s_median'])} | {_fmt(r['queue_wait_s_median'])} | "
            f"{_fmt(r['retry_count_median'])} | {_fmt(r['empty_response_median'])} | "
            f"{_fmt(r['valid_tasks_median'], 1)} | {_fmt(r['valid_task_rate_median'], 3)} | "
            f"{_fmt(r['unique_code_hashes_median'], 1)} | {_fmt(r['llm_s_per_valid_median'])} |")
    lines.append("")
    lines.append("## 2. 串行基线（max_in_flight=1）对比\n")
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(f"{r['provider']}/{r['model']}", []).append(r)
    for model_key, model_rows in by_model.items():
        base = next((r for r in model_rows if r["max_in_flight"] == 1), None)
        if base is None:
            continue
        lines.append(f"### {model_key}\n")
        lines.append(f"串行基线墙钟：{_fmt(base['wall_clock_s_median'])}s，有效率：{_fmt(base['valid_task_rate_median'], 3)}，"
                     f"retry：{_fmt(base['retry_count_median'])}，空响应：{_fmt(base['empty_response_median'])}，"
                     f"LLM s/有效：{_fmt(base['llm_s_per_valid_median'])}\n")
        for r in model_rows:
            if r["max_in_flight"] == 1:
                continue
            wall_delta = None
            if base["wall_clock_s_median"] and r["wall_clock_s_median"] is not None:
                wall_delta = (r["wall_clock_s_median"] - base["wall_clock_s_median"]) / base["wall_clock_s_median"] * 100
            lines.append(f"- max_in_flight={r['max_in_flight']}：墙钟 {_fmt(r['wall_clock_s_median'])}s "
                         f"({_fmt(wall_delta, 1)}% vs 串行)，有效率 {_fmt(r['valid_task_rate_median'], 3)}，"
                         f"retry {_fmt(r['retry_count_median'])}，空响应 {_fmt(r['empty_response_median'])}，"
                         f"LLM s/有效 {_fmt(r['llm_s_per_valid_median'])}")
        lines.append("")
    lines.append("## 3. 门禁核查\n")
    lines.append("- 有效任务率是否不低于串行基线（超过统计波动）？\n- 空响应率是否不恶化？\n- retry/repair 是否无异常增加？\n- GPU0 最低剩余显存是否安全？（需服务器证据，另附）\n- 服务端是否无持续 rate limit / queue collapse？\n")
    lines.append("## 4. 遗留限制\n\n（由运行者填写：重复次数、是否缺 235B 臂、GPU 证据、manifest SHA 等。）\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--conclusion", required=True, choices=FINAL_CONCLUSIONS)
    parser.add_argument("--output", default="LLM_REPLAY_REPORT.md")
    args = parser.parse_args(argv)
    results = collect_results(args.results_dir)
    if not results:
        raise SystemExit("no RESULT.json files found")
    agg = aggregate(results)
    md = render_report(agg, conclusion=args.conclusion)
    out = Path(args.output)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({agg['group_count']} config groups, {len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
