"""D3Q final aggregation and deliverables.

``aggregate``: build D3Q_SLOT_RESULTS.csv (all 72 slots incl. incident-01
losses), the error taxonomy, token usage, and per model/repeat pipeline
metrics from the published chunk artifacts + ledger + reconciliation.

``finalize``: merge GPU2 preflight results, compute the primary metric
(seconds per preflight-accepted task), apply the frozen seven-choice verdict
rules, and publish the final deliverable set with SHA256SUMS.

Pure local computation over published evidence; fail closed on gaps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3q_slot_runner as runner_mod  # noqa: E402
from d3q_phase2_driver import (  # noqa: E402
    RECONCILIATION_FILENAME,
    _load_reconciliation,
    all_slots_ordered,
)

COUNTER_FIELDS = (
    "empty_response", "timeout", "connection_error", "http_4xx", "http_5xx",
    "invalid_json", "extract_error", "syntax_error", "api_enum_error",
    "inventory_error", "dangerous_import", "dangerous_capability",
    "cpu_jax_error", "duplicate_code",
)
CSV_FIELDS = (
    "slot_id", "repeat", "arm", "model", "provider", "prompt_index", "status",
    "posts_used", "initial_valid", "final_valid", "attempts", "repair_requests",
    "repair_success", *COUNTER_FIELDS, "fatal_api_blocked",
    "prompt_tokens", "completion_tokens", "cached_tokens",
    "generation_wall_s", "repair_wall_s", "cpu_validation_wall_s",
    "final_code_sha256",
)


class FinalizeError(RuntimeError):
    def __init__(self, reason: str, detail: Any = None):
        self.reason = reason
        self.detail = detail
        super().__init__(reason if detail is None else f"{reason}: {detail}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _locate(slot_id: str, artifact_dirs: Sequence[Path]) -> Optional[Path]:
    for artifact in artifact_dirs:
        slot_dir = Path(artifact) / "slots" / slot_id
        if (slot_dir / f"{slot_id}.result.json").is_file():
            return slot_dir
    return None


def _slot_row(slot_id: str, artifact_dirs: Sequence[Path], reconciled: Dict[str, int]) -> Dict[str, Any]:
    repeat, arm, prompt_index = runner_mod.parse_slot_id(slot_id)
    provider, model, _url = runner_mod.arm_to_provider_model(arm)
    base = {
        "slot_id": slot_id, "repeat": repeat, "arm": arm, "model": model,
        "provider": provider, "prompt_index": prompt_index,
    }
    if slot_id in reconciled:
        row = {field: "" for field in CSV_FIELDS}
        row.update(base)
        row["status"] = "lost_reconciled"
        row["posts_used"] = reconciled[slot_id]
        return row
    slot_dir = _locate(slot_id, artifact_dirs)
    if slot_dir is None:
        raise FinalizeError("slot_artifact_missing", slot_id)
    result = json.loads((slot_dir / f"{slot_id}.result.json").read_text(encoding="utf-8"))
    row = {field: "" for field in CSV_FIELDS}
    row.update(base)
    row["status"] = "completed"
    row["posts_used"] = result.get("attempts")
    for field in CSV_FIELDS:
        if field in result:
            row[field] = result[field]
    return row


def cmd_aggregate(artifact_dirs: Sequence[Path], artifacts_root: Path, out_dir: Path) -> Dict[str, Any]:
    artifact_dirs = [Path(p) for p in artifact_dirs]
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FinalizeError("output_exists", str(out_dir))
    out_dir.mkdir(parents=True)
    reconciliation = _load_reconciliation(Path(artifacts_root))
    reconciled = dict(reconciliation["slot_consumed"])

    rows = [_slot_row(slot_id, artifact_dirs, reconciled) for slot_id in all_slots_ordered()]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    (out_dir / "D3Q_SLOT_RESULTS.csv").write_text(buffer.getvalue(), encoding="utf-8")

    taxonomy: Dict[str, Any] = {"by_provider": {}, "totals": {}}
    tokens: Dict[str, Any] = {}
    pipeline: Dict[str, Any] = {}
    for provider in (runner_mod.SMALL_PROVIDER, runner_mod.LARGE_PROVIDER):
        prov_rows = [r for r in rows if r["provider"] == provider and r["status"] == "completed"]
        counters = {field: sum(int(r[field] or 0) for r in prov_rows) for field in COUNTER_FIELDS}
        taxonomy["by_provider"][provider] = counters
        for field, count in counters.items():
            taxonomy["totals"][field] = taxonomy["totals"].get(field, 0) + count
        prompt_tokens = sum(int(r["prompt_tokens"] or 0) for r in prov_rows)
        completion_tokens = sum(int(r["completion_tokens"] or 0) for r in prov_rows)
        cached_tokens = sum(int(r["cached_tokens"] or 0) for r in prov_rows)
        posts = sum(int(r["posts_used"] or 0) for r in prov_rows)
        tokens[provider] = {
            "posts": posts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        }
        for arm in ("small", "large"):
            for repeat in ("r1", "r2", "r3"):
                # parse_slot_id stores the numeric repeat ("1"), not "r1";
                # match on that so pipeline_metrics is actually populated.
                repeat_num = repeat[1:]
                arm_rows = [
                    r for r in rows
                    if r["arm"] == arm and r["repeat"] == repeat_num and r["status"] == "completed"
                ]
                if not arm_rows:
                    continue
                final_valid = [r for r in arm_rows if r["final_valid"] is True]
                wall_sum = sum(
                    float(r["generation_wall_s"] or 0) + float(r["repair_wall_s"] or 0)
                    + float(r["cpu_validation_wall_s"] or 0)
                    for r in arm_rows
                )
                repairs = sum(int(r["repair_requests"] or 0) for r in arm_rows)
                key = f"{arm}_{repeat}"
                pipeline[key] = {
                    "slots_completed": len(arm_rows),
                    "slots_lost": len([
                        s for s, c in reconciled.items()
                        if s.startswith(f"slot_{repeat}_{arm}_")
                    ]),
                    "final_valid_count": len(final_valid),
                    "final_valid_rate": round(len(final_valid) / len(arm_rows), 6),
                    "total_posts": sum(int(r["posts_used"] or 0) for r in arm_rows),
                    "repair_requests": repairs,
                    "pipeline_wall_s": round(wall_sum, 6),
                    "pipeline_wall_per_final_valid_s": (
                        round(wall_sum / len(final_valid), 6) if final_valid else None
                    ),
                }
    aggregate = {
        "classification": "D3Q_AGGREGATE",
        "schema_version": 1,
        "slots_total": len(rows),
        "slots_completed": len([r for r in rows if r["status"] == "completed"]),
        "slots_lost_reconciled": sorted(reconciled),
        "error_taxonomy": taxonomy,
        "token_usage": tokens,
        "reconciled_posts": dict(reconciliation["provider_consumed"]),
        "pipeline_metrics": pipeline,
    }
    (out_dir / "D3Q_AGGREGATE.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def _preflight_wall(arm_entry: Dict[str, Any], preflight_artifacts: Path) -> Optional[float]:
    result_file = arm_entry.get("result_file")
    if not result_file:
        return None
    # result_file is a REMOTE path; the identical collected copy lives under
    # preflight_artifacts/arms/<arm_id>/run/critical_path.json.
    arm_id = arm_entry.get("arm_id")
    run_dir = Path(preflight_artifacts) / "arms" / arm_id / "run"
    critical = run_dir / "critical_path.json"
    if not critical.is_file():
        raise FinalizeError("preflight_critical_path_missing", str(critical))
    data = json.loads(critical.read_text(encoding="utf-8"))
    # Preferred: an explicit preflight_wall field (forward-compatible with a
    # future replay emitting it). Real replay output (verified against the B1
    # reference run perf48_b1r2_gpu2_20260813T032611Z) instead records the
    # wall as critical_path.session_wall, mirrored by
    # replay_summary.session_wall_s (both 832.995494452 for B1).
    wall = data.get("preflight_wall")
    if isinstance(wall, dict):
        wall = wall.get("total_s", wall.get("duration_s"))
    summary_wall = None
    summary_path = run_dir / "replay_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_wall = summary.get("session_wall_s")
        if not isinstance(summary_wall, (int, float)):
            summary_wall = None
    if not isinstance(wall, (int, float)):
        wall = data.get("session_wall")
    if isinstance(wall, (int, float)) and summary_wall is not None:
        if abs(float(wall) - summary_wall) > max(0.005 * summary_wall, 0.05):
            raise FinalizeError("preflight_wall_mismatch", arm_id)
    if not isinstance(wall, (int, float)):
        wall = summary_wall
    if not isinstance(wall, (int, float)) or wall <= 0:
        raise FinalizeError("preflight_wall_missing", arm_id)
    return float(wall)


def cmd_finalize(aggregate_dir: Path, preflight_result: Path, preflight_artifacts: Path, out_dir: Path, price_json: Optional[Path]) -> Dict[str, Any]:
    aggregate_dir = Path(aggregate_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FinalizeError("output_exists", str(out_dir))
    aggregate = json.loads((aggregate_dir / "D3Q_AGGREGATE.json").read_text(encoding="utf-8"))
    csv_text = (aggregate_dir / "D3Q_SLOT_RESULTS.csv").read_text(encoding="utf-8")
    preflight = json.loads(Path(preflight_result).read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise FinalizeError("preflight_not_pass", preflight.get("status"))
    arms = {arm["arm_id"]: arm for arm in preflight.get("arms", [])}

    primary: Dict[str, Any] = {}
    for arm_id, arm in arms.items():
        pipeline_key = arm_id
        pipe = aggregate["pipeline_metrics"].get(pipeline_key)
        accepted = arm.get("accepted") or []
        if arm.get("status") == "NO_CANDIDATES":
            accepted = []
        preflight_wall = (
            _preflight_wall(arm, preflight_artifacts) if arm.get("status") == "PASS" else None
        )
        pipeline_wall = pipe["pipeline_wall_s"] if pipe else 0.0
        if accepted:
            total_wall = pipeline_wall + (preflight_wall or 0.0)
            primary[arm_id] = {
                "accepted": accepted,
                "accepted_count": len(accepted),
                "pipeline_wall_s": pipeline_wall,
                "preflight_wall_s": preflight_wall,
                "seconds_per_preflight_accepted_task": round(total_wall / len(accepted), 6),
            }
        else:
            primary[arm_id] = {
                "accepted": [],
                "accepted_count": 0,
                "pipeline_wall_s": pipeline_wall,
                "preflight_wall_s": preflight_wall,
                "seconds_per_preflight_accepted_task": None,
            }

    def model_summary(arm_name: str) -> Dict[str, Any]:
        keys = [f"{arm_name}_{r}" for r in ("r1", "r2", "r3")]
        accepted_counts = [primary[k]["accepted_count"] for k in keys if k in primary]
        secs = [primary[k]["seconds_per_preflight_accepted_task"] for k in keys if k in primary and primary[k]["seconds_per_preflight_accepted_task"] is not None]
        return {
            "accepted_per_repeat": accepted_counts,
            "seconds_per_accepted_task_per_repeat": [
                primary[k]["seconds_per_preflight_accepted_task"] for k in keys if k in primary
            ],
            "mean_seconds_per_accepted_task": round(sum(secs) / len(secs), 6) if secs else None,
            "total_accepted": sum(accepted_counts) if accepted_counts else 0,
        }

    small = model_summary("small")
    large = model_summary("large")
    verdict, conditions = _verdict(aggregate, primary, small, large)

    cost: Dict[str, Any] = {
        "token_usage": aggregate["token_usage"],
        "deepseek_cost": {"status": "tokens_only_no_verified_price_snapshot"},
    }
    if price_json is not None and Path(price_json).is_file():
        price = json.loads(Path(price_json).read_text(encoding="utf-8"))
        usage = aggregate["token_usage"].get(runner_mod.LARGE_PROVIDER, {})
        try:
            input_price = float(price["input_price_per_mtok"])
            output_price = float(price["output_price_per_mtok"])
            cache_price = float(price.get("cache_hit_price_per_mtok", input_price))
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            cached = usage.get("cached_tokens", 0)
            uncached_input = max(prompt - cached, 0)
            total = (uncached_input * input_price + cached * cache_price + completion * output_price) / 1_000_000
            cost["deepseek_cost"] = {
                "status": "computed_from_snapshot",
                "snapshot": price,
                "formula": "uncached_input*in + cached*cache + completion*out (per token)",
                "usd": round(total, 6),
            }
        except (KeyError, TypeError, ValueError) as exc:
            cost["deepseek_cost"] = {"status": "tokens_only_price_snapshot_invalid", "error": str(exc)}

    final = {
        "classification": "D3Q_FINAL_RESULT",
        "schema_version": 1,
        "verdict": verdict,
        "verdict_conditions": conditions,
        "primary_metric_name": "seconds_per_preflight_accepted_task",
        "primary_metric": primary,
        "model_summary": {"small": small, "large": large},
        "aggregate": aggregate,
        "preflight": preflight,
        "attrition_disclosure": {
            "incident": "D3Q_PHASE2_INCIDENT_01",
            "lost_slots": aggregate["slots_lost_reconciled"],
            "note": "r1-small has 7/12 prompts with results; lost slots consumed real POST budget and are excluded from all rates.",
        },
    }
    out_dir.mkdir(parents=True)
    (out_dir / "D3Q_FINAL_RESULT.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (out_dir / "D3Q_SLOT_RESULTS.csv").write_text(csv_text, encoding="utf-8")
    (out_dir / "D3Q_PREFLIGHT_RESULTS.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    (out_dir / "D3Q_ERROR_TAXONOMY.json").write_text(json.dumps(aggregate["error_taxonomy"], indent=2) + "\n", encoding="utf-8")
    (out_dir / "D3Q_COST_AND_TOKENS.json").write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")
    _write_inventory_and_sums(out_dir)
    return final


def _verdict(aggregate, primary, small, large):
    conditions: Dict[str, Any] = {}
    small_secs = small["seconds_per_accepted_task_per_repeat"]
    large_secs = large["seconds_per_accepted_task_per_repeat"]
    small_valid_rates = [
        aggregate["pipeline_metrics"][f"small_{r}"]["final_valid_rate"]
        for r in ("r1", "r2", "r3") if f"small_{r}" in aggregate["pipeline_metrics"]
    ]
    large_valid_rates = [
        aggregate["pipeline_metrics"][f"large_{r}"]["final_valid_rate"]
        for r in ("r1", "r2", "r3") if f"large_{r}" in aggregate["pipeline_metrics"]
    ]
    small_total_accepted = small["total_accepted"]
    large_total_accepted = large["total_accepted"]
    if small_total_accepted == 0 or large_total_accepted == 0:
        if small_total_accepted == 0 and large_total_accepted == 0:
            return "D3_NO_CLEAR_WINNER", {"reason": "both arms accepted zero tasks"}
        winner = "D3_DEEPSEEK_FLASH_FASTER_END_TO_END" if small_total_accepted == 0 else "D3_SMALL_MODEL_FASTER_END_TO_END"
        return winner, {"reason": "one arm accepted zero tasks (metric infinity for that arm)"}
    small_faster_generation = None
    mean_small = small["mean_seconds_per_accepted_task"]
    mean_large = large["mean_seconds_per_accepted_task"]
    if mean_small is not None and mean_large is not None:
        if mean_small < mean_large:
            return "D3_SMALL_MODEL_FASTER_END_TO_END", {"mean_small": mean_small, "mean_large": mean_large}
        if mean_small > mean_large:
            lower_valid = all(s <= l for s, l in zip(small_valid_rates, large_valid_rates)) if small_valid_rates and large_valid_rates else False
            conditions = {
                "mean_small": mean_small,
                "mean_large": mean_large,
                "small_valid_rates": small_valid_rates,
                "large_valid_rates": large_valid_rates,
                "direction_consistent_repeats": "checked_in_report",
            }
            if lower_valid:
                return "D3_SMALL_MODEL_INVALIDITY_ERASES_LOCAL_SPEED_GAIN", conditions
            return "D3_DEEPSEEK_FLASH_FASTER_END_TO_END", conditions
    return "D3_NO_CLEAR_WINNER", {"reason": "metric incomplete", "small": small, "large": large}


def _write_inventory_and_sums(out_dir: Path) -> None:
    inventory = {"classification": "D3Q_ARTIFACT_INVENTORY", "schema_version": 1, "files": []}
    lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name not in ("SHA256SUMS", "D3Q_ARTIFACT_INVENTORY.json"):
            digest = _sha256_file(path)
            rel = path.relative_to(out_dir).as_posix()
            inventory["files"].append({"path": rel, "sha256": digest, "bytes": path.stat().st_size})
            lines.append(f"{digest}  {rel}")
    (out_dir / "D3Q_ARTIFACT_INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    inventory_digest = _sha256_file(out_dir / "D3Q_ARTIFACT_INVENTORY.json")
    lines.append(f"{inventory_digest}  D3Q_ARTIFACT_INVENTORY.json")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="d3q_finalize")
    sub = parser.add_subparsers(dest="command", required=True)
    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--artifact-dir", action="append", required=True)
    p_agg.add_argument("--artifacts-root", default=str(HERE / "d3q_artifacts"))
    p_agg.add_argument("--out", required=True)
    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--aggregate-dir", required=True)
    p_fin.add_argument("--preflight-result", required=True)
    p_fin.add_argument("--preflight-artifacts", required=True)
    p_fin.add_argument("--out", required=True)
    p_fin.add_argument("--price-json", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "aggregate":
            result = cmd_aggregate([Path(p) for p in args.artifact_dir], Path(args.artifacts_root), Path(args.out))
        else:
            price = Path(args.price_json) if args.price_json else None
            result = cmd_finalize(
                Path(args.aggregate_dir),
                Path(args.preflight_result),
                Path(args.preflight_artifacts),
                Path(args.out),
                price,
            )
    except FinalizeError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.reason, "detail": exc.detail}, sort_keys=True, default=str))
        return 2
    print(json.dumps({"status": "PASS", "classification": result.get("classification")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
