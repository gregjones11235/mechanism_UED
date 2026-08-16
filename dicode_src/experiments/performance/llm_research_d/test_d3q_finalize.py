"""Local tests for d3q_finalize wall extraction (real replay shapes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3q_finalize as fin  # noqa: E402


def _arm(tmp_path, arm_id, critical=None, summary=None):
    run_dir = tmp_path / "arms" / arm_id / "run"
    run_dir.mkdir(parents=True)
    if critical is not None:
        (run_dir / "critical_path.json").write_text(json.dumps(critical), encoding="utf-8")
    if summary is not None:
        (run_dir / "replay_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {"arm_id": arm_id, "result_file": "/remote/ignored/RESULT.json", "status": "PASS"}


def test_wall_real_replay_shape(tmp_path):
    # Shape verified against B1 reference run: session_wall in critical_path,
    # session_wall_s in replay_summary, no preflight_wall key.
    arm = _arm(
        tmp_path, "large_r1",
        critical={"session_wall": 832.995494452, "covered_union": 832.986454148},
        summary={"session_wall_s": 832.995494452, "run_id": "x"},
    )
    assert fin._preflight_wall(arm, tmp_path) == pytest.approx(832.995494452)


def test_wall_explicit_preflight_wall_key(tmp_path):
    arm = _arm(tmp_path, "large_r1", critical={"preflight_wall": 100.5})
    assert fin._preflight_wall(arm, tmp_path) == 100.5


def test_wall_summary_only(tmp_path):
    arm = _arm(tmp_path, "small_r1", critical={"covered_union": 50.0}, summary={"session_wall_s": 51.0})
    assert fin._preflight_wall(arm, tmp_path) == 51.0


def test_wall_missing_fail_closed(tmp_path):
    arm = _arm(tmp_path, "small_r1", critical={"covered_union": 50.0})
    with pytest.raises(fin.FinalizeError):
        fin._preflight_wall(arm, tmp_path)


def test_wall_mismatch_fail_closed(tmp_path):
    arm = _arm(
        tmp_path, "small_r1",
        critical={"session_wall": 900.0},
        summary={"session_wall_s": 500.0},
    )
    with pytest.raises(fin.FinalizeError):
        fin._preflight_wall(arm, tmp_path)


def test_wall_critical_missing_fail_closed(tmp_path):
    arm = {"arm_id": "ghost", "result_file": "/remote/RESULT.json", "status": "PASS"}
    with pytest.raises(fin.FinalizeError):
        fin._preflight_wall(arm, tmp_path)


def test_wall_no_result_file_returns_none(tmp_path):
    assert fin._preflight_wall({"arm_id": "x"}, tmp_path) is None


def test_aggregate_pipeline_metrics_populated(tmp_path, monkeypatch):
    # Regression for the 20260816T083433Z defect: parse_slot_id returns the
    # numeric repeat ("1"), so filtering rows by "r1" never matched and
    # pipeline_metrics stayed empty -> pipeline_wall_s=0 in the final result.
    slots = ["slot_r1_small_p00", "slot_r1_large_p00"]
    monkeypatch.setattr(fin, "all_slots_ordered", lambda: slots)
    monkeypatch.setattr(
        fin, "_load_reconciliation",
        lambda root: {"slot_consumed": {}, "provider_consumed": {"ollama": 0, "deepseek_official": 0}},
    )
    art = tmp_path / "art"
    for sid, wall in (("slot_r1_small_p00", 10.0), ("slot_r1_large_p00", 20.0)):
        slot_dir = art / "slots" / sid
        slot_dir.mkdir(parents=True)
        (slot_dir / f"{sid}.result.json").write_text(json.dumps({
            "attempts": 1, "initial_valid": True, "final_valid": True,
            "generation_wall_s": wall, "repair_wall_s": 1.0, "cpu_validation_wall_s": 2.0,
        }), encoding="utf-8")
    out = tmp_path / "agg"
    agg = fin.cmd_aggregate([art], tmp_path, out)
    pm = agg["pipeline_metrics"]
    assert set(pm.keys()) == {"small_r1", "large_r1"}
    assert pm["small_r1"]["pipeline_wall_s"] == 13.0
    assert pm["large_r1"]["pipeline_wall_s"] == 23.0
    assert pm["small_r1"]["final_valid_rate"] == 1.0
    assert pm["small_r1"]["slots_completed"] == 1
