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
