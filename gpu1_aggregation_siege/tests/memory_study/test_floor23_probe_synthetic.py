"""SYNTHETIC end-to-end probe tests (jax-free): schema validation, tier3
library integration, primary-event cross-check, fail-closed paths, CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dicode.memory_study.floor23_probe import (
    RESULT_SCHEMA_ID,
    SCENARIO,
    CandidateRuntime,
    load_tier3_library,
    make_synthetic_candidate,
    run_floor23_probe,
    synthetic_states,
)
from dicode.memory_study.ho_capture_bank import generate_synthetic_capture_bank
from dicode.memory_study.ho_contract import HOMode, FailClosed

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "gpu1_aggregation_siege" / "scripts" / "run_memory_study_floor23.py"


def _fixtures():
    states = synthetic_states(4, seed=0)
    _, captures = generate_synthetic_capture_bank(3, 4, 8, seed=777)
    runtimes = [
        make_synthetic_candidate("SYN_CAND_A", success_bias=0.60),
        make_synthetic_candidate("SYN_CAND_B", success_bias=0.35),
    ]
    return states, captures, runtimes


def test_tier3_library_loads_from_repo_root():
    metrics, predicates = load_tier3_library(REPO_ROOT)
    assert metrics.FRONT == SCENARIO
    assert predicates.front_floor_transition_reached(2, 3) is True
    assert predicates.front_floor_transition_reached(2, 2) is False


def test_tier3_library_missing_fails_closed(tmp_path):
    with pytest.raises(FailClosed, match="TIER3_TOOLING_MISSING"):
        load_tier3_library(tmp_path)


def test_synthetic_end_to_end(tmp_path):
    states, captures, runtimes = _fixtures()
    summary = run_floor23_probe(states, runtimes, captures, tmp_path,
                                run_mode="synthetic", probe_seed=0)
    assert summary["results_written"] == len(states) * len(runtimes) * 3
    assert summary["num_states"] == len(states)
    assert len(summary["arms"]) == len(runtimes) * 3
    # every arm reports the frozen primary metric name and a sane value
    for arm in summary["arms"]:
        pm = arm["primary_metric"]
        assert pm["metric"] == "P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START"
        assert pm["valid_starts"] == len(states)
        assert pm["value"] is None or 0.0 <= pm["value"] <= 1.0
    # per-result provenance files exist and carry schema + PASS receipts
    files = list((tmp_path / "results").glob("*.json"))
    assert len(files) == summary["results_written"]
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        assert doc["schema"] == RESULT_SCHEMA_ID
        assert doc["scenario"] == SCENARIO
        assert doc["receipt"]["verdict"] == "PASS"
        assert doc["capture_bank_hash"] == captures[0].bank_hash
        assert doc["episode"]["valid_start"] is True


def test_synthetic_probe_is_repeatable(tmp_path):
    states, captures, runtimes = _fixtures()
    s1 = run_floor23_probe(states, runtimes, captures, tmp_path / "r1",
                           run_mode="synthetic", probe_seed=0)
    s2 = run_floor23_probe(states, runtimes, captures, tmp_path / "r2",
                           run_mode="synthetic", probe_seed=0)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


def test_primary_event_cross_check_fails_closed(tmp_path):
    states, captures, _ = _fixtures()

    def lying_rollout(state, memory):
        return {
            "scenario": SCENARIO, "valid_start": True,
            "episode_id": state["state_id"],
            "front_floor_transition_reached": True,   # claims success...
            "graph_distance_progress": 1.0,
            "from_level": 2, "to_level": 2,           # ...but never left floor 2
        }

    lying = CandidateRuntime(candidate_id="LIAR", params={"w": [1.0]},
                             initial_memory=(), step_fn=lambda p, m, o: m,
                             rollout_fn=lying_rollout)
    with pytest.raises(FailClosed, match="PRIMARY_EVENT_INCONSISTENT"):
        run_floor23_probe(states[:1], [lying], captures, tmp_path,
                          run_mode="synthetic")


def test_invalid_start_episode_fails_closed(tmp_path):
    states, captures, _ = _fixtures()

    def bad_rollout(state, memory):
        return {"scenario": SCENARIO, "valid_start": False,
                "episode_id": state["state_id"],
                "front_floor_transition_reached": False,
                "graph_distance_progress": 0.0}

    bad = CandidateRuntime(candidate_id="BADSTART", params={"w": [1.0]},
                           initial_memory=(), step_fn=lambda p, m, o: m,
                           rollout_fn=bad_rollout)
    with pytest.raises(FailClosed, match="EPISODE_INVALID_START"):
        run_floor23_probe(states[:1], [bad], captures, tmp_path,
                          run_mode="synthetic")


def test_duplicate_candidate_id_fails_closed(tmp_path):
    states, captures, runtimes = _fixtures()
    dup = [runtimes[0], CandidateRuntime(
        candidate_id=runtimes[0].candidate_id, params={"w": [1.0]},
        initial_memory=(), step_fn=lambda p, m, o: m,
        rollout_fn=runtimes[0].rollout_fn)]
    with pytest.raises(FailClosed, match="DUPLICATE_CANDIDATE_ID"):
        run_floor23_probe(states[:1], dup, captures, tmp_path,
                          run_mode="synthetic")


def test_empty_inputs_fail_closed(tmp_path):
    states, captures, runtimes = _fixtures()
    with pytest.raises(FailClosed, match="PROBE_NO_STATES"):
        run_floor23_probe([], runtimes, captures, tmp_path)
    with pytest.raises(FailClosed, match="PROBE_NO_CANDIDATES"):
        run_floor23_probe(states, [], captures, tmp_path)
    with pytest.raises(FailClosed, match="PROBE_NO_CAPTURES"):
        run_floor23_probe(states, runtimes, [], tmp_path)


def test_unknown_run_mode_fails_closed(tmp_path):
    states, captures, runtimes = _fixtures()
    with pytest.raises(FailClosed, match="UNKNOWN_RUN_MODE"):
        run_floor23_probe(states, runtimes, captures, tmp_path,
                          run_mode="hybrid")


def test_cli_synthetic_end_to_end(tmp_path):
    out = tmp_path / "cli_out"
    proc = subprocess.run(
        [sys.executable, str(CLI), "--mode", "synthetic",
         "--out-root", str(out), "--num-states", "3", "--seed", "5"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["status"] == "SYNTHETIC_PROBE_COMPLETE"
    assert doc["results_written"] == 3 * 2 * 3
    assert (out / "summary.json").is_file()


def test_cli_real_mode_blocked_locally(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(CLI), "--mode", "real",
         "--out-root", str(tmp_path / "real_out")],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    doc = json.loads(proc.stdout)
    assert doc["status"] == "BLOCKED"
    assert doc["reason"] in ("REAL_MODE_ASSETS_MISSING",
                             "REAL_MODE_BLOCKED_LOCAL_NO_JAX")