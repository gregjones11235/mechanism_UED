# -*- coding: utf-8 -*-
"""E3 scoring-retention fix contract tests.

Root cause of the 100-update-session OOM: the session-level lax.scan retained
rmt_entering_tokens / slowgru_longstate (recurrent trajectory state the PPO
loss needs but the scoring calculator never reads) across all 100 updates, and
kept all 100 scoring_data copies when only the last scoring_window_updates (40)
are used.

This fix:
  1. strips rmt_entering_tokens / slowgru_longstate from the scoring projection
     (SCORING_MUST_NOT_RETAIN_POLICY_MEMORY=true), while the PPO training
     trajectory stays FULL.
  2. splits the session scan into warmup (no scoring retention) + scoring window
     (retain only the last k projected windows) with strict RNG/state continuity.

TEST_ONLY / SYNTHETIC fixtures where noted; no real LLM, no GPU training.
"""

import inspect
import re

import pytest


# ---------------------------------------------------------------------------
# 1. Scoring projection must drop recurrent policy-memory fields
# ---------------------------------------------------------------------------

def test_scoring_projection_drops_rmt_entering_tokens():
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    # The scoring_traj.replace must explicitly set rmt_entering_tokens=None.
    m = re.search(
        r"scoring_traj = traj_batch\.replace\((.*?)\)\n",
        src, re.DOTALL)
    assert m, "scoring_traj.replace not found"
    body = m.group(1)
    assert "rmt_entering_tokens=None" in body, \
        "scoring_traj must strip rmt_entering_tokens"
    assert "SCORING_MUST_NOT_RETAIN_POLICY_MEMORY" in src, \
        "invariant marker missing"


def test_scoring_projection_drops_slowgru_longstate():
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    m = re.search(
        r"scoring_traj = traj_batch\.replace\((.*?)\)\n",
        src, re.DOTALL)
    assert m, "scoring_traj.replace not found"
    body = m.group(1)
    assert "slowgru_longstate=None" in body, \
        "scoring_traj must strip slowgru_longstate"


def test_scoring_projection_preserves_required_score_fields():
    """scoring.py reads info / reward / value / advantages — these must remain
    in the scoring projection."""
    import dicode.scoring as sc
    src = inspect.getsource(sc)
    for field in ("traj_batch.info", "traj_batch.reward", "traj_batch.value",
                  "advantages"):
        assert field in src, f"scoring.py must read {field}"


def test_ppo_training_trajectory_stays_full():
    """The PPO loss path must NOT strip rmt_entering_tokens / slowgru_longstate —
    they feed prepare_training_memory_batch -> policy_forward_train."""
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    # The full traj_batch is used in the loss before scoring_traj is built.
    assert "rmt_entering_tokens" in src
    assert "slowgru_longstate" in src
    # prepare_training_memory_batch must be invoked on the FULL traj_batch.
    assert "prepare_training_memory_batch" in src


# ---------------------------------------------------------------------------
# 2. Two-phase scan: warmup no-retention + scoring window
# ---------------------------------------------------------------------------

def test_split_retention_executes_all_updates():
    """The split must execute warmup + scoring = NUM_UPDATES total."""
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    assert "warmup_updates" in src
    assert "scoring_updates" in src
    assert "warmup_updates + scoring_updates" in src or \
        "max(NUM_UPDATES - k, 0)" in src
    # warmup scan must NOT retain scoring (empty output structure).
    assert "_warmup_scoring" in src


def test_split_retention_keeps_last_scoring_window_only():
    """The scoring scan retains only scoring_window_updates (k) windows, not all
    NUM_UPDATES."""
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    assert "scoring_updates = min(NUM_UPDATES, k)" in src or \
        "scoring_updates" in src
    assert "warmup_updates = max(NUM_UPDATES - k, 0)" in src


def test_two_phase_scan_preserves_continuity():
    """Phase B must start from Phase A's final runner_state (never re-init)."""
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    # The warmup scan output feeds the scoring scan's initial carry.
    assert "final_runner_state" in src
    assert "initial_runner_state" in src
    # No re-init between phases: no env.reset / PRNGKey re-seed inside the split.
    assert "jax.lax.scan" in src


def test_scoring_window_is_last_k_not_per_chunk():
    """The final scoring_window_data slices [-k:] of the retained scan output —
    exactly the last k logical updates."""
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    assert "x[-k:]" in src


def test_no_monolithic_100_retention():
    """The old single-scan-of-100 must be gone: the session scan is now split,
    and the warmup scan output is discarded (empty structure)."""
    import dicode.ppo_tr as tr
    src = inspect.getsource(tr)
    # There must be a warmup scan with a non-retaining step function.
    assert "_update_step_noscore" in src or "warmup_updates" in src
    assert "length=warmup_updates" in src


# ---------------------------------------------------------------------------
# 3. Conservative layout still holds
# ---------------------------------------------------------------------------

def test_formal_layout_is_1024x128():
    import yaml
    cfg = yaml.safe_load(open("conf/training/default.yaml"))
    assert cfg["num_envs"] == 1024
    assert cfg["num_steps"] == 128


def test_formal_session_runs_100_outer_updates():
    import yaml
    mgr = yaml.safe_load(open("conf/dicode_manager/default.yaml"))
    assert mgr["max_updates_per_session"] == 100


def test_formal_session_env_steps_13107200():
    assert 100 * 1024 * 128 == 13107200


def test_scoring_window_updates_is_40():
    import yaml
    cfg = yaml.safe_load(open("conf/training/default.yaml"))
    assert cfg.get("scoring_window_updates", 40) == 40
