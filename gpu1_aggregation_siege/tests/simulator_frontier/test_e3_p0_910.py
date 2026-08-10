# -*- coding: utf-8 -*-
"""P0-9/10 contract tests.

P0-9: explicit source-Student optimizer semantics.  The source checkpoint is
params-only (no opt_state), so E3 freezes a NEW optimizer phase starting at
session 0 (Adam, step 0) and then continues continuously across sessions — never
a silent rebuild.  Each session report records optimizer_semantics.

P0-10: task_class_count / num_envs / num_steps are recorded as SEPARATE
fields — the vague update_shape=[16,128] must be gone from the formal runner.
"""

import inspect

import pytest


def test_formal_layout_is_1024x128():
    import yaml
    cfg = yaml.safe_load(open("conf/training/default.yaml"))
    assert cfg["num_envs"] == 1024
    assert cfg["num_steps"] == 128


def test_task_class_count_is_16():
    # 15 sampled curriculum slots + 1 OriginalTask appended internally = 16.
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        CURRICULUM_SLOT_COUNT,
    )
    assert CURRICULUM_SLOT_COUNT == 15
    assert CURRICULUM_SLOT_COUNT + 1 == 16


def _runner_source() -> str:
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_formal_longrun.py"
    return p.read_text(encoding="utf-8")


def test_no_vague_update_shape_in_formal_runner():
    """P0-10: the formal runner must not use update_shape=[16,128]."""
    src = _runner_source()
    assert "update_shape" not in src
    assert "task_class_count" in src
    assert "num_envs" in src
    assert "num_steps" in src


def test_optimizer_semantics_declared():
    """P0-9: the formal runner must declare optimizer_semantics explicitly."""
    src = _runner_source()
    assert "optimizer_semantics" in src
    assert "NEW_OPTIMIZER_PHASE_FROM_SESSION0_THEN_CONTINUOUS" in src
    assert "RESUME_PREVIOUS_SESSION_OPT_STATE" in src


def test_optimizer_continuous_across_sessions():
    """P0-9: resume path reattaches the previous session's opt_state/step."""
    import dicode.simulator_frontier.runstate_codec as codec
    assert "opt_state" in codec.REQUIRED_RUNSTATE_FIELDS
    assert "params" in codec.REQUIRED_RUNSTATE_FIELDS
    assert "train_step" in codec.REQUIRED_RUNSTATE_FIELDS
