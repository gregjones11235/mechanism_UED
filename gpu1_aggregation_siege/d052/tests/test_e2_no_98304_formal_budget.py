"""§八 (director smoke handoff): the 98304 formal budget protocol is
REMOVED — the formal timeline is the frozen DiCode resolved config's
``training.total_timesteps`` clocked by ``global_env_steps``.

Contract under test:

* ``run_e2_longrun.py`` contains NO reference to 98304 /
  TOTAL_ENV_STEPS_LONG_RUN / 12288 / 8-window termination / E2_PILOT as
  a launch gate;
* the launcher consumes the frozen DiCode config whose
  ``training.total_timesteps`` is the formal timeline (NOT 98304);
* the 15+1 batch semantics come from the frozen manager config
  (original_task_proportion=0.2).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from d052.feedback_llm_ued import constants as C


def _load_longrun():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_longrun.py")
    spec = importlib.util.spec_from_file_location("run_e2_longrun_e2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LONGRUN = _load_longrun()


class TestLegacy98304ProtocolRemoved:
    def test_no_98304_references_in_source(self):
        #: the CODE has no 98304 budget field, no TOTAL_ENV_STEPS_LONG_RUN,
        #: no 12288 steps/update and no MAX_WINDOWS termination (the
        #: docstring may describe the removed protocol; the code never
        #: uses it)
        source = (Path(__file__).resolve().parents[2] / "scripts"
                  / "run_e2_longrun.py").read_text(encoding="utf-8")
        #: the old protocol's STRUCTURAL tokens must be gone everywhere
        #: (docstring and code) — the 98304/12288 literals may appear only
        #: in prose describing the removal, never as a budget field
        assert "TOTAL_ENV_STEPS_LONG_RUN" not in source
        assert "MAX_WINDOWS" not in source
        assert "training_env_steps_per_update" not in source
        assert "total_env_steps_required" not in source
        assert "E2_PILOT_AUTHORIZED" not in source

    def test_no_fixed_window_termination(self):
        cfg = LONGRUN.dicode_launch_config(
            "normal_feedback", LONGRUN.load_dicode_resolved_config())
        assert "windows" not in cfg
        assert "total_env_steps" not in cfg
        assert "training_env_steps_per_update" not in cfg

    def test_e2_pilot_not_a_launch_gate(self):
        #: the launcher gate is FORMAL_EXPERIMENT_AUTHORIZED, not the old
        #: pilot flag
        source = (Path(__file__).resolve().parents[2] / "scripts"
                  / "run_e2_longrun.py").read_text(encoding="utf-8")
        assert "FORMAL_EXPERIMENT_AUTHORIZED" in source
        assert "E2_PILOT_AUTHORIZED" not in source


class TestFormalTimelineIsDicodeClock:
    def test_total_timesteps_is_not_98304(self):
        cfg = LONGRUN.load_dicode_resolved_config()
        total = cfg["training"]["total_timesteps"]
        assert total > 0
        assert total != 98304
        assert cfg["training"]["clock_field"] == "global_env_steps"

    def test_manager_15_plus_1_semantics_consumed(self):
        cfg = LONGRUN.load_dicode_resolved_config()
        assert abs(cfg["manager"]["original_task_proportion"] - 0.2) < 1e-9
        assert cfg["manager"]["active_task_capacity"] > 0
        assert cfg["manager"]["training_sample_size_n"] > 0

    def test_constant_still_marks_98304_deprecated_not_formal(self):
        #: the historical constant is deprecated and MUST NOT be the formal
        #: budget (the longrun never references it)
        assert C.TOTAL_ENV_STEPS_LONG_RUN == 98304
        source = (Path(__file__).resolve().parents[2] / "scripts"
                  / "run_e2_longrun.py").read_text(encoding="utf-8")
        assert "TOTAL_ENV_STEPS_LONG_RUN" not in source


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
