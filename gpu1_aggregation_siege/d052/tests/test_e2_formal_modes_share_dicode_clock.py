"""§八 (director smoke handoff): the three E2 modes SHARE the frozen DiCode
config and the SAME training clock — E2 mode changes ONLY the Feedback
View, never the training clock.

Contract under test:

* normal / no-feedback / shuffled launch configs carry byte-identical
  dicode_config (total_timesteps, clock field, manager section) and
  identical auxiliary per-window budgets;
* only ``feedback_view_label`` differs (normal / masked / permuted);
* ``assert_modes_share_dicode_clock`` returns [] and the report says
  ``modes_share_dicode_clock`` true.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from d052.feedback_llm_ued import constants as C


def _load_longrun():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_longrun.py")
    spec = importlib.util.spec_from_file_location("run_e2_longrun_e2b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LONGRUN = _load_longrun()


def _configs():
    dicode = LONGRUN.load_dicode_resolved_config()
    return [LONGRUN.dicode_launch_config(name, dicode)
            for name in sorted(LONGRUN.MODE_TO_LOOP_MODE)]


class TestModesShareDicodeClock:
    def test_identical_dicode_config_across_modes(self):
        configs = _configs()
        dumps = {hashlib.sha256(
            json.dumps(cfg["dicode_config"], sort_keys=True).encode())
            .hexdigest() for cfg in configs}
        assert len(dumps) == 1
        total = {cfg["dicode_config"]["training"]["total_timesteps"]
                 for cfg in configs}
        assert len(total) == 1 and next(iter(total)) > 0

    def test_identical_auxiliary_budgets_across_modes(self):
        configs = _configs()
        assert len({cfg["llm_calls_per_window"] for cfg in configs}) == 1
        assert len({cfg["probe_transitions_per_window"]
                    for cfg in configs}) == 1

    def test_only_feedback_view_differs(self):
        configs = _configs()
        views = {cfg["feedback_view_label"] for cfg in configs}
        assert views == {"normal", "masked", "permuted"}
        for cfg in configs:
            loop = cfg["loop_mode"]
            if cfg["feedback_view_label"] == "normal":
                assert loop == C.MODE_NORMAL_FEEDBACK
            elif cfg["feedback_view_label"] == "masked":
                assert loop == C.MODE_STATIC_LLM
            else:
                assert loop == C.MODE_SHUFFLED_FEEDBACK

    def test_assert_modes_share_clock_passes(self):
        assert LONGRUN.assert_modes_share_dicode_clock(_configs()) == []

    def test_report_says_modes_share_clock(self, capsys):
        code = LONGRUN.main(["--mode", "normal_feedback", "--check-only"])
        assert code == 0
        out = capsys.readouterr().out
        assert '"modes_share_dicode_clock": true' in out


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
