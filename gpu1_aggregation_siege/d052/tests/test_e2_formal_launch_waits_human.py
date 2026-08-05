"""§八 (director smoke handoff): the FORMAL launch waits for a HUMAN-
approved Formal Manifest — direction two never auto-starts a formal run.

Contract under test:

* a bare launch attempt is refused (exit 1) with
  FORMAL_EXPERIMENT_AUTHORIZED=false;
* ``--formal-manifest-only`` PREPARES the Formal Manifest preview (exit
  0, status PREPARED_ONLY_NOT_AUTHORIZED) and NEVER launches;
* check-only validates the DiCode clock consumption without launching.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from d052.feedback_llm_ued import constants as C


def _load_longrun():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_longrun.py")
    spec = importlib.util.spec_from_file_location("run_e2_longrun_e2c", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LONGRUN = _load_longrun()


class TestFormalLaunchWaitsHuman:
    def test_launch_refused_formal_not_authorized(self, capsys):
        code = LONGRUN.main(["--mode", "normal_feedback"])
        assert code == 1
        err = capsys.readouterr().err
        assert "FORMAL_EXPERIMENT_AUTHORIZED=false" in err

    def test_formal_manifest_only_prepares_and_never_launches(self, tmp_path,
                                                              capsys):
        out_path = tmp_path / "formal_manifest.json"
        code = LONGRUN.main(["--mode", "normal_feedback",
                             "--formal-manifest-only",
                             "--manifest-out", str(out_path)])
        assert code == 0
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
        assert manifest["kind"] == "DICODE_FORMAL_MANIFEST_PREVIEW"
        assert manifest["status"] == "PREPARED_ONLY_NOT_AUTHORIZED"
        assert manifest["formal_experiment_authorized"] is False
        err = capsys.readouterr().err
        assert "FORMAL MANIFEST PREPARED" in err

    def test_check_only_validates_clock_without_launch(self, capsys):
        code = LONGRUN.main(["--mode", "normal_feedback", "--check-only"])
        assert code == 0
        out = capsys.readouterr().out
        report, _ = json.JSONDecoder().raw_decode(out)
        assert report["legacy_98304_budget_removed"] is True
        assert report["formal_experiment_authorized"] is False
        assert report["modes_share_dicode_clock"] is True

    def test_report_out_writes_the_report(self, tmp_path, capsys):
        out_path = tmp_path / "report_out.json"
        LONGRUN.main(["--mode", "no_feedback_control", "--check-only",
                      "--report-out", str(out_path)])
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["feedback_view_label"] == "masked"


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.FORMAL_EXPERIMENT_AUTHORIZED is False
        assert C.E2_REAL_SMOKE_AUTHORIZED is False
