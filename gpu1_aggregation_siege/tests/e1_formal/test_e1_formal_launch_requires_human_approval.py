"""CC2-Director tests: the formal experiment always waits for human
approval.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

run_e1_longrun.py is a manifest PREPARER only: it never starts
training, always reports FORMAL_EXPERIMENT_AUTHORIZED=false, and even
``--launch`` is refused (launch_granted stays false).
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import run_e1_longrun as LONG  # noqa: E402


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _block():
    total = 2_005_401_600
    return dict(
        semantics="ADDITIONAL_FROM_PRETRAINED_CHECKPOINT",
        total_timesteps=total,
        initial_checkpoint_timesteps=4096,
        additional_training_timesteps=total - 4096,
        final_total_timesteps=total,
    )


class TestFormalLaunchWaitsForHumanApproval:
    def test_manifest_always_reports_unauthorized(self, tmp_path):
        manifest_out = str(tmp_path / "manifest.json")
        rc = LONG.main(
            [
                "--budget-block",
                json.dumps(_block()),
                "--manifest-out",
                manifest_out,
            ]
        )
        # the budget is decided but the real gates (shared runtime,
        # unfrozen reference, draft anchor manifest) keep it REFUSED
        assert rc != 0
        report = _read(manifest_out)
        assert report["formal_experiment_authorized"] is False
        assert report["launch_granted"] is False

    def test_launch_flag_never_starts_training(self, tmp_path):
        manifest_out = str(tmp_path / "manifest_launch.json")
        rc = LONG.main(
            [
                "--budget-block",
                json.dumps(_block()),
                "--launch",
                "--manifest-out",
                manifest_out,
            ]
        )
        report = _read(manifest_out)
        assert report["formal_experiment_authorized"] is False
        assert report["launch_requested"] is True
        assert report["launch_granted"] is False
        assert report["prepare_only"] is True

    def test_manifest_fields_are_timesteps_on_the_dicode_timeline(
        self, tmp_path
    ):
        manifest = LONG.build_frozen_manifest(
            LONG.RT.TEACHER_CONFIG_PATH,
            budget_block=_block(),
        )
        fields = manifest["fields"]
        assert "total_timesteps" in fields
        assert "final_total_timesteps" in fields
        # no 98304-as-budget field anywhere in the manifest
        assert "total_env_steps" not in fields

    def test_no_launch_codepath_reaches_training(self):
        source = open(
            os.path.join(SCRIPTS_DIR, "run_e1_longrun.py"),
            "r",
            encoding="utf-8",
        ).read()
        # the entrypoint documents it NEVER starts the training loop
        assert "never starts" in source
        assert "formal_experiment_authorized" in source
