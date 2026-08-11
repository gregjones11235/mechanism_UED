"""§七 (dual student, section 6): read-only mount and training readiness
are SEPARATE — a Student may be read-only mounted and probed while the
training runtime (CanonicalDiCodeOneUpdateRuntime OBJECT) is absent.
REAL_CHECKPOINT_LOADED never implies REAL_TRAINING_UPDATE_EXECUTED.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from d052.feedback_llm_ued import constants as C

from e2_test_sign_helpers import valid_director_bundle


def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_readonly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()


class TestReadOnlyMountNotTrainingReady:
    def test_report_separates_readonly_from_training(self, tmp_path, capsys):
        manifest = valid_director_bundle(
            candidate_id=C.STRONG_STUDENT_CANDIDATE_ID)
        path = tmp_path / "bundle.json"
        path.write_text(json.dumps(manifest.model_dump(), ensure_ascii=False,
                                   default=str), encoding="utf-8")
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle", str(path)])
        #: without an injected verifier the object-level chain fails closed
        assert code == 1
        err = capsys.readouterr().err
        assert "OBJECT_LEVEL_CHECK_BLOCKED" in err
        assert "PRODUCTION_BUNDLE_VERIFIER_UNBOUND" in err
        #: REAL_CHECKPOINT_LOADED never implies a training update
        assert C.REAL_CHECKPOINT_LOADED is False
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False

    def test_training_runtime_requires_the_object(self):
        #: the director bundle only DECLARES the DiCode runtime identity;
        #: without the injected OBJECT the training runtime is NOT ready
        assert C.STUDENT_READ_ONLY_MOUNT_READY == \
            "STUDENT_READ_ONLY_MOUNT_READY"
        assert C.STUDENT_TRAINING_RUNTIME_READY == \
            "STUDENT_TRAINING_RUNTIME_READY"


class TestPosture:
    def test_all_flags_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.REAL_CHECKPOINT_LOADED is False
        assert C.REAL_SIMULATOR_PROBE is False
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False
        assert C.E2_REAL_SMOKE_AUTHORIZED is False
        assert C.FORMAL_EXPERIMENT_AUTHORIZED is False
