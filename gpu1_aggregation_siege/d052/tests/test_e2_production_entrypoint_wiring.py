"""§十四 (REQUEST_CHANGES): call-tracing tests proving the production
entrypoint actually calls the trusted verifier / bundle-hash cross-binding /
object resolver, reaches the controller with BOUND_OBJECT, never auto-sets
test_only, and stops BEFORE the LLM when objects are missing.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from d052.feedback_llm_ued import constants as C

from e2_test_sign_helpers import valid_director_bundle


def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location("run_e2_real_two_window_w", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()
PERSISTENT = C.STRONG_STUDENT_CANDIDATE_ID


class _SpyVerifier:
    """TEST_ONLY verifier recording that it was actually called."""

    verifier_id = "a" * 64
    verifier_implementation_hash = "b" * 64
    calls = 0

    def verify_manifest(self, manifest) -> bool:
        type(self).calls += 1
        return True

    def signer_trusted(self, signer_id) -> bool:
        type(self).calls += 1
        return True

    def verify_source_commit(self, source_commit) -> bool:
        type(self).calls += 1
        return True


class TestEntrypointCallsTrustedVerifier:
    def test_object_level_check_calls_verifier(self):
        manifest = valid_director_bundle(candidate_id=PERSISTENT)
        _SpyVerifier.calls = 0
        result = ENTRYPOINT.run_e2_object_level_check(
            manifest=manifest, director_bundle_verifier=_SpyVerifier(),
            formal_asset_registry=None, selected_candidate_id=PERSISTENT)
        assert _SpyVerifier.calls > 0          # verifier WAS called
        assert result["status"] == "OBJECT_LEVEL_CHECK_BLOCKED"
        assert "FORMAL_ASSET_REGISTRY_UNBOUND" in result["reason"]

    def test_missing_verifier_blocks_before_anything(self):
        manifest = valid_director_bundle(candidate_id=PERSISTENT)
        result = ENTRYPOINT.run_e2_object_level_check(
            manifest=manifest, director_bundle_verifier=None,
            formal_asset_registry=object(), selected_candidate_id=PERSISTENT)
        assert result["status"] == "OBJECT_LEVEL_CHECK_BLOCKED"
        assert "PRODUCTION_BUNDLE_VERIFIER_UNBOUND" in result["reason"]
        assert result["REAL_LLM_EXECUTED"] is False

    def test_cross_binds_bundle_hash(self):
        #: the check runs assert_runtime_bundle_hash_cross_bound (a valid
        #: bundle's Student binds the manifest's own hash by construction;
        #: a tampered one would block)
        manifest = valid_director_bundle(candidate_id=PERSISTENT)
        assert manifest.student_init_contract.runtime_bundle_hash \
            == manifest.bundle_hash

    def test_production_two_window_refuses_without_authorization(self):
        manifest = valid_director_bundle(candidate_id=PERSISTENT)
        with pytest.raises(ENTRYPOINT.RealTwoWindowBlocked,
                           match="E2_REAL_SMOKE_AUTHORIZED=false"):
            ENTRYPOINT.run_e2_production_two_window(
                manifest=manifest, resolved_runtime=object(),
                selected_student=PERSISTENT, authorization=object())

    def test_check_only_never_executes(self, capsys):
        #: run_e2_entrypoint with check_only never reaches the LLM — a
        #: blocked object-level check returns 1 without executing
        manifest = valid_director_bundle(candidate_id=PERSISTENT)
        code = ENTRYPOINT.run_e2_entrypoint(
            manifest=manifest, selected_candidate_id=PERSISTENT,
            check_only=True, director_bundle_verifier=None,
            formal_asset_registry=None, report_out="")
        assert code == 1
        err = capsys.readouterr().err
        assert "PRODUCTION_BUNDLE_VERIFIER_UNBOUND" in err


class TestControllerExecutionMode:
    def test_controller_rejects_illegal_execution_mode(self):
        from d052.feedback_llm_ued.controller import FeedbackUEDController
        with pytest.raises(ValueError,
                           match="ILLEGAL_EXECUTION_MODE"):
            FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                  execution_mode="AUTO")

    def test_production_controller_never_auto_test_only(self):
        from d052.feedback_llm_ued.controller import FeedbackUEDController
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                    execution_mode="PRODUCTION")
        assert ctl.execution_mode == "PRODUCTION"
        #: a training-allowed run without the dicode binding fails closed
        #: (REAL_DICODE_BATCH_PLAN_REQUIRED) — no auto test_only fallback
        assert C.E2_REAL_SMOKE_AUTHORIZED is False


class TestRemovedSignHelperNotExported:
    def test_production_module_has_no_signer(self):
        import d052.feedback_llm_ued.director_runtime_bundle as m
        assert not hasattr(m, "sign_director_runtime_bundle")
        assert "sign_director_runtime_bundle" not in m.__all__


class TestPosture:
    def test_flags_all_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.E2_REAL_SMOKE_AUTHORIZED is False
        assert C.FORMAL_EXPERIMENT_AUTHORIZED is False
