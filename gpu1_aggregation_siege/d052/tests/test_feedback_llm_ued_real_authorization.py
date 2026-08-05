"""§19 seam coverage: the runtime authorization grant channel.

Contract under test:

* the grants are strictly LAYERED — any inconsistent grant set
  (real_envcoder without real_llm_backend; real_probe without both
  lower; real_training without all three lower) is refused at
  construction (INCONSISTENT_RUNTIME_GRANTS);
* ``assert_real_mode_servicable``: a real LLM grant without an injected
  transport fails closed (REAL_MODE_BLOCKED_NO_LLM_BACKEND), any grant
  with missing shared assets fails closed (BLOCKED_WAITING_SHARED_
  RUNTIME), no grant / everything present is a no-op;
* the controller refuses a real-authorized run on a mock/replay backend
  (REAL_RUN_BACKEND_NOT_REAL — mock-impersonating-real is rejected);
* the grant set survives the controller snapshot -> restore round-trip
  (P0-6: an all-false set restores the historical constants-only gate,
  a granted set restores EXECUTION_MODE_REAL);
* the round constants stay False throughout (no passing test flips a
  REAL_* flag).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import persistence as P
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.llm_backend import (
    DeterministicMockFeedbackBackend,
    RealBackendAdapter,
)
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
    RuntimeAuthorizationBlocked,
    assert_real_mode_servicable,
    empty_authorization,
)


def _scripted_transport(role, prompt):
    from d052.feedback_llm_ued.feedback_contracts import extract_context
    from test_feedback_llm_ued_envcoder_sequence import ROLE_MODULES
    import json
    context = extract_context(prompt)
    module = ROLE_MODULES[role]
    raw = json.dumps(module.mock_rule(context), sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False, default=str)
    return __import__(
        "d052.feedback_llm_ued.real_call_journal",
        fromlist=["RealTransportResult"]).RealTransportResult(
        raw=raw, request_id=f"test-req-{role}")


class TestGrantLadder:
    def test_all_grants_default_false(self):
        auth = empty_authorization()
        assert auth.any_grant() is False
        assert RealRuntimeAuthorization().any_grant() is False

    def test_full_ladder_is_consistent(self):
        auth = RealRuntimeAuthorization(real_llm_backend=True,
                                        real_envcoder=True,
                                        real_probe=True,
                                        real_training=True)
        assert auth.any_grant() is True
        assert "real_training=True" in auth.describe()

    @pytest.mark.parametrize("kwargs", [
        dict(real_envcoder=True),
        dict(real_probe=True),
        dict(real_probe=True, real_llm_backend=True),
        dict(real_training=True),
        dict(real_training=True, real_llm_backend=True,
             real_envcoder=True),
    ])
    def test_inconsistent_grant_sets_refused(self, kwargs):
        with pytest.raises(RuntimeAuthorizationBlocked,
                           match="INCONSISTENT_RUNTIME_GRANTS"):
            RealRuntimeAuthorization(**kwargs)


class TestServicableCheck:
    def test_real_llm_without_transport_fails_closed(self):
        auth = RealRuntimeAuthorization(real_llm_backend=True)
        with pytest.raises(
                RuntimeAuthorizationBlocked,
                match=C.REAL_MODE_BLOCKED_NO_LLM_BACKEND):
            assert_real_mode_servicable(authorization=auth,
                                        llm_transport=None)

    def test_any_grant_with_missing_assets_fails_closed(self):
        auth = RealRuntimeAuthorization(real_llm_backend=True)
        with pytest.raises(
                RuntimeAuthorizationBlocked,
                match=C.BLOCKED_WAITING_SHARED_RUNTIME):
            assert_real_mode_servicable(
                authorization=auth, llm_transport=object(),
                missing_assets=["shared StudentAdapter",
                                "shared ReferenceAdapter"])

    def test_no_grants_is_a_noop(self):
        assert_real_mode_servicable(
            authorization=empty_authorization(), llm_transport=None,
            missing_assets=["anything"])

    def test_complete_is_a_noop(self):
        auth = RealRuntimeAuthorization(real_llm_backend=True,
                                        real_envcoder=True,
                                        real_probe=True,
                                        real_training=True)
        assert_real_mode_servicable(
            authorization=auth, llm_transport=object(),
            missing_assets=())


class TestControllerPosture:
    def test_real_auth_without_real_backend_refused(self):
        #: mock-impersonating-real: real LLM calls are authorized but the
        #: backend is the deterministic MOCK — refused fail-closed
        auth = RealRuntimeAuthorization(real_llm_backend=True)
        with pytest.raises(RuntimeError, match="REAL_RUN_BACKEND_NOT_REAL"):
            FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                  backend=DeterministicMockFeedbackBackend(),
                                  runtime_authorization=auth)

    def test_full_grants_with_real_backend_constructs(self):
        auth = RealRuntimeAuthorization(real_llm_backend=True,
                                        real_envcoder=True,
                                        real_probe=True,
                                        real_training=True)
        backend = RealBackendAdapter(_scripted_transport,
                                     backend_id="test.backend.v1",
                                     model_id="test-model.v1",
                                     authorized=True)
        from test_feedback_llm_ued_two_window_update_count import (
            ScriptedRealProbeRunner,
        )
        from types import SimpleNamespace
        ctl = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK, backend=backend,
            probe_runner=ScriptedRealProbeRunner(),
            runtime_authorization=auth,
            student_init_contract=SimpleNamespace(
                candidate_id=C.STRONG_STUDENT_CANDIDATE_ID,
                architecture_family="RMT16",
                memory_family="RMT16_ORIGINAL", carry_mode="PERSISTENT",
                parameter_tree_hash="ab" * 32, checkpoint_global_step=98304),
            training_contract=SimpleNamespace(
                run_one_optimizer_update=lambda **kw: None,
                save_checkpoint=lambda **kw: "hash",
                load_checkpoint=lambda **kw: None,
                verify_director_round_trip=lambda **kw: None),
            real_env_coder_callable=(
                lambda **kw: __import__(
                    "test_feedback_llm_ued_envcoder_sequence",
                    fromlist=["passed_artifact"]).passed_artifact(
                    kw["window"], kw["plan_id"], n_calls=1)))
        assert ctl.launch_decision.real_llm_calls_allowed is True
        assert ctl.launch_decision.training_allowed is True


class TestSnapshotRestoreOfGrants:
    def test_granted_set_restores_real_authorization(self, tmp_path):
        #: the persistence API re-injects the backend seam only — the
        #: restorable grant boundary is real_llm_backend (probe/training
        #: grants additionally require the probe runner + student binding +
        #: training contract, which the launcher re-injects; the
        #: persistence layer documents that re-injection contract)
        auth = RealRuntimeAuthorization(real_llm_backend=True)
        backend = RealBackendAdapter(_scripted_transport,
                                     backend_id="test.backend.v1",
                                     model_id="test-model.v1",
                                     authorized=True)
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK, backend=backend,
                                    runtime_authorization=auth)
        path = str(tmp_path / "ctl_grants.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(
            path, backend=RealBackendAdapter(_scripted_transport,
                                             backend_id="test.backend.v1",
                                             model_id="test-model.v1",
                                             authorized=True))
        assert restored.runtime_authorization.real_llm_backend is True
        assert restored.launch_decision.real_llm_calls_allowed is True
        assert restored.runtime_authorization.real_training is False

    def test_all_false_grants_restore_historical_gate(self, tmp_path):
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        path = str(tmp_path / "ctl_mock.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(path)
        assert restored.launch_decision.real_llm_calls_allowed is False
        assert restored.runtime_authorization.any_grant() is False


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
