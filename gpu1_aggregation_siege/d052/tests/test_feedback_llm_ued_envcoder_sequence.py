"""P0-6 (§19 seam coverage): exact sequence accounting for the real
EnvCoder chain.

The global call sequence must be STRICTLY monotonic: a real EnvCoder
window consumes EXACTLY ``n_calls`` sequence slots (the unique-template
call plus every bounded repair re-call) — never one. Under-counting a
repair sequence would silently re-use sequence numbers, so a missing or
zero ``n_calls`` fails closed, and consumption happens even when the
artifact later fails the PASSED check (the calls were actually made).
Snapshot/restore must continue from the exact persisted sequence head.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: the injected
real-envcoder callable is scripted and returns pre-built PASSED artifacts
— NO real LLM call happens, and NO passing test flips a REAL_* flag.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import persistence as P
from d052.feedback_llm_ued import (
    behavior_auditor,
    causal_failure_analyst,
    critic_skeptic,
    explorer,
    intervention_tutor,
    student_modeler,
)
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.llm_backend import RealBackendAdapter
from d052.feedback_llm_ued.real_call_journal import RealTransportResult
from d052.feedback_llm_ued.real_env_coder import (
    STATUS_FAILED,
    STATUS_PASSED,
    RealEnvCoderArtifact,
)
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
)

TEST_BACKEND_ID = "TEST_ONLY_BACKEND"
TEST_MODEL_ID = "TEST_ONLY_MODEL"
BOARD_CALLS = C.BOARD_CALLS_PER_WINDOW          # 6

ROLE_MODULES = {
    student_modeler.ROLE: student_modeler,
    behavior_auditor.ROLE: behavior_auditor,
    causal_failure_analyst.ROLE: causal_failure_analyst,
    intervention_tutor.ROLE: intervention_tutor,
    explorer.ROLE: explorer,
    critic_skeptic.ROLE: critic_skeptic,
}


def scripted_board_transport():
    """TEST_ONLY / SYNTHETIC real transport: deterministic mock-rule text
    shaped like a provider response — NO real LLM is ever called."""
    def transport(role, prompt):
        context = extract_context(prompt)
        module = ROLE_MODULES[role]
        raw = json.dumps(module.mock_rule(context), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False,
                         default=str)
        return RealTransportResult(raw=raw,
                                   request_id=f"test-req-{role}")
    return transport


def passed_artifact(window: int, plan_id: str, *, n_calls: int,
                    overall_status: str = STATUS_PASSED
                    ) -> RealEnvCoderArtifact:
    return RealEnvCoderArtifact(
        window=window, plan_id=plan_id,
        spec_hash=text_sha256(f"spec-w{window}-{plan_id}"),
        backend_id=TEST_BACKEND_ID, model_id=TEST_MODEL_ID,
        n_calls=n_calls, repair_attempts=n_calls - 1,
        logical_call_ids=[f"lcid-w{window}-{i}" for i in range(n_calls)],
        envelope_request_hashes=["ab" * 32] * n_calls,
        directive_artifacts=[], overall_status=overall_status, blockers=[])


def scripted_real_env_coder(n_calls_by_window):
    """TEST_ONLY stand-in for execute_real_env_coder: returns a PASSED
    artifact with the scripted n_calls and records the sequence it saw."""
    calls = []

    def _callable(*, window, plan_id, directives, sequence):
        calls.append(dict(window=window, plan_id=plan_id,
                          n_directives=len(list(directives)),
                          sequence=sequence))
        return passed_artifact(window, plan_id,
                               n_calls=n_calls_by_window.get(window, 1))

    _callable.calls = calls
    return _callable


def make_controller(env_coder) -> FeedbackUEDController:
    #: real LLM + real envcoder grants (the grant set refuses
    #: real_envcoder without real_llm_backend — the EnvCoder IS an LLM
    #: call); the board runs on a scripted REAL-kind backend, the probe
    #: stays symbolic (no real_probe grant) and training stays
    #: unauthorized: this isolates the envcoder sequence seam
    authorization = RealRuntimeAuthorization(real_llm_backend=True,
                                             real_envcoder=True)
    backend = RealBackendAdapter(scripted_board_transport(),
                                 backend_id=TEST_BACKEND_ID,
                                 model_id=TEST_MODEL_ID,
                                 authorized=True)
    return FeedbackUEDController(
        C.MODE_NORMAL_FEEDBACK, backend=backend,
        runtime_authorization=authorization,
        real_env_coder_callable=env_coder)


class TestExactSequenceAccounting:
    def test_repair_sequences_consumed_exactly(self):
        env_coder = scripted_real_env_coder({0: 3, 1: 1})
        controller = make_controller(env_coder)
        summary = controller.run(max_windows=2)

        #: window 0: board consumes 0..5, envcoder starts at 6 and
        #: consumes 6,7,8 -> window 1 board starts at 9
        assert [c["sequence"] for c in env_coder.calls] == [6, 15]
        envelope_sequences = [e.sequence for e in controller.envelopes]
        assert envelope_sequences == [0, 1, 2, 3, 4, 5,
                                      9, 10, 11, 12, 13, 14]
        #: strict monotonicity — no sequence number is ever re-used
        assert all(a < b for a, b in zip(envelope_sequences,
                                         envelope_sequences[1:]))
        assert controller._sequence == 16
        assert [w["env_coder_call_count"] for w in summary.windows] == [3, 1]
        assert summary.n_windows == 2

    def test_single_call_matches_symbolic_accounting(self):
        #: with n_calls=1 the real chain consumes exactly what the symbolic
        #: coder consumes — the historical accounting, unchanged
        env_coder = scripted_real_env_coder({})
        controller = make_controller(env_coder)
        controller.run(max_windows=2)
        assert [c["sequence"] for c in env_coder.calls] == [6, 13]
        assert controller._sequence == 14

        symbolic = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        symbolic.run(max_windows=2)
        envcoder_sequences = [e.sequence for e in symbolic.envelopes
                              if e.role == C.ROLE_ENV_CODER]
        assert envcoder_sequences == [6, 13]
        assert symbolic._sequence == controller._sequence

    def test_missing_call_count_fails_closed(self):
        def no_count(*, window, plan_id, directives, sequence):
            #: an artifact shape WITHOUT n_calls — accounting must refuse
            return SimpleNamespace(overall_status=STATUS_PASSED,
                                   directive_artifacts=[])

        controller = make_controller(no_count)
        with pytest.raises(RuntimeError,
                           match="REAL_ENVCODER_CALL_COUNT_MISSING"):
            controller.run(max_windows=2)

    def test_failed_artifact_still_consumed_its_sequences(self):
        #: the calls were MADE even though the artifact is not PASSED —
        #: the sequence consumption precedes the PASSED check
        def failing(*, window, plan_id, directives, sequence):
            return passed_artifact(window, plan_id, n_calls=2,
                                   overall_status=STATUS_FAILED)

        controller = make_controller(failing)
        with pytest.raises(RuntimeError, match="REAL_ENVCODER_NOT_PASSED"):
            controller.run(max_windows=2)
        assert controller._sequence == BOARD_CALLS + 2


class TestSnapshotRestoreContinuity:
    def test_restore_continues_from_exact_sequence_head(self, tmp_path):
        env_coder = scripted_real_env_coder({0: 3, 1: 2})
        controller = make_controller(env_coder)
        controller.run(max_windows=1)
        assert controller._sequence == BOARD_CALLS + 3

        path = str(tmp_path / "ctl_sequence.json")
        P.save_controller(controller, path)
        #: the launcher re-injects the (non-serializable) real backend and
        #: envcoder seam; the runtime grants come back from the snapshot
        restored = P.load_controller(
            path, backend=RealBackendAdapter(scripted_board_transport(),
                                             backend_id=TEST_BACKEND_ID,
                                             model_id=TEST_MODEL_ID,
                                             authorized=True))
        assert restored._sequence == controller._sequence == 9
        assert restored.runtime_authorization.real_envcoder is True
        assert restored.launch_decision.real_llm_calls_allowed is True

        restored._real_env_coder_callable = env_coder
        summary = restored.run(max_windows=2)
        assert [c["sequence"] for c in env_coder.calls] == [6, 15]
        #: 6 (board0) + 3 (envcoder0) + 6 (board1) + 2 (envcoder1)
        assert restored._sequence == 17
        assert [w["env_coder_call_count"] for w in summary.windows] == [3, 2]

        #: same accounting as the uninterrupted run
        uninterrupted = make_controller(
            scripted_real_env_coder({0: 3, 1: 2}))
        u_summary = uninterrupted.run(max_windows=2)
        assert uninterrupted._sequence == restored._sequence
        assert ([w["env_coder_call_count"] for w in u_summary.windows]
                == [w["env_coder_call_count"] for w in summary.windows])

    def test_all_false_grants_restore_historical_gate(self, tmp_path):
        #: a mock-dry-run snapshot carries the all-false grant set and
        #: restores with the historical constants-only gate (unchanged)
        controller = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        controller.run(max_windows=1)
        path = str(tmp_path / "ctl_mock.json")
        P.save_controller(controller, path)
        restored = P.load_controller(path)
        assert restored.launch_decision.real_llm_calls_allowed is False
        assert restored.runtime_authorization.real_envcoder is False
        restored.run(max_windows=2)


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
