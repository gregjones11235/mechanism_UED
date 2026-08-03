"""P0-1 backend abstraction: UsageStats, launch gate, replay/real seams.

Fake-real integration tests only — no real API is contacted, no frozen replay
corpus is committed: the replay corpus is recorded inside each test by a
RecordingBackend wrapped around the deterministic mock.
"""
import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_MOCK_DRY_RUN,
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
    LaunchGateBlocked,
)
from d052.feedback_llm_ued.llm_backend import (
    BackendBlocked,
    BackendCallFailed,
    DeterministicMockFeedbackBackend,
    LLMBackend,
    RealBackendAdapter,
    RecordingBackend,
    ReplayBackend,
    ReplayMiss,
    UsageStats,
    assert_no_real_llm_usage,
)


class TestUsageStats:
    def test_total_counts_served_kinds_only(self):
        u = UsageStats(real_calls=1, replay_calls=2, mock_calls=3,
                       failed_calls=5)
        assert u.total_calls == 6          # failed calls are NOT served calls
        with pytest.raises(AssertionError, match="REAL_LLM_CALLS_FORBIDDEN"):
            u.assert_no_real()
        UsageStats(mock_calls=4).assert_no_real()     # no real -> passes
        assert_no_real_llm_usage(UsageStats(replay_calls=2))

    def test_snapshot_is_a_copy(self):
        u = UsageStats(mock_calls=2)
        snap = u.snapshot()
        u.mock_calls += 3
        assert snap.mock_calls == 2 and u.mock_calls == 5


class TestLaunchGate:
    def test_mock_dry_run_allows_mock_and_replay_only(self):
        gate = FeedbackLaunchGate(EXECUTION_MODE_MOCK_DRY_RUN)
        d = gate.evaluate()
        assert d.backend_kinds_allowed == (C.BACKEND_KIND_MOCK,
                                           C.BACKEND_KIND_REPLAY)
        assert d.real_llm_calls_allowed is False
        assert d.real_simulator_probe_allowed is False
        assert d.training_allowed is False
        assert d.final_formal_run_allowed is False
        gate.assert_backend_allowed(C.BACKEND_KIND_MOCK)
        gate.assert_backend_allowed(C.BACKEND_KIND_REPLAY)
        with pytest.raises(LaunchGateBlocked, match="BACKEND_KIND_NOT_ALLOWED"):
            gate.assert_backend_allowed(C.BACKEND_KIND_REAL)

    def test_real_mode_still_blocked_by_round_flags(self, monkeypatch):
        gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
        d = gate.evaluate()
        assert d.real_llm_calls_allowed is False       # flag False this round
        with pytest.raises(LaunchGateBlocked):
            gate.assert_backend_allowed(C.BACKEND_KIND_REAL)
        # flag flipped (director decision) -> real backend becomes allowed
        monkeypatch.setattr(C, "REAL_LLM_CALLS_AUTHORIZED", True)
        assert gate.evaluate().real_llm_calls_allowed is True
        gate.assert_backend_allowed(C.BACKEND_KIND_REAL)

    def test_capability_asserts_fail_closed(self):
        gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
        with pytest.raises(LaunchGateBlocked,
                           match="REAL_SIMULATOR_PROBE_NOT_ALLOWED"):
            gate.assert_real_probe_allowed()
        with pytest.raises(LaunchGateBlocked, match="TRAINING_NOT_ALLOWED"):
            gate.assert_training_allowed()
        with pytest.raises(LaunchGateBlocked,
                           match="FINAL_FORMAL_RUN_NOT_ALLOWED"):
            gate.assert_final_formal_run_allowed()

    def test_unknown_mode_and_kind_rejected(self):
        with pytest.raises(ValueError, match="UNKNOWN_EXECUTION_MODE"):
            FeedbackLaunchGate("self_training")
        with pytest.raises(ValueError, match="UNKNOWN_BACKEND_KIND"):
            FeedbackLaunchGate().assert_backend_allowed("carrier_pigeon")


class TestMockBackendUsage:
    def test_protocol_conformance_and_kinds(self):
        mock = DeterministicMockFeedbackBackend()
        replay = ReplayBackend({})
        assert isinstance(mock, LLMBackend)
        assert isinstance(replay, LLMBackend)
        assert mock.kind == C.BACKEND_KIND_MOCK
        assert replay.kind == C.BACKEND_KIND_REPLAY

    def test_mock_usage_accounting(self):
        backend = DeterministicMockFeedbackBackend()
        ctx = dict(window=1, hypotheses=[], feedback=[])
        from d052.feedback_llm_ued import feedback_diagnostician
        prompt = feedback_diagnostician.build_prompt(ctx)
        backend.complete(C.ROLE_FEEDBACK_DIAGNOSTICIAN, prompt)
        assert backend.usage.mock_calls == 1
        assert backend.usage.real_calls == 0
        assert backend.usage.total_calls == 1
        assert_no_real_llm_usage(backend.usage)


class TestReplayBackend:
    def test_miss_fails_closed(self):
        backend = ReplayBackend({})
        with pytest.raises(ReplayMiss, match="REPLAY_MISS"):
            backend.complete("any_role", "any prompt")
        assert backend.usage.total_calls == 0

    def test_hit_serves_recorded_raw_and_counts_replay(self):
        corpus = {("r", "sha"): "RAW-RESPONSE"}
        backend = ReplayBackend(corpus)
        from d052.bagr_ued.hashing import text_sha256
        backend._corpus[("r", text_sha256("p"))] = "RAW-2"
        assert backend.complete("r", "p") == "RAW-2"
        assert backend.usage.replay_calls == 1
        assert backend.usage.total_calls == 1

    def test_record_then_replay_equivalence_same_test(self):
        """Record a mock run in-test, replay it, get byte-identical prompts."""
        from d052.feedback_llm_ued import feedback_diagnostician
        mock = DeterministicMockFeedbackBackend()
        recorder = RecordingBackend(mock)
        ctx = dict(window=2, hypotheses=[], feedback=[])
        prompt = feedback_diagnostician.build_prompt(ctx)
        raw1 = recorder.complete(C.ROLE_FEEDBACK_DIAGNOSTICIAN, prompt)
        replay = ReplayBackend(recorder.to_replay_corpus())
        raw2 = replay.complete(C.ROLE_FEEDBACK_DIAGNOSTICIAN, prompt)
        assert raw1 == raw2
        assert replay.usage.replay_calls == 1
        assert isinstance(recorder, LLMBackend)
        assert recorder.kind == C.BACKEND_KIND_MOCK      # delegates inward


class TestRealBackendAdapterFakeReal:
    """Fake-real integration: a scripted transport stands in for the API."""

    def _make(self, transport, max_retries=2):
        return RealBackendAdapter(transport, backend_id="fake.real.v1",
                                  model_id="fake-model.v1",
                                  authorized=True, max_retries=max_retries)

    def test_unauthorized_construction_fails_closed(self):
        with pytest.raises(BackendBlocked, match="REAL_LLM_BACKEND_BLOCKED"):
            RealBackendAdapter(lambda r, p: "x", backend_id="b", model_id="m",
                               authorized=C.REAL_LLM_CALLS_AUTHORIZED)

    def test_success_counts_real_calls(self):
        backend = self._make(lambda role, prompt: f"resp:{role}")
        assert backend.complete("r", "p") == "resp:r"
        assert backend.usage.real_calls == 1
        assert backend.usage.failed_calls == 0

    def test_retry_then_success(self):
        calls = []

        def flaky(role, prompt):
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("transient")
            return "ok"

        backend = self._make(flaky, max_retries=2)
        assert backend.complete("r", "p") == "ok"
        assert backend.usage.real_calls == 1
        assert backend.usage.failed_calls == 2

    def test_exhausted_retries_raise_and_count_failures(self):
        def always_fail(role, prompt):
            raise ConnectionError("down")

        backend = self._make(always_fail, max_retries=1)
        with pytest.raises(BackendCallFailed, match="REAL_LLM_CALL_FAILED"):
            backend.complete("r", "p")
        assert backend.usage.real_calls == 0
        assert backend.usage.failed_calls == 2      # 1 + max_retries attempts

    def test_empty_response_counts_as_failure(self):
        backend = self._make(lambda role, prompt: "", max_retries=0)
        with pytest.raises(BackendCallFailed):
            backend.complete("r", "p")
        assert backend.usage.failed_calls == 1

    def test_credentials_never_stored(self):
        secret = "sk-live-DO-NOT-STORE"

        def transport(role, prompt):
            _ = secret                       # lives ONLY in this closure
            return "ok"

        backend = self._make(transport)
        backend.complete("r", "p")
        dumped = json.dumps(backend.usage.snapshot().__dict__, default=str)
        assert secret not in dumped
        assert not hasattr(backend, "_transport_credentials")


class TestControllerReplayEquivalence:
    """A recorded normal run replays to a byte-identical summary."""

    WINDOWS = 4

    def test_replay_summary_matches_recorded_run(self):
        recorder = RecordingBackend(DeterministicMockFeedbackBackend())
        ctl1 = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK, backend=recorder)
        s1 = ctl1.run(max_windows=self.WINDOWS)

        replay = ReplayBackend(recorder.to_replay_corpus())
        ctl2 = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK, backend=replay)
        s2 = ctl2.run(max_windows=self.WINDOWS)

        assert json.dumps(s1.to_dict(), sort_keys=True) == \
            json.dumps(s2.to_dict(), sort_keys=True)
        assert replay.usage.replay_calls == recorder.usage.mock_calls
        assert replay.usage.real_calls == 0
        assert replay.usage.total_calls > 0
