"""P0-0 (CC3 follow-up audit): execution-mode-aware six-role board usage.

The old ``run_review_board`` ended with an unconditional
``backend.usage.assert_no_real()`` — a check that hard-blocked the
legitimate REAL board path on window 1 and only ever inspected cumulative
state. The contract now is a MODE-AWARE USAGE DELTA taken around the six
role calls (snapshot before, exact delta after):

* mock backend   : real Δ=0, replay Δ=0, mock Δ=6
* replay backend : real Δ=0, mock Δ=0, replay Δ=6
* real backend   : real Δ=6, mock Δ=0, replay Δ=0

plus: any role failure propagates (the window is never marked
board-complete), and the controller's end-of-run honesty check is NOT a
substitute for this role-local check (both stay in force).

FIXTURES ARE TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION. The "real"
backend below is a ``RealBackendAdapter`` over an in-process SYNTHETIC
transport that reuses the deterministic mock rules — NO real LLM is ever
called, no credential exists anywhere in this file, and no REAL_* flag is
flipped by any test here.
"""
import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import assemble_board_context
from d052.feedback_llm_ued.feedback_view import NullFeedbackView
from d052.feedback_llm_ued.llm_backend import (
    DeterministicMockFeedbackBackend,
    RealBackendAdapter,
    RecordingBackend,
    ReplayBackend,
    UsageStats,
)
from d052.feedback_llm_ued.review_board import (
    BOARD_USAGE_DELTA_BY_KIND,
    BoardUsageDeltaMismatch,
    expected_board_usage_delta,
    run_review_board,
    verify_board_usage_delta,
)

#: TEST_ONLY / SYNTHETIC — never a real LLM endpoint
SYNTHETIC_BACKEND_ID = "test.synthetic_real.feedback_llm_ued.v1"
SYNTHETIC_MODEL_ID = "synthetic-rule-model.v1"


def _static_case():
    """The minimal board case: window 0, static mode, empty view — the six
    roles always run and produce a valid BoardOutput on any honest backend."""
    ctx = assemble_board_context(NullFeedbackView(), window=0,
                                 mode=C.MODE_STATIC_LLM)
    return dict(window=0, mode=C.MODE_STATIC_LLM, board_context=ctx,
                view=NullFeedbackView(), hypotheses=[], sequence_start=0)


def _synthetic_real_backend(max_retries=0):
    """A kind='real' backend served by an in-process synthetic transport.

    NOT_REAL_EXECUTION: the transport derives each response from the
    deterministic mock rules (a private mock instance whose OWN usage
    counters are irrelevant to the board's backend). This exercises the
    real adapter seam (retry budget, usage accounting, journal hook) without
    ever touching a paid endpoint.
    """
    source = DeterministicMockFeedbackBackend()

    def transport(role, prompt):
        return source.complete(role, prompt)

    return RealBackendAdapter(transport, backend_id=SYNTHETIC_BACKEND_ID,
                              model_id=SYNTHETIC_MODEL_ID, authorized=True,
                              max_retries=max_retries)


class TestExpectedDeltaContract:
    def test_delta_contract_per_kind(self):
        six = C.BOARD_CALLS_PER_WINDOW
        assert expected_board_usage_delta(C.BACKEND_KIND_MOCK) == \
            dict(real_calls=0, replay_calls=0, mock_calls=six)
        assert expected_board_usage_delta(C.BACKEND_KIND_REPLAY) == \
            dict(real_calls=0, replay_calls=six, mock_calls=0)
        assert expected_board_usage_delta(C.BACKEND_KIND_REAL) == \
            dict(real_calls=six, replay_calls=0, mock_calls=0)
        #: exactly six calls in EVERY mode — compute-matched by kind
        for delta in BOARD_USAGE_DELTA_BY_KIND.values():
            assert sum(delta.values()) == six

    def test_unknown_backend_kind_fails_closed(self):
        with pytest.raises(ValueError, match="UNKNOWN_BACKEND_KIND"):
            expected_board_usage_delta("symbolic")


class TestMockAndReplayDeltas:
    def test_mock_board_adds_exactly_six_mock_calls(self):
        backend = DeterministicMockFeedbackBackend()
        out = run_review_board(backend=backend, **_static_case())
        assert out.board_call_count == 6
        assert backend.usage.mock_calls == 6
        assert backend.usage.real_calls == 0
        assert backend.usage.replay_calls == 0

    def test_replay_board_adds_exactly_six_replay_calls(self):
        #: record a corpus in-process, then replay it (fail-closed backend)
        recorder = RecordingBackend(DeterministicMockFeedbackBackend())
        run_review_board(backend=recorder, **_static_case())
        backend = ReplayBackend(recorder.to_replay_corpus())
        out = run_review_board(backend=backend, **_static_case())
        assert out.board_call_count == 6
        assert backend.usage.replay_calls == 6
        assert backend.usage.real_calls == 0
        assert backend.usage.mock_calls == 0


class TestRealBoardUsage:
    def test_six_real_successes_are_not_blocked(self):
        """Negative test (a) of the audit: six legitimate real calls must
        NOT be refused by the legacy assert_no_real — the mode-aware delta
        accepts exactly them. (SYNTHETIC transport, NOT_REAL_EXECUTION.)"""
        backend = _synthetic_real_backend()
        out = run_review_board(backend=backend, **_static_case())
        assert out.board_call_count == 6
        assert backend.usage.real_calls == 6
        assert backend.usage.mock_calls == 0
        assert backend.usage.replay_calls == 0

    def test_end_of_run_check_is_not_replaced(self):
        """Point 6 of the audit: the board's role-local delta check does NOT
        replace the run-level honesty surface — the cumulative
        ``assert_no_real`` still refuses the very same usage when a
        non-real run claims it (both checks stay in force, independently)."""
        backend = _synthetic_real_backend()
        run_review_board(backend=backend, **_static_case())
        with pytest.raises(AssertionError,
                           match="REAL_LLM_CALLS_FORBIDDEN"):
            backend.usage.assert_no_real()

    def test_real_run_with_one_mixed_mock_call_rejected(self):
        """Negative test (b): a silent mock fallback inside a real board —
        five real completions plus ONE served as mock — must be refused."""

        class MixedKindBackend:
            #: claims real, silently serves one completion from the mock path
            kind = C.BACKEND_KIND_REAL
            backend_id = "test.mixed.feedback_llm_ued.v1"
            model_id = SYNTHETIC_MODEL_ID

            def __init__(self):
                self._source = DeterministicMockFeedbackBackend()
                self.usage = UsageStats()
                self._calls = 0

            def complete(self, role, prompt):
                raw = self._source.complete(role, prompt)
                if self._calls == 3:            # the silent fallback
                    self.usage.mock_calls += 1
                else:
                    self.usage.real_calls += 1
                self._calls += 1
                return raw

        with pytest.raises(BoardUsageDeltaMismatch,
                           match="BOARD_USAGE_DELTA_MISMATCH"):
            run_review_board(backend=MixedKindBackend(), **_static_case())

    def test_real_run_with_fewer_than_six_calls_rejected(self):
        """Negative test (c): five real completions are not a board. The
        integration path fails via the sixth call; the validator itself
        refuses the short delta."""

        class ShortBackend:
            kind = C.BACKEND_KIND_REAL
            backend_id = "test.short.feedback_llm_ued.v1"
            model_id = SYNTHETIC_MODEL_ID

            def __init__(self):
                self._source = DeterministicMockFeedbackBackend()
                self.usage = UsageStats()
                self._calls = 0

            def complete(self, role, prompt):
                self._calls += 1
                if self._calls > 5:
                    raise RuntimeError(
                        "TRANSPORT_REFUSED_SIXTH_CALL: synthetic short run")
                self.usage.real_calls += 1
                return self._source.complete(role, prompt)

        #: integration: the failure propagates — no BoardOutput exists, so
        #: the window can never be marked board-complete
        with pytest.raises(RuntimeError, match="TRANSPORT_REFUSED_SIXTH_CALL"):
            run_review_board(backend=ShortBackend(), **_static_case())

        #: validator level: an observed short delta is refused directly
        class _RealKinded:
            kind = C.BACKEND_KIND_REAL
            usage = UsageStats(real_calls=5)

        with pytest.raises(BoardUsageDeltaMismatch,
                           match="BOARD_USAGE_DELTA_MISMATCH"):
            verify_board_usage_delta(_RealKinded(), before=UsageStats())

    def test_real_run_with_more_than_six_calls_rejected(self):
        """Negative test (d): extra completions inside the board window are
        just as illegal as missing ones."""

        class DoubleCountingBackend:
            kind = C.BACKEND_KIND_REAL
            backend_id = "test.double.feedback_llm_ued.v1"
            model_id = SYNTHETIC_MODEL_ID

            def __init__(self):
                self._inner = _synthetic_real_backend()

            @property
            def usage(self):
                return self._inner.usage

            def complete(self, role, prompt):
                raw = self._inner.complete(role, prompt)
                #: one spurious extra completion smuggled into the window
                self._inner.usage.real_calls += 1
                return raw

        with pytest.raises(BoardUsageDeltaMismatch,
                           match="BOARD_USAGE_DELTA_MISMATCH"):
            run_review_board(backend=DoubleCountingBackend(),
                             **_static_case())

        class _RealKinded:
            kind = C.BACKEND_KIND_REAL
            usage = UsageStats(real_calls=7)

        with pytest.raises(BoardUsageDeltaMismatch,
                           match="BOARD_USAGE_DELTA_MISMATCH"):
            verify_board_usage_delta(_RealKinded(), before=UsageStats())

    def test_role_failure_mid_board_is_not_board_complete(self):
        """Point 5 of the audit: any role failure propagates — the board
        produces no output, so the window cannot be marked board-complete."""

        class FailingAtRoleThree:
            kind = C.BACKEND_KIND_REAL
            backend_id = "test.failrole.feedback_llm_ued.v1"
            model_id = SYNTHETIC_MODEL_ID

            def __init__(self):
                self._source = DeterministicMockFeedbackBackend()
                self.usage = UsageStats()
                self._calls = 0

            def complete(self, role, prompt):
                self._calls += 1
                if self._calls == 3:
                    raise RuntimeError("ROLE_CALL_FAILED: synthetic fault")
                self.usage.real_calls += 1
                return self._source.complete(role, prompt)

        with pytest.raises(RuntimeError, match="ROLE_CALL_FAILED"):
            run_review_board(backend=FailingAtRoleThree(), **_static_case())

    def test_replay_miss_is_a_role_failure_not_a_board(self):
        recorder = RecordingBackend(DeterministicMockFeedbackBackend())
        run_review_board(backend=recorder, **_static_case())
        corpus = dict(recorder.to_replay_corpus())
        corpus.pop(next(iter(corpus)))           # one role cannot replay
        backend = ReplayBackend(corpus)
        with pytest.raises(KeyError, match="REPLAY_MISS"):
            run_review_board(backend=backend, **_static_case())

    def test_snapshot_is_taken_before_the_first_role(self):
        """The delta must be the board's OWN: pre-existing usage on the
        backend (an earlier window's six calls) may not break the check."""
        backend = DeterministicMockFeedbackBackend()
        run_review_board(backend=backend, **_static_case())   # window A
        assert backend.usage.mock_calls == 6
        out = run_review_board(backend=backend, **_static_case())  # window B
        assert out.board_call_count == 6
        assert backend.usage.mock_calls == 12     # cumulative, still honest
