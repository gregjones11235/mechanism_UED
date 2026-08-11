"""LLM backends for the feedback-adaptive loop (P0-1 abstraction).

The controller talks ONLY to the :class:`LLMBackend` Protocol
(``backend_id`` / ``model_id`` / ``kind`` / ``usage`` / ``complete``) and to
:class:`UsageStats` — never to mock-only attributes. Three backends share the
same usage accounting:

* ``DeterministicMockFeedbackBackend`` (kind="mock"): deterministic rule-based
  responses derived from the structured prompt context. ``real_calls`` stays 0.
* ``ReplayBackend`` (kind="replay"): replays a recorded corpus keyed by
  ``(role, sha256(prompt))``; a miss FAILS CLOSED (``ReplayMiss``).
* ``RealBackendAdapter`` (kind="real"): seam for a real LLM API. Refuses to
  construct unless explicitly authorized; credentials live ONLY inside the
  injected ``transport`` closure (never stored here); failed attempts are
  counted and retried up to ``max_retries``.

``REAL_LLM_CALLS_AUTHORIZED=false`` this round, so the loop runs on the mock
backend and :func:`assert_no_real_llm_usage` is the single end-of-run check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple, \
    runtime_checkable

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.real_call_journal import (
    RealCallJournal,
    default_logical_call_id,
    normalize_transport_result,
)


class BackendBlocked(RuntimeError):
    """A backend refused to exist under this round's authorization."""


class BackendCallFailed(RuntimeError):
    """A real backend call exhausted its retry budget."""


class ReplayMiss(KeyError):
    """ReplayBackend asked for a (role, prompt) pair absent from the corpus."""


@dataclass
class UsageStats:
    """Unified usage accounting shared by Mock / Replay / Real backends."""

    real_calls: int = 0
    replay_calls: int = 0
    mock_calls: int = 0
    failed_calls: int = 0

    @property
    def total_calls(self) -> int:
        """Every served completion, regardless of kind (failed excluded)."""
        return self.real_calls + self.replay_calls + self.mock_calls

    def assert_no_real(self) -> None:
        if self.real_calls != 0:
            raise AssertionError(
                f"REAL_LLM_CALLS_FORBIDDEN: real_calls={self.real_calls} "
                f"(REAL_LLM_CALLS_AUTHORIZED="
                f"{C.REAL_LLM_CALLS_AUTHORIZED} this round)")

    def snapshot(self) -> "UsageStats":
        return UsageStats(self.real_calls, self.replay_calls,
                          self.mock_calls, self.failed_calls)


def assert_no_real_llm_usage(usage: UsageStats) -> None:
    """Single end-of-run honesty check, backend-kind agnostic."""
    usage.assert_no_real()


@runtime_checkable
class LLMBackend(Protocol):
    backend_id: str
    model_id: str
    kind: str

    @property
    def usage(self) -> UsageStats: ...

    def complete(self, role: str, prompt: str) -> str: ...


class DeterministicMockFeedbackBackend:
    """Rule-based, seedless-deterministic mock backend (kind="mock")."""

    backend_id = C.MOCK_BACKEND_ID
    model_id = C.MOCK_MODEL_ID
    kind = C.BACKEND_KIND_MOCK

    def __init__(self) -> None:
        self._usage = UsageStats()
        self._rules: Dict[str, Callable[[dict], dict]] = dict(_DEFAULT_RULES)

    @property
    def usage(self) -> UsageStats:
        return self._usage

    # -- compatibility shims (pre-P0-1 call sites; usage-based is canonical)
    @property
    def mock_calls(self) -> int:
        return self._usage.mock_calls

    @property
    def real_calls(self) -> int:
        return self._usage.real_calls

    def assert_no_real_calls(self) -> None:
        self._usage.assert_no_real()

    def complete(self, role: str, prompt: str) -> str:
        if role not in self._rules:
            raise KeyError(f"UNKNOWN_ROLE_FOR_MOCK_BACKEND: {role}")
        context = extract_context(prompt)
        out = self._rules[role](context)
        self._usage.mock_calls += 1          # mock — NOT a real LLM call
        return json.dumps(out, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)


#: short alias used by newer modules
MockFeedbackBackend = DeterministicMockFeedbackBackend


class RecordingBackend:
    """Wraps any backend and records ``(role, sha256(prompt)) -> raw``.

    Used by tests to build a replay corpus inside the same test (no frozen
    corpus is ever committed), so replay-equivalence stays self-healing across
    behavior-changing commits.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self._recordings: Dict[Tuple[str, str], str] = {}

    @property
    def backend_id(self) -> str:
        return self._inner.backend_id

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def kind(self) -> str:
        return self._inner.kind

    @property
    def usage(self) -> UsageStats:
        return self._inner.usage

    def complete(self, role: str, prompt: str) -> str:
        raw = self._inner.complete(role, prompt)
        self._recordings[(role, text_sha256(prompt))] = raw
        return raw

    def to_replay_corpus(self) -> Dict[Tuple[str, str], str]:
        return dict(self._recordings)


class ReplayBackend:
    """Replays a recorded corpus, fail-closed on any miss (kind="replay")."""

    backend_id = C.REPLAY_BACKEND_ID
    model_id = C.REPLAY_MODEL_ID
    kind = C.BACKEND_KIND_REPLAY

    def __init__(self, corpus: Mapping[Tuple[str, str], str]) -> None:
        self._corpus: Dict[Tuple[str, str], str] = dict(corpus)
        self._usage = UsageStats()

    @property
    def usage(self) -> UsageStats:
        return self._usage

    def complete(self, role: str, prompt: str) -> str:
        key = (role, text_sha256(prompt))
        try:
            raw = self._corpus[key]
        except KeyError:
            raise ReplayMiss(
                f"REPLAY_MISS: role={role} prompt_sha256={key[1]} "
                f"(corpus has {len(self._corpus)} entries; replay is "
                f"fail-closed, no silent fallback)") from None
        self._usage.replay_calls += 1
        return raw


class RealBackendAdapter:
    """Seam for a real LLM API (kind="real").

    * Construction FAILS CLOSED unless ``authorized`` is True; call sites are
      expected to pass the round flag (``C.REAL_LLM_CALLS_AUTHORIZED``) or an
      explicit runtime grant from the production entrypoint.
    * Credentials/API keys live ONLY inside the injected ``transport``
      closure — this class never sees, stores or logs them.
    * Failed attempts (exception or empty response) increment
      ``usage.failed_calls`` and are retried up to ``max_retries`` extra
      attempts; exhausting the budget raises ``BackendCallFailed``.
    * Optional audit ``journal`` (a ``real_call_journal.RealCallJournal``):
      when present, every served call is journaled (role / model / backend /
      logical call id / API request id / prompt+response hashes / token
      usage / retry count), duplicate successful calls are refused before
      the transport is ever invoked, and the retry count cannot exceed the
      journal's cap — the master-directive LLM call rules, enforced in code.
    """

    kind = C.BACKEND_KIND_REAL

    def __init__(self,
                 transport: Callable[[str, str], object],
                 *,
                 backend_id: str,
                 model_id: str,
                 authorized: bool,
                 max_retries: int = 2,
                 journal: Optional[RealCallJournal] = None) -> None:
        if not authorized:
            raise BackendBlocked(
                "REAL_LLM_BACKEND_BLOCKED: REAL_LLM_CALLS_AUTHORIZED="
                f"{C.REAL_LLM_CALLS_AUTHORIZED} this round; a real backend "
                "may not be constructed")
        if max_retries < 0:
            raise ValueError(f"NEGATIVE_MAX_RETRIES: {max_retries}")
        if journal is not None and max_retries > journal.retry_cap:
            raise ValueError(
                f"RETRY_CAP_MISMATCH: max_retries={max_retries} exceeds the "
                f"journal retry cap {journal.retry_cap}")
        self.backend_id = backend_id
        self.model_id = model_id
        self._transport = transport
        self._max_retries = max_retries
        self._journal = journal
        self._usage = UsageStats()

    @property
    def usage(self) -> UsageStats:
        return self._usage

    @property
    def journal(self) -> Optional[RealCallJournal]:
        return self._journal

    def record_schema_outcome(self, role: str, prompt: str, *, status: str,
                              window: int = -1, sequence: int = -1,
                              artifact_binding: str = ""):
        """P0-5: the caller-side parse verdict of one transported call.

        Every real call must end in a PARSED or SCHEMA_FAILED outcome
        entry; PARSED closes the logical_call_id (any later activity under
        it is refused as a duplicate successful call). No-op without a
        journal.
        """
        if self._journal is None:
            return None
        logical_call_id = default_logical_call_id(role, prompt,
                                                  self.backend_id)
        return self._journal.record_schema_outcome(
            logical_call_id, status=status, window=window,
            sequence=sequence, artifact_binding=artifact_binding)

    def complete(self, role: str, prompt: str) -> str:
        prompt_sha = text_sha256(prompt)
        logical_call_id = default_logical_call_id(role, prompt,
                                                  self.backend_id)
        if self._journal is not None:
            #: refuse a repeat of an already-successful call BEFORE spending
            #: any transport attempt (forbidden duplicate successful call)
            self._journal.assert_open(logical_call_id)
        attempts = self._max_retries + 1
        last_error: object = None
        for attempt in range(attempts):
            try:
                result = self._transport(role, prompt)
            except Exception as exc:          # transport failure -> retry
                self._usage.failed_calls += 1
                last_error = exc
                continue
            try:
                raw, request_id, token_usage, usage_status = \
                    normalize_transport_result(result)
            except ValueError as exc:         # empty / malformed -> retry
                self._usage.failed_calls += 1
                last_error = exc
                continue
            self._usage.real_calls += 1
            if self._journal is not None:
                self._journal.record_transport(
                    logical_call_id=logical_call_id, role=role,
                    backend_id=self.backend_id, model_id=self.model_id,
                    request_id=request_id, prompt_sha256=prompt_sha,
                    response_sha256=text_sha256(raw),
                    token_usage=token_usage,
                    token_usage_status=usage_status,
                    retry_count=attempt)
            return raw
        raise BackendCallFailed(
            f"REAL_LLM_CALL_FAILED: role={role} after {attempts} attempts; "
            f"last_error={last_error!r}")


# ---------------------------------------------------------------------------
# rule registry (imported at the bottom so role modules can import this file's
# Protocol without a cycle)
# ---------------------------------------------------------------------------
from d052.feedback_llm_ued import (  # noqa: E402
    behavior_auditor,
    causal_failure_analyst,
    critic_skeptic,
    env_coder,
    explorer,
    intervention_tutor,
    student_modeler,
)

_DEFAULT_RULES: Dict[str, Callable[[dict], dict]] = {
    # six-role Review Board (C6) — every role always registered; the board
    # runs all six every window, in every comparison mode. The legacy
    # Diagnostician/Designer/Reviewer roles were abolished with C8 and are
    # NOT registered (complete() raises UNKNOWN_ROLE_FOR_MOCK_BACKEND).
    C.ROLE_STUDENT_MODELER: student_modeler.mock_rule,
    C.ROLE_BEHAVIOR_AUDITOR: behavior_auditor.mock_rule,
    C.ROLE_CAUSAL_FAILURE_ANALYST: causal_failure_analyst.mock_rule,
    C.ROLE_INTERVENTION_TUTOR: intervention_tutor.mock_rule,
    C.ROLE_EXPLORER: explorer.mock_rule,
    C.ROLE_CRITIC_SKEPTIC: critic_skeptic.mock_rule,
    # independent EnvCoder — the 7th LLM-family call of every window (C7)
    C.ROLE_ENV_CODER: env_coder.mock_rule,
}
