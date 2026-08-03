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
from typing import Callable, Dict, Mapping, Protocol, Tuple, runtime_checkable

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import extract_context


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
      expected to pass the round flag (``C.REAL_LLM_CALLS_AUTHORIZED``).
    * Credentials/API keys live ONLY inside the injected ``transport``
      closure — this class never sees, stores or logs them.
    * Failed attempts (exception or empty response) increment
      ``usage.failed_calls`` and are retried up to ``max_retries`` extra
      attempts; exhausting the budget raises ``BackendCallFailed``.
    """

    kind = C.BACKEND_KIND_REAL

    def __init__(self,
                 transport: Callable[[str, str], str],
                 *,
                 backend_id: str,
                 model_id: str,
                 authorized: bool,
                 max_retries: int = 2) -> None:
        if not authorized:
            raise BackendBlocked(
                "REAL_LLM_BACKEND_BLOCKED: REAL_LLM_CALLS_AUTHORIZED="
                f"{C.REAL_LLM_CALLS_AUTHORIZED} this round; a real backend "
                "may not be constructed")
        if max_retries < 0:
            raise ValueError(f"NEGATIVE_MAX_RETRIES: {max_retries}")
        self.backend_id = backend_id
        self.model_id = model_id
        self._transport = transport
        self._max_retries = max_retries
        self._usage = UsageStats()

    @property
    def usage(self) -> UsageStats:
        return self._usage

    def complete(self, role: str, prompt: str) -> str:
        attempts = self._max_retries + 1
        last_error: object = None
        for _ in range(attempts):
            try:
                raw = self._transport(role, prompt)
            except Exception as exc:          # transport failure -> retry
                self._usage.failed_calls += 1
                last_error = exc
                continue
            if not isinstance(raw, str) or not raw:
                self._usage.failed_calls += 1
                last_error = ValueError("EMPTY_REAL_LLM_RESPONSE")
                continue
            self._usage.real_calls += 1
            return raw
        raise BackendCallFailed(
            f"REAL_LLM_CALL_FAILED: role={role} after {attempts} attempts; "
            f"last_error={last_error!r}")


# ---------------------------------------------------------------------------
# rule registry (imported at the bottom so role modules can import this file's
# Protocol without a cycle)
# ---------------------------------------------------------------------------
from d052.feedback_llm_ued import (  # noqa: E402
    adaptive_designer,
    adversarial_reviewer,
    feedback_diagnostician,
)

_DEFAULT_RULES: Dict[str, Callable[[dict], dict]] = {
    C.ROLE_FEEDBACK_DIAGNOSTICIAN: feedback_diagnostician.mock_rule,
    C.ROLE_ADAPTIVE_ENVIRONMENT_DESIGNER: adaptive_designer.mock_rule,
    C.ROLE_ADVERSARIAL_REVIEWER: adversarial_reviewer.mock_rule,
}
