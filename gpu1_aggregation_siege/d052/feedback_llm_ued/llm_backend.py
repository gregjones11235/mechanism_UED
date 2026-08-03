"""Deterministic mock LLM backend for the feedback-adaptive loop.

REAL_LLM_CALLS_AUTHORIZED=false this round, so the loop runs against a
DETERMINISTIC rule-based mock backend. Each role's response is derived
deterministically from the structured context embedded in its prompt (between
the FEEDBACK_LLM_UED context markers). ``real_calls`` stays 0 and the
controller asserts it at the end; ``mock_calls`` counts what was served.

The per-role derivation rules live in the role modules (``mock_rule``) and are
registered here — the backend is the ONLY place role dispatch happens, and a
real backend can later be substituted behind the same LLMBackend Protocol
without touching role code.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Protocol, runtime_checkable

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import extract_context


@runtime_checkable
class LLMBackend(Protocol):
    backend_id: str
    model_id: str
    real_calls: int

    def complete(self, role: str, prompt: str) -> str: ...


class DeterministicMockFeedbackBackend:
    """Rule-based, seedless-deterministic mock backend (real_calls == 0)."""

    backend_id = C.MOCK_BACKEND_ID
    model_id = C.MOCK_MODEL_ID

    def __init__(self) -> None:
        self.real_calls = 0
        self.mock_calls = 0
        self._rules: Dict[str, Callable[[dict], dict]] = dict(_DEFAULT_RULES)

    def complete(self, role: str, prompt: str) -> str:
        if role not in self._rules:
            raise KeyError(f"UNKNOWN_ROLE_FOR_MOCK_BACKEND: {role}")
        context = extract_context(prompt)
        out = self._rules[role](context)
        self.mock_calls += 1          # mock — NOT a real LLM call
        return json.dumps(out, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)

    def assert_no_real_calls(self) -> None:
        if self.real_calls != 0:
            raise AssertionError(
                f"REAL_LLM_CALLS_FORBIDDEN: real_calls={self.real_calls} "
                f"(REAL_LLM_CALLS_AUTHORIZED=false this round)")


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
