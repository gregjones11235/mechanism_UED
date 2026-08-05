"""TEST_ONLY / SYNTHETIC LLM client factory.  NOT_REAL_EXECUTION.

This module exists ONLY so the two-LLM runtime descriptor contract can be
exercised by the dedicated tests.  It is never referenced by any production
entry point: the trusted-signer entrypoint allowlist rejects it for
production bundles (see the runtime-security audit).  Do not import it
outside tests, and never use it as a real LLM client.
"""

from __future__ import annotations

from typing import Any, Mapping

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


class _SyntheticCompleteClient:
    """A dummy ``complete`` surface for the TEST_ONLY factory."""

    def __init__(self, role: str) -> None:
        self._role = role

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError(
            "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT: the TEST_ONLY client "
            "factory is never called on a production path; a real director-"
            "approved client provider is required")


def synthetic_two_llm_client_factory(roles):
    """TEST_ONLY factory: return one dummy client per requested role.

    The factory itself is importable and source-hash bound so the descriptor
    mint/verify/build contract can be proven; the returned clients never
    complete a real call (fail closed if ever invoked).
    """
    return {role: _SyntheticCompleteClient(role) for role in roles}
