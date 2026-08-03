"""E1 binding of the CC4 shared StudentInitContract (D14).

E1 consumes EXACTLY the same identity-only contract consumer as the
demoted static_llm direction — there is no second loader, no second
registry, and no E1-private student identity mechanism. This module is
a thin binding layer:

* re-exports the frozen ``static_llm.student_init_contract`` consumer;
* pins the ONE strong-Student candidate id for the E1 pipeline;
* offers ``consume_e1_student_contract`` = consume + pinned-identity
  assertion in a single fail-closed step.

Identity values still come verbatim from the frozen CC4 contract
mapping; nothing is defaulted or guessed here either.
"""
from __future__ import annotations

from typing import Any

from ..static_llm.student_init_contract import (
    CONTRACT_SCHEMA_VERSION,
    PINNED_STUDENT_CANDIDATE_ID,
    StudentContractError,
    StudentInitContract,
    assert_pinned_candidate,
    consume_student_init_contract,
    contract_field_names,
)

__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "PINNED_STUDENT_CANDIDATE_ID",
    "StudentContractError",
    "StudentInitContract",
    "assert_pinned_candidate",
    "consume_e1_student_contract",
    "consume_student_init_contract",
    "contract_field_names",
]


def consume_e1_student_contract(
    mapping: Any, context: str
) -> StudentInitContract:
    """Consume the CC4 contract AND assert the pinned E1 strong Student.

    Raises ``StudentContractError`` (greppable codes) on any identity
    violation, and ``STUDENT_ID_MISMATCH`` if the mapping does not
    refer to the ONE pinned candidate.
    """
    contract = consume_student_init_contract(mapping, context)
    assert_pinned_candidate(contract)
    return contract
