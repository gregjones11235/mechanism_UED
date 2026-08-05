"""CC2 follow-up P0-15: the 98304 training-budget semantics.

The 98304 total was pinned as a "supervisor constant" and could be
applied as a compute-matched value without any explicit decision. The
budget now has EXPLICIT SEMANTICS and must come from a director
decision block::

    budget = resolve_training_budget(block)   # or raises UNDECIDED

``training_budget_semantics`` is exactly one of::

    TOTAL_FROM_COMMON_INITIALIZATION       (fresh common init,
                                            initial steps == 0)
    ADDITIONAL_FROM_PRETRAINED_CHECKPOINT  (from a pretrained
                                            checkpoint, initial steps
                                            > 0)

The block must carry all three step fields and they must be
consistent (final_total == initial + additional, additional > 0,
final_total > 0). Any absent / inconsistent / unfrozen decision
resolves to ``BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`` — 98304 is
NEVER applied implicitly as a compute match, and a longrun never
starts until the decision is frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: the two explicit budget semantics (the ONLY admitted values)
TOTAL_FROM_COMMON_INITIALIZATION = "TOTAL_FROM_COMMON_INITIALIZATION"
ADDITIONAL_FROM_PRETRAINED_CHECKPOINT = (
    "ADDITIONAL_FROM_PRETRAINED_CHECKPOINT"
)
BUDGET_SEMANTICS = (
    TOTAL_FROM_COMMON_INITIALIZATION,
    ADDITIONAL_FROM_PRETRAINED_CHECKPOINT,
)

#: the honest gate state when no director decision exists
BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION = (
    "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION"
)

#: the four decision fields (all REQUIRED, no defaults)
BUDGET_FIELD_NAMES = (
    "semantics",
    "initial_checkpoint_env_steps",
    "additional_training_env_steps",
    "final_total_env_steps",
)

# fail-closed codes (greppable)
BUDGET_BAD_TYPE = "BUDGET_BAD_TYPE"
BUDGET_UNDECIDED = "BUDGET_UNDECIDED"
BUDGET_UNKNOWN_SEMANTICS = "BUDGET_UNKNOWN_SEMANTICS"
BUDGET_INCONSISTENT = "BUDGET_INCONSISTENT"
BUDGET_ZERO = "BUDGET_ZERO"


class BudgetError(E1SchemaError):
    """Fail-closed budget violation; ``code`` is greppable."""


@dataclass(frozen=True)
class TrainingBudget:
    """The director-frozen training budget (immutable, hash-bound)."""

    semantics: str
    initial_checkpoint_env_steps: int
    additional_training_env_steps: int
    final_total_env_steps: int
    budget_hash: str


def resolve_training_budget(block: Any, ctx: str) -> TrainingBudget:
    """Resolve the director's budget decision fail-closed.

    ``block`` is the director's decision mapping; anything absent,
    malformed, inconsistent or carrying unknown semantics raises —
    the caller records ``BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION``.
    """
    if not isinstance(block, Mapping):
        raise BudgetError(
            BUDGET_UNDECIDED,
            f"{ctx}: no director training-budget decision exists "
            f"({BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION}); the 98304 "
            "total is NEVER applied implicitly as a compute match",
        )
    missing = sorted(f for f in BUDGET_FIELD_NAMES if f not in block)
    if missing:
        raise BudgetError(
            BUDGET_UNDECIDED,
            f"{ctx}: the director budget decision is incomplete "
            f"(missing {missing}); "
            f"{BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION}",
        )
    semantics = block["semantics"]
    if semantics not in BUDGET_SEMANTICS:
        raise BudgetError(
            BUDGET_UNKNOWN_SEMANTICS,
            f"{ctx}: training_budget_semantics must be one of "
            f"{list(BUDGET_SEMANTICS)}, got {semantics!r}",
        )
    steps: dict = {}
    for field in (
        "initial_checkpoint_env_steps",
        "additional_training_env_steps",
        "final_total_env_steps",
    ):
        value = block[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BudgetError(
                BUDGET_BAD_TYPE,
                f"{ctx}: {field} must be a non-negative int, got "
                f"{value!r}",
            )
        steps[field] = value
    initial = steps["initial_checkpoint_env_steps"]
    additional = steps["additional_training_env_steps"]
    final_total = steps["final_total_env_steps"]
    if final_total != initial + additional:
        raise BudgetError(
            BUDGET_INCONSISTENT,
            f"{ctx}: final_total_env_steps ({final_total}) != "
            f"initial ({initial}) + additional ({additional}); the "
            "budget arithmetic must close exactly",
        )
    if additional <= 0:
        raise BudgetError(
            BUDGET_ZERO,
            f"{ctx}: additional_training_env_steps must be > 0; a "
            "longrun that trains nothing is not a longrun",
        )
    if final_total <= 0:
        raise BudgetError(
            BUDGET_ZERO,
            f"{ctx}: final_total_env_steps must be > 0, got "
            f"{final_total}",
        )
    if (
        semantics == TOTAL_FROM_COMMON_INITIALIZATION
        and initial != 0
    ):
        raise BudgetError(
            BUDGET_INCONSISTENT,
            f"{ctx}: TOTAL_FROM_COMMON_INITIALIZATION requires "
            "initial_checkpoint_env_steps == 0 (fresh common init), "
            f"got {initial}",
        )
    budget_hash = canonical_sha256(
        {
            "semantics": semantics,
            "initial_checkpoint_env_steps": initial,
            "additional_training_env_steps": additional,
            "final_total_env_steps": final_total,
        }
    )
    return TrainingBudget(
        semantics=semantics,
        initial_checkpoint_env_steps=initial,
        additional_training_env_steps=additional,
        final_total_env_steps=final_total,
        budget_hash=budget_hash,
    )


def require_budget_decided(budget: Any, ctx: str) -> TrainingBudget:
    """Refuse to start a longrun until the budget is decided."""
    if not isinstance(budget, TrainingBudget):
        raise BudgetError(
            BUDGET_UNDECIDED,
            f"{ctx}: training budget is not decided "
            f"({BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION}); a longrun "
            "never starts on an unresolved 98304",
        )
    return budget
