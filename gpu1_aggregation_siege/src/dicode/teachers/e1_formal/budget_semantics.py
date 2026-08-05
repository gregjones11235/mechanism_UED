"""CC2-Director: the training budget on the DiCode timeline.

The ONLY training timeline is the DiCode native protocol
(``config.training.total_timesteps`` from the frozen
``conf/training/default.yaml``). 方向一 no longer defines any fixed
longrun horizon (98304 is gone as a formal budget — it may appear
ONLY in checkpoint paths / checkpoint steps / Student candidate
identity)::

    budget = resolve_training_budget(block, frozen_total_timesteps=...)

``training_budget_semantics`` is exactly one of::

    TOTAL_FROM_COMMON_INITIALIZATION       (fresh common init,
                                            initial == 0)
    ADDITIONAL_FROM_PRETRAINED_CHECKPOINT  (from a pretrained
                                            checkpoint, initial > 0)

The director's decision block carries the DiCode timeline fields:

* ``total_timesteps`` — MUST equal the frozen DiCode resolved config
  value (``BUDGET_TIMELINE_MISMATCH`` otherwise; a local horizon is
  never a substitute);
* ``initial_checkpoint_timesteps`` + ``additional_training_timesteps``
  = ``final_total_timesteps`` (exact arithmetic), additional > 0.

Any absent / inconsistent / unfrozen decision resolves to
``BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`` and a formal experiment
never starts until the decision is frozen.
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

#: the five decision fields (all REQUIRED, no defaults)
BUDGET_FIELD_NAMES = (
    "semantics",
    "total_timesteps",
    "initial_checkpoint_timesteps",
    "additional_training_timesteps",
    "final_total_timesteps",
)

# fail-closed codes (greppable)
BUDGET_BAD_TYPE = "BUDGET_BAD_TYPE"
BUDGET_UNDECIDED = "BUDGET_UNDECIDED"
BUDGET_UNKNOWN_SEMANTICS = "BUDGET_UNKNOWN_SEMANTICS"
BUDGET_INCONSISTENT = "BUDGET_INCONSISTENT"
BUDGET_ZERO = "BUDGET_ZERO"
BUDGET_TIMELINE_MISMATCH = "BUDGET_TIMELINE_MISMATCH"


class BudgetError(E1SchemaError):
    """Fail-closed budget violation; ``code`` is greppable."""


@dataclass(frozen=True)
class TrainingBudget:
    """The director-frozen training budget on the DiCode timeline
    (immutable, hash-bound)."""

    semantics: str
    total_timesteps: int
    initial_checkpoint_timesteps: int
    additional_training_timesteps: int
    final_total_timesteps: int
    budget_hash: str


def resolve_training_budget(
    block: Any, *, frozen_total_timesteps: int, ctx: str
) -> TrainingBudget:
    """Resolve the director's budget decision fail-closed.

    ``frozen_total_timesteps`` is the value from the frozen DiCode
    resolved config (conf/training/default.yaml) — the ONLY admitted
    training timeline.
    """
    if not isinstance(block, Mapping):
        raise BudgetError(
            BUDGET_UNDECIDED,
            f"{ctx}: no director training-budget decision exists "
            f"({BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION}); the formal "
            "experiment never starts on an unresolved budget",
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
        "total_timesteps",
        "initial_checkpoint_timesteps",
        "additional_training_timesteps",
        "final_total_timesteps",
    ):
        value = block[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BudgetError(
                BUDGET_BAD_TYPE,
                f"{ctx}: {field} must be a non-negative int, got "
                f"{value!r}",
            )
        steps[field] = value
    if steps["total_timesteps"] != frozen_total_timesteps:
        raise BudgetError(
            BUDGET_TIMELINE_MISMATCH,
            f"{ctx}: total_timesteps {steps['total_timesteps']} != the "
            f"frozen DiCode resolved config value "
            f"{frozen_total_timesteps}; a local longrun horizon is "
            "never a substitute for the DiCode timeline",
        )
    initial = steps["initial_checkpoint_timesteps"]
    additional = steps["additional_training_timesteps"]
    final_total = steps["final_total_timesteps"]
    if final_total != initial + additional:
        raise BudgetError(
            BUDGET_INCONSISTENT,
            f"{ctx}: final_total_timesteps ({final_total}) != "
            f"initial ({initial}) + additional ({additional}); the "
            "budget arithmetic must close exactly",
        )
    if additional <= 0:
        raise BudgetError(
            BUDGET_ZERO,
            f"{ctx}: additional_training_timesteps must be > 0; an "
            "experiment that trains nothing is not a formal experiment",
        )
    if final_total <= 0:
        raise BudgetError(
            BUDGET_ZERO,
            f"{ctx}: final_total_timesteps must be > 0, got "
            f"{final_total}",
        )
    if semantics == TOTAL_FROM_COMMON_INITIALIZATION and initial != 0:
        raise BudgetError(
            BUDGET_INCONSISTENT,
            f"{ctx}: TOTAL_FROM_COMMON_INITIALIZATION requires "
            "initial_checkpoint_timesteps == 0 (fresh common init), "
            f"got {initial}",
        )
    budget_hash = canonical_sha256(
        {
            "semantics": semantics,
            "total_timesteps": steps["total_timesteps"],
            "initial_checkpoint_timesteps": initial,
            "additional_training_timesteps": additional,
            "final_total_timesteps": final_total,
        }
    )
    return TrainingBudget(
        semantics=semantics,
        total_timesteps=steps["total_timesteps"],
        initial_checkpoint_timesteps=initial,
        additional_training_timesteps=additional,
        final_total_timesteps=final_total,
        budget_hash=budget_hash,
    )


def require_budget_decided(budget: Any, ctx: str) -> TrainingBudget:
    """Refuse to start a formal experiment until the budget is decided."""
    if not isinstance(budget, TrainingBudget):
        raise BudgetError(
            BUDGET_UNDECIDED,
            f"{ctx}: training budget is not decided "
            f"({BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION}); a formal "
            "experiment never starts on an unresolved budget",
        )
    return budget
