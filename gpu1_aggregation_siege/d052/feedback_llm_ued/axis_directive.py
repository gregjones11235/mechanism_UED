"""AxisDirective — the controlled environment specification (P1-1, C5).

The six-role board proposes mutations as EXPLICIT controlled-experiment
directives: which axis moves, from which level to which level, in which
direction, what is held constant, which predicted signature the probe should
observe, and whether the directive is a treatment or a held control. This is
the board -> EnvCoder contract: from C8 onward a candidate's axis
configuration must derive from a directive, never from an index rotation
(the historical ``i % len(axes)`` / ``i % 3`` pattern is abolished as a
source of truth).

All vocabulary is environment-level (TaskParams induction knobs only) and
every rule fails closed — a malformed directive is a hard error, never a
silent coercion.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.environment_generator import (
    AXIS_LEVELS,
    FAMILY_AXES,
)
from d052.schemas.common import CanonicalModel

#: old_level sentinel: the axis was never measured before this directive
LEVEL_NONE = "none"
OLD_LEVELS = frozenset(set(AXIS_LEVELS) | {LEVEL_NONE})

DIRECTION_INCREASE = "increase"
DIRECTION_DECREASE = "decrease"
DIRECTION_HOLD = "hold"
DIRECTIONS = frozenset({DIRECTION_INCREASE, DIRECTION_DECREASE,
                        DIRECTION_HOLD})

ROLE_TREATMENT = "treatment"
ROLE_CONTROL = "control"
EXPERIMENT_ROLES = frozenset({ROLE_TREATMENT, ROLE_CONTROL})

_LEVEL_RANK = {level: i for i, level in enumerate(AXIS_LEVELS)}


class AxisDirective(CanonicalModel):
    """One controlled axis movement proposed by the review board."""

    directive_id: str = Field(min_length=1)
    source_window: int = Field(ge=0)
    environment_family: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    old_level: str = Field(min_length=1)
    new_level: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    experiment_control_role: str = Field(min_length=1)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    expected_next_signature: Dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    directive_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "AxisDirective":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if self.axis not in C.MUTATION_AXES:
            raise ValueError(f"ILLEGAL_DIRECTIVE_AXIS: {self.axis!r}")
        family_axes = FAMILY_AXES.get(self.environment_family)
        if family_axes is not None and self.axis not in family_axes:
            raise ValueError(
                f"AXIS_NOT_IN_FAMILY: axis={self.axis!r} is not an "
                f"induction knob of {self.environment_family!r}")
        legal_held = (set(family_axes) - {self.axis}) if family_axes else set()
        for held_axis, held_level in self.held_constant_axes.items():
            if family_axes is not None and held_axis not in legal_held:
                raise ValueError(
                    f"ILLEGAL_HELD_AXIS: {held_axis!r} for family "
                    f"{self.environment_family!r}")
            if held_level not in AXIS_LEVELS:
                raise ValueError(
                    f"ILLEGAL_HELD_LEVEL: {held_axis!r}={held_level!r}")
        if self.old_level not in OLD_LEVELS:
            raise ValueError(f"ILLEGAL_OLD_LEVEL: {self.old_level!r}")
        if self.new_level not in AXIS_LEVELS:
            raise ValueError(f"ILLEGAL_NEW_LEVEL: {self.new_level!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"ILLEGAL_DIRECTION: {self.direction!r}")
        if self.experiment_control_role not in EXPERIMENT_ROLES:
            raise ValueError(
                f"ILLEGAL_EXPERIMENT_ROLE: {self.experiment_control_role!r}")

        if self.experiment_control_role == ROLE_CONTROL:
            if self.direction != DIRECTION_HOLD or \
                    self.new_level != self.old_level:
                raise ValueError(
                    "CONTROL_DIRECTIVE_MUST_HOLD: a control re-measures the "
                    "existing setting (direction=hold, new_level=old_level)")
        else:                                    # treatment
            if self.direction not in (DIRECTION_INCREASE, DIRECTION_DECREASE):
                raise ValueError(
                    "TREATMENT_DIRECTIVE_NEEDS_DIRECTION: increase/decrease")
            if self.old_level == LEVEL_NONE:
                pass                             # first measurement: any level
            else:
                if self.new_level == self.old_level:
                    raise ValueError(
                        "NO_OP_TREATMENT_DIRECTIVE: new_level == old_level")
                moved_up = (_LEVEL_RANK[self.new_level]
                            > _LEVEL_RANK[self.old_level])
                if moved_up != (self.direction == DIRECTION_INCREASE):
                    raise ValueError(
                        "DIRECTION_LEVEL_MISMATCH: "
                        f"{self.old_level}->{self.new_level} is not "
                        f"{self.direction!r}")

        if not self.expected_next_signature:
            raise ValueError(
                "EMPTY_EXPECTED_SIGNATURE: a directive must predict the "
                "signature its probe is meant to observe")
        for key, value in self.expected_next_signature.items():
            if not isinstance(value, (int, float)) or \
                    not math.isfinite(float(value)):
                raise ValueError(
                    f"NON_FINITE_EXPECTATION: {key}={value!r}")

        # C14: an externally carried directive_hash is recomputed and
        # compared verbatim (CONTENT_HASH_MISMATCH fails closed)
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="directive_hash",
                                       carried=self.directive_hash,
                                       kind="AxisDirective")
        object.__setattr__(self, "directive_hash", computed)
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("directive_hash", None)
        return canonical_sha256(payload)


def candidate_axis_config(directive: AxisDirective
                          ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Derive the candidate's (axis_values, held_constant_axes) from the
    directive — the single legal source of a candidate's mutation settings.

    A treatment applies ``new_level``; a control re-measures the held
    ``old_level``. Everything else stays exactly as the directive holds it.
    """
    level = (directive.new_level if directive.experiment_control_role
             == ROLE_TREATMENT else directive.old_level)
    return ({directive.axis: level}, dict(directive.held_constant_axes))


def assert_directive_batch_legal(directives: Sequence[AxisDirective]) -> None:
    """Batch invariants: unique ids; at most ONE treatment per
    (source_window, family, axis) — two competing movements of the same knob
    in one window would destroy the controlled-experiment reading."""
    seen_ids: set = set()
    seen_treatments: set = set()
    for d in directives:
        if d.directive_id in seen_ids:
            raise ValueError(f"DUPLICATE_DIRECTIVE_ID: {d.directive_id!r}")
        seen_ids.add(d.directive_id)
        if d.experiment_control_role == ROLE_TREATMENT:
            key = (d.source_window, d.environment_family, d.axis)
            if key in seen_treatments:
                raise ValueError(
                    f"DUPLICATE_AXIS_DIRECTIVE: window={key[0]} "
                    f"family={key[1]!r} axis={key[2]!r} has more than one "
                    "treatment directive")
            seen_treatments.add(key)
