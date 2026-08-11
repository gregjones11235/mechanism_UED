"""PlanRevisionRecord — the audit trail between plan_k and plan_{k+1}.

Every time the AdaptiveEnvironmentDesigner emits a new plan, a revision record
binds: which feedback ids the change is BASED ON, which plan it supersedes,
which families were retained/mutated/retired/newly-explored, the per-family
budget (slot) change, and a reason for EACH modification.

The honesty rule of the whole direction is encoded here as a hard validator:
a plan / modification that cites NO feedback_id may ONLY be labelled
EXPLORATION — it can never masquerade as a feedback-driven adjustment. The
record-level label is FORCED by the cited-id union, and record-level ids MUST
equal the union over modifications, so neither direction of mismatch parses.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel

#: label forced onto any revision that cites at least one feedback id
FEEDBACK_DRIVEN_LABEL = "FEEDBACK_DRIVEN"
REVISION_LABELS = frozenset({C.EXPLORATION_LABEL, FEEDBACK_DRIVEN_LABEL})


class PlanModification(CanonicalModel):
    """One family-level modification with its budget change + justification."""

    environment_family: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    based_on_feedback_ids: List[str] = Field(default_factory=list)
    is_exploration: bool = False
    #: None = family absent from the previous plan (new this revision)
    old_slots: Optional[int] = Field(default=None, ge=0)
    new_slots: int = Field(ge=0)

    @model_validator(mode="after")
    def _legality(self) -> "PlanModification":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if self.decision not in C.DESIGNER_DECISIONS:
            raise ValueError(f"ILLEGAL_DECISION: {self.decision!r}")
        if not self.based_on_feedback_ids:
            if not self.is_exploration:
                raise ValueError(
                    "EXPLORATION_LABEL_REQUIRED: a modification with no cited "
                    "feedback_id may only be EXPLORATION, never a "
                    "feedback-driven adjustment")
            if self.decision not in C.EXPLORATION_DECISIONS:
                raise ValueError(
                    f"EXPLORATION_DECISION_ONLY: an uncited modification may "
                    f"only use {sorted(C.EXPLORATION_DECISIONS)}, got "
                    f"{self.decision!r}")
        elif self.is_exploration:
            raise ValueError(
                "MASQUERADE_FORBIDDEN: a modification citing feedback_ids is "
                "feedback-driven and may not be labelled exploration")
        return self


class PlanRevisionRecord(CanonicalModel):
    """Hash-bound record of one plan_k -> plan_{k+1} transition."""

    revision_id: str = Field(min_length=1)
    window: int = Field(ge=0)
    mode: str = Field(min_length=1)
    #: "" only for the initial plan of a run (no predecessor yet)
    previous_plan_id: str = ""
    new_plan_id: str = Field(min_length=1)
    based_on_feedback_ids: List[str] = Field(default_factory=list)
    modifications: List[PlanModification] = Field(default_factory=list)
    label: str = C.EXPLORATION_LABEL
    record_hash: str = ""

    @model_validator(mode="after")
    def _consistency_and_hash(self) -> "PlanRevisionRecord":
        if self.mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {self.mode!r}")
        if self.label not in REVISION_LABELS:
            raise ValueError(f"ILLEGAL_REVISION_LABEL: {self.label!r}")
        union: Set[str] = set()
        for mod in self.modifications:
            union.update(mod.based_on_feedback_ids)
        if set(self.based_on_feedback_ids) != union:
            raise ValueError(
                f"FEEDBACK_ID_MISMATCH: record-level based_on_feedback_ids "
                f"{sorted(set(self.based_on_feedback_ids))} must equal the "
                f"union over modifications {sorted(union)}")
        expected_label = (FEEDBACK_DRIVEN_LABEL if union
                          else C.EXPLORATION_LABEL)
        if self.label != expected_label:
            raise ValueError(
                f"REVISION_LABEL_FORCED: cited-feedback union "
                f"{'is empty' if not union else 'is non-empty'}, so the label "
                f"must be {expected_label!r}, got {self.label!r}")
        # C14: an externally carried record_hash is recomputed and compared
        # verbatim (CONTENT_HASH_MISMATCH fails closed)
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="record_hash",
                                       carried=self.record_hash,
                                       kind="PlanRevisionRecord")
        object.__setattr__(self, "record_hash", computed)
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("record_hash", None)
        return canonical_sha256(payload)


def budget_changes(revision: PlanRevisionRecord) -> List[Dict[str, object]]:
    """Explicit per-family budget (slot) delta view of a revision."""
    out: List[Dict[str, object]] = []
    for mod in revision.modifications:
        old = mod.old_slots if mod.old_slots is not None else 0
        out.append(dict(
            environment_family=mod.environment_family,
            decision=mod.decision,
            old_slots=old,
            new_slots=mod.new_slots,
            delta=mod.new_slots - old,
            is_exploration=mod.is_exploration,
            based_on_feedback_ids=list(mod.based_on_feedback_ids),
        ))
    return out


def assert_feedback_ids_known(revision: PlanRevisionRecord,
                              known_ids) -> None:
    """Fail-closed cross-check that every cited feedback id actually exists
    (caller passes ``SimulatorFeedbackStore.ids()`` or equivalent)."""
    missing = [fid for fid in revision.based_on_feedback_ids
               if fid not in known_ids]
    if missing:
        raise ValueError(
            f"UNKNOWN_FEEDBACK_ID: {sorted(set(missing))} — revision "
            f"{revision.revision_id!r} cites feedback that does not exist in "
            f"the SimulatorFeedbackStore")
