"""BudgetAllocator: 12 UED slots + 4 global anchors (task sections 13 / 14).

    UED_ACTIVE_SLOTS      = 12   (filled from the Soft Copeland ranking)
    GLOBAL_CANONICAL_ANCHORS = 4  (fixed GLOBAL anchors; NOT from proposals)

Honest shortfall: if the ranking provides fewer than 12 eligible descriptors,
the plan records INSUFFICIENT with a note — NO backfill, NO k-reduction, NO
re-LLM (same discipline as d052.selectors.base). Anchors are reserved
unconditionally so GLOBAL coverage never collapses to Tier3-only
(GLOBAL_UED_SLOTS_MINIMUM enforced). Deterministic allocation_hash.
"""
from __future__ import annotations

from typing import List

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.soft_copeland import CopelandRanking
from d052.schemas.common import CanonicalModel


class BudgetPlan(CanonicalModel):
    ued_slots: List[str] = Field(default_factory=list)
    anchor_slots: List[str] = Field(default_factory=list)
    status: str = Field(pattern=r"^(OK|INSUFFICIENT)$")
    shortfall_note: str = ""
    ued_active_slots_target: int = C.UED_ACTIVE_SLOTS
    global_anchor_count: int = C.GLOBAL_CANONICAL_ANCHORS
    allocation_hash: str = ""

    @model_validator(mode="after")
    def _invariants(self) -> "BudgetPlan":
        if len(self.anchor_slots) != C.GLOBAL_CANONICAL_ANCHORS:
            raise ValueError(
                f"ANCHOR_COUNT: expected {C.GLOBAL_CANONICAL_ANCHORS} global "
                f"anchors, got {len(self.anchor_slots)}")
        if len(self.ued_slots) > C.UED_ACTIVE_SLOTS:
            raise ValueError(
                f"UED_SLOT_OVERFLOW: > {C.UED_ACTIVE_SLOTS} UED slots")
        if len(set(self.ued_slots)) != len(self.ued_slots):
            raise ValueError("DUPLICATE_UED_SLOT")
        if set(self.ued_slots) & set(self.anchor_slots):
            raise ValueError("UED_ANCHOR_SLOT_COLLISION")
        if len(self.ued_slots) < C.GLOBAL_UED_SLOTS_MINIMUM:
            raise ValueError(
                f"GLOBAL_UED_SLOTS_MINIMUM_VIOLATED: need >= "
                f"{C.GLOBAL_UED_SLOTS_MINIMUM} UED slot(s)")
        if self.status == "INSUFFICIENT" and not self.shortfall_note:
            raise ValueError("MISSING_SHORTFALL_NOTE")
        return self


class BudgetAllocator:
    def __init__(self, ued_active_slots: int = C.UED_ACTIVE_SLOTS,
                 anchor_ids: tuple = C.GLOBAL_CANONICAL_ANCHOR_IDS) -> None:
        if len(anchor_ids) != C.GLOBAL_CANONICAL_ANCHORS:
            raise ValueError("ANCHOR_ID_COUNT")
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("DUPLICATE_ANCHOR_IDS")
        self.ued_active_slots = ued_active_slots
        self.anchor_ids = list(anchor_ids)

    def allocate(self, ranking: CopelandRanking) -> BudgetPlan:
        ranked_ids = [e.environment_id for e in ranking.entries]
        chosen = ranked_ids[:self.ued_active_slots]
        if len(chosen) >= self.ued_active_slots:
            status, note = "OK", ""
        else:
            status = "INSUFFICIENT"
            note = (f"only {len(chosen)} eligible descriptors for "
                    f"{self.ued_active_slots} UED slots; NO backfill / "
                    f"NO k-reduction / NO re-LLM")
        plan = BudgetPlan(ued_slots=chosen, anchor_slots=list(self.anchor_ids),
                          status=status, shortfall_note=note)
        object.__setattr__(plan, "allocation_hash", canonical_sha256(
            plan.model_dump(exclude={"allocation_hash"})))
        return plan
