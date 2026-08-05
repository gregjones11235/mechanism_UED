"""P0-16 (director smoke handoff, section 4): the Canonical DiCode 15+1
training batch plan.

The window k+1 final selection (12 dynamic + 4 anchors) is converted into
the director-declared DiCode batch contract:

* 12 dynamic task ids (the funnel's non-anchor final-batch members);
* 3 NON-TARGET anchor task ids (the director-declared curriculum anchors);
* = 15 curriculum task ids — the ONLY ids that enter ``batch_candidate_ids``;
* the OriginalTask is appended ONCE internally by the director-shared
  CanonicalDiCodeOneUpdateRuntime — it NEVER enters batch_candidate_ids,
  is NEVER duplicated, and the batch totals 16 tasks with
  original_task_proportion = 0.20 (the other 15 share 0.80).

Direction two NEVER implements a second PPO/optimizer: the window k+1
update is executed exclusively by the director's shared
CanonicalDiCodeOneUpdateRuntime (consumed via the Runtime Bundle).
"""
from __future__ import annotations

from typing import List, Sequence

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel


class CanonicalDiCodeTrainingBatchPlan(CanonicalModel):
    """One window's immutable DiCode 15+1 training batch plan.

    ``batch_candidate_ids`` = the 15 curriculum task ids ONLY (the
    OriginalTask is appended internally by the DiCode runtime, never
    listed here, never duplicated). ``original_appended_by`` names the
    director-shared runtime identity that performs the append + the single
    optimizer update.
    """

    model_config = {"frozen": True}

    window: int = Field(ge=0)
    dynamic_task_ids: List[str] = Field(default_factory=list)
    non_target_anchor_ids: List[str] = Field(default_factory=list)
    curriculum_task_ids: List[str] = Field(default_factory=list)
    original_task_id: str = Field(min_length=1)
    original_task_proportion: float = Field(ge=0.0, le=1.0)
    total_task_count: int = Field(ge=0)
    original_appended_by: str = Field(min_length=1)
    plan_hash: str = ""

    @property
    def batch_candidate_ids(self) -> List[str]:
        """The ids the ONE update consumes — the 15 curriculum tasks only;
        the OriginalTask is appended internally by DiCode (never here)."""
        return list(self.curriculum_task_ids)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "CanonicalDiCodeTrainingBatchPlan":
        dynamic = list(self.dynamic_task_ids)
        anchors = list(self.non_target_anchor_ids)
        curriculum = list(self.curriculum_task_ids)
        if len(dynamic) != C.DICODE_CURRICULUM_DYNAMIC:
            raise ValueError(
                f"DICODE_DYNAMIC_COUNT_MISMATCH: {len(dynamic)} != "
                f"{C.DICODE_CURRICULUM_DYNAMIC}")
        if len(anchors) != C.DICODE_CURRICULUM_NON_TARGET_ANCHORS:
            raise ValueError(
                f"DICODE_ANCHOR_COUNT_MISMATCH: {len(anchors)} != "
                f"{C.DICODE_CURRICULUM_NON_TARGET_ANCHORS}")
        if len(curriculum) != C.DICODE_CURRICULUM_TASK_COUNT:
            raise ValueError(
                f"DICODE_CURRICULUM_COUNT_MISMATCH: {len(curriculum)} != "
                f"{C.DICODE_CURRICULUM_TASK_COUNT}")
        if len(set(curriculum)) != len(curriculum):
            raise ValueError("DICODE_DUPLICATE_CURRICULUM_TASK")
        #: the OriginalTask must NEVER be a curriculum task (it is appended
        #: internally once) and must NEVER be duplicated
        if self.original_task_id in curriculum:
            raise ValueError(
                "DICODE_ORIGINAL_IN_BATCH: the OriginalTask must NOT enter "
                "batch_candidate_ids — it is appended internally by DiCode")
        if self.original_task_id in dynamic or \
                self.original_task_id in anchors:
            raise ValueError(
                "DICODE_ORIGINAL_DUPLICATED: the OriginalTask is one of the "
                "curriculum tasks — it must be appended ONCE only")
        if self.total_task_count != C.DICODE_BATCH_TOTAL_TASKS:
            raise ValueError(
                f"DICODE_TOTAL_COUNT_MISMATCH: {self.total_task_count} != "
                f"{C.DICODE_BATCH_TOTAL_TASKS}")
        if abs(self.original_task_proportion
               - C.DICODE_ORIGINAL_TASK_PROPORTION) > 1e-9:
            raise ValueError(
                f"DICODE_ORIGINAL_PROPORTION_MISMATCH: "
                f"{self.original_task_proportion} != "
                f"{C.DICODE_ORIGINAL_TASK_PROPORTION}")
        if not self.plan_hash:
            raise ValueError(
                "DICODE_BATCH_PLAN_UNSIGNED: the batch plan must carry the "
                "recomputable plan_hash")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="plan_hash",
                                       carried=self.plan_hash,
                                       kind="CanonicalDiCodeTrainingBatchPlan")
        object.__setattr__(self, "plan_hash", computed)
        return self


def build_dicode_batch_plan(*, window: int, final_batch_ids: Sequence[str],
                            anchor_ids: Sequence[str],
                            non_target_anchor_ids: Sequence[str],
                            original_task_id: str,
                            original_appended_by: str
                            ) -> CanonicalDiCodeTrainingBatchPlan:
    """Convert the window's final selection into the DiCode 15+1 batch.

    The final batch (12 dynamic + 4 anchors): the 12 dynamic ids are the
    final-batch members that are NOT anchors (nor the OriginalTask); the
    plan binds exactly the director-declared 3 NON-TARGET anchors — the
    remaining (target) anchor is excluded from the curriculum; the
    OriginalTask is declared separately and appended ONCE internally by
    the DiCode runtime (never in batch_candidate_ids). Fail-closed on any
    structural violation.
    """
    anchor_set = set(anchor_ids)
    dynamic_ids = [cid for cid in final_batch_ids
                   if cid not in anchor_set and cid != original_task_id]
    if len(dynamic_ids) != C.DICODE_CURRICULUM_DYNAMIC:
        raise ValueError(
            "DICODE_DYNAMIC_COUNT_MISMATCH: final batch produced "
            f"{len(dynamic_ids)} dynamic ids (non-anchor members) != "
            f"{C.DICODE_CURRICULUM_DYNAMIC}")
    undeclared = [a for a in non_target_anchor_ids if a not in anchor_set]
    if undeclared:
        raise ValueError(
            "DICODE_NON_TARGET_ANCHOR_NOT_IN_MANIFEST: "
            f"{sorted(undeclared)} are not among the shared manifest "
            "anchors")
    #: the plan is direction-two's own artifact: its recomputable content
    #: hash is computed over the exact canonical payload (the model's
    #: validator re-verifies it fail-closed)
    body = dict(
        window=window,
        dynamic_task_ids=list(dynamic_ids),
        non_target_anchor_ids=list(non_target_anchor_ids),
        curriculum_task_ids=(list(dynamic_ids)
                             + list(non_target_anchor_ids)),
        original_task_id=original_task_id,
        original_task_proportion=C.DICODE_ORIGINAL_TASK_PROPORTION,
        total_task_count=C.DICODE_BATCH_TOTAL_TASKS,
        original_appended_by=original_appended_by)
    body.setdefault(
        "protocol_version",
        CanonicalDiCodeTrainingBatchPlan.model_fields["protocol_version"]
        .default)
    return CanonicalDiCodeTrainingBatchPlan(
        **body, plan_hash=canonical_sha256(body))


__all__ = ["CanonicalDiCodeTrainingBatchPlan", "build_dicode_batch_plan"]
