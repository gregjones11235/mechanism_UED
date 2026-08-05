"""CC2-Director: the shared DiCode training batch protocol (15 + 1).

方向一's selection result is 12 dynamic candidates. DiCode's native
batch protocol freezes the shared translation — this module is the
SINGLE shared contract (方向一 establishes it; DiCode consumes it via
``run_session_training(sampled_task_ids=...)``)::

    12 dynamic  +  3 non-target anchors  =  15 curriculum_task_ids
    + DiCode-appended OriginalTask (original_craftax)
    = 16 total tasks in the session

Rules (all mechanical, fail-closed):

* ``dynamic_task_ids`` is EXACTLY the 12 selected candidates;
* ``non_target_anchor_ids`` is EXACTLY the 3 shared non-target
  anchors (``task_1, task_2, task_3`` — the registered anchors
  EXCLUDING ``original_craftax``);
* ``curriculum_task_ids`` is EXACTLY the 15 = dynamic + non-target
  anchors (order: dynamic first, then the anchors);
* ``target_task_id == "original_craftax"`` and is NEVER part of the
  curriculum / never passed into ``sampled_task_ids`` — it appears
  EXACTLY ONCE, as the DiCode-appended target;
* ``target_probability == 0.20`` (frozen from
  ``conf/dicode_manager/default.yaml`` ``original_task_proportion``);
* the plan binds ``selection_attestation_hash`` and
  ``anchor_manifest_hash`` and folds everything into ``plan_hash``.

方向一 does NOT implement a second copy of the shared batch protocol;
DiCode's own timeline (``config.training.total_timesteps``,
``global_env_steps``, ``global_update_step``, session index) is the
only training time axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError
from .layout import ANCHOR_TASK_IDS, ORIGINAL_ANCHOR_TASK_ID

#: the frozen target identity + probability (DiCode native protocol)
DICODE_TARGET_TASK_ID = "original_craftax"
#: frozen from conf/dicode_manager/default.yaml original_task_proportion
DICODE_TARGET_PROBABILITY = 0.20

DICODE_NUM_DYNAMIC = 12
DICODE_NUM_NON_TARGET_ANCHORS = 3
DICODE_NUM_CURRICULUM = 15

#: the shared non-target anchors (registered anchors minus the target)
DICODE_NON_TARGET_ANCHOR_IDS = tuple(
    aid for aid in ANCHOR_TASK_IDS if aid != ORIGINAL_ANCHOR_TASK_ID
)
assert len(DICODE_NON_TARGET_ANCHOR_IDS) == DICODE_NUM_NON_TARGET_ANCHORS

# fail-closed codes (greppable)
PLAN_BAD_TYPE = "PLAN_BAD_TYPE"
PLAN_COUNT = "PLAN_COUNT"
PLAN_DUPLICATE = "PLAN_DUPLICATE"
PLAN_TARGET_MISSING = "PLAN_TARGET_MISSING"
PLAN_TARGET_DUPLICATED = "PLAN_TARGET_DUPLICATED"
PLAN_PROBABILITY = "PLAN_PROBABILITY"
PLAN_ANCHOR_MISMATCH = "PLAN_ANCHOR_MISMATCH"
PLAN_BINDING_MISMATCH = "PLAN_BINDING_MISMATCH"
PLAN_HASH_MISMATCH = "PLAN_HASH_MISMATCH"


class DiCodePlanError(E1SchemaError):
    """Fail-closed DiCode batch-plan violation; ``code`` is
    greppable."""


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DiCodePlanError(
            PLAN_BAD_TYPE,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


@dataclass(frozen=True)
class CanonicalDiCodeTrainingBatchPlan:
    """The shared 15+1 training batch plan (immutable, hash-bound)."""

    dynamic_task_ids: Tuple[str, ...]        # exactly 12
    non_target_anchor_ids: Tuple[str, ...]   # exactly 3
    curriculum_task_ids: Tuple[str, ...]     # exactly 15 = 12 + 3
    target_task_id: str                      # original_craftax
    target_probability: float                # 0.20
    selection_attestation_hash: str
    anchor_manifest_hash: str
    plan_hash: str


def _plan_payload(
    *,
    dynamic_task_ids: Tuple[str, ...],
    non_target_anchor_ids: Tuple[str, ...],
    curriculum_task_ids: Tuple[str, ...],
    target_task_id: str,
    target_probability: float,
    selection_attestation_hash: str,
    anchor_manifest_hash: str,
) -> dict:
    return {
        "protocol": "CanonicalDiCodeTrainingBatchPlan",
        "dynamic_task_ids": list(dynamic_task_ids),
        "non_target_anchor_ids": list(non_target_anchor_ids),
        "curriculum_task_ids": list(curriculum_task_ids),
        "target_task_id": target_task_id,
        "target_probability": target_probability,
        "selection_attestation_hash": selection_attestation_hash,
        "anchor_manifest_hash": anchor_manifest_hash,
    }


def _validate_shape(
    *,
    dynamic_task_ids: Tuple[str, ...],
    non_target_anchor_ids: Tuple[str, ...],
    curriculum_task_ids: Tuple[str, ...],
    target_task_id: str,
    target_probability: float,
    ctx: str,
) -> None:
    if len(dynamic_task_ids) != DICODE_NUM_DYNAMIC:
        raise DiCodePlanError(
            PLAN_COUNT,
            f"{ctx}: dynamic_task_ids must be exactly "
            f"{DICODE_NUM_DYNAMIC}, got {len(dynamic_task_ids)}",
        )
    if len(set(dynamic_task_ids)) != len(dynamic_task_ids):
        raise DiCodePlanError(
            PLAN_DUPLICATE,
            f"{ctx}: duplicate dynamic task id in the selected set",
        )
    if tuple(non_target_anchor_ids) != DICODE_NON_TARGET_ANCHOR_IDS:
        raise DiCodePlanError(
            PLAN_ANCHOR_MISMATCH,
            f"{ctx}: non_target_anchor_ids must be the shared "
            f"{list(DICODE_NON_TARGET_ANCHOR_IDS)}, got "
            f"{list(non_target_anchor_ids)}",
        )
    if len(curriculum_task_ids) != DICODE_NUM_CURRICULUM:
        raise DiCodePlanError(
            PLAN_COUNT,
            f"{ctx}: curriculum_task_ids must be exactly "
            f"{DICODE_NUM_CURRICULUM} (12 dynamic + 3 anchors), got "
            f"{len(curriculum_task_ids)}",
        )
    expected_curriculum = tuple(
        list(dynamic_task_ids) + list(non_target_anchor_ids)
    )
    if tuple(curriculum_task_ids) != expected_curriculum:
        raise DiCodePlanError(
            PLAN_COUNT,
            f"{ctx}: curriculum_task_ids must be dynamic ids followed "
            "by the non-target anchors in order",
        )
    if target_task_id != DICODE_TARGET_TASK_ID:
        raise DiCodePlanError(
            PLAN_TARGET_MISSING,
            f"{ctx}: target_task_id must be {DICODE_TARGET_TASK_ID!r}, "
            f"got {target_task_id!r}",
        )
    if target_task_id in curriculum_task_ids:
        raise DiCodePlanError(
            PLAN_TARGET_DUPLICATED,
            f"{ctx}: the OriginalTask {target_task_id!r} is NEVER part "
            "of the curriculum / never passed into sampled_task_ids; "
            "DiCode appends it exactly once",
        )
    # the target appears exactly once overall (as the target only)
    all_ids = list(curriculum_task_ids) + [target_task_id]
    if all_ids.count(target_task_id) != 1:
        raise DiCodePlanError(
            PLAN_TARGET_DUPLICATED,
            f"{ctx}: the OriginalTask appears {all_ids.count(target_task_id)} "
            "time(s); it must appear EXACTLY ONCE",
        )
    if not isinstance(target_probability, (int, float)) or isinstance(
        target_probability, bool
    ):
        raise DiCodePlanError(
            PLAN_PROBABILITY,
            f"{ctx}: target_probability must be a number, got "
            f"{target_probability!r}",
        )
    if float(target_probability) != DICODE_TARGET_PROBABILITY:
        raise DiCodePlanError(
            PLAN_PROBABILITY,
            f"{ctx}: target_probability must be the frozen "
            f"{DICODE_TARGET_PROBABILITY} (conf/dicode_manager "
            f"original_task_proportion), got {target_probability}",
        )


def build_canonical_dicode_training_batch_plan(
    *,
    selection_attestation: Any,
    anchor_manifest_hash: str,
    ctx: str,
    non_target_anchor_ids: Optional[Tuple[str, ...]] = None,
) -> CanonicalDiCodeTrainingBatchPlan:
    """Translate ONE attested selection into the shared 15+1 plan."""
    from .selection_attestation import SelectionAttestation

    if not isinstance(selection_attestation, SelectionAttestation):
        raise DiCodePlanError(
            PLAN_BAD_TYPE,
            f"{ctx}: selection_attestation must be a "
            f"SelectionAttestation, got {type(selection_attestation).__name__}",
        )
    dynamic_task_ids = tuple(selection_attestation.selected_ids)
    non_target = (
        tuple(non_target_anchor_ids)
        if non_target_anchor_ids is not None
        else DICODE_NON_TARGET_ANCHOR_IDS
    )
    curriculum_task_ids = tuple(list(dynamic_task_ids) + list(non_target))
    _validate_shape(
        dynamic_task_ids=dynamic_task_ids,
        non_target_anchor_ids=non_target,
        curriculum_task_ids=curriculum_task_ids,
        target_task_id=DICODE_TARGET_TASK_ID,
        target_probability=DICODE_TARGET_PROBABILITY,
        ctx=ctx,
    )
    selection_hash = _require_sha64(
        selection_attestation.attestation_hash,
        "selection_attestation_hash",
        ctx,
    )
    anchor_hash = _require_sha64(
        anchor_manifest_hash, "anchor_manifest_hash", ctx
    )
    plan_hash = canonical_sha256(
        _plan_payload(
            dynamic_task_ids=dynamic_task_ids,
            non_target_anchor_ids=non_target,
            curriculum_task_ids=curriculum_task_ids,
            target_task_id=DICODE_TARGET_TASK_ID,
            target_probability=DICODE_TARGET_PROBABILITY,
            selection_attestation_hash=selection_hash,
            anchor_manifest_hash=anchor_hash,
        )
    )
    return CanonicalDiCodeTrainingBatchPlan(
        dynamic_task_ids=dynamic_task_ids,
        non_target_anchor_ids=non_target,
        curriculum_task_ids=curriculum_task_ids,
        target_task_id=DICODE_TARGET_TASK_ID,
        target_probability=DICODE_TARGET_PROBABILITY,
        selection_attestation_hash=selection_hash,
        anchor_manifest_hash=anchor_hash,
        plan_hash=plan_hash,
    )


def verify_canonical_dicode_training_batch_plan(
    plan: Any,
    *,
    selection_attestation: Any,
    anchor_manifest_hash: str,
    ctx: str,
) -> None:
    """Re-derive the plan fail-closed against its sources."""
    from .selection_attestation import SelectionAttestation

    if not isinstance(plan, CanonicalDiCodeTrainingBatchPlan):
        raise DiCodePlanError(
            PLAN_BAD_TYPE,
            f"{ctx}: expected a CanonicalDiCodeTrainingBatchPlan, got "
            f"{type(plan).__name__}",
        )
    if not isinstance(selection_attestation, SelectionAttestation):
        raise DiCodePlanError(
            PLAN_BAD_TYPE,
            f"{ctx}: selection_attestation must be a "
            f"SelectionAttestation, got {type(selection_attestation).__name__}",
        )
    _validate_shape(
        dynamic_task_ids=plan.dynamic_task_ids,
        non_target_anchor_ids=plan.non_target_anchor_ids,
        curriculum_task_ids=plan.curriculum_task_ids,
        target_task_id=plan.target_task_id,
        target_probability=plan.target_probability,
        ctx=ctx,
    )
    if (
        plan.selection_attestation_hash
        != selection_attestation.attestation_hash
    ):
        raise DiCodePlanError(
            PLAN_BINDING_MISMATCH,
            f"{ctx}: plan binds selection attestation "
            f"{plan.selection_attestation_hash!r} but the attestation "
            f"is {selection_attestation.attestation_hash!r}",
        )
    if plan.anchor_manifest_hash != anchor_manifest_hash:
        raise DiCodePlanError(
            PLAN_BINDING_MISMATCH,
            f"{ctx}: plan binds anchor manifest "
            f"{plan.anchor_manifest_hash!r} but the live manifest is "
            f"{anchor_manifest_hash!r}",
        )
    recomputed = canonical_sha256(
        _plan_payload(
            dynamic_task_ids=plan.dynamic_task_ids,
            non_target_anchor_ids=plan.non_target_anchor_ids,
            curriculum_task_ids=plan.curriculum_task_ids,
            target_task_id=plan.target_task_id,
            target_probability=plan.target_probability,
            selection_attestation_hash=plan.selection_attestation_hash,
            anchor_manifest_hash=plan.anchor_manifest_hash,
        )
    )
    if recomputed != plan.plan_hash:
        raise DiCodePlanError(
            PLAN_HASH_MISMATCH,
            f"{ctx}: plan_hash {plan.plan_hash!r} != recomputed "
            f"{recomputed!r} (tampered plan)",
        )
