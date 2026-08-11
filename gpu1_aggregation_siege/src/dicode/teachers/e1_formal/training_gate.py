"""C13 fail-closed training gate between the E1 teacher batch and
``run_session_training``.

Supervisor REQUEST_CHANGES fix: a blocked E1 teacher used to return an
anchors-only batch that the legacy loop then trained anyway. That was a
sneak path: while the shared anchor manifest is DRAFT or no real
Student/Reference dual probes exist, NOTHING may train — zero PPO
updates, zero global/env-step progress.

This module is the second line of defense (the teacher itself never
marks a blocked batch ``training_permitted``; this gate re-verifies the
batch structure independently, so even a buggy/forged "permitted"
batch cannot reach training):

* ``training_permitted`` must be the LITERAL ``True`` — any other
  value (missing, truthy string, 1) is blocked;
* a permitted batch must be EXACTLY 12 dynamic + 4 shared anchors in
  canonical order — an anchors-only batch, a wrong anchor set, a
  duplicate id, or a missing/non-covering layout is refused;
* a blocked batch carries its ``blocked_codes`` through the exception
  so the loop can report them.

Pure standard library; fail-closed with greppable codes. NO jax, NO
craftax, NO I/O.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from . import layout
from .anchor_manifest import NUM_SHARED_ANCHORS
from .schemas import E1SchemaError

#: the batch was not explicitly permitted => zero training updates
TRAINING_GATE_BLOCKED = "TRAINING_GATE_BLOCKED"
#: a "permitted" batch failed the structural re-check (sneak attempt)
TRAINING_GATE_BAD_BATCH = "TRAINING_GATE_BAD_BATCH"
#: gates are clear but no verified previous-window batch exists
TRAINING_BLOCKED_NO_VERIFIED_BATCH = "TRAINING_BLOCKED_NO_VERIFIED_BATCH"

_EXPECTED_BATCH_SIZE = layout.NUM_DYNAMIC_SLOTS + NUM_SHARED_ANCHORS


class TrainingGateError(E1SchemaError):
    """Fail-closed training-gate violation; ``code`` is greppable.

    ``codes`` carries the batch's own blocked codes (when present) so
    the caller can report WHY training was refused.
    """

    def __init__(
        self, code: str, detail: str, codes: Tuple[str, ...] = ()
    ) -> None:
        super().__init__(code, detail)
        self.codes = tuple(codes)


def _blocked_codes_of(e1_batch: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = e1_batch.get("blocked_codes")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        item for item in raw if isinstance(item, str) and item.strip()
    )


def enforce_training_gate(e1_batch: Any) -> List[str]:
    """Return the 16 task ids ONLY if training is explicitly permitted.

    Raises ``TrainingGateError`` otherwise — the caller must NOT invoke
    ``run_session_training`` on any raise path. This is the guarantee
    "blocked hard gate => zero training updates, zero step progress".
    """
    ctx = "e1_formal.training_gate"
    if not isinstance(e1_batch, Mapping):
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: batch must be a mapping, got "
            f"{type(e1_batch).__name__}",
        )
    if e1_batch.get("training_permitted") is not True:
        codes = _blocked_codes_of(e1_batch)
        raise TrainingGateError(
            TRAINING_GATE_BLOCKED,
            f"{ctx}: teacher batch is not training_permitted; zero "
            "training updates this session (blocked codes: "
            f"{list(codes) or ['unspecified']})",
            codes=codes,
        )

    # --- permitted: re-verify the structure independently -------------
    raw_ids = e1_batch.get("task_ids")
    if not isinstance(raw_ids, (list, tuple)):
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: permitted batch needs a task_ids sequence, got "
            f"{type(raw_ids).__name__}",
        )
    ids = list(raw_ids)
    if len(ids) != _EXPECTED_BATCH_SIZE:
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: a permitted batch must be exactly "
            f"{layout.NUM_DYNAMIC_SLOTS} dynamic + {NUM_SHARED_ANCHORS} "
            f"shared anchors = {_EXPECTED_BATCH_SIZE} tasks, got "
            f"{len(ids)}. An anchors-only batch is a sneak path and is "
            "refused.",
        )
    dynamic_ids = ids[: layout.NUM_DYNAMIC_SLOTS]
    anchor_ids = ids[layout.NUM_DYNAMIC_SLOTS :]
    if tuple(anchor_ids) != layout.ANCHOR_TASK_IDS:
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: trailing anchors {anchor_ids} != canonical "
            f"{list(layout.ANCHOR_TASK_IDS)} (original_craftax last)",
        )
    for i, task_id in enumerate(dynamic_ids):
        if not isinstance(task_id, str) or not task_id.strip():
            raise TrainingGateError(
                TRAINING_GATE_BAD_BATCH,
                f"{ctx}: dynamic slot {i} must be a non-empty str, got "
                f"{task_id!r}",
            )
    if len(set(dynamic_ids)) != len(dynamic_ids):
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: duplicate dynamic task ids in {dynamic_ids}",
        )
    overlap = sorted(set(dynamic_ids) & set(layout.ANCHOR_TASK_IDS))
    if overlap:
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: dynamic ids overlap the shared anchors: {overlap}",
        )

    layout_map = e1_batch.get("layout")
    if not isinstance(layout_map, Mapping):
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: a permitted batch must carry its pinned layout "
            f"mapping, got {type(layout_map).__name__}",
        )
    if set(layout_map) != set(ids):
        raise TrainingGateError(
            TRAINING_GATE_BAD_BATCH,
            f"{ctx}: layout keys must cover exactly the 16 batch tasks; "
            "a legacy distribution may never substitute the pinned "
            "layout of a permitted batch",
        )
    return ids
