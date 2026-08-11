"""Stage 9 (part): 12 dynamic + 4 shared-anchor training layout (D8).

Batch structure (supervisor standing direction): 12 dynamic candidates
plus the 4 registered standard-reset anchors ``[task_1, task_2,
task_3, original_craftax]`` (original ALWAYS last; anchors enter the
batch as registered — the teacher never modifies them).

Weight split with anchor mass ``beta`` and original share ``s`` of the
anchor mass::

    w_dynamic  = (1 - beta) / 12        (each of the 12 dynamic slots)
    w_original = beta * s               (original_craftax)
    w_seed     = beta * (1 - s) / 3     (each of task_1/2/3)

``beta`` and ``s`` are PINNED this round (plan D8): beta = 1/4,
s = 2/5, giving the exact rational weights 1/16 (= 0.0625), 1/10 and
1/20 per slot, which sum to EXACTLY 1 (checked with exact integer
arithmetic, never float tolerance). Deviating from the pinned
constants fails closed — there is no alternative layout this round.

``legacy_distribution_mirror`` is a pure-python BYTE MIRROR of the
legacy ``training._calculate_task_distribution`` expression structure,
used by the byte-identity equivalence tests (C11); the legacy path in
``training.py`` itself is left verbatim.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Sequence

from .schemas import E1SchemaError

#: registered standard-reset anchors; original_craftax ALWAYS last
ANCHOR_TASK_IDS = ("task_1", "task_2", "task_3", "original_craftax")
ORIGINAL_ANCHOR_TASK_ID = "original_craftax"

#: pinned layout constants (plan D8)
ANCHOR_MASS = 0.25       # beta: total mass reserved for the 4 anchors
ORIGINAL_SHARE = 0.4     # s: share of anchor mass going to original_craftax
NUM_DYNAMIC_SLOTS = 12

#: exact rational form of the pinned constants (binary floats 0.25 and
#: 0.4 are NOT exact rationals; the weights are derived from THESE)
_PINNED_ANCHOR_MASS = Fraction(1, 4)
_PINNED_ORIGINAL_SHARE = Fraction(2, 5)

# fail-closed codes
LAYOUT_BAD_TYPE = "LAYOUT_BAD_TYPE"
LAYOUT_BAD_DYNAMIC_SET = "LAYOUT_BAD_DYNAMIC_SET"
LAYOUT_DUPLICATE_TASK = "LAYOUT_DUPLICATE_TASK"
LAYOUT_WEIGHTS_INVALID = "LAYOUT_WEIGHTS_INVALID"


class LayoutError(E1SchemaError):
    """Fail-closed layout violation; ``code`` is greppable."""


def _pinned_rational_weights() -> Dict[str, Fraction]:
    """Exact rational weights in canonical order (original last)."""
    beta = _PINNED_ANCHOR_MASS
    share = _PINNED_ORIGINAL_SHARE
    w_dynamic = (1 - beta) / NUM_DYNAMIC_SLOTS
    w_original = beta * share
    w_seed = beta * (1 - share) / 3
    weights: Dict[str, Fraction] = {}
    for anchor_id in ANCHOR_TASK_IDS:
        weights[anchor_id] = (
            w_original if anchor_id == ORIGINAL_ANCHOR_TASK_ID else w_seed
        )
    weights["__dynamic__"] = w_dynamic
    return weights


def anchor_weights() -> Dict[str, float]:
    """Per-anchor float weights in canonical order (original last)."""
    rational = _pinned_rational_weights()
    return {
        anchor_id: float(rational[anchor_id])
        for anchor_id in ANCHOR_TASK_IDS
    }


def build_training_layout(
    dynamic_task_ids: Sequence[str],
    *,
    anchor_mass: float = ANCHOR_MASS,
    original_share: float = ORIGINAL_SHARE,
) -> Dict[str, float]:
    """Deterministic 12 dynamic + 4 anchor weight layout.

    Requires EXACTLY ``NUM_DYNAMIC_SLOTS`` unique non-empty dynamic ids
    (a blocked batch trains nothing and never reaches this function —
    C13; there is no anchors-only trainable batch) and the PINNED
    layout constants. Returns an ordered mapping dynamic... -> task_1 ->
    task_2 -> task_3 -> original_craftax whose weights are strictly
    positive and sum to exactly 1.
    """
    if isinstance(anchor_mass, bool) or not isinstance(
        anchor_mass, (int, float)
    ):
        raise LayoutError(LAYOUT_BAD_TYPE, "anchor_mass must be a number")
    if isinstance(original_share, bool) or not isinstance(
        original_share, (int, float)
    ):
        raise LayoutError(LAYOUT_BAD_TYPE, "original_share must be a number")
    if float(anchor_mass) != ANCHOR_MASS or float(original_share) != ORIGINAL_SHARE:
        raise LayoutError(
            LAYOUT_WEIGHTS_INVALID,
            "only the pinned layout constants are admissible this round "
            f"(anchor_mass={ANCHOR_MASS}, original_share={ORIGINAL_SHARE});"
            f" got anchor_mass={anchor_mass!r}, "
            f"original_share={original_share!r}",
        )
    if not isinstance(dynamic_task_ids, (list, tuple)):
        raise LayoutError(
            LAYOUT_BAD_DYNAMIC_SET,
            "dynamic_task_ids must be a sequence of task ids",
        )
    if len(dynamic_task_ids) != NUM_DYNAMIC_SLOTS:
        raise LayoutError(
            LAYOUT_BAD_DYNAMIC_SET,
            f"layout requires exactly {NUM_DYNAMIC_SLOTS} dynamic task "
            f"ids, got {len(dynamic_task_ids)}",
        )
    cleaned = []
    for task_id in dynamic_task_ids:
        if not isinstance(task_id, str) or not task_id.strip():
            raise LayoutError(
                LAYOUT_BAD_DYNAMIC_SET,
                f"dynamic task id must be non-empty str, got {task_id!r}",
            )
        cleaned.append(task_id.strip())
    if len(set(cleaned)) != len(cleaned):
        raise LayoutError(
            LAYOUT_DUPLICATE_TASK, f"duplicate dynamic task id in {cleaned}"
        )
    overlap = sorted(set(cleaned) & set(ANCHOR_TASK_IDS))
    if overlap:
        raise LayoutError(
            LAYOUT_DUPLICATE_TASK,
            f"dynamic task ids collide with anchor ids: {overlap}",
        )

    rational = _pinned_rational_weights()
    w_dynamic = float(rational["__dynamic__"])
    layout: Dict[str, float] = {tid: w_dynamic for tid in cleaned}
    layout.update(anchor_weights())

    # exact-integer sanity: strictly positive and sum EXACTLY 1
    if any(value <= 0.0 for value in layout.values()):
        raise LayoutError(
            LAYOUT_WEIGHTS_INVALID, "layout weight must be strictly positive"
        )
    total = sum(
        rational["__dynamic__"] for _ in cleaned
    ) + sum(rational[anchor_id] for anchor_id in ANCHOR_TASK_IDS)
    if total != Fraction(1):
        raise LayoutError(
            LAYOUT_WEIGHTS_INVALID,
            "layout weights must sum to exactly 1 (exact arithmetic)",
        )
    return layout


def legacy_distribution_mirror(
    num_curriculum_tasks: int, original_proportion: float
) -> list:
    """BYTE MIRROR of legacy ``training._calculate_task_distribution``.

    Mirrors the legacy expression structure verbatim in pure python
    (``jnp.concatenate([full(n, other), [original]])`` then normalize)
    so byte-identity tests can prove the E1 teacher changes nothing on
    the legacy path. Do NOT call from the E1 runtime path.
    """
    if num_curriculum_tasks > 0:
        other_proportion = (1.0 - original_proportion) / num_curriculum_tasks
        proportions = [other_proportion] * num_curriculum_tasks + [
            original_proportion
        ]
    else:
        proportions = [1.0]
    total = sum(proportions)
    return [p / total for p in proportions]
