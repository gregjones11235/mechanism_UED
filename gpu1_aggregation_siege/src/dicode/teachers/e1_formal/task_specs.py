"""Stage 3: canonical TaskSpec compiler.

Compiles the SURVIVING intervention families of a COMPLETE review
window into canonical TaskSpecs:

* ``spec_id = {window_id}::{family_id}::v{variant}``;
* every spec is bound to the ``window_hash`` that produced it;
* goal achievements are canonicalized against the official craftax-67
  ``REGISTRY`` — the ONE sanctioned d052 import in the E1 runtime
  (pure stdlib; unknown targets fail closed, no fuzzy matching);
* ``spec_hash`` = canonical sha256 of the spec CONTENT (dedup
  identity; variant-independent); ``artifact_id = {spec_hash}::v{v}``
  is what the EnvCoder produces and what the K1 counter dedupes.

Deterministic scale limits (plan D9): <= 8 surviving families, K = 2
variants per spec, <= 10 specs per window (overflow is truncated
deterministically in family-major order and recorded, never silent),
=> <= 20 artifacts per window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from d052.achievements import REGISTRY, AchievementError

from ..static_llm.guards import raise_if_forbidden
from .board import ReviewWindow, WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .schemas import E1SchemaError

MAX_WINDOW_SPECS = 10
DEFAULT_VARIANTS_PER_SPEC = 2

UNKNOWN_ACHIEVEMENT = "UNKNOWN_ACHIEVEMENT"
EMPTY_GOAL_SET = "EMPTY_GOAL_SET"
DEDUPED_SPEC = "DEDUPED_SPEC"
SPECS_TRUNCATED_TO_CAP = "SPECS_TRUNCATED_TO_CAP"
TASK_SPEC_VOID_WINDOW = "TASK_SPEC_VOID_WINDOW"


class TaskSpecError(E1SchemaError):
    """Fail-closed TaskSpec violation; ``code`` is greppable."""


@dataclass(frozen=True)
class TaskSpec:
    """One canonical, hash-identified task specification."""

    spec_id: str
    window_id: str
    window_hash: str
    family_id: str
    variant: int
    description: str
    target_achievements: Tuple[str, ...]  # canonical REGISTRY names, sorted
    axis_changes: Tuple[Dict[str, str], ...]
    constant_axes: Tuple[str, ...]
    scaffolding: str
    student_must_do: str
    spec_hash: str  # content identity (variant-independent)
    artifact_id: str  # spec_hash + variant (EnvCoder/K1 identity)


@dataclass(frozen=True)
class CompileResult:
    """Specs plus deterministic notes (dedup / truncation)."""

    specs: Tuple[TaskSpec, ...]
    notes: Tuple[Dict[str, str], ...]


def _canonical_targets(targets: Tuple[str, ...], ctx: str) -> Tuple[str, ...]:
    if len(targets) == 0:
        raise TaskSpecError(
            EMPTY_GOAL_SET, f"{ctx}: intervention family has an empty goal set"
        )
    canonical = set()
    for target in targets:
        try:
            canonical.add(REGISTRY.resolve(target))
        except AchievementError as e:
            raise TaskSpecError(
                getattr(e, "code", UNKNOWN_ACHIEVEMENT),
                f"{ctx}: goal achievement {target!r} is not in the official "
                "craftax-67 REGISTRY (unknown_target_policy=error; no fuzzy "
                "matching)",
            ) from e
    return tuple(sorted(canonical))


def _spec_content_hash(window_hash: str, family: Mapping[str, Any], targets: Tuple[str, ...]) -> str:
    payload = {
        "window_hash": window_hash,
        "description": family["description"],
        "target_achievements": list(targets),
        "axis_changes": [dict(c) for c in family.get("axis_changes", ())],
        "constant_axes": list(family.get("constant_axes", ())),
        "scaffolding": family["scaffolding"],
        "student_must_do": family["student_must_do"],
    }
    return canonical_sha256(payload)


def compile_task_specs(
    window: ReviewWindow,
    *,
    variants_per_spec: int = DEFAULT_VARIANTS_PER_SPEC,
) -> CompileResult:
    """Compile surviving families of a COMPLETE window into TaskSpecs.

    Deterministic: family-major order, dedup by (spec_hash, variant),
    hard cap ``MAX_WINDOW_SPECS`` with recorded truncation notes.
    """
    if not isinstance(window, ReviewWindow):
        raise TaskSpecError(
            TASK_SPEC_VOID_WINDOW,
            f"compile requires a ReviewWindow, got {type(window).__name__}",
        )
    if window.status != WINDOW_STATUS_COMPLETE:
        raise TaskSpecError(
            TASK_SPEC_VOID_WINDOW,
            f"window {window.window_id} is void ({window.void_code}); a void "
            "window can never produce task specs",
        )
    if isinstance(variants_per_spec, bool) or not isinstance(variants_per_spec, int):
        raise ValueError("variants_per_spec must be an int")
    if variants_per_spec < 1:
        raise ValueError("variants_per_spec must be >= 1")

    plan = dict(window.role_results)["intervention_tutor"]
    families_by_id = {f["family_id"]: f for f in plan["families"]}

    specs = []
    notes = []
    seen = set()
    for family_id in window.surviving_families:
        family = families_by_id[family_id]
        ctx = f"window {window.window_id} family {family_id}"
        # defense in depth: board outputs were already guard-scanned
        raise_if_forbidden(dict(family), ctx)
        targets = _canonical_targets(
            tuple(family.get("target_achievements", ())), ctx
        )
        spec_hash = _spec_content_hash(window.window_hash, family, targets)
        for variant in range(variants_per_spec):
            if (spec_hash, variant) in seen:
                notes.append(
                    {
                        "family_id": family_id,
                        "variant": str(variant),
                        "spec_hash": spec_hash,
                        "note": DEDUPED_SPEC,
                    }
                )
                continue
            if len(specs) >= MAX_WINDOW_SPECS:
                notes.append(
                    {
                        "family_id": family_id,
                        "variant": str(variant),
                        "spec_hash": spec_hash,
                        "note": SPECS_TRUNCATED_TO_CAP,
                    }
                )
                continue
            seen.add((spec_hash, variant))
            specs.append(
                TaskSpec(
                    spec_id=f"{window.window_id}::{family_id}::v{variant}",
                    window_id=window.window_id,
                    window_hash=window.window_hash,
                    family_id=family_id,
                    variant=variant,
                    description=family["description"],
                    target_achievements=targets,
                    axis_changes=tuple(
                        dict(c) for c in family.get("axis_changes", ())
                    ),
                    constant_axes=tuple(family.get("constant_axes", ())),
                    scaffolding=family["scaffolding"],
                    student_must_do=family["student_must_do"],
                    spec_hash=spec_hash,
                    artifact_id=f"{spec_hash}::v{variant}",
                )
            )
    return CompileResult(specs=tuple(specs), notes=tuple(notes))
