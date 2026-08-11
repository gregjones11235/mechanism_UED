"""Stage 3: canonical TaskTemplate / TaskSpec compiler.

Compiles the SURVIVING intervention families of a COMPLETE review
window into canonical task artifacts (round-3 P0-2):

* one FAMILY is one TEMPLATE: ``template_hash`` is the canonical
  sha256 of the family CONTENT (variant-independent); the EnvCoder is
  invoked ONCE per unique template and its artifact identity is
  ``template_artifact_id = {template_hash}::tpl`` (this is what the
  K1 counter dedupes);
* each template expands deterministically into ``variants_per_spec``
  TaskSpecs; variant content is derived WITHOUT any LLM by
  ``derive_variant_params`` (exact rational levels via ``Fraction``);
* ``spec_id = {window_id}::{family_id}::v{variant}``;
* every spec is bound to the ``window_hash`` that produced it AND to
  its ``template_hash``;
* goal achievements are canonicalized against the official craftax-67
  ``REGISTRY`` — the ONE sanctioned d052 import in the E1 runtime
  (pure stdlib; unknown targets fail closed, no fuzzy matching);
* ``spec_hash`` = canonical sha256 of {window_hash, template_hash,
  variant, variant_params} — variants are therefore content-distinct;
  ``artifact_id = {spec_hash}::v{variant}`` identifies the compiled
  per-variant artifact.

Deterministic scale limits: <= ``MAX_WINDOW_TEMPLATES`` (10) unique
family templates per window, each with K = ``DEFAULT_VARIANTS_PER_SPEC``
(2) variants => <= ``MAX_WINDOW_SPEC_POOL`` (20) specs per window.
Overflow is truncated deterministically in family-major order and
recorded with notes — never silent, never padded with stubs. A window
whose compiled artifact count stays below the 12 dynamic slots is
refused WHOLESALE downstream (INSUFFICIENT_DYNAMIC_ARTIFACTS).

Honest scale note: the committed board contract (static_llm
``MAX_INTERVENTION_FAMILIES``) caps a REAL window at 8 families, so on
the production path the compiler sees <= 8 templates / <= 16 specs;
``MAX_WINDOW_TEMPLATES`` / ``MAX_WINDOW_SPEC_POOL`` are defense-in-depth
backstops. The 12 dynamic slots stay reachable whenever a window yields
>= 6 unique families (6 x 2 variants = 12).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Mapping, Tuple

from d052.achievements import REGISTRY, AchievementError

from ..static_llm.guards import raise_if_forbidden
from .board import ReviewWindow, WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: unique family templates per window (the EnvCoder call budget)
MAX_WINDOW_TEMPLATES = 10
#: deterministic spec-pool ceiling (templates x variants)
MAX_WINDOW_SPEC_POOL = 20
DEFAULT_VARIANTS_PER_SPEC = 2

UNKNOWN_ACHIEVEMENT = "UNKNOWN_ACHIEVEMENT"
EMPTY_GOAL_SET = "EMPTY_GOAL_SET"
DEDUPED_TEMPLATE = "DEDUPED_TEMPLATE"
TEMPLATES_TRUNCATED_TO_CAP = "TEMPLATES_TRUNCATED_TO_CAP"
POOL_TRUNCATED_TO_CAP = "POOL_TRUNCATED_TO_CAP"
TASK_SPEC_VOID_WINDOW = "TASK_SPEC_VOID_WINDOW"
TASK_SPEC_BAD_AXIS_CHANGE = "TASK_SPEC_BAD_AXIS_CHANGE"


class TaskSpecError(E1SchemaError):
    """Fail-closed TaskSpec violation; ``code`` is greppable."""


@dataclass(frozen=True)
class TaskTemplate:
    """One unique family template (EnvCoder invocation identity)."""

    family_id: str
    template_hash: str
    template_artifact_id: str  # f"{template_hash}::tpl"


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
    template_hash: str  # family content identity (variant-independent)
    template_artifact_id: str  # EnvCoder/K1 identity for the template
    variant_params: Tuple[Tuple[str, str], ...]  # deterministic derivation
    spec_hash: str  # {window_hash, template_hash, variant, variant_params}
    artifact_id: str  # spec_hash + variant (compiled-artifact identity)


@dataclass(frozen=True)
class CompileResult:
    """Templates, specs and deterministic notes (dedup / truncation)."""

    templates: Tuple[TaskTemplate, ...]
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


def _template_content_hash(
    window_hash: str, family: Mapping[str, Any], targets: Tuple[str, ...]
) -> str:
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


def _require_axis_str(
    change: Mapping[str, Any], key: str, ctx: str
) -> str:
    value = change.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskSpecError(
            TASK_SPEC_BAD_AXIS_CHANGE,
            f"{ctx}: axis_changes entry field {key!r} must be a non-empty "
            f"string, got {value!r} (no defaults)",
        )
    return value.strip()


def derive_variant_params(
    family: Mapping[str, Any], variant: int, variants_per_spec: int, ctx: str
) -> Tuple[Tuple[str, str], ...]:
    """Deterministically derive variant-differentiating params (no LLM).

    For every axis change ``{axis, from_value, to_value}`` the variant
    sits at the exact rational level ``Fraction(variant, V-1)`` between
    ``from_value`` (level 0) and ``to_value`` (level 1); with a single
    variant the level is 0. A family without axis changes is
    differentiated by ``variant_index`` alone. Fails closed on any
    malformed axis-change entry.
    """
    if isinstance(variant, bool) or not isinstance(variant, int) or variant < 0:
        raise TaskSpecError(
            TASK_SPEC_BAD_AXIS_CHANGE,
            f"{ctx}: variant must be an int >= 0, got {variant!r}",
        )
    if (
        isinstance(variants_per_spec, bool)
        or not isinstance(variants_per_spec, int)
        or variants_per_spec < 1
    ):
        raise TaskSpecError(
            TASK_SPEC_BAD_AXIS_CHANGE,
            f"{ctx}: variants_per_spec must be an int >= 1, got "
            f"{variants_per_spec!r}",
        )
    axis_changes = family.get("axis_changes", ())
    if not isinstance(axis_changes, (list, tuple)):
        raise TaskSpecError(
            TASK_SPEC_BAD_AXIS_CHANGE,
            f"{ctx}: axis_changes must be a list, got "
            f"{type(axis_changes).__name__}",
        )
    if len(axis_changes) == 0:
        return (("variant_index", str(variant)),)
    level = (
        Fraction(variant, variants_per_spec - 1)
        if variants_per_spec > 1
        else Fraction(0)
    )
    params = []
    for i, change in enumerate(axis_changes):
        if not isinstance(change, Mapping):
            raise TaskSpecError(
                TASK_SPEC_BAD_AXIS_CHANGE,
                f"{ctx}: axis_changes[{i}] must be a mapping, got "
                f"{type(change).__name__}",
            )
        unknown = sorted(k for k in change if k not in {"axis", "from_value", "to_value"})
        if unknown:
            raise TaskSpecError(
                TASK_SPEC_BAD_AXIS_CHANGE,
                f"{ctx}: axis_changes[{i}] unknown field(s) {unknown}",
            )
        axis = _require_axis_str(change, "axis", f"{ctx}.axis_changes[{i}]")
        from_value = _require_axis_str(
            change, "from_value", f"{ctx}.axis_changes[{i}]"
        )
        to_value = _require_axis_str(
            change, "to_value", f"{ctx}.axis_changes[{i}]"
        )
        params.append((f"{axis}:level", str(level)))
        params.append((f"{axis}:from", from_value))
        params.append((f"{axis}:to", to_value))
    return tuple(params)


def compile_task_specs(
    window: ReviewWindow,
    *,
    variants_per_spec: int = DEFAULT_VARIANTS_PER_SPEC,
) -> CompileResult:
    """Compile surviving families of a COMPLETE window.

    Deterministic: family-major order; dedup by ``template_hash``
    (first family wins); template cap ``MAX_WINDOW_TEMPLATES`` and pool
    cap ``MAX_WINDOW_SPEC_POOL`` with recorded truncation notes.
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

    templates = []
    specs = []
    notes = []
    seen_templates = set()
    for family_id in window.surviving_families:
        family = families_by_id[family_id]
        ctx = f"window {window.window_id} family {family_id}"
        # defense in depth: board outputs were already guard-scanned
        raise_if_forbidden(dict(family), ctx)
        targets = _canonical_targets(
            tuple(family.get("target_achievements", ())), ctx
        )
        template_hash = _template_content_hash(
            window.window_hash, family, targets
        )
        template_artifact_id = f"{template_hash}::tpl"
        if template_hash in seen_templates:
            notes.append(
                {
                    "family_id": family_id,
                    "template_hash": template_hash,
                    "note": DEDUPED_TEMPLATE,
                }
            )
            continue
        if len(seen_templates) >= MAX_WINDOW_TEMPLATES:
            notes.append(
                {
                    "family_id": family_id,
                    "template_hash": template_hash,
                    "note": TEMPLATES_TRUNCATED_TO_CAP,
                }
            )
            continue
        seen_templates.add(template_hash)
        templates.append(
            TaskTemplate(
                family_id=family_id,
                template_hash=template_hash,
                template_artifact_id=template_artifact_id,
            )
        )
        for variant in range(variants_per_spec):
            if len(specs) >= MAX_WINDOW_SPEC_POOL:
                notes.append(
                    {
                        "family_id": family_id,
                        "variant": str(variant),
                        "template_hash": template_hash,
                        "note": POOL_TRUNCATED_TO_CAP,
                    }
                )
                continue
            variant_params = derive_variant_params(
                family, variant, variants_per_spec, ctx
            )
            spec_hash = canonical_sha256(
                {
                    "window_hash": window.window_hash,
                    "template_hash": template_hash,
                    "variant": variant,
                    "variant_params": [list(p) for p in variant_params],
                }
            )
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
                    template_hash=template_hash,
                    template_artifact_id=template_artifact_id,
                    variant_params=variant_params,
                    spec_hash=spec_hash,
                    artifact_id=f"{spec_hash}::v{variant}",
                )
            )
    return CompileResult(
        templates=tuple(templates), specs=tuple(specs), notes=tuple(notes)
    )
