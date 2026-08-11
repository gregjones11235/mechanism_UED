"""Static-LLM-UED V1 contract schemas and provenance enum (design contract S4-S7).

Pure standard library: no pydantic, no jax, no craftax. Every validator is a
pure, deterministic, FAIL-CLOSED function that raises ``SchemaError`` with a
greppable ``.code`` instead of silently coercing malformed LLM output.

Two concerns live here:

1. ``Provenance`` — the data-admissibility contract that keeps formal
   evaluation data (FORMAL_FRONT / FORMAL_BACK / FORMAL_FULL) out of the
   teacher. Only TRAINING and NORMAL_TRAINING_FEEDBACK data may ever reach the
   BehaviorDiagnostician / CurriculumDesigner / selector / archive-priority
   updates. Unknown or missing provenance fails closed.

2. Structured role outputs — frozen-dataclass contracts plus ``parse_*``
   functions for the BehaviorDiagnostician (``Diagnosis``), the
   CurriculumDesigner (``InterventionPlan``) and the env-code planner
   (``EnvTemplate``). Limits follow the design contract:
   <=3 weaknesses, <=3 hypotheses per weakness, <=6 hypotheses total,
   <=8 intervention families, <=3 axis changes per family, <=2 explorations.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple

# ---------------------------------------------------------------------------
# Schema versions (part of every plan-cache key; bump on any contract change)
# ---------------------------------------------------------------------------
DIAGNOSIS_SCHEMA_VERSION = "static_llm_ued.diagnosis.v1"
INTERVENTION_PLAN_SCHEMA_VERSION = "static_llm_ued.intervention_plan.v1"
ENV_TEMPLATE_SCHEMA_VERSION = "static_llm_ued.env_template.v1"

# ---------------------------------------------------------------------------
# Contract limits (design contract S5/S7)
# ---------------------------------------------------------------------------
MAX_WEAKNESSES = 3
MAX_HYPOTHESES_PER_WEAKNESS = 3
MAX_TOTAL_HYPOTHESES = 6
MAX_INTERVENTION_FAMILIES = 8
MAX_AXIS_CHANGES_PER_FAMILY = 3
MAX_EXPLORATION_PROPOSALS = 2

#: Machine-readable limit manifest (auditable; mirrored by the validators).
DIAGNOSIS_SCHEMA = {
    "schema_version": DIAGNOSIS_SCHEMA_VERSION,
    "max_weaknesses": MAX_WEAKNESSES,
    "max_hypotheses_per_weakness": MAX_HYPOTHESES_PER_WEAKNESS,
    "max_total_hypotheses": MAX_TOTAL_HYPOTHESES,
    "required_fields": (
        "weaknesses",
        "hypotheses",
        "reuse_previous_direction",
        "overall_confidence",
    ),
}
INTERVENTION_PLAN_SCHEMA = {
    "schema_version": INTERVENTION_PLAN_SCHEMA_VERSION,
    "max_families": MAX_INTERVENTION_FAMILIES,
    "max_axis_changes_per_family": MAX_AXIS_CHANGES_PER_FAMILY,
    "max_explorations": MAX_EXPLORATION_PROPOSALS,
    "required_fields": ("families", "explorations"),
}
ENV_TEMPLATE_SCHEMA = {
    "schema_version": ENV_TEMPLATE_SCHEMA_VERSION,
    "required_fields": ("template_id", "family_id", "task_description"),
    "optional_fields": ("code_constraints", "example_task_ids"),
}


# ---------------------------------------------------------------------------
# Provenance (design contract S4)
# ---------------------------------------------------------------------------
class Provenance(str, Enum):
    """Origin class of any data offered to the static teacher."""

    #: metrics written by the normal PPO training loop (archive history)
    TRAINING = "TRAINING"
    #: per-session training-window metrics + original-task skill SRs extracted
    #: from the TRAINING WINDOW (this is NOT a formal evaluation)
    NORMAL_TRAINING_FEEDBACK = "NORMAL_TRAINING_FEEDBACK"
    #: formal FRONT evaluation output — NEVER admissible as teacher evidence
    FORMAL_FRONT = "FORMAL_FRONT"
    #: formal BACK evaluation output — NEVER admissible as teacher evidence
    FORMAL_BACK = "FORMAL_BACK"
    #: formal FULL evaluation output — NEVER admissible as teacher evidence
    FORMAL_FULL = "FORMAL_FULL"


ADMISSIBLE_TEACHER_PROVENANCES = frozenset(
    {Provenance.TRAINING, Provenance.NORMAL_TRAINING_FEEDBACK}
)
FORMAL_PROVENANCES = frozenset(
    {Provenance.FORMAL_FRONT, Provenance.FORMAL_BACK, Provenance.FORMAL_FULL}
)


class SchemaError(Exception):
    """Fail-closed contract violation carrying a greppable ``code``."""

    # provenance codes
    PROVENANCE_MISSING = "PROVENANCE_MISSING"
    UNKNOWN_PROVENANCE = "UNKNOWN_PROVENANCE"
    FORMAL_PROVENANCE_REJECTED = "FORMAL_PROVENANCE_REJECTED"
    # generic field codes
    MISSING_FIELD = "MISSING_FIELD"
    EMPTY_FIELD = "EMPTY_FIELD"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    BAD_TYPE = "BAD_TYPE"
    DUPLICATE_ID = "DUPLICATE_ID"
    # diagnosis codes
    TOO_MANY_WEAKNESSES = "TOO_MANY_WEAKNESSES"
    TOO_MANY_HYPOTHESES_PER_WEAKNESS = "TOO_MANY_HYPOTHESES_PER_WEAKNESS"
    TOO_MANY_TOTAL_HYPOTHESES = "TOO_MANY_TOTAL_HYPOTHESES"
    ORPHAN_HYPOTHESIS = "ORPHAN_HYPOTHESIS"
    CONFIDENCE_OUT_OF_RANGE = "CONFIDENCE_OUT_OF_RANGE"
    BAD_PRIORITY = "BAD_PRIORITY"
    # intervention plan codes
    TOO_MANY_FAMILIES = "TOO_MANY_FAMILIES"
    TOO_MANY_AXIS_CHANGES = "TOO_MANY_AXIS_CHANGES"
    TOO_MANY_EXPLORATIONS = "TOO_MANY_EXPLORATIONS"
    AXIS_OVERLAP = "AXIS_OVERLAP"
    NO_FAMILIES = "NO_FAMILIES"
    UNKNOWN_FAMILY = "UNKNOWN_FAMILY"
    BAD_TASK_ID = "BAD_TASK_ID"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


# ---------------------------------------------------------------------------
# Provenance helpers (fail-closed)
# ---------------------------------------------------------------------------
def parse_provenance(value: Any) -> Provenance:
    """Parses a provenance label; missing/unknown values fail closed."""
    if value is None:
        raise SchemaError(
            SchemaError.PROVENANCE_MISSING,
            "data offered to the static teacher carries no provenance label",
        )
    if isinstance(value, Provenance):
        return value
    try:
        return Provenance(value)
    except (ValueError, TypeError) as e:
        raise SchemaError(
            SchemaError.UNKNOWN_PROVENANCE,
            f"unknown provenance label {value!r}; refusing fail-closed",
        ) from e


def is_admissible_provenance(value: Any) -> bool:
    """True only for provenances the teacher may consume."""
    try:
        return parse_provenance(value) in ADMISSIBLE_TEACHER_PROVENANCES
    except SchemaError:
        return False


def assert_admissible_provenance(value: Any, context: str) -> Provenance:
    """Fail-closed admissibility gate for teacher evidence.

    Returns the parsed ``Provenance`` when admissible; raises ``SchemaError``
    with a greppable code for missing, unknown, or FORMAL provenance.
    """
    provenance = parse_provenance(value)
    if provenance in FORMAL_PROVENANCES:
        raise SchemaError(
            SchemaError.FORMAL_PROVENANCE_REJECTED,
            f"{context}: formal evaluation data ({provenance.value}) must never "
            f"enter the static teacher; formal evaluation is read-only final "
            f"judgement only",
        )
    if provenance not in ADMISSIBLE_TEACHER_PROVENANCES:
        raise SchemaError(
            SchemaError.UNKNOWN_PROVENANCE,
            f"{context}: provenance {provenance.value} is not admissible",
        )
    return provenance


# ---------------------------------------------------------------------------
# Small strict field helpers
# ---------------------------------------------------------------------------
def _require_mapping(obj: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise SchemaError(
            SchemaError.BAD_TYPE, f"{context}: expected a mapping, got {type(obj).__name__}"
        )
    return obj


def _check_unknown_fields(
    obj: Mapping[str, Any], allowed: Sequence[str], context: str
) -> None:
    allowed_set = set(allowed)
    unknown = sorted(k for k in obj if k not in allowed_set)
    if unknown:
        raise SchemaError(
            SchemaError.UNKNOWN_FIELD, f"{context}: unknown field(s) {unknown}"
        )


def _require_nonempty_str(obj: Mapping[str, Any], key: str, context: str) -> str:
    if key not in obj:
        raise SchemaError(SchemaError.MISSING_FIELD, f"{context}: missing field {key!r}")
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(
            SchemaError.EMPTY_FIELD,
            f"{context}: field {key!r} must be a non-empty string, got {value!r}",
        )
    return value.strip()


def _require_str_tuple(
    obj: Mapping[str, Any], key: str, context: str, *, default: Tuple[str, ...] = ()
) -> Tuple[str, ...]:
    if key not in obj:
        return default
    value = obj[key]
    if not isinstance(value, (list, tuple)):
        raise SchemaError(
            SchemaError.BAD_TYPE, f"{context}: field {key!r} must be a list of strings"
        )
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SchemaError(
                SchemaError.EMPTY_FIELD,
                f"{context}: field {key!r}[{i}] must be a non-empty string",
            )
        out.append(item.strip())
    return tuple(out)


def _require_bool(obj: Mapping[str, Any], key: str, context: str) -> bool:
    if key not in obj:
        raise SchemaError(SchemaError.MISSING_FIELD, f"{context}: missing field {key!r}")
    value = obj[key]
    if not isinstance(value, bool):
        raise SchemaError(
            SchemaError.BAD_TYPE, f"{context}: field {key!r} must be a bool, got {value!r}"
        )
    return value


def _check_unique_ids(ids: Sequence[str], context: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise SchemaError(SchemaError.DUPLICATE_ID, f"{context}: duplicate id {i!r}")
        seen.add(i)


_TASK_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Diagnosis (BehaviorDiagnostician output)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Weakness:
    weakness_id: str
    name: str
    evidence_refs: Tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    weakness_id: str
    statement: str


@dataclass(frozen=True)
class Diagnosis:
    weaknesses: Tuple[Weakness, ...]
    hypotheses: Tuple[Hypothesis, ...]
    reuse_previous_direction: bool
    overall_confidence: float
    schema_version: str = DIAGNOSIS_SCHEMA_VERSION


def validate_diagnosis(diagnosis: Diagnosis) -> Diagnosis:
    """Validates a ``Diagnosis``; raises ``SchemaError`` fail-closed."""
    if not diagnosis.weaknesses:
        raise SchemaError(
            SchemaError.EMPTY_FIELD,
            "diagnosis: at least one weakness is required (reuse the previous "
            "direction instead of emitting an empty diagnosis)",
        )
    if len(diagnosis.weaknesses) > MAX_WEAKNESSES:
        raise SchemaError(
            SchemaError.TOO_MANY_WEAKNESSES,
            f"diagnosis: {len(diagnosis.weaknesses)} weaknesses > {MAX_WEAKNESSES}",
        )
    _check_unique_ids([w.weakness_id for w in diagnosis.weaknesses], "diagnosis.weaknesses")
    for w in diagnosis.weaknesses:
        if not (1 <= w.priority <= MAX_WEAKNESSES):
            raise SchemaError(
                SchemaError.BAD_PRIORITY,
                f"diagnosis: weakness {w.weakness_id} priority {w.priority} outside "
                f"[1, {MAX_WEAKNESSES}]",
            )

    if len(diagnosis.hypotheses) > MAX_TOTAL_HYPOTHESES:
        raise SchemaError(
            SchemaError.TOO_MANY_TOTAL_HYPOTHESES,
            f"diagnosis: {len(diagnosis.hypotheses)} hypotheses > {MAX_TOTAL_HYPOTHESES}",
        )
    _check_unique_ids([h.hypothesis_id for h in diagnosis.hypotheses], "diagnosis.hypotheses")

    weakness_ids = {w.weakness_id for w in diagnosis.weaknesses}
    per_weakness: dict[str, int] = {}
    for h in diagnosis.hypotheses:
        if h.weakness_id not in weakness_ids:
            raise SchemaError(
                SchemaError.ORPHAN_HYPOTHESIS,
                f"diagnosis: hypothesis {h.hypothesis_id} references unknown weakness "
                f"{h.weakness_id!r}",
            )
        per_weakness[h.weakness_id] = per_weakness.get(h.weakness_id, 0) + 1
    for weakness_id, count in sorted(per_weakness.items()):
        if count > MAX_HYPOTHESES_PER_WEAKNESS:
            raise SchemaError(
                SchemaError.TOO_MANY_HYPOTHESES_PER_WEAKNESS,
                f"diagnosis: weakness {weakness_id} has {count} hypotheses > "
                f"{MAX_HYPOTHESES_PER_WEAKNESS}",
            )

    if not (
        isinstance(diagnosis.overall_confidence, float)
        and math.isfinite(diagnosis.overall_confidence)
        and 0.0 <= diagnosis.overall_confidence <= 1.0
    ):
        raise SchemaError(
            SchemaError.CONFIDENCE_OUT_OF_RANGE,
            f"diagnosis: overall_confidence {diagnosis.overall_confidence!r} not in [0, 1]",
        )
    return diagnosis


def parse_diagnosis(obj: Any) -> Diagnosis:
    """Parses raw (LLM-produced) JSON into a validated ``Diagnosis``."""
    obj = _require_mapping(obj, "diagnosis")
    _check_unknown_fields(
        obj,
        (
            "weaknesses",
            "hypotheses",
            "reuse_previous_direction",
            "overall_confidence",
            "schema_version",
        ),
        "diagnosis",
    )
    if "weaknesses" not in obj or not isinstance(obj["weaknesses"], (list, tuple)):
        raise SchemaError(SchemaError.MISSING_FIELD, "diagnosis: missing list field 'weaknesses'")
    if "hypotheses" not in obj or not isinstance(obj["hypotheses"], (list, tuple)):
        raise SchemaError(SchemaError.MISSING_FIELD, "diagnosis: missing list field 'hypotheses'")

    weaknesses = []
    for i, w in enumerate(obj["weaknesses"]):
        w = _require_mapping(w, f"diagnosis.weaknesses[{i}]")
        _check_unknown_fields(
            w, ("weakness_id", "name", "evidence_refs", "priority"),
            f"diagnosis.weaknesses[{i}]",
        )
        weaknesses.append(
            Weakness(
                weakness_id=_require_nonempty_str(w, "weakness_id", f"diagnosis.weaknesses[{i}]"),
                name=_require_nonempty_str(w, "name", f"diagnosis.weaknesses[{i}]"),
                evidence_refs=_require_str_tuple(w, "evidence_refs", f"diagnosis.weaknesses[{i}]"),
                priority=int(w["priority"]) if isinstance(w.get("priority"), int) and not
                isinstance(w.get("priority"), bool) else _bad_priority(i, w.get("priority")),
            )
        )

    hypotheses = []
    for i, h in enumerate(obj["hypotheses"]):
        h = _require_mapping(h, f"diagnosis.hypotheses[{i}]")
        _check_unknown_fields(
            h, ("hypothesis_id", "weakness_id", "statement"), f"diagnosis.hypotheses[{i}]"
        )
        hypotheses.append(
            Hypothesis(
                hypothesis_id=_require_nonempty_str(
                    h, "hypothesis_id", f"diagnosis.hypotheses[{i}]"
                ),
                weakness_id=_require_nonempty_str(
                    h, "weakness_id", f"diagnosis.hypotheses[{i}]"
                ),
                statement=_require_nonempty_str(h, "statement", f"diagnosis.hypotheses[{i}]"),
            )
        )

    confidence = obj.get("overall_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise SchemaError(
            SchemaError.BAD_TYPE,
            f"diagnosis: overall_confidence must be a number, got {confidence!r}",
        )

    return validate_diagnosis(
        Diagnosis(
            weaknesses=tuple(weaknesses),
            hypotheses=tuple(hypotheses),
            reuse_previous_direction=_require_bool(
                obj, "reuse_previous_direction", "diagnosis"
            ),
            overall_confidence=float(confidence),
        )
    )


def _bad_priority(index: int, value: Any) -> int:
    raise SchemaError(
        SchemaError.BAD_TYPE,
        f"diagnosis.weaknesses[{index}]: priority must be an int, got {value!r}",
    )


# ---------------------------------------------------------------------------
# InterventionPlan (CurriculumDesigner output)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisChange:
    axis: str
    from_value: str
    to_value: str


@dataclass(frozen=True)
class InterventionFamily:
    family_id: str
    description: str
    target_achievements: Tuple[str, ...]
    axis_changes: Tuple[AxisChange, ...]
    constant_axes: Tuple[str, ...]
    scaffolding: str
    student_must_do: str


@dataclass(frozen=True)
class ExplorationProposal:
    proposal_id: str
    description: str
    axis_changes: Tuple[AxisChange, ...]


@dataclass(frozen=True)
class InterventionPlan:
    families: Tuple[InterventionFamily, ...]
    explorations: Tuple[ExplorationProposal, ...]
    schema_version: str = INTERVENTION_PLAN_SCHEMA_VERSION


def _parse_axis_changes(raw: Any, context: str) -> Tuple[AxisChange, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise SchemaError(
            SchemaError.BAD_TYPE, f"{context}: axis_changes must be a list"
        )
    if len(raw) > MAX_AXIS_CHANGES_PER_FAMILY:
        raise SchemaError(
            SchemaError.TOO_MANY_AXIS_CHANGES,
            f"{context}: {len(raw)} axis changes > {MAX_AXIS_CHANGES_PER_FAMILY}",
        )
    changes = []
    for i, c in enumerate(raw):
        c = _require_mapping(c, f"{context}.axis_changes[{i}]")
        _check_unknown_fields(c, ("axis", "from_value", "to_value"), f"{context}.axis_changes[{i}]")
        changes.append(
            AxisChange(
                axis=_require_nonempty_str(c, "axis", f"{context}.axis_changes[{i}]"),
                from_value=_require_nonempty_str(c, "from_value", f"{context}.axis_changes[{i}]"),
                to_value=_require_nonempty_str(c, "to_value", f"{context}.axis_changes[{i}]"),
            )
        )
    return tuple(changes)


def validate_intervention_plan(plan: InterventionPlan) -> InterventionPlan:
    """Validates an ``InterventionPlan``; raises ``SchemaError`` fail-closed."""
    if not plan.families:
        raise SchemaError(
            SchemaError.NO_FAMILIES,
            "intervention plan: at least one intervention family is required",
        )
    if len(plan.families) > MAX_INTERVENTION_FAMILIES:
        raise SchemaError(
            SchemaError.TOO_MANY_FAMILIES,
            f"intervention plan: {len(plan.families)} families > {MAX_INTERVENTION_FAMILIES}",
        )
    if len(plan.explorations) > MAX_EXPLORATION_PROPOSALS:
        raise SchemaError(
            SchemaError.TOO_MANY_EXPLORATIONS,
            f"intervention plan: {len(plan.explorations)} explorations > "
            f"{MAX_EXPLORATION_PROPOSALS}",
        )
    _check_unique_ids([f.family_id for f in plan.families], "intervention_plan.families")
    _check_unique_ids(
        [e.proposal_id for e in plan.explorations], "intervention_plan.explorations"
    )
    for f in plan.families:
        changed = {c.axis for c in f.axis_changes}
        overlap = changed & set(f.constant_axes)
        if overlap:
            raise SchemaError(
                SchemaError.AXIS_OVERLAP,
                f"intervention plan: family {f.family_id} lists axes both changed and "
                f"constant: {sorted(overlap)}",
            )
    return plan


def parse_intervention_plan(obj: Any) -> InterventionPlan:
    """Parses raw (LLM-produced) JSON into a validated ``InterventionPlan``."""
    obj = _require_mapping(obj, "intervention_plan")
    _check_unknown_fields(
        obj, ("families", "explorations", "schema_version"), "intervention_plan"
    )
    if "families" not in obj or not isinstance(obj["families"], (list, tuple)):
        raise SchemaError(
            SchemaError.MISSING_FIELD, "intervention_plan: missing list field 'families'"
        )

    families = []
    for i, f in enumerate(obj["families"]):
        f = _require_mapping(f, f"intervention_plan.families[{i}]")
        _check_unknown_fields(
            f,
            (
                "family_id",
                "description",
                "target_achievements",
                "axis_changes",
                "constant_axes",
                "scaffolding",
                "student_must_do",
            ),
            f"intervention_plan.families[{i}]",
        )
        ctx = f"intervention_plan.families[{i}]"
        families.append(
            InterventionFamily(
                family_id=_require_nonempty_str(f, "family_id", ctx),
                description=_require_nonempty_str(f, "description", ctx),
                target_achievements=_require_str_tuple(f, "target_achievements", ctx),
                axis_changes=_parse_axis_changes(f.get("axis_changes"), ctx),
                constant_axes=_require_str_tuple(f, "constant_axes", ctx),
                scaffolding=_require_nonempty_str(f, "scaffolding", ctx),
                student_must_do=_require_nonempty_str(f, "student_must_do", ctx),
            )
        )

    explorations = []
    for i, e in enumerate(obj.get("explorations", [])):
        e = _require_mapping(e, f"intervention_plan.explorations[{i}]")
        _check_unknown_fields(
            e, ("proposal_id", "description", "axis_changes"),
            f"intervention_plan.explorations[{i}]",
        )
        ctx = f"intervention_plan.explorations[{i}]"
        explorations.append(
            ExplorationProposal(
                proposal_id=_require_nonempty_str(e, "proposal_id", ctx),
                description=_require_nonempty_str(e, "description", ctx),
                axis_changes=_parse_axis_changes(e.get("axis_changes"), ctx),
            )
        )

    return validate_intervention_plan(
        InterventionPlan(families=tuple(families), explorations=tuple(explorations))
    )


# ---------------------------------------------------------------------------
# EnvTemplate (per-candidate authoring instruction for the env coder)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvTemplate:
    template_id: str
    family_id: str
    task_description: str
    code_constraints: Tuple[str, ...] = ()
    example_task_ids: Tuple[str, ...] = ()
    schema_version: str = ENV_TEMPLATE_SCHEMA_VERSION


def validate_env_template(template: EnvTemplate) -> EnvTemplate:
    for tid in template.example_task_ids:
        if not _TASK_ID_RE.match(tid):
            raise SchemaError(
                SchemaError.BAD_TASK_ID,
                f"env template {template.template_id}: example task id {tid!r} is not "
                f"a legal task identifier",
            )
    return template


def parse_env_template(obj: Any) -> EnvTemplate:
    """Parses raw JSON into a validated ``EnvTemplate``."""
    obj = _require_mapping(obj, "env_template")
    _check_unknown_fields(
        obj,
        (
            "template_id",
            "family_id",
            "task_description",
            "code_constraints",
            "example_task_ids",
            "schema_version",
        ),
        "env_template",
    )
    return validate_env_template(
        EnvTemplate(
            template_id=_require_nonempty_str(obj, "template_id", "env_template"),
            family_id=_require_nonempty_str(obj, "family_id", "env_template"),
            task_description=_require_nonempty_str(obj, "task_description", "env_template"),
            code_constraints=_require_str_tuple(obj, "code_constraints", "env_template"),
            example_task_ids=_require_str_tuple(obj, "example_task_ids", "env_template"),
        )
    )
