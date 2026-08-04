"""E1 formal provenance tiers, error base and review-window invariants.

Reuses (never modifies) ``dicode.teachers.static_llm.schemas``. Adds:

* the E1 provenance tiering (design contract G1-G5 / section seven):
  - LLM_ROLE_ADMISSIBLE: the ONLY labels that may enter any prompt
    (TRAINING, NORMAL_TRAINING_FEEDBACK);
  - SELECTOR_ADMISSIBLE: LLM set + CANDIDATE_EVALUATION (candidate
    evaluation enters ONLY the Copeland selection side, never an LLM);
  - FORMAL_* rejected at BOTH tiers fail-closed (double layer together
    with the F5 content scanner);
* ``E1SchemaError`` — fail-closed error base carrying greppable codes.
"""
from __future__ import annotations

from typing import Any

from ..static_llm.schemas import Provenance, SchemaError

#: provenance label stamped on the candidate-evaluation seam (stage 6).
#: Defined here (not in the committed static_llm enum, which is frozen).
PROVENANCE_CANDIDATE_EVALUATION = "CANDIDATE_EVALUATION"

#: The ONLY provenances that may enter any LLM prompt.
LLM_ROLE_ADMISSIBLE_PROVENANCES = frozenset(
    {
        Provenance.TRAINING.value,
        Provenance.NORMAL_TRAINING_FEEDBACK.value,
    }
)

#: Candidate evaluation is admissible ONLY on the selection side.
SELECTOR_ADMISSIBLE_PROVENANCES = LLM_ROLE_ADMISSIBLE_PROVENANCES | {
    PROVENANCE_CANDIDATE_EVALUATION
}

FORMAL_PROVENANCE_LABELS = frozenset(
    {
        Provenance.FORMAL_FRONT.value,
        Provenance.FORMAL_BACK.value,
        Provenance.FORMAL_FULL.value,
    }
)

_ALL_KNOWN_LABELS = (
    LLM_ROLE_ADMISSIBLE_PROVENANCES
    | SELECTOR_ADMISSIBLE_PROVENANCES
    | FORMAL_PROVENANCE_LABELS
)


class E1SchemaError(SchemaError):
    """Fail-closed E1 contract violation; ``code`` is greppable."""


class E1Code:
    """Greppable E1 fail-closed codes (module-level registry)."""

    # provenance tiering
    LLM_PROVENANCE_VIOLATION = "LLM_PROVENANCE_VIOLATION"
    SELECTOR_PROVENANCE_VIOLATION = "SELECTOR_PROVENANCE_VIOLATION"
    # canonical encoding
    CANONICAL_UNSUPPORTED_TYPE = "CANONICAL_UNSUPPORTED_TYPE"
    # json extraction from LLM text
    JSON_NOT_FOUND = "JSON_NOT_FOUND"
    JSON_PARSE_FAILED = "JSON_PARSE_FAILED"
    # flags / manifest
    FLAG_MANIFEST_MISMATCH = "FLAG_MANIFEST_MISMATCH"
    FLAGS_BAD_TYPE = "FLAGS_BAD_TYPE"
    FLAGS_UNKNOWN_FIELD = "FLAGS_UNKNOWN_FIELD"
    FLAGS_MISSING_FIELD = "FLAGS_MISSING_FIELD"
    # review window (stage 2)
    INCOMPLETE_REVIEW_WINDOW = "INCOMPLETE_REVIEW_WINDOW"
    LLM_BUDGET_EXCEEDED = "LLM_BUDGET_EXCEEDED"
    ALL_FAMILIES_VETOED = "ALL_FAMILIES_VETOED"
    # dynamic-slot integrity (stage 3/4, round-3 P0-2): fewer than 12
    # real compiled dynamic artifacts => the whole window is refused;
    # stub/placeholder slot padding is structurally absent.
    INSUFFICIENT_DYNAMIC_ARTIFACTS = "INSUFFICIENT_DYNAMIC_ARTIFACTS"
    # invocation gate / selection degradation (G2/G3)
    SELECTION_BLOCKED_NO_REAL_EVIDENCE = "SELECTION_BLOCKED_NO_REAL_EVIDENCE"


def normalize_e1_provenance(value: Any, context: str) -> str:
    """Normalize a provenance value to its canonical label, fail-closed."""
    if value is None:
        raise E1SchemaError(
            SchemaError.PROVENANCE_MISSING,
            f"{context}: data offered to E1 carries no provenance label",
        )
    label = value.value if isinstance(value, Provenance) else value
    if not isinstance(label, str) or label not in _ALL_KNOWN_LABELS:
        raise E1SchemaError(
            SchemaError.UNKNOWN_PROVENANCE,
            f"{context}: unknown provenance label {value!r}; refusing fail-closed",
        )
    return label


def assert_llm_role_admissible(value: Any, context: str) -> str:
    """Return the label only if it may enter an LLM prompt (fail-closed)."""
    label = normalize_e1_provenance(value, context)
    if label in FORMAL_PROVENANCE_LABELS:
        raise E1SchemaError(
            SchemaError.FORMAL_PROVENANCE_REJECTED,
            f"{context}: formal evaluation data ({label}) must never enter "
            "an E1 LLM prompt; formal evaluation is read-only final "
            "judgement only",
        )
    if label not in LLM_ROLE_ADMISSIBLE_PROVENANCES:
        raise E1SchemaError(
            E1Code.LLM_PROVENANCE_VIOLATION,
            f"{context}: provenance {label!r} is not LLM-role admissible "
            f"(allowed: {sorted(LLM_ROLE_ADMISSIBLE_PROVENANCES)})",
        )
    return label


def assert_selector_admissible(value: Any, context: str) -> str:
    """Return the label only if it may enter the Copeland selection side."""
    label = normalize_e1_provenance(value, context)
    if label in FORMAL_PROVENANCE_LABELS:
        raise E1SchemaError(
            SchemaError.FORMAL_PROVENANCE_REJECTED,
            f"{context}: formal evaluation data ({label}) must never enter "
            "the E1 selector; formal evaluation is read-only final "
            "judgement only",
        )
    if label not in SELECTOR_ADMISSIBLE_PROVENANCES:
        raise E1SchemaError(
            E1Code.SELECTOR_PROVENANCE_VIOLATION,
            f"{context}: provenance {label!r} is not selector-admissible "
            f"(allowed: {sorted(SELECTOR_ADMISSIBLE_PROVENANCES)})",
        )
    return label
