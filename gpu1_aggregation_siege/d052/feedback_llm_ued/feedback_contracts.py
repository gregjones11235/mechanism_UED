"""Shared canonical contracts for the feedback loop.

Everything that crosses a module boundary is a ``CanonicalModel``
(extra=forbid, protocol_version pinned) so a forged / legacy / mis-shaped
record is a hard error, never a silent coercion. Hashing reuses
``d052.bagr_ued.hashing.canonical_sha256`` (single source of truth).

The LLM roles consume/emit JSON folded under bounded context markers (same
pattern as the BA-BAGR-UED board) so the deterministic mock backend and any
future real backend parse an identical block.
"""
from __future__ import annotations

import json
from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel, validate_sha256_hex

#: markers delimiting the machine-readable context block inside a role prompt
CONTEXT_OPEN = "<<<FEEDBACK_LLM_UED_CONTEXT_JSON_V1_OPEN>>>"
CONTEXT_CLOSE = "<<<FEEDBACK_LLM_UED_CONTEXT_JSON_V1_CLOSE>>>"


class CandidateEnvironment(CanonicalModel):
    """A legal, mock-namespaced, environment-level TaskParams CANDIDATE.

    Field set == MOCK_TASKPARAMS_FIELD_WHITELIST minus candidate_hash
    (computed). extra=forbid makes field invention a hard error; every
    mutation axis must be a legal environment-induction knob.
    """

    candidate_id: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    axis_values: Dict[str, str] = Field(default_factory=dict)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    variant_id: str = Field(min_length=1)
    variant_kind: str = Field(min_length=1)
    mutation_axes: List[str] = Field(default_factory=list)
    distinguishes_hypothesis_ids: List[str] = Field(default_factory=list)
    provenance: Dict[str, object] = Field(default_factory=dict)
    real_adapter_status: str = C.REAL_SIMULATOR_PROBE_STATUS
    legality_hint: str = ("MOCK_ONLY — convert through the real TaskParams "
                          "adapter once unblocked; do not execute directly")
    candidate_hash: str = ""

    @model_validator(mode="after")
    def _whitelist_and_hash(self) -> "CandidateEnvironment":
        allowed = set(C.MOCK_TASKPARAMS_FIELD_WHITELIST)
        extra = set(self.model_dump()) - allowed
        if extra:
            raise ValueError(
                f"UNAUTHORIZED_CANDIDATE_FIELD: {sorted(extra)} — real "
                f"TaskParams fields are UNKNOWN "
                f"(REAL_SIMULATOR_PROBE={C.REAL_SIMULATOR_PROBE_STATUS}); "
                f"guessing them is forbidden")
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        for a in self.mutation_axes:
            if a not in C.MUTATION_AXES:
                raise ValueError(f"ILLEGAL_CANDIDATE_AXIS: {a!r}")
        if not self.candidate_hash:
            payload = self.model_dump()
            payload.pop("candidate_hash", None)
            object.__setattr__(self, "candidate_hash",
                               canonical_sha256(payload))
        return self


class ProbeMetrics(CanonicalModel):
    """Coarse EPISODE-LEVEL statistics from one probe stage.

    Deliberately only the allowed Reference/Student aggregate signals — no
    action sequence / trajectory / hidden state / logits may be expressed in
    this type (extra=forbid). This is the shape the comparator, ledger and
    LLM roles ever see.
    """

    stage: str = Field(min_length=1)                    # "fast" | "full"
    student_success_rate: float = Field(ge=0.0, le=1.0)
    student_behavior_activation: float = Field(ge=0.0, le=1.0)
    student_front_progress: float = Field(ge=0.0, le=1.0)
    reference_success_rate: float = Field(ge=0.0, le=1.0)
    reference_mean_progress: float = Field(ge=0.0, le=1.0)
    reference_behavior_activation: float = Field(ge=0.0, le=1.0)
    global_retention: float = Field(ge=0.0, le=1.0)
    regret: float = Field(ge=0.0)
    learnability: float = Field(ge=0.0, le=1.0)
    simulator_transitions: int = Field(ge=0)
    too_hard: bool = False
    too_easy: bool = False
    probe_source: str = C.SOURCE_CANDIDATE_PROBE

    @model_validator(mode="after")
    def _source_allowed(self) -> "ProbeMetrics":
        if self.probe_source not in C.ALLOWED_LOOP_SOURCES:
            raise ValueError(
                f"PROBE_SOURCE_NOT_ALLOWED: {self.probe_source!r}")
        return self


class FamilyAllocation(CanonicalModel):
    """One environment family's slot/budget in a curriculum plan."""

    environment_family: str = Field(min_length=1)
    slots: int = Field(ge=0)
    decision: str = Field(min_length=1)
    based_on_feedback_ids: List[str] = Field(default_factory=list)
    reason: str = Field(default="")
    is_exploration: bool = False

    @model_validator(mode="after")
    def _decision_legal(self) -> "FamilyAllocation":
        if self.decision not in C.DESIGNER_DECISIONS:
            raise ValueError(f"ILLEGAL_DECISION: {self.decision!r}")
        return self


class CurriculumPlan(CanonicalModel):
    """plan_k / plan_{k+1}: the designer's environment-level curriculum."""

    plan_id: str = Field(min_length=1)
    window: int = Field(ge=0)
    previous_plan_id: str = ""
    mode: str = Field(min_length=1)
    allocations: List[FamilyAllocation] = Field(default_factory=list)
    retained_families: List[str] = Field(default_factory=list)
    mutated_families: List[str] = Field(default_factory=list)
    retired_families: List[str] = Field(default_factory=list)
    explored_families: List[str] = Field(default_factory=list)
    plan_hash: str = ""

    @model_validator(mode="after")
    def _hash(self) -> "CurriculumPlan":
        if self.mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {self.mode!r}")
        if not self.plan_hash:
            payload = self.model_dump()
            payload.pop("plan_hash", None)
            object.__setattr__(self, "plan_hash", canonical_sha256(payload))
        return self

    def signature(self) -> Dict[str, object]:
        """A comparable, order-independent fingerprint of the plan (used for
        the normal-vs-shuffled plan-difference metric)."""
        return dict(
            allocations=sorted(
                [dict(family=a.environment_family, slots=a.slots,
                      decision=a.decision) for a in self.allocations],
                key=lambda d: d["family"]),
            retired=sorted(self.retired_families),
            explored=sorted(self.explored_families))


def plan_signature_hash(plan: CurriculumPlan) -> str:
    return canonical_sha256(plan.signature())


def bind_hash(payload: Dict[str, object]) -> str:
    """Content hash used to bind LLM request/response + provenance records."""
    return canonical_sha256(payload)


def build_role_prompt(body: str, context: Dict[str, object]) -> str:
    """Wrap the machine-readable context block in the bounded markers.

    ``body`` is the role's natural-language instruction; the context JSON is
    folded between CONTEXT_OPEN/CONTEXT_CLOSE so the mock (and any future real)
    backend parses an identical block. Deterministic key ordering.
    """
    context_json = json.dumps(context, sort_keys=True, ensure_ascii=False,
                              default=str)
    return f"{body}\n{CONTEXT_OPEN}\n{context_json}\n{CONTEXT_CLOSE}\n"


def extract_context(prompt: str) -> Dict[str, object]:
    """Pull the machine-readable context block back out of a role prompt."""
    start = prompt.find(CONTEXT_OPEN)
    end = prompt.find(CONTEXT_CLOSE)
    if start < 0 or end < 0 or end < start:
        raise ValueError("MISSING_CONTEXT_BLOCK: prompt carries no "
                         "FEEDBACK_LLM_UED_CONTEXT_JSON_V1 block")
    return json.loads(prompt[start + len(CONTEXT_OPEN):end].strip())


class FeedbackRoleEnvelope(CanonicalModel):
    """Identity binding for one feedback-loop LLM invocation (audit-grade)."""

    role: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    window: int = Field(ge=0)
    sequence: int = Field(ge=0)
    request_hash: str
    response_hash: str
    raw_response: str
    parsed_json: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _hashes(self) -> "FeedbackRoleEnvelope":
        validate_sha256_hex(self.request_hash, "request_hash")
        validate_sha256_hex(self.response_hash, "response_hash")
        return self

    @staticmethod
    def make(*, role: str, prompt_version: str, backend_id: str, model_id: str,
             window: int, sequence: int, prompt: str, raw_response: str,
             parsed_dump: dict) -> "FeedbackRoleEnvelope":
        return FeedbackRoleEnvelope(
            role=role, prompt_version=prompt_version, backend_id=backend_id,
            model_id=model_id, window=window, sequence=sequence,
            request_hash=canonical_sha256(
                {"role": role, "prompt_version": prompt_version,
                 "prompt": prompt}),
            response_hash=text_sha256(raw_response),
            raw_response=raw_response, parsed_json=parsed_dump)


__all__ = [
    "CONTEXT_OPEN", "CONTEXT_CLOSE", "CandidateEnvironment", "ProbeMetrics",
    "FamilyAllocation", "CurriculumPlan", "plan_signature_hash", "bind_hash",
    "validate_sha256_hex", "build_role_prompt", "extract_context",
    "FeedbackRoleEnvelope",
]
