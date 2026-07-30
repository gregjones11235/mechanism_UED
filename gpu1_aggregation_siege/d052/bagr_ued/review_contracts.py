"""Shared review-board contracts: RoleEnvelope + request/response hashing.

Every role output that enters the Reconciler is wrapped in a RoleEnvelope
binding the parsed output to: the role, its pinned prompt version, the
backend/model identity, the request+response content hashes, and a monotonic
sequence number. The Reconciler later binds each reconciled item to these
hashes (task section 10) so a replay can prove which role output, prompt, and
backend produced it.

NOTE: envelopes record identity only. LLM (mock) outputs generate CANDIDATE
hypotheses — they never directly override the selector or curriculum (that is
the Reconciler + Soft Copeland + Budget chain's deterministic job).
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.schemas.common import CanonicalModel, validate_sha256_hex

#: markers delimiting the machine-readable context block inside a role prompt
CONTEXT_OPEN = "<<<BAGR_UED_CONTEXT_JSON_V1_OPEN>>>"
CONTEXT_CLOSE = "<<<BAGR_UED_CONTEXT_JSON_V1_CLOSE>>>"


class RoleEnvelope(CanonicalModel):
    """Identity binding for one role invocation (audit-grade, replayable)."""

    role: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    request_hash: str
    response_hash: str
    raw_response: str
    parsed_json: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _hashes(self) -> "RoleEnvelope":
        validate_sha256_hex(self.request_hash, "request_hash")
        validate_sha256_hex(self.response_hash, "response_hash")
        return self

    @staticmethod
    def make(*, role: str, prompt_version: str, backend_id: str, model_id: str,
             sequence: int, prompt: str, raw_response: str,
             parsed_dump: dict) -> "RoleEnvelope":
        return RoleEnvelope(
            role=role,
            prompt_version=prompt_version,
            backend_id=backend_id,
            model_id=model_id,
            sequence=sequence,
            request_hash=canonical_sha256(
                {"role": role, "prompt_version": prompt_version, "prompt": prompt}),
            response_hash=text_sha256(raw_response),
            raw_response=raw_response,
            parsed_json=parsed_dump,
        )


class ReviewBoardOutput(CanonicalModel):
    """All envelopes of one review-board window, in the fixed role order."""

    bundle_id: str = Field(min_length=1)
    envelopes: List[RoleEnvelope] = Field(default_factory=list)
    #: TrajectorySupervisionGuard verdict over the WHOLE board output
    supervision_guard_status: str = "NOT_CHECKED"
    #: FormalEvaluationLeakageGuard verdict over the board INPUT context
    leakage_guard_status: str = "NOT_CHECKED"
    real_llm_calls: int = Field(ge=0)
