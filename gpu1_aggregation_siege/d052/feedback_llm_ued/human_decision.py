"""HumanDecisionArtifact — the audit-grade record of a REQUEST_CONTROL stop
(C11).

When a window's six-role board requests human control (the critic escalates
and/or the InterventionTutor proposes REQUEST_CONTROL), the loop HALTS: no
execution batch is produced, no probe runs, nothing else is applied
autonomously. The halt is recorded here as a deterministic, hash-bound
artifact:

* WHICH sources triggered the escalation (critic / tutor);
* the critic's objections and global risk;
* the tutor's REQUEST_CONTROL families and their cited feedback ids —
  resolved to REAL store ids by the controller before this artifact is
  built (under the shuffled mode the board cites anonymized ids);
* the full BoardOutput binding (``board_hash``), so a human reviewer can
  recompute every detail of the escalated window.

No wall-clock timestamps: the artifact is a pure function of the escalated
window's frozen state, so a replayed run reproduces it byte-for-byte.
"""
from __future__ import annotations

from typing import List

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel, is_sha256_hex

#: legal escalation sources (board roles whose output can halt the loop)
TRIGGER_CRITIC = C.ROLE_CRITIC_SKEPTIC
TRIGGER_INTERVENTION_TUTOR = C.ROLE_INTERVENTION_TUTOR
LEGAL_TRIGGER_SOURCES = frozenset({TRIGGER_CRITIC, TRIGGER_INTERVENTION_TUTOR})


class HumanDecisionArtifact(CanonicalModel):
    """One REQUEST_CONTROL stop, frozen for human review."""

    artifact_id: str = Field(min_length=1)
    window: int = Field(ge=0)
    mode: str
    trigger_sources: List[str] = Field(default_factory=list)
    global_risk: str
    critic_objections: List[str] = Field(default_factory=list)
    #: families of the tutor's REQUEST_CONTROL proposals (sorted)
    request_control_families: List[str] = Field(default_factory=list)
    #: REAL store ids cited by those proposals (resolved by the controller)
    cited_feedback_ids: List[str] = Field(default_factory=list)
    #: binds the complete BoardOutput of the escalated window
    board_hash: str
    reason: str = ""
    artifact_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "HumanDecisionArtifact":
        if self.mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {self.mode!r}")
        if not self.trigger_sources:
            raise ValueError(
                "EMPTY_TRIGGER_SOURCES: a REQUEST_CONTROL artifact must "
                "name at least one escalating board role")
        for source in self.trigger_sources:
            if source not in LEGAL_TRIGGER_SOURCES:
                raise ValueError(f"ILLEGAL_TRIGGER_SOURCE: {source!r}")
        if not is_sha256_hex(self.board_hash):
            raise ValueError(
                f"ILLEGAL_BOARD_HASH: {self.board_hash!r}")
        if TRIGGER_INTERVENTION_TUTOR in self.trigger_sources and \
                not self.request_control_families:
            raise ValueError(
                "TUTOR_TRIGGER_WITHOUT_PROPOSALS: the tutor triggered this "
                "escalation but no REQUEST_CONTROL family is recorded")
        expected_id = f"hda-w{self.window:02d}-{self.board_hash[:16]}"
        if self.artifact_id != expected_id:
            raise ValueError(
                f"ARTIFACT_ID_MISMATCH: {self.artifact_id!r} != "
                f"{expected_id!r}")
        if not self.artifact_hash:
            payload = self.model_dump()
            payload.pop("artifact_hash", None)
            object.__setattr__(self, "artifact_hash",
                               canonical_sha256(payload))
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("artifact_hash", None)
        return canonical_sha256(payload)
