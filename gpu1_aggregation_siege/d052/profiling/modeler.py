"""Modeler (student-state) judgment + the machine-facts / LLM-interpretation firewall.

The Modeler runs ONCE per session over the student state (not per-candidate). It is
handed FACTS, not verdicts (source: modeler.py _build_state_evidence "We hand the
modeler facts, not verdicts"). The firewall is structural:

  * ``MachineFacts`` is the deterministic input (held-out SR series, forgetting
    prefilter, snapshot count). It carries NO mastery tier labels -- tiers are a
    deterministic downstream derivation and the source mandates they are NEVER
    passed to the LLM/selector.
  * ``ModelerJudgment`` carries the LLM-produced interpretation (student_state,
    recommendation, guidance, siege foci) SEPARATELY, plus an evidence_check that
    lets downstream code age guidance to stale/contradicted.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.achievements import REGISTRY
from d052.schemas.common import CanonicalModel


class StudentState(str, Enum):
    NORMAL_EARLY = "NORMAL_EARLY"
    RISING = "RISING"
    STALLED = "STALLED"
    NOISY = "NOISY"
    FORGETTING = "FORGETTING"
    MASTERED = "MASTERED"


class Recommendation(str, Enum):
    DEPTH = "DEPTH"
    BREADTH = "BREADTH"
    CONSOLIDATE = "CONSOLIDATE"


class EvidenceCheck(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NO_EVIDENCE = "no_evidence"


class MachineFacts(CanonicalModel):
    """Deterministic facts handed to the Modeler. NO tier labels, NO verdicts."""

    latest_sr: Dict[str, float] = Field(default_factory=dict)   # canonical name -> SR
    recent_series: List[Dict[str, float]] = Field(default_factory=list)
    forgetting_prefilter: List[str] = Field(default_factory=list)  # canonical names
    num_snapshots: int = Field(ge=0)

    @model_validator(mode="after")
    def _canonical_names(self) -> "MachineFacts":
        for name in list(self.latest_sr) :
            REGISTRY.resolve(name)  # unknown -> AchievementError
        for snap in self.recent_series:
            for name in snap:
                REGISTRY.resolve(name)
        for name in self.forgetting_prefilter:
            REGISTRY.resolve(name)
        # firewall: this payload must not smuggle in tier labels
        return self


class ModelerJudgment(CanonicalModel):
    """One Modeler session output (LLM interpretation + grounding evidence_check)."""

    machine_facts: MachineFacts
    student_state: StudentState
    recommendation: Recommendation
    guidance: str = ""
    #: canonical achievements the modeler flags as siege foci
    siege_foci: List[str] = Field(default_factory=list)
    evidence_check: EvidenceCheck
    provider: Optional[str] = None
    exact_model_id: Optional[str] = None
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _validate_foci(self) -> "ModelerJudgment":
        resolved = []
        for name in self.siege_foci:
            resolved.append(REGISTRY.resolve(name))  # unknown -> error
        if len(set(resolved)) != len(resolved):
            raise ValueError("DUPLICATE_SIEGE_FOCUS")
        object.__setattr__(self, "siege_foci", sorted(set(resolved)))
        return self

    def assert_firewall(self) -> None:
        """Document/enforce that interpretation is grounded in machine_facts.

        A judgment whose evidence_check is CONTRADICTED must not be acted on as
        supported; downstream code ages such guidance to stale.
        """
        if self.evidence_check is EvidenceCheck.CONTRADICTED and self.guidance:
            # not an error, but callers must treat guidance as stale
            return
