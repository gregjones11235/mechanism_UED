"""Trajectory evidence boundary + symbolic adapter (task section 3).

Training trajectories are READ-ONLY EVIDENCE. This module defines:

  * the admissible evidence envelope (what a trajectory may carry into the
    review board): symbolic actions + action semantic CLASSES, symbolic
    state-change summaries, atomic env events, limited anomaly windows —
    NEVER raw Craftax action integers or raw state leaf indices resolved
    inside this package (those must be resolved by an EXTERNAL symbolic
    adapter; the mock adapter below validates that contract);
  * the evidence source enum with the ALLOWED / FORBIDDEN split consumed by
    FormalEvaluationLeakageGuard (formal FRONT/BACK/FULL, frozen bank, and
    evaluation-certificate private state are all rejected fail-closed);
  * span / bundle schemas every downstream role consumes by ID, never by
    raw payload.

Forbidden inputs rejected here (fail-closed codes):
  RAW_STATE_LEAF_INDEX_FORBIDDEN   — a state field keyed by a numeric leaf index
  RAW_ACTION_INT_UNRESOLVED        — an action given only as a bare integer with
                                     no symbolic vocabulary resolution
  FORBIDDEN_PAYLOAD_KEY            — a frozen-state / bank payload key present
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.schemas.common import CanonicalModel

StateValue = Union[str, int, float, bool]

#: state field names that would betray a raw frozen-state / bank payload
_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "frozen_state", "bank_state", "front_state_payload", "back_state_payload",
    "full_state_payload", "private_state", "evaluation_certificate",
    "front_bank_states", "back_bank_states", "expert_action_sequence",
})

_LEAF_INDEX_KEY = re.compile(r"^(state\[|leaf_|idx_|index_|i\d+$|\d+$)")


class EvidenceSource(str, Enum):
    """Provenance of trajectory evidence. ALLOWED vs FORBIDDEN is a hard split."""

    GENERATIVE_TRAINING_ENV = C.SOURCE_GENERATIVE_TRAINING_ENV        # ALLOWED
    SYNTHETIC_TEST_TRACE = C.SOURCE_SYNTHETIC_TEST_TRACE              # ALLOWED
    FORMAL_FRONT = C.SOURCE_FORMAL_FRONT                              # FORBIDDEN
    FORMAL_BACK = C.SOURCE_FORMAL_BACK                                # FORBIDDEN
    FORMAL_FULL = C.SOURCE_FORMAL_FULL                                # FORBIDDEN
    FROZEN_BANK = C.SOURCE_FROZEN_BANK                                # FORBIDDEN
    FORMAL_EVALUATION_CERTIFICATE_PRIVATE_STATE = \
        C.SOURCE_FORMAL_CERT_PRIVATE_STATE                            # FORBIDDEN

    @property
    def admissible(self) -> bool:
        return self.value in C.ALLOWED_EVIDENCE_SOURCES


class TrajectoryEvidenceError(Exception):
    RAW_STATE_LEAF_INDEX_FORBIDDEN = "RAW_STATE_LEAF_INDEX_FORBIDDEN"
    RAW_ACTION_INT_UNRESOLVED = "RAW_ACTION_INT_UNRESOLVED"
    FORBIDDEN_PAYLOAD_KEY = "FORBIDDEN_PAYLOAD_KEY"
    UNKNOWN_SYMBOLIC_ACTION = "UNKNOWN_SYMBOLIC_ACTION"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class StepRecord(CanonicalModel):
    """One training step, fully SYMBOLIC (no raw ints/leaf indices).

    ``state_summary`` carries only symbolic change summaries (distance BANDS,
    need BANDS, progress ordinals, flags) — the external adapter discretizes
    raw state BEFORE it reaches this package.
    """

    step_index: int = Field(ge=0)
    symbolic_action: str = Field(min_length=1)
    #: semantic classes of the action (e.g. rest_class, combat_class) — detectors
    #: key off CLASSES, never off action integers
    action_semantic_classes: List[str] = Field(default_factory=list)
    state_summary: Dict[str, StateValue] = Field(default_factory=dict)
    #: atomic env events at/after this step (damage_taken, chased, died,
    #: timeout, no_effect, achievement_<name>, ...) — deterministic labels
    env_events: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_raw_state(self) -> "StepRecord":
        for k in self.state_summary:
            kl = k.lower()
            if kl in _FORBIDDEN_PAYLOAD_KEYS:
                raise TrajectoryEvidenceError(
                    TrajectoryEvidenceError.FORBIDDEN_PAYLOAD_KEY,
                    f"step {self.step_index}: state_summary key {k!r} is a "
                    f"forbidden frozen-state/bank payload key")
            if _LEAF_INDEX_KEY.match(kl):
                raise TrajectoryEvidenceError(
                    TrajectoryEvidenceError.RAW_STATE_LEAF_INDEX_FORBIDDEN,
                    f"step {self.step_index}: state_summary key {k!r} looks "
                    f"like a raw state leaf index; only symbolic summary "
                    f"fields may enter the review board")
        return self


class EpisodeEvidence(CanonicalModel):
    """One episode of current-Student generative-training evidence."""

    episode_id: str = Field(min_length=1)
    source: EvidenceSource
    steps: List[StepRecord] = Field(default_factory=list)
    #: death / timeout / success / abandoned / None(unknown)
    outcome: Optional[str] = None
    meta: Dict[str, StateValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _source_admissible(self) -> "EpisodeEvidence":
        if not self.source.admissible:
            # FormalEvaluationLeakageGuard raises with the specific source; this
            # is the schema-level backstop so a forbidden episode can never even
            # be constructed inside the package.
            from d052.bagr_ued.formal_evaluation_leakage_guard import (
                FormalEvaluationLeakageGuard)
            FormalEvaluationLeakageGuard().assert_admissible_source(self.source)
        return self


class EvidenceSpan(CanonicalModel):
    """A limited window [start_step, end_step] inside one episode."""

    episode_id: str = Field(min_length=1)
    start_step: int = Field(ge=0)
    end_step: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "EvidenceSpan":
        if self.end_step < self.start_step:
            raise ValueError(
                f"SPAN_INVERTED: end_step={self.end_step} < "
                f"start_step={self.start_step}")
        return self

    @property
    def span_hash_payload(self) -> dict:
        return {"episode_id": self.episode_id, "start_step": self.start_step,
                "end_step": self.end_step}


class TrajectoryEvidenceBundle(CanonicalModel):
    """A batch of episodes admitted as read-only evidence for one review window."""

    bundle_id: str = Field(min_length=1)
    source: EvidenceSource
    symbolic_adapter_version: str = Field(min_length=1)
    episodes: List[EpisodeEvidence] = Field(default_factory=list)
    #: set by the adapter: FormalEvaluationLeakageGuard verdict on intake
    leakage_guard_status: str = "NOT_CHECKED"

    @model_validator(mode="after")
    def _consistent_source(self) -> "TrajectoryEvidenceBundle":
        for ep in self.episodes:
            if ep.source is not self.source:
                raise ValueError(
                    f"SOURCE_MIX_FORBIDDEN: bundle source {self.source.value} "
                    f"but episode {ep.episode_id} has {ep.source.value}")
        return self

    def episode(self, episode_id: str) -> EpisodeEvidence:
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return ep
        raise KeyError(f"UNKNOWN_EPISODE: {episode_id}")


# ---------------------------------------------------------------------------
# Symbolic adapter: raw -> symbolic resolution is EXTERNAL to this package.
# Detectors consume only the symbolic contract below.
# ---------------------------------------------------------------------------

class ActionVocabulary(CanonicalModel):
    """Externally-supplied action semantics. No integer is hardcoded in package
    code: the (raw int -> symbolic name + semantic classes) table is DATA owned
    by the caller (the real adapter is an external dependency)."""

    version: str = Field(min_length=1)
    #: symbolic action name -> semantic classes (rest_class, combat_class, ...)
    semantics: Dict[str, List[str]] = Field(min_length=1)
    #: optional raw-int resolution table (raw int -> symbolic name); the mock
    #: adapter uses it only when a caller hands over raw ints
    raw_int_map: Dict[int, str] = Field(default_factory=dict)

    def classes_of(self, symbolic_action: str) -> List[str]:
        if symbolic_action not in self.semantics:
            raise TrajectoryEvidenceError(
                TrajectoryEvidenceError.UNKNOWN_SYMBOLIC_ACTION,
                f"symbolic action {symbolic_action!r} not in vocabulary "
                f"version={self.version}")
        return list(self.semantics[symbolic_action])


class MockSymbolicAdapter:
    """Mock external symbolic adapter (architecture tests only).

    Enforces: (a) raw actions must resolve through the injected vocabulary —
    a bare int with no vocabulary entry fails closed; (b) raw state keys must
    be symbolic names — numeric leaf-index keys fail closed.
    """

    version = "mock_symbolic_adapter.v1"

    def __init__(self, vocabulary: ActionVocabulary) -> None:
        self.vocabulary = vocabulary

    def resolve_action(self, raw_action: Union[int, str]) -> "tuple[str, List[str]]":
        if isinstance(raw_action, str):
            name = raw_action
        else:
            name = self.vocabulary.raw_int_map.get(int(raw_action))
            if name is None:
                raise TrajectoryEvidenceError(
                    TrajectoryEvidenceError.RAW_ACTION_INT_UNRESOLVED,
                    f"raw action int {raw_action} has no symbolic resolution "
                    f"in vocabulary {self.vocabulary.version}; hardcoded "
                    f"Craftax action integers are forbidden")
        return name, self.vocabulary.classes_of(name)

    def summarize_state(self, raw_state: Dict[str, StateValue]) -> Dict[str, StateValue]:
        for k in raw_state:
            if _LEAF_INDEX_KEY.match(k.lower()):
                raise TrajectoryEvidenceError(
                    TrajectoryEvidenceError.RAW_STATE_LEAF_INDEX_FORBIDDEN,
                    f"raw state key {k!r} looks like a state leaf index; the "
                    f"external adapter must discretize to symbolic summary "
                    f"fields before evidence enters this package")
            if k.lower() in _FORBIDDEN_PAYLOAD_KEYS:
                raise TrajectoryEvidenceError(
                    TrajectoryEvidenceError.FORBIDDEN_PAYLOAD_KEY,
                    f"raw state key {k!r} is a forbidden frozen-state/bank "
                    f"payload key")
        return dict(raw_state)
