"""Unified selector interface schemas.

All selectors (baseline rungs S0/S1/S2, Soft/Budgeted Copeland, Auction raw/
budgeted) share ONE config + ONE result contract so they are comparable and
bit-for-bit replayable. Determinism is mandatory: ``seed`` is required and the
``selection_hash`` is a content hash of (selector, policy, k, seed, selected_ids),
so identical inputs reproduce identical selections.

Critic consumption policy (task §Critic 判断消费协议):
  hard_veto (default) : critic_reject=True excludes the candidate; a shortfall
                        yields INSUFFICIENT_ELIGIBLE_CANDIDATES -- NO backfill,
                        NO k-reduction, NO re-LLM.
  soft_penalty        : critic signal enters the score as a penalty.
  score_only          : critic signal recorded but not applied to eligibility.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import List, Optional

from pydantic import Field, model_validator

from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.roles import ScoringRole


class SelectorType(str, Enum):
    S0_CANONICAL_BASELINE = "S0_CANONICAL_BASELINE"   # deterministic, no LLM
    S1_THREE_ROLE = "S1_THREE_ROLE"                   # tutor/critic/explorer
    S2_FOUR_ROLE_MODELER = "S2_FOUR_ROLE_MODELER"     # + modeler
    SOFT_COPELAND = "SOFT_COPELAND"
    BUDGETED_SOFT_COPELAND = "BUDGETED_SOFT_COPELAND"
    AUCTION_RAW = "AUCTION_RAW"
    AUCTION_BUDGETED = "AUCTION_BUDGETED"


class CriticPolicy(str, Enum):
    HARD_VETO = "hard_veto"      # default
    SOFT_PENALTY = "soft_penalty"
    SCORE_ONLY = "score_only"


class SelectionStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_ELIGIBLE_CANDIDATES = "INSUFFICIENT_ELIGIBLE_CANDIDATES"


_BUDGETED = {SelectorType.BUDGETED_SOFT_COPELAND, SelectorType.AUCTION_BUDGETED}


def compute_selection_hash(selector: str, critic_policy: str, k: int, seed: int,
                           selected_ids: List[str]) -> str:
    payload = {
        "selector": selector,
        "critic_policy": critic_policy,
        "k": k,
        "seed": seed,
        "selected_ids": sorted(selected_ids),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class SelectorConfig(CanonicalModel):
    """Configuration for one selector run (shared across all selector types)."""

    selector: SelectorType
    critic_policy: CriticPolicy = CriticPolicy.HARD_VETO
    k: int = Field(gt=0)
    seed: int                       # required for determinism
    roles: List[ScoringRole] = Field(default_factory=list)
    #: required for budgeted selectors; forbidden otherwise
    budget: Optional[float] = None

    @model_validator(mode="after")
    def _validate_config(self) -> "SelectorConfig":
        if self.selector in _BUDGETED and self.budget is None:
            raise ValueError(
                f"MISSING_BUDGET: selector {self.selector.value} requires budget")
        if self.selector not in _BUDGETED and self.budget is not None:
            raise ValueError(
                f"UNEXPECTED_BUDGET: selector {self.selector.value} is not budgeted")
        if self.selector is SelectorType.S0_CANONICAL_BASELINE and self.roles:
            raise ValueError(
                "S0_NO_ROLES: S0_CANONICAL_BASELINE is deterministic and uses no "
                "LLM roles")
        if self.selector in (SelectorType.S1_THREE_ROLE,
                             SelectorType.S2_FOUR_ROLE_MODELER) and not self.roles:
            raise ValueError(
                f"MISSING_ROLES: selector {self.selector.value} requires roles")
        return self


class SelectionResult(CanonicalModel):
    """The manifest a selector emits (audit-grade, replayable)."""

    selector: SelectorType
    critic_policy: CriticPolicy
    k_requested: int = Field(gt=0)
    seed: int
    candidate_count_in: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    selected_ids: List[str]
    rejected_by_critic: List[str] = Field(default_factory=list)
    selection_status: SelectionStatus
    selection_hash: str
    #: honest note when status != OK (why shortfall; what was NOT done)
    shortfall_note: str = ""

    @model_validator(mode="after")
    def _validate_result(self) -> "SelectionResult":
        validate_sha256_hex(self.selection_hash, "selection_hash")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("DUPLICATE_SELECTION: selected_ids repeat")
        expected = compute_selection_hash(
            self.selector.value, self.critic_policy.value, self.k_requested,
            self.seed, self.selected_ids)
        if self.selection_hash != expected:
            raise ValueError(
                f"SELECTION_HASH_MISMATCH: expected {expected}, "
                f"got {self.selection_hash}")
        if self.selection_status is SelectionStatus.OK:
            if len(self.selected_ids) != self.k_requested:
                raise ValueError(
                    f"STATUS_OK_REQUIRES_K: status=OK but selected "
                    f"{len(self.selected_ids)} != k={self.k_requested}")
        else:  # INSUFFICIENT_ELIGIBLE_CANDIDATES
            if len(self.selected_ids) >= self.k_requested:
                raise ValueError(
                    "INSUFFICIENT_BUT_FULL: status=INSUFFICIENT yet selected >= k "
                    "(backfill/k-reduction is forbidden)")
            if not self.shortfall_note:
                raise ValueError(
                    "MISSING_SHORTFALL_NOTE: an INSUFFICIENT result must record why")
        if self.eligible_count > self.candidate_count_in:
            raise ValueError("ELIGIBLE_EXCEEDS_INPUT")
        return self
