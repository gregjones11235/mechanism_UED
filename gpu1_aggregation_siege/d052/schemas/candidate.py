"""Candidate / TaskParams / CandidatePool schemas (shared frozen pool).

Salvages the legacy D052 candidate field set -- ``task_id``, ``chash``,
``task_params``, ``target_achievements`` -- but makes it strict and deterministic:

  * ``target_achievements`` are validated against the canonical 67 (unknown ->
    error; empty -> error); max 4 targets, enforced as a HARD error (never a
    silent cap).
  * ``chash`` is a 64-char sha256 content hash over a canonical serialization of
    (task_id, sorted canonical target names, task_params). It is REQUIRED and
    VERIFIED against recomputation (HASH_MISMATCH on disagreement) so a
    candidate's identity cannot drift from its contents.
  * ``legacy_short_id`` reproduces the old ``sha256(f"{id}:{sorted(names)}")[:16]``
    scheme for traceability to voided D052 artifacts (bridge only; not an identity).
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Literal

from pydantic import Field, field_validator, model_validator

from d052.schemas.achievements import AchievementRef
from d052.schemas.common import (
    CanonicalModel,
    validate_finite,
    validate_sha256_hex,
)

#: Hard, documented cap on targets per candidate (legacy D052 used [:4]).
MAX_TARGET_ACHIEVEMENTS = 4


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_candidate_chash(task_id: str, canonical_names: List[str],
                            task_params: dict) -> str:
    """Deterministic 64-char sha256 content hash for a candidate.

    Input is a canonical JSON of {task_id, targets(sorted canonical), task_params}.
    Depends only on content, so identical candidates hash identically across runs.
    """
    payload = {
        "task_id": task_id,
        "target_achievements": sorted(canonical_names),
        "task_params": task_params,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_legacy_short_id(task_id: str, canonical_names: List[str]) -> str:
    """Bridge to the voided D052 short id: sha256(f"{id}:{sorted(names)}")[:16]."""
    s = f"{task_id}:{sorted(canonical_names)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class TaskParams(CanonicalModel):
    """The four D052 candidate task knobs (all positive, finite multipliers)."""

    passive_spawn_multiplier: float
    melee_spawn_multiplier: float
    mob_health_multiplier: float
    mob_damage_multiplier: float

    @model_validator(mode="after")
    def _check(self) -> "TaskParams":
        for fname in ("passive_spawn_multiplier", "melee_spawn_multiplier",
                      "mob_health_multiplier", "mob_damage_multiplier"):
            v = validate_finite(getattr(self, fname), fname)
            if v <= 0:
                raise ValueError(
                    f"NON_POSITIVE: {fname} must be > 0, got {v}")
        return self


class Candidate(CanonicalModel):
    """One curriculum candidate (salvaged D052 field set, made strict)."""

    task_id: str = Field(min_length=1)
    chash: str
    task_params: TaskParams
    target_achievements: List[AchievementRef] = Field(min_length=1)
    #: computed; the old short id for traceability only
    legacy_short_id: str = ""

    @field_validator("chash")
    @classmethod
    def _chash_format(cls, v: str) -> str:
        return validate_sha256_hex(v, "chash")

    @model_validator(mode="after")
    def _validate_targets_and_hash(self) -> "Candidate":
        names = [a.name for a in self.target_achievements]
        distinct = sorted(set(names))
        if len(distinct) != len(names):
            raise ValueError(
                "DUPLICATE_TARGET: target_achievements must not repeat "
                f"(got {names})")
        if len(distinct) > MAX_TARGET_ACHIEVEMENTS:
            raise ValueError(
                f"MAX_TARGETS_EXCEEDED: at most {MAX_TARGET_ACHIEVEMENTS} target "
                f"achievements (hard error, not a silent cap); got {len(distinct)}")
        expected = compute_candidate_chash(
            self.task_id, distinct, self.task_params.model_dump())
        if self.chash != expected:
            raise ValueError(
                f"HASH_MISMATCH: chash does not match deterministic content hash; "
                f"expected {expected}, got {self.chash}")
        object.__setattr__(
            self, "legacy_short_id",
            compute_legacy_short_id(self.task_id, distinct))
        return self

    @property
    def canonical_target_names(self) -> List[str]:
        return sorted({a.name for a in self.target_achievements})


class CandidatePool(CanonicalModel):
    """A shared, frozen candidate pool consumed identically by every selector.

    ``frozen`` is pinned True (candidate_pool_mode=shared_frozen). ``pool_hash``
    is REQUIRED and verified as the sha256 over the ordered list of candidate
    chashes, so the pool identity is tamper-evident and selectors can hard-fail on
    a mismatch (as production_dispatcher already does on pool_hash).
    """

    pool_id: str = Field(min_length=1)
    pool_hash: str
    candidate_count: int = Field(ge=0)
    candidates: List[Candidate]
    frozen: Literal[True] = True

    @field_validator("pool_hash")
    @classmethod
    def _pool_hash_format(cls, v: str) -> str:
        return validate_sha256_hex(v, "pool_hash")

    @model_validator(mode="after")
    def _validate_pool(self) -> "CandidatePool":
        if self.candidate_count != len(self.candidates):
            raise ValueError(
                f"COUNT_MISMATCH: candidate_count={self.candidate_count} != "
                f"len(candidates)={len(self.candidates)}")
        # task_ids must be unique within a pool
        ids = [c.task_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("DUPLICATE_TASK_ID: task_ids repeat within pool")
        expected = hashlib.sha256(
            _canonical_json([c.chash for c in self.candidates]).encode("utf-8")
        ).hexdigest()
        if self.pool_hash != expected:
            raise ValueError(
                f"POOL_HASH_MISMATCH: pool_hash != sha256(ordered candidate "
                f"chashes); expected {expected}, got {self.pool_hash}")
        return self

    @staticmethod
    def hash_candidates(candidates: List[Candidate]) -> str:
        """Helper to compute the pool_hash a CandidatePool requires."""
        return hashlib.sha256(
            _canonical_json([c.chash for c in candidates]).encode("utf-8")
        ).hexdigest()
