"""Candidate validator — turn a raw salvaged-style candidate dict into a strict,
content-hashed canonical ``Candidate``, fail-closed.

Raw shape (the legacy D052 field set we salvage)::

    {"task_id": str,
     "task_params": {"passive_spawn_multiplier": float,
                     "melee_spawn_multiplier": float,
                     "mob_health_multiplier": float,
                     "mob_damage_multiplier": float},
     "target_achievements": [name, ...]}

Rules (NO_SILENT_SCHEMA_COERCION / unknown_target_policy=error /
empty_goal_policy=error):
  * unknown / extra top-level keys -> error
  * unknown achievement name -> error (audited aliases still resolve)
  * empty target set -> error
  * > 4 targets -> error (hard, not a silent cap)
  * the canonical content hash (chash) is COMPUTED, never trusted from input; a
    supplied chash that disagrees -> error
"""
from __future__ import annotations

from typing import Any, List, Mapping

from d052.achievements import REGISTRY, AchievementError
from d052.schemas.candidate import Candidate, TaskParams, compute_candidate_chash

_ALLOWED_RAW_KEYS = frozenset({"task_id", "task_params", "target_achievements"})
_REQUIRED_RAW_KEYS = _ALLOWED_RAW_KEYS


class CandidateValidationError(Exception):
    """Fail-closed candidate validation error with a stable ``code``."""

    UNKNOWN_RAW_KEY = "UNKNOWN_RAW_KEY"
    MISSING_RAW_KEY = "MISSING_RAW_KEY"
    INVALID_TYPE = "INVALID_TYPE"
    DUPLICATE_TARGET = "DUPLICATE_TARGET"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def validate_target_names(names: Any) -> List[str]:
    """Validate + canonicalize a target-name list. Empty -> error; unknown -> error.

    Returns sorted distinct canonical names. Raises CandidateValidationError on
    wrong type; registry errors (unknown/empty) surface as ValueError from the
    schema layer when the Candidate is built, but we also guard type here.
    """
    if not isinstance(names, (list, tuple)):
        raise CandidateValidationError(
            CandidateValidationError.INVALID_TYPE,
            f"target_achievements must be a list, got {type(names).__name__}")
    if not names:
        raise AchievementError(
            AchievementError.EMPTY_GOAL_SET,
            "target achievement set is empty (empty_goal_policy=error)")
    # resolve each (audited alias allow-list; unknown -> AchievementError)
    resolved = [REGISTRY.resolve(n) for n in names]
    if len(set(resolved)) != len(resolved):
        raise CandidateValidationError(
            CandidateValidationError.DUPLICATE_TARGET,
            f"duplicate target achievements after canonical resolution: {resolved}")
    return sorted(resolved)


def canonicalize_candidate(raw: Mapping[str, Any]) -> Candidate:
    """Build a strict Candidate from a raw salvaged dict (computes chash)."""
    if not isinstance(raw, Mapping):
        raise CandidateValidationError(
            CandidateValidationError.INVALID_TYPE,
            f"candidate must be a mapping, got {type(raw).__name__}")

    keys = set(raw.keys())
    extra = keys - _ALLOWED_RAW_KEYS
    if extra:
        raise CandidateValidationError(
            CandidateValidationError.UNKNOWN_RAW_KEY,
            f"unexpected candidate keys {sorted(extra)}; allowed="
            f"{sorted(_ALLOWED_RAW_KEYS)} (no silent coercion)")
    missing = _REQUIRED_RAW_KEYS - keys
    if missing:
        raise CandidateValidationError(
            CandidateValidationError.MISSING_RAW_KEY,
            f"missing candidate keys {sorted(missing)}")

    task_id = raw["task_id"]
    if not isinstance(task_id, str) or not task_id:
        raise CandidateValidationError(
            CandidateValidationError.INVALID_TYPE, "task_id must be a non-empty str")

    tp_raw = raw["task_params"]
    if not isinstance(tp_raw, Mapping):
        raise CandidateValidationError(
            CandidateValidationError.INVALID_TYPE,
            "task_params must be a mapping")
    task_params = TaskParams(**dict(tp_raw))  # validates positive/finite/extra

    canonical_names = validate_target_names(raw["target_achievements"])
    chash = compute_candidate_chash(task_id, canonical_names, task_params.model_dump())

    # Candidate re-validates everything (targets canonical, hash matches, caps).
    return Candidate(
        task_id=task_id,
        chash=chash,
        task_params=task_params,
        target_achievements=[{"name": n} for n in canonical_names],
    )
