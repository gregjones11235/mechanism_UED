"""AchievementRef — a single validated canonical achievement reference."""
from __future__ import annotations

from typing import Optional

from pydantic import field_validator, model_validator

from d052.achievements import REGISTRY
from d052.schemas.common import CanonicalModel, resolve_canonical_name


class AchievementRef(CanonicalModel):
    """One legal target achievement.

    ``name`` is resolved through the explicit, audited alias allow-list and stored
    in canonical form; ``canonical_id`` and ``goal_vector_index`` are auto-filled
    and are always equal (confirmed: canonical_id == goal_vector_index == enum
    value). Unknown names raise (unknown_target_policy=error); nothing is dropped.
    """

    name: str
    canonical_id: Optional[int] = None
    goal_vector_index: Optional[int] = None

    @field_validator("name")
    @classmethod
    def _resolve_name(cls, v: str) -> str:
        return resolve_canonical_name(v)

    @model_validator(mode="after")
    def _fill_ids(self) -> "AchievementRef":
        cid = REGISTRY.canonical_id(self.name)
        object.__setattr__(self, "canonical_id", cid)
        object.__setattr__(self, "goal_vector_index", cid)
        return self
