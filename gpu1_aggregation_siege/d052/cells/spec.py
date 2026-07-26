"""CellSpec — the immutable, content-addressed definition of one training cell.

A cell is the unit of per-cell authorization & launch (task §Cell 注册/验证/准备/
授权/按需启动). Its identity is a sha256 over ~20 canonical fields
(``IDENTITY_FIELDS``); two specs with identical content hash identically, so a
cell's identity cannot drift from its definition. State is NOT part of identity
(it lives in the registry record and mutates through the lifecycle).

All canonical_v2 pins are enforced as Literals: a cell that disagrees with the
frozen config is a configuration error, not a variant.
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Literal

from pydantic import Field, field_validator, model_validator

from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.selector import SelectorConfig

#: The canonical environment this cell targets (frozen baseline evidence).
ENVIRONMENT_VERSION = "craftax==1.4.5"

#: The ~20 content fields that define a cell's identity (ordered, fixed).
IDENTITY_FIELDS = (
    "cell_id",                       # 1
    "protocol_version",              # 2
    "hypothesis",                    # 3
    "pool_id",                       # 4
    "pool_hash",                     # 5
    "selector",                      # 6  (selector type)
    "critic_policy",                 # 7
    "k",                             # 8
    "seed",                          # 9
    "roles",                         # 10
    "budget",                        # 11
    "achievement_schema",            # 12
    "conditioning_type",             # 13
    "conditioning_dimension",        # 14
    "student_obs_dim",               # 15
    "candidate_pool_mode",           # 16
    "score_normalization",           # 17
    "candidate_ids",                 # 18
    "selection_hash",                # 19
    "environment_version",           # 20
    "intended_total_timesteps",      # 21
    "output_dir",                    # 22
)


class CellSpec(CanonicalModel):
    cell_id: str = Field(min_length=1)
    protocol_version: Literal["canonical_v2"]
    title: str = ""
    hypothesis: str = ""
    pool_id: str = Field(min_length=1)
    pool_hash: str
    selector: SelectorConfig
    achievement_schema: Literal["craftax_67_v1"] = "craftax_67_v1"
    conditioning_type: Literal["achievement_multi_hot"] = "achievement_multi_hot"
    conditioning_dimension: Literal[67] = 67
    student_obs_dim: Literal[8335] = 8335
    candidate_pool_mode: Literal["shared_frozen"] = "shared_frozen"
    score_normalization: Literal["rank_percentile_v1"] = "rank_percentile_v1"
    #: the selected candidate task_ids this cell will train (binds to a selection)
    candidate_ids: List[str] = Field(min_length=1)
    selection_hash: str
    environment_version: str = ENVIRONMENT_VERSION
    #: FROZEN INTENT ONLY -- nothing in this phase executes any timestep.
    intended_total_timesteps: int = Field(ge=0)
    output_dir: str = Field(min_length=1)
    created_by: str = Field(min_length=1)

    @field_validator("pool_hash", "selection_hash")
    @classmethod
    def _hashes(cls, v: str) -> str:
        return validate_sha256_hex(v, "hash")

    @model_validator(mode="after")
    def _unique_candidates(self) -> "CellSpec":
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("DUPLICATE_CANDIDATE_ID: candidate_ids repeat")
        return self

    def identity_payload(self) -> dict:
        """The ordered, canonical dict hashed into the cell identity."""
        return {
            "cell_id": self.cell_id,
            "protocol_version": self.protocol_version,
            "hypothesis": self.hypothesis,
            "pool_id": self.pool_id,
            "pool_hash": self.pool_hash,
            "selector": self.selector.selector.value,
            "critic_policy": self.selector.critic_policy.value,
            "k": self.selector.k,
            "seed": self.selector.seed,
            "roles": sorted(r.value for r in self.selector.roles),
            "budget": self.selector.budget,
            "achievement_schema": self.achievement_schema,
            "conditioning_type": self.conditioning_type,
            "conditioning_dimension": self.conditioning_dimension,
            "student_obs_dim": self.student_obs_dim,
            "candidate_pool_mode": self.candidate_pool_mode,
            "score_normalization": self.score_normalization,
            "candidate_ids": sorted(self.candidate_ids),
            "selection_hash": self.selection_hash,
            "environment_version": self.environment_version,
            "intended_total_timesteps": self.intended_total_timesteps,
            "output_dir": self.output_dir,
        }

    def identity_hash(self) -> str:
        """Deterministic sha256 over IDENTITY_FIELDS (content-addressed identity)."""
        payload = self.identity_payload()
        assert set(payload) == set(IDENTITY_FIELDS), "identity fields drifted"
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
