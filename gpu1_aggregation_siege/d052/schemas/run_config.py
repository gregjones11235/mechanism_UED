"""RunConfig — the top-level canonical_v2 run configuration.

``protocol_version`` is REQUIRED here (no default): a run config that omits it
fails to parse, mirroring the entry-point gate (MISSING_PROTOCOL_VERSION). The
frozen canonical constants are pinned with ``Literal`` types so a config that
disagrees with achievement_schema / conditioning / obs_dim / pool mode is a type
error, not a variant.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.selector import SelectorConfig


class RunConfig(CanonicalModel):
    # REQUIRED (no default) -> missing protocol_version fails to parse
    protocol_version: Literal["canonical_v2"]

    run_id: str = Field(min_length=1)
    seed: int

    # --- frozen canonical constants (pinned via Literal) ---
    achievement_schema: Literal["craftax_67_v1"] = "craftax_67_v1"
    conditioning_type: Literal["achievement_multi_hot"] = "achievement_multi_hot"
    conditioning_dimension: Literal[67] = 67
    student_obs_dim: Literal[8335] = 8335
    candidate_pool_mode: Literal["shared_frozen"] = "shared_frozen"
    score_normalization: Literal["rank_percentile_v1"] = "rank_percentile_v1"
    unknown_target_policy: Literal["error"] = "error"
    empty_goal_policy: Literal["error"] = "error"
    fallback_policy: Literal["error"] = "error"

    # --- shared frozen pool binding ---
    pool_id: str = Field(min_length=1)
    pool_hash: str

    # --- selector ---
    selector: SelectorConfig

    # --- output isolation ---
    output_dir: str = Field(min_length=1)

    # canonical runs are never legacy
    allow_legacy_d052: Literal[False] = False

    @model_validator(mode="after")
    def _validate_run(self) -> "RunConfig":
        validate_sha256_hex(self.pool_hash, "pool_hash")
        return self
