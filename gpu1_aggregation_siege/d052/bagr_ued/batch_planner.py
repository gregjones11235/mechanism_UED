"""BatchPlanner: the 2048-transition update plan (task section 14).

    NUM_ENVS=16  x  ROLLOUT_LENGTH=128  =  TRANSITIONS_PER_UPDATE=2048
    every REVIEW_INTERVAL_UPDATES=4 updates (4 x 2048 = 8192 transitions):
        one Review Board window (event extraction -> failure-behavior review
        -> environment proposal -> archive refresh DRY-RUN)

This round the planner only PLANS (dry run); it never triggers rollouts. The
schedule is pure arithmetic and fully deterministic.
"""
from __future__ import annotations

from typing import List

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.schemas.common import CanonicalModel


class ReviewWindow(CanonicalModel):
    window_index: int = Field(ge=0)
    after_update: int = Field(ge=1)
    cumulative_transitions: int = Field(ge=1)


class BatchPlan(CanonicalModel):
    num_envs: int = C.NUM_ENVS
    rollout_length: int = C.ROLLOUT_LENGTH
    transitions_per_update: int = C.TRANSITIONS_PER_UPDATE
    total_updates: int = Field(ge=1)
    review_interval_updates: int = C.REVIEW_INTERVAL_UPDATES
    review_interval_transitions: int = C.REVIEW_INTERVAL_TRANSITIONS
    review_windows: List[ReviewWindow] = Field(default_factory=list)

    @model_validator(mode="after")
    def _arithmetic(self) -> "BatchPlan":
        if self.transitions_per_update != self.num_envs * self.rollout_length:
            raise ValueError(
                f"TRANSITION_ARITHMETIC: {self.num_envs} x "
                f"{self.rollout_length} != {self.transitions_per_update}")
        if self.review_interval_transitions != \
                self.review_interval_updates * self.transitions_per_update:
            raise ValueError("REVIEW_INTERVAL_ARITHMETIC")
        return self


class BatchPlanner:
    def plan(self, total_updates: int) -> BatchPlan:
        windows = []
        for u in range(1, total_updates + 1):
            if u % C.REVIEW_INTERVAL_UPDATES == 0:
                windows.append(ReviewWindow(
                    window_index=len(windows),
                    after_update=u,
                    cumulative_transitions=u * C.TRANSITIONS_PER_UPDATE))
        return BatchPlan(total_updates=total_updates, review_windows=windows)
