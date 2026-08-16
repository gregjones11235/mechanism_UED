"""FrontierSpec: deterministic output of the Frontier Locator (no LLM)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from ..runtime.hashing import hash_payload


@dataclass(frozen=True)
class FrontierSpec:
    skill_family: str
    tier: str
    probe_id: str
    mastered_before: Optional[str]
    failing_here: str
    status: str
    rollout_horizon: int
    success_predicate: str
    progress_metric: str
    priority: int
    allowed_variations: tuple
    spec_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def finalize(spec: FrontierSpec) -> FrontierSpec:
    body = {k: v for k, v in spec.to_dict().items() if k != "spec_hash"}
    return FrontierSpec(**{**body, "spec_hash": hash_payload(body)})