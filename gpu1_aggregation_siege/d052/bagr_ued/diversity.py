"""Diversity scoring (task sections 1 / 12).

Deterministic novelty of a descriptor relative to what is already in the
archive + the 4 global canonical anchors: normalized Hamming distance over
the (axis -> level) signature, averaged against every baseline signature.
1.0 = differs from everything seen; 0.0 = duplicate. Deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import Field

from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.schemas.common import CanonicalModel


class DiversityScore(CanonicalModel):
    descriptor_id: str = Field(min_length=1)
    diversity: float = Field(ge=0.0, le=1.0)
    baseline_count: int = Field(ge=0)


def signature(descriptor: TaskParamsDescriptor) -> Dict[str, str]:
    """The (axis -> level) signature; controlled axes read as 'baseline'."""
    sig = dict(descriptor.mock_control_values)
    sig.update(descriptor.mock_axis_values)
    return sig


def _hamming(a: Dict[str, str], b: Dict[str, str]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    diff = sum(1 for k in keys if a.get(k, "baseline") != b.get(k, "baseline"))
    return diff / len(keys)


def compute_diversity(descriptors: List[TaskParamsDescriptor],
                      baseline_signatures: Sequence[Dict[str, str]]
                      ) -> List[DiversityScore]:
    baselines = list(baseline_signatures)
    out: List[DiversityScore] = []
    for d in sorted(descriptors, key=lambda x: x.descriptor_id):
        sig = signature(d)
        if not baselines:
            div = 1.0
        else:
            div = sum(_hamming(sig, b) for b in baselines) / len(baselines)
        out.append(DiversityScore(descriptor_id=d.descriptor_id,
                                  diversity=round(div, 6),
                                  baseline_count=len(baselines)))
    return out
