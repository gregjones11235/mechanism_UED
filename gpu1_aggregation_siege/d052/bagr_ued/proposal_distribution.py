"""ProposalDistribution (task sections 1 / 11).

A deterministic, seeded sampling distribution over legal TaskParams
descriptors. Weights come from the reconciliation confidence of the
hypotheses a descriptor discriminates, penalized by the critic's soft
selection signal (kept separate from critic rejections — a rejected
descriptor never enters the support at all). Sampling uses
random.Random(seed) ONLY — no system entropy, no time, fully replayable.

NOTE: final environment VALUE is never decided by this distribution or by LLM
judgment — it must be validated by real rollout evidence (this round: dry run,
so scores are mock-evidence based and labeled as such downstream).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

from pydantic import Field

from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.hashing import canonical_sha256
from d052.schemas.common import CanonicalModel

DEFAULT_TEMPERATURE = 1.0


class ProposalWeights(CanonicalModel):
    descriptor_id: str
    weight: float = Field(ge=0.0)


class ProposalDistribution:
    def __init__(self, descriptors: List[TaskParamsDescriptor],
                 confidence_by_hypothesis: Dict[str, float],
                 critic_penalty_by_intervention: Dict[str, float],
                 temperature: float = DEFAULT_TEMPERATURE) -> None:
        self.descriptors = sorted(descriptors, key=lambda d: d.descriptor_id)
        self.weights: Dict[str, float] = {}
        for d in self.descriptors:
            confs = [confidence_by_hypothesis.get(h, 0.3)
                     for h in d.distinguishes_hypothesis_ids]
            base = sum(confs) / len(confs) if confs else 0.3
            penalties = [critic_penalty_by_intervention.get(i, 0.0)
                         for i in d.provenance.get("source_intervention_ids", [])]
            pen = max(penalties) if penalties else 0.0
            # control descriptor carries the base context: neutral weight
            if d.mock_variant_kind == "control":
                w = 0.5
            else:
                w = max(0.05, base * (1.0 - min(1.0, pen)))
            self.weights[d.descriptor_id] = round(w, 6)
        self.temperature = temperature

    def probabilities(self) -> Dict[str, float]:
        scaled = {k: math.exp(math.log(v) / self.temperature)
                  for k, v in self.weights.items()}
        total = sum(scaled.values())
        return {k: v / total for k, v in scaled.items()}

    def sample(self, n: int, seed: int) -> List[str]:
        """Deterministic seeded sampling WITH replacement -> descriptor ids."""
        probs = self.probabilities()
        ids = sorted(probs)
        cum = []
        acc = 0.0
        for i in ids:
            acc += probs[i]
            cum.append(acc)
        rng = random.Random(seed)
        picks: List[str] = []
        for _ in range(n):
            r = rng.random() * acc
            for idx, c in enumerate(cum):
                if r <= c:
                    picks.append(ids[idx])
                    break
            else:
                picks.append(ids[-1])
        return picks

    def distribution_hash(self) -> str:
        return canonical_sha256({"weights": self.weights,
                                 "temperature": self.temperature})
