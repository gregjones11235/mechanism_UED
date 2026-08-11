"""CounterfactualEnvironmentBuilder (task section 11).

From the board's ACCEPTED intervention hypotheses, builds:
  * one CONTROL environment (all axes at baseline),
  * single-axis mutation variants (one axis moved off baseline, everything
    else controlled),
  * a small capped factorial combination (degree <= 2) to separate e.g.
    threat-perception vs resource-pressure vs memory vs distribution problems.

Every variant is a symbolic Global-TaskParams-level descriptor (axis levels
are symbolic — baseline/low/high — because the REAL TaskParams adapter is
BLOCKED_EXTERNAL_DEPENDENCY; no numeric field is invented). Variants carry the
hypothesis ids they discriminate, so the downstream scorer can attribute
behavioral movement to a cause.

Deterministic: same reconciliation -> same plan (variant ids are content
hashes; ordering is sorted).
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.review_reconciler import ReconciliationResult
from d052.schemas.common import CanonicalModel

MAX_FACTORIAL_DEGREE = 2
AXIS_LEVELS = ("low", "high")


class EnvironmentVariant(CanonicalModel):
    variant_id: str = Field(min_length=1)
    base_env_family: str = Field(min_length=1)
    kind: str = Field(pattern=r"^(control|single_axis|factorial)$")
    axis_values: Dict[str, str] = Field(default_factory=dict)
    controlled_variables: List[str] = Field(default_factory=list)
    distinguishes_hypothesis_ids: List[str] = Field(default_factory=list)
    source_intervention_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _legal(self) -> "EnvironmentVariant":
        for a in list(self.axis_values) + self.controlled_variables:
            if a not in C.MUTATION_AXES:
                raise ValueError(f"ILLEGAL_VARIANT_AXIS: {a!r}")
        overlap = set(self.axis_values) & set(self.controlled_variables)
        if overlap:
            raise ValueError(f"VARIANT_AXIS_CONFLICT: {sorted(overlap)}")
        return self


class CounterfactualPlan(CanonicalModel):
    bundle_id: str = Field(min_length=1)
    base_env_family: str = "generative_training_env_family"
    control: EnvironmentVariant
    variants: List[EnvironmentVariant] = Field(default_factory=list)
    factorial_degree_cap: int = MAX_FACTORIAL_DEGREE
    plan_hash: str = ""

    def all_variants(self) -> List[EnvironmentVariant]:
        return [self.control] + self.variants

    def finalize_hash(self) -> "CounterfactualPlan":
        payload = self.model_dump()
        payload.pop("plan_hash", None)
        object.__setattr__(self, "plan_hash", canonical_sha256(payload))
        return self


class CounterfactualEnvironmentBuilder:
    def build(self, reconciliation: ReconciliationResult,
              interventions: List[dict]) -> CounterfactualPlan:
        accepted_ids = {i.item_id for i in
                        reconciliation.accepted_intervention_hypotheses
                        if i.decision == "accepted"}
        accepted = [itv for itv in interventions
                    if itv["intervention_id"] in accepted_ids]
        accepted.sort(key=lambda x: x["intervention_id"])

        # hypothesis -> intervention provenance for discrimination bookkeeping
        def vids(itv: dict) -> str:
            return itv["intervention_id"]

        control = EnvironmentVariant(
            variant_id="variant:control",
            base_env_family="generative_training_env_family",
            kind="control",
            axis_values={},
            controlled_variables=sorted(C.MUTATION_AXES),
            distinguishes_hypothesis_ids=[],
            source_intervention_ids=[vids(i) for i in accepted])

        variants: List[EnvironmentVariant] = []
        axis_to_hyps: Dict[str, set] = {}
        axis_to_itv: Dict[str, set] = {}
        for itv in accepted:
            for a in itv["mutation_axes"]:
                axis_to_hyps.setdefault(a, set()).update(
                    itv["target_hypothesis_ids"])
                axis_to_itv.setdefault(a, set()).add(itv["intervention_id"])

        all_axes = sorted(axis_to_hyps)
        # single-axis variants (both levels)
        for a in all_axes:
            for level in AXIS_LEVELS:
                payload = dict(axis=a, level=level)
                variants.append(EnvironmentVariant(
                    variant_id=f"variant:single:{canonical_sha256(payload)[:12]}",
                    base_env_family="generative_training_env_family",
                    kind="single_axis",
                    axis_values={a: level},
                    controlled_variables=sorted(set(C.MUTATION_AXES) - {a}),
                    distinguishes_hypothesis_ids=sorted(axis_to_hyps[a]),
                    source_intervention_ids=sorted(axis_to_itv[a])))
        # one small factorial combination (degree 2, first two axes) — capped
        if len(all_axes) >= MAX_FACTORIAL_DEGREE:
            pair = all_axes[:MAX_FACTORIAL_DEGREE]
            payload = dict(pair=pair, levels=list(AXIS_LEVELS))
            hyps = set().union(*[axis_to_hyps[a] for a in pair])
            itvs = set().union(*[axis_to_itv[a] for a in pair])
            variants.append(EnvironmentVariant(
                variant_id=f"variant:factorial:{canonical_sha256(payload)[:12]}",
                base_env_family="generative_training_env_family",
                kind="factorial",
                axis_values={pair[0]: AXIS_LEVELS[0], pair[1]: AXIS_LEVELS[1]},
                controlled_variables=sorted(set(C.MUTATION_AXES) - set(pair)),
                distinguishes_hypothesis_ids=sorted(hyps),
                source_intervention_ids=sorted(itvs)))

        variants.sort(key=lambda v: v.variant_id)
        plan = CounterfactualPlan(bundle_id=reconciliation.bundle_id,
                                  control=control, variants=variants)
        return plan.finalize_hash()
