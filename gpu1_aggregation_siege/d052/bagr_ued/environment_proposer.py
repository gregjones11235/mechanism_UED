"""GlobalTaskParamsProposer + MOCK TaskParams adapter (task section 11).

The REAL Global TaskParams adapter is an external dependency this package
does not own:

    REAL_TASKPARAMS_ADAPTER = BLOCKED_EXTERNAL_DEPENDENCY

So this module proposes TaskParams DESCRIPTORS through a MOCK adapter whose
field set is an explicit mock-namespaced whitelist. It NEVER guesses real
field names (extra=forbid + LegalityGate): a descriptor is a promise "these
axes must move", not a claim about real TaskParams layout. When the real
adapter is delivered, descriptors convert through it — not before.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.counterfactual_environment import (
    CounterfactualPlan,
    EnvironmentVariant,
)
from d052.bagr_ued.hashing import canonical_sha256
from d052.schemas.common import CanonicalModel


class TaskParamsDescriptor(CanonicalModel):
    """A legal, mock-namespaced Global TaskParams proposal.

    Field set == MOCK_TASKPARAMS_FIELD_WHITELIST minus descriptor_hash (which
    is computed). extra=forbid makes field invention a hard error.
    """

    descriptor_id: str = Field(min_length=1)
    mock_env_family: str = Field(min_length=1)
    mock_axis_values: Dict[str, str] = Field(default_factory=dict)
    mock_control_values: Dict[str, str] = Field(default_factory=dict)
    mock_variant_id: str = Field(min_length=1)
    mock_variant_kind: str = Field(min_length=1)
    mutation_axes: List[str] = Field(default_factory=list)
    distinguishes_hypothesis_ids: List[str] = Field(default_factory=list)
    provenance: Dict[str, object] = Field(default_factory=dict)
    real_adapter_status: str = C.REAL_TASKPARAMS_ADAPTER
    legality_hint: str = "MOCK_ONLY — convert through the real TaskParams " \
                         "adapter once unblocked; do not execute directly"
    descriptor_hash: str = ""

    @model_validator(mode="after")
    def _whitelist_and_hash(self) -> "TaskParamsDescriptor":
        allowed = set(C.MOCK_TASKPARAMS_FIELD_WHITELIST)
        extra = set(self.model_dump()) - allowed
        if extra:
            raise ValueError(
                f"UNAUTHORIZED_DESCRIPTOR_FIELD: {sorted(extra)} — real "
                f"TaskParams fields are UNKNOWN "
                f"(REAL_TASKPARAMS_ADAPTER={C.REAL_TASKPARAMS_ADAPTER}); "
                f"guessing them is forbidden")
        if self.real_adapter_status != C.REAL_TASKPARAMS_ADAPTER:
            raise ValueError("DESCRIPTOR_ADAPTER_MARKER_MISSING")
        for a in self.mutation_axes:
            if a not in C.MUTATION_AXES:
                raise ValueError(f"ILLEGAL_DESCRIPTOR_AXIS: {a!r}")
        if not self.descriptor_hash:
            payload = self.model_dump()
            payload.pop("descriptor_hash", None)
            object.__setattr__(self, "descriptor_hash", canonical_sha256(payload))
        return self


class MockTaskParamsAdapter:
    """Mock adapter standing in for BLOCKED_EXTERNAL_DEPENDENCY."""

    real_adapter_status = C.REAL_TASKPARAMS_ADAPTER

    def to_descriptor(self, variant: EnvironmentVariant, *,
                      plan_hash: str, bundle_id: str) -> TaskParamsDescriptor:
        return TaskParamsDescriptor(
            descriptor_id=f"tpd:{variant.variant_id}",
            mock_env_family=variant.base_env_family,
            mock_axis_values=dict(variant.axis_values),
            mock_control_values={a: "baseline"
                                 for a in variant.controlled_variables},
            mock_variant_id=variant.variant_id,
            mock_variant_kind=variant.kind,
            mutation_axes=sorted(variant.axis_values),
            distinguishes_hypothesis_ids=variant.distinguishes_hypothesis_ids,
            provenance=dict(plan_hash=plan_hash, bundle_id=bundle_id,
                            source_intervention_ids=variant.source_intervention_ids),
        )


class GlobalTaskParamsProposer:
    """Turns a CounterfactualPlan into legal TaskParams descriptors."""

    def __init__(self, adapter: MockTaskParamsAdapter | None = None) -> None:
        self.adapter = adapter or MockTaskParamsAdapter()

    def propose(self, plan: CounterfactualPlan) -> List[TaskParamsDescriptor]:
        descriptors = [self.adapter.to_descriptor(v, plan_hash=plan.plan_hash,
                                                  bundle_id=plan.bundle_id)
                       for v in plan.all_variants()]
        descriptors.sort(key=lambda d: d.descriptor_id)
        return descriptors
