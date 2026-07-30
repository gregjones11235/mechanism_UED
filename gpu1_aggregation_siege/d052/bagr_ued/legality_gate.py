"""LegalityGate for TaskParams descriptors (task sections 11 / 15).

Fail-closed validation of every proposed descriptor BEFORE it may enter the
scoring/selection chain:

  * field set must be exactly the mock whitelist (no invented real fields);
  * mutation axes must be within the legal vocabulary;
  * the BLOCKED_EXTERNAL_DEPENDENCY adapter marker must be present;
  * TrajectorySupervisionGuard must pass (no action/reward/policy content
    smuggled into a descriptor);
  * FormalEvaluationLeakageGuard must pass (no formal provenance in
    descriptor provenance).

Any violation -> LegalityViolation with a specific code. ``screen`` filters a
list and returns (legal, rejected_with_reasons) — rejected descriptors are
recorded, never silently dropped.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from d052.bagr_ued import constants as C
from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard)
from d052.bagr_ued.trajectory_supervision_guard import TrajectorySupervisionGuard


class LegalityViolation(Exception):
    ILLEGAL_PROPOSAL_FIELD = "ILLEGAL_PROPOSAL_FIELD"
    ILLEGAL_PROPOSAL_AXIS = "ILLEGAL_PROPOSAL_AXIS"
    ILLEGAL_PROPOSAL_ADAPTER_MARKER = "ILLEGAL_PROPOSAL_ADAPTER_MARKER"
    ILLEGAL_PROPOSAL_SUPERVISION = "ILLEGAL_PROPOSAL_SUPERVISION"
    ILLEGAL_PROPOSAL_LEAKAGE = "ILLEGAL_PROPOSAL_LEAKAGE"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class LegalityGate:
    def __init__(self) -> None:
        self.supervision = TrajectorySupervisionGuard()
        self.leakage = FormalEvaluationLeakageGuard()

    def assert_legal(self, descriptor: TaskParamsDescriptor) -> None:
        dump = descriptor.model_dump()
        extra = set(dump) - set(C.MOCK_TASKPARAMS_FIELD_WHITELIST)
        if extra:
            raise LegalityViolation(
                LegalityViolation.ILLEGAL_PROPOSAL_FIELD,
                f"descriptor {descriptor.descriptor_id} carries fields "
                f"outside the mock whitelist: {sorted(extra)}")
        for a in descriptor.mutation_axes:
            if a not in C.MUTATION_AXES:
                raise LegalityViolation(
                    LegalityViolation.ILLEGAL_PROPOSAL_AXIS,
                    f"descriptor {descriptor.descriptor_id} axis {a!r} is "
                    f"not a legal mutation axis")
        if descriptor.real_adapter_status != C.REAL_TASKPARAMS_ADAPTER:
            raise LegalityViolation(
                LegalityViolation.ILLEGAL_PROPOSAL_ADAPTER_MARKER,
                f"descriptor {descriptor.descriptor_id} must carry "
                f"real_adapter_status={C.REAL_TASKPARAMS_ADAPTER}")
        sup = self.supervision.scan(dump, label=f"descriptor:"
                                                f"{descriptor.descriptor_id}")
        if not sup["passed"]:
            raise LegalityViolation(
                LegalityViolation.ILLEGAL_PROPOSAL_SUPERVISION,
                f"descriptor {descriptor.descriptor_id} failed the "
                f"TrajectorySupervisionGuard: {sup['findings']}")
        leak = self.leakage.scan(dump, label=f"descriptor:"
                                             f"{descriptor.descriptor_id}")
        if not leak["passed"]:
            raise LegalityViolation(
                LegalityViolation.ILLEGAL_PROPOSAL_LEAKAGE,
                f"descriptor {descriptor.descriptor_id} failed the "
                f"FormalEvaluationLeakageGuard: {leak['findings']}")

    def screen(self, descriptors: List[TaskParamsDescriptor]
               ) -> Tuple[List[TaskParamsDescriptor], List[Dict[str, str]]]:
        legal: List[TaskParamsDescriptor] = []
        rejected: List[Dict[str, str]] = []
        for d in descriptors:
            try:
                self.assert_legal(d)
                legal.append(d)
            except LegalityViolation as e:
                rejected.append(dict(descriptor_id=d.descriptor_id,
                                     code=e.code, message=str(e)))
        return legal, rejected
