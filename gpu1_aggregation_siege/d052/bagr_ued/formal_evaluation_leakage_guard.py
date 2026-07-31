"""Guard B — FormalEvaluationLeakageGuard (task sections 3 / 15).

Rejects, fail-closed, ANY trajectory evidence whose provenance is formal
evaluation state entering the review board:

    FORMAL_FRONT / FORMAL_BACK / FORMAL_FULL / FROZEN_BANK /
    FORMAL_EVALUATION_CERTIFICATE_PRIVATE_STATE

Allowed provenance is ONLY the current Student's generative-training
trajectories (GENERATIVE_TRAINING_ENV) and synthetic test traces
(SYNTHETIC_TEST_TRACE). The guard checks BOTH the typed source enum and any
free-form provenance strings/keys inside an object (defense in depth: a
forbidden source hiding in a ``data_source`` / ``origin`` string field is
caught as well as a frozen-payload key like ``front_bank_states``).

Independent of Guard A by design: the two guards are separate modules with
separate codes so a regression can name exactly which boundary broke.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from d052.bagr_ued import constants as C
from d052.bagr_ued.trajectory_evidence import EvidenceSource


class FormalLeakageViolation(Exception):
    FORMAL_EVALUATION_LEAKAGE = "FORMAL_EVALUATION_LEAKAGE"
    FORBIDDEN_PROVENANCE_KEY = "FORBIDDEN_PROVENANCE_KEY"
    SOURCE_NOT_DECLARED = "SOURCE_NOT_DECLARED"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


_PROVENANCE_KEYS = frozenset({
    "source", "provenance", "origin", "data_source", "trajectory_source",
    "evidence_source", "bank", "bank_source",
})

#: CC1 audit fix1 (§6): bank_blob / formal_state_blob / formal_state_payload /
#: state_payload are the same forbidden formal-state payload family — mirrored
#: here as well as into Guard A's alias vocabulary (defense in depth: the two
#: guards stay independent modules with independent code paths).
_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "front_bank_states", "back_bank_states", "frozen_state", "private_state",
    "evaluation_certificate", "front_state_payload", "back_state_payload",
    "full_state_payload", "expert_action_sequence",
    "bank_blob", "formal_state_blob", "formal_state_payload", "state_payload",
})

_FORBIDDEN_VALUE_TOKENS = frozenset(
    {s.lower() for s in C.FORBIDDEN_EVIDENCE_SOURCES})


class FormalEvaluationLeakageGuard:
    """Stateless scanner for formal-evaluation provenance leakage."""

    def assert_admissible_source(self, source: Union[EvidenceSource, str]) -> None:
        value = source.value if isinstance(source, EvidenceSource) else str(source)
        if value in C.FORBIDDEN_EVIDENCE_SOURCES:
            raise FormalLeakageViolation(
                FormalLeakageViolation.FORMAL_EVALUATION_LEAKAGE,
                f"evidence source {value!r} is formal evaluation / frozen-bank "
                f"state and may NOT enter the review board (allowed: "
                f"{sorted(C.ALLOWED_EVIDENCE_SOURCES)})")
        if value not in C.ALLOWED_EVIDENCE_SOURCES:
            raise FormalLeakageViolation(
                FormalLeakageViolation.SOURCE_NOT_DECLARED,
                f"evidence source {value!r} is neither explicitly allowed nor "
                f"known; fail-closed (allowed: "
                f"{sorted(C.ALLOWED_EVIDENCE_SOURCES)})")

    def scan(self, obj: Any, *, label: str = "input") -> Dict[str, Any]:
        findings: List[Dict[str, str]] = []
        self._walk(obj, path="$", findings=findings)
        return dict(guard="FormalEvaluationLeakageGuard",
                    label=label,
                    passed=not findings,
                    findings=findings)

    def assert_clean(self, obj: Any, *, label: str = "input") -> Dict[str, Any]:
        report = self.scan(obj, label=label)
        if report["passed"]:
            return report
        first = report["findings"][0]
        raise FormalLeakageViolation(first["code"],
                                     f"{label}: {first['code']} at "
                                     f"{first['path']} ({first['detail']}); "
                                     f"total_findings={len(report['findings'])}")

    # -- internals ----------------------------------------------------------
    def _walk(self, obj: Any, *, path: str, findings: List[Dict[str, str]]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in _FORBIDDEN_PAYLOAD_KEYS:
                    findings.append(dict(
                        code=FormalLeakageViolation.FORBIDDEN_PROVENANCE_KEY,
                        path=f"{path}.{k}",
                        detail=f"forbidden frozen-state/bank payload key "
                               f"{k!r}"))
                if kl in _PROVENANCE_KEYS and isinstance(v, str):
                    if v.lower() in _FORBIDDEN_VALUE_TOKENS:
                        findings.append(dict(
                            code=FormalLeakageViolation.FORMAL_EVALUATION_LEAKAGE,
                            path=f"{path}.{k}",
                            detail=f"provenance value {v!r} is a forbidden "
                                   f"formal-evaluation source"))
                self._walk(v, path=f"{path}.{k}", findings=findings)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._walk(v, path=f"{path}[{i}]", findings=findings)
        elif isinstance(obj, str):
            low = obj.lower()
            for tok in _FORBIDDEN_VALUE_TOKENS:
                if re.search(rf"\b{re.escape(tok)}\b", low):
                    findings.append(dict(
                        code=FormalLeakageViolation.FORMAL_EVALUATION_LEAKAGE,
                        path=path,
                        detail=f"free-text provenance mentions forbidden "
                               f"source {tok!r}"))
        elif hasattr(obj, "model_dump"):
            self._walk(obj.model_dump(), path=path, findings=findings)
