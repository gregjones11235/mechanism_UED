"""Fail-closed isolation guards for the feedback loop (task §4 / §6).

Two independent hard boundaries, both enforced by raising (never by silently
dropping):

  A. ``FormalSourceIsolationGuard`` — the formal data domains (FORMAL_FRONT /
     FORMAL_BACK / FORMAL_FULL) may NEVER enter the HypothesisLedger, the LLM
     roles, the generator, the selector, or the Student optimizer. Training
     and candidate probes carry a SEPARATE source enum; anything labelled
     formal fails closed at the door.

  B. ``ReferenceOutputGuard`` — Reference probe results may expose ONLY coarse
     episode-level statistics. An action sequence / trajectory / waypoint /
     hidden state / logits carrier must never reach the Student or an LLM
     prompt as action guidance.

Both are pure (no simulator / LLM dependency) and unit-testable.
"""
from __future__ import annotations

from typing import Any, Dict, List

from d052.feedback_llm_ued import constants as C


class FormalIsolationError(Exception):
    FORMAL_SOURCE_FORBIDDEN = "FORMAL_SOURCE_FORBIDDEN"
    REFERENCE_CARRIER_FORBIDDEN = "REFERENCE_CARRIER_FORBIDDEN"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class FormalSourceIsolationGuard:
    """Refuses any formal-evaluation source from entering the loop."""

    def assert_allowed_source(self, source: str, *, label: str) -> str:
        if source in C.FORMAL_FORBIDDEN_SOURCES:
            raise FormalIsolationError(
                FormalIsolationError.FORMAL_SOURCE_FORBIDDEN,
                f"{label}: formal evaluation source {source!r} may not enter "
                f"the feedback loop (ledger/LLM/generator/selector/optimizer)")
        if source not in C.ALLOWED_LOOP_SOURCES:
            raise FormalIsolationError(
                FormalIsolationError.UNKNOWN_SOURCE,
                f"{label}: source {source!r} is not an allowed loop source "
                f"({sorted(C.ALLOWED_LOOP_SOURCES)})")
        return source

    def assert_record_clean(self, record: Dict[str, Any], *, label: str) -> None:
        """Scan a record's source-ish fields for a formal domain."""
        for key in ("source", "evidence_source", "provenance_source",
                    "data_source"):
            value = record.get(key)
            if isinstance(value, str):
                self.assert_allowed_source(value, label=f"{label}.{key}")
        prov = record.get("provenance")
        if isinstance(prov, dict):
            for key in ("source", "evidence_source"):
                value = prov.get(key)
                if isinstance(value, str):
                    self.assert_allowed_source(
                        value, label=f"{label}.provenance.{key}")


class ReferenceOutputGuard:
    """Refuses Reference action-guidance carriers in Student/LLM payloads."""

    def scan(self, node: Any, *, label: str,
             _path: str = "$") -> List[dict]:
        findings: List[dict] = []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and \
                        k.lower() in C.REFERENCE_FORBIDDEN_CARRIERS:
                    findings.append(dict(
                        code=FormalIsolationError.REFERENCE_CARRIER_FORBIDDEN,
                        path=f"{_path}.{k}",
                        detail=f"Reference carrier {k!r} is forbidden from "
                               f"Student/LLM consumption (action guidance)"))
                findings.extend(self.scan(v, label=label,
                                          _path=f"{_path}.{k}"))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                findings.extend(self.scan(v, label=label,
                                          _path=f"{_path}[{i}]"))
        return findings

    def assert_clean(self, node: Any, *, label: str) -> Dict[str, Any]:
        findings = self.scan(node, label=label)
        if findings:
            first = findings[0]
            raise FormalIsolationError(
                FormalIsolationError.REFERENCE_CARRIER_FORBIDDEN,
                f"{label}: {first['code']} at {first['path']} "
                f"({first['detail']}); total_findings={len(findings)}")
        return dict(passed=True, findings=[])
