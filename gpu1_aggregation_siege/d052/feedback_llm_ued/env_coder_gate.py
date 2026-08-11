"""compile / reset / step gates for EnvCoder output (C7).

Nothing the EnvCoder emits reaches the probe funnel unless ALL THREE gates
pass — fail closed, never silently coerced:

1. **compile** — the coded batch covers the directive batch 1:1, each entry
   binds the SOURCE directive's content hash (recomputed comparison), and
   every code manifest is a well-formed symbolic symbol;
2. **reset** — every coded environment declares a reset contract
   (``reset(seed)->state``);
3. **step** — every coded environment declares a step contract
   (``step(action)->(state,reward,terminal,info)``).

A failed gate blocks the probe stage entirely (``assert_passed`` raises
``EnvCoderGateBlocked`` listing every blocker).
"""
from __future__ import annotations

from typing import List, Sequence

from pydantic import Field

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued.env_coder import (
    CODE_SYMBOL_PREFIX,
    CodedDirective,
    EnvCoderOutput,
)
from d052.schemas.common import CanonicalModel

RESET_CONTRACT_MARKER = "reset(seed)->state"
STEP_CONTRACT_MARKER = "step(action)->"


class EnvCoderGateBlocked(RuntimeError):
    """A candidate batch failed the compile/reset/step gates."""


class GateReport(CanonicalModel):
    """The three gate verdicts for one EnvCoder batch (audit-grade)."""

    window: int = Field(ge=0)
    compile_passed: bool = False
    reset_passed: bool = False
    step_passed: bool = False
    blockers: List[str] = Field(default_factory=list)
    gate_hash: str = ""

    @property
    def passed(self) -> bool:
        return self.compile_passed and self.reset_passed and self.step_passed

    def compute_hash(self) -> str:
        payload = self.model_dump()
        payload.pop("gate_hash", None)
        return canonical_sha256(payload)


def _compile_gate(directives: Sequence, coded: Sequence[CodedDirective],
                  blockers: List[str]) -> bool:
    expected = {d.directive_id: d for d in directives}
    seen: set = set()
    ok = True
    for entry in coded:
        if entry.directive_id in seen:
            blockers.append(
                f"DUPLICATE_CODED_DIRECTIVE: {entry.directive_id!r}")
            ok = False
            continue
        seen.add(entry.directive_id)
        source = expected.get(entry.directive_id)
        if source is None:
            blockers.append(
                f"UNAUTHORIZED_CODED_DIRECTIVE: {entry.directive_id!r} was "
                f"not in the board's directive batch")
            ok = False
            continue
        if entry.directive_hash != source.directive_hash:
            blockers.append(
                f"DIRECTIVE_HASH_MISMATCH: coded entry "
                f"{entry.directive_id!r} binds hash "
                f"{entry.directive_hash[:12]}... but the source directive "
                f"hashes to {source.directive_hash[:12]}...")
            ok = False
        if not entry.code_symbol.startswith(CODE_SYMBOL_PREFIX):
            blockers.append(
                f"EMPTY_CODE_SYMBOL: {entry.directive_id!r} carries no "
                f"well-formed code manifest")
            ok = False
    missing = sorted(set(expected) - seen)
    for directive_id in missing:
        blockers.append(f"DIRECTIVE_NOT_CODED: {directive_id!r}")
        ok = False
    return ok


def _reset_gate(coded: Sequence[CodedDirective],
                blockers: List[str]) -> bool:
    ok = True
    for entry in coded:
        if RESET_CONTRACT_MARKER not in entry.reset_contract:
            blockers.append(
                f"RESET_CONTRACT_MISSING: {entry.directive_id!r} declares "
                f"no {RESET_CONTRACT_MARKER!r} contract")
            ok = False
    return ok


def _step_gate(coded: Sequence[CodedDirective],
               blockers: List[str]) -> bool:
    ok = True
    for entry in coded:
        if STEP_CONTRACT_MARKER not in entry.step_contract:
            blockers.append(
                f"STEP_CONTRACT_MISSING: {entry.directive_id!r} declares "
                f"no {STEP_CONTRACT_MARKER!r} contract")
            ok = False
    return ok


class EnvCoderGate:
    """Evaluates the three gates over one EnvCoder batch."""

    def evaluate(self, *, window: int, directives: Sequence,
                 output: EnvCoderOutput) -> GateReport:
        blockers: List[str] = []
        compile_ok = _compile_gate(directives, output.coded, blockers)
        reset_ok = _reset_gate(output.coded, blockers)
        step_ok = _step_gate(output.coded, blockers)
        report = GateReport(window=window,
                            compile_passed=compile_ok,
                            reset_passed=reset_ok,
                            step_passed=step_ok,
                            blockers=blockers)
        object.__setattr__(report, "gate_hash", report.compute_hash())
        return report

    def assert_passed(self, report: GateReport) -> None:
        if not report.passed:
            raise EnvCoderGateBlocked(
                "ENVCODER_GATE_BLOCKED: " + "; ".join(report.blockers))
