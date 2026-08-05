# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-17): the long-run launch blocker list is DYNAMIC — a pure
function of actual evidence (real preflight blockers, Reference/anchor
designation state, the director's signed training-budget decision and the
audit flag).  The budget semantics are exactly two values and the decision's
total is cross-bound to the frozen budget.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.longrun_gate import (
    BUDGET_DECISION_SCHEMA,
    TRAINING_BUDGET_ADDITIONAL_FROM_PRETRAINED_CHECKPOINT,
    TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION,
    LongRunBudgetDecision,
    budget_decision_from_payload,
    evaluate_launch_blockers,
    mint_longrun_budget_decision,
    verify_longrun_budget_decision,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

_DEC = dict(decision_id="budget-001", director_id="director/cc4",
            signature_ref="controller-signature/budget",
            budget_semantics=TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION,
            total_env_steps=98304)


class TestBudgetDecision:
    def test_positive_mint_verifies_and_rebuilds(self):
        decision = mint_longrun_budget_decision(**_DEC)
        verify_longrun_budget_decision(decision)
        payload = {**_DEC, "schema": BUDGET_DECISION_SCHEMA,
                   "decision_hash": decision.decision_hash}
        rebuilt = budget_decision_from_payload(payload)
        assert rebuilt.decision_hash == decision.decision_hash

    def test_synthetic_signature_refused_at_mint_and_payload(self):
        with pytest.raises(InvalidEvidenceError):
            mint_longrun_budget_decision(**{**_DEC,
                                            "signature_ref": "SYNTHETIC_SIGNATURE_x"})
        decision = mint_longrun_budget_decision(**_DEC)
        payload = {**_DEC, "schema": BUDGET_DECISION_SCHEMA,
                   "decision_hash": decision.decision_hash,
                   "signature_ref": "SYNTHETIC_SIGNATURE_x"}
        with pytest.raises(InvalidEvidenceError):
            budget_decision_from_payload(payload)

    def test_unknown_semantics_refused(self):
        with pytest.raises(InvalidEvidenceError):
            mint_longrun_budget_decision(**{**_DEC, "budget_semantics": "HALF_AND_HALF"})

    def test_both_semantics_are_admissible(self):
        for semantics in (TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION,
                          TRAINING_BUDGET_ADDITIONAL_FROM_PRETRAINED_CHECKPOINT):
            decision = mint_longrun_budget_decision(**{**_DEC, "budget_semantics": semantics})
            verify_longrun_budget_decision(decision)

    def test_tampered_decision_rejected(self):
        decision = mint_longrun_budget_decision(**_DEC)
        verify_longrun_budget_decision(decision)
        import dataclasses
        tampered = dataclasses.replace(decision)
        object.__setattr__(tampered, "total_env_steps", 999)
        with pytest.raises(InvalidEvidenceError):
            verify_longrun_budget_decision(tampered)


class TestDynamicGate:
    def test_fully_unbound_gate_reports_all_blockers(self):
        blockers = evaluate_launch_blockers(
            preflight_blockers=None,
            reference_candidate_id="PENDING_CONTROLLER_DESIGNATION",
            anchor_manifest_ref="PENDING_CONTROLLER_SIGNED_SHARED_ANCHOR_MANIFEST",
            budget_decision=None, audit_approved=False)
        assert "BLOCKED_E3_PREFLIGHT_NOT_EVALUATED" in blockers
        assert "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION" in blockers
        assert "BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION" in blockers
        assert "BLOCKED_AUDIT_APPROVAL_NOT_GRANTED" in blockers

    def test_fully_bound_gate_reports_zero_blockers(self):
        decision = mint_longrun_budget_decision(**_DEC)
        blockers = evaluate_launch_blockers(
            preflight_blockers=(),
            reference_candidate_id="REF_RMT16_98304",
            anchor_manifest_ref="controller-signed/anchors-v1",
            budget_decision=decision, audit_approved=True)
        assert blockers == ()

    def test_real_preflight_blockers_carried_over(self):
        decision = mint_longrun_budget_decision(**_DEC)
        blockers = evaluate_launch_blockers(
            preflight_blockers=("BLOCKED_TRAINING_SURFACE_PENDING_R9",),
            reference_candidate_id="REF_RMT16_98304",
            anchor_manifest_ref="controller-signed/anchors-v1",
            budget_decision=decision, audit_approved=True)
        assert blockers == ("BLOCKED_TRAINING_SURFACE_PENDING_R9",)

    def test_missing_decision_and_audit_block(self):
        decision = mint_longrun_budget_decision(**_DEC)
        no_decision = evaluate_launch_blockers(
            preflight_blockers=(), reference_candidate_id="REF_RMT16_98304",
            anchor_manifest_ref="controller-signed/anchors-v1",
            budget_decision=None, audit_approved=True)
        assert "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION" in no_decision
        no_audit = evaluate_launch_blockers(
            preflight_blockers=(), reference_candidate_id="REF_RMT16_98304",
            anchor_manifest_ref="controller-signed/anchors-v1",
            budget_decision=decision, audit_approved=False)
        assert "BLOCKED_AUDIT_APPROVAL_NOT_GRANTED" in no_audit


def _load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLongrunEntrypoint:
    def test_no_args_blocks_with_dynamic_reasons(self, tmp_path):
        script = _load_script("run_e3_longrun")
        assert script.main([f"--out={tmp_path}"]) == script.BLOCKED
        report_path = tmp_path / "e3_longrun_entrypoint.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "BLOCKED_E3_PREFLIGHT_NOT_EVALUATED" in report["launch_blockers"]
        assert "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION" in report["launch_blockers"]

    def test_preflight_and_decision_shrink_the_blockers(self, tmp_path):
        script = _load_script("run_e3_longrun")
        pre_path = tmp_path / "preflight.json"
        pre_path.write_text(json.dumps({"ready": True, "gates": {}, "blockers": []}),
                            encoding="utf-8")
        decision = mint_longrun_budget_decision(**_DEC)
        decision_path = tmp_path / "decision.json"
        decision_path.write_text(json.dumps({
            "schema": BUDGET_DECISION_SCHEMA, **_DEC,
            "decision_hash": decision.decision_hash}), encoding="utf-8")
        assert script.main([f"--out={tmp_path}",
                            f"--preflight-report={pre_path}",
                            f"--budget-decision={decision_path}"]) == script.BLOCKED
        report = json.loads((tmp_path / "e3_longrun_entrypoint.json").read_text(encoding="utf-8"))
        assert "BLOCKED_E3_PREFLIGHT_NOT_EVALUATED" not in report["launch_blockers"]
        assert "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION" not in report["launch_blockers"]
        assert report["frozen_config"]["training_budget_semantics"] == \
            TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION

    def test_cross_bound_budget_total_fails(self, tmp_path):
        script = _load_script("run_e3_longrun")
        decision = mint_longrun_budget_decision(**{**_DEC, "total_env_steps": 12345})
        decision_path = tmp_path / "decision.json"
        decision_path.write_text(json.dumps({
            "schema": BUDGET_DECISION_SCHEMA,
            **_DEC, "total_env_steps": 12345,
            "decision_hash": decision.decision_hash}), encoding="utf-8")
        assert script.main([f"--out={tmp_path}",
                            f"--budget-decision={decision_path}"]) == script.FAIL
