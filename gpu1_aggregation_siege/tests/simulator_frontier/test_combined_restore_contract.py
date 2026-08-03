"""Fail-closed tests for the R4c combined fresh-process restore CONTRACT.

Contract only: no real combined restore executes this round (policy-memory/
history restore is Phase 2; the audited CC2 pkl carries params+manifest
only).  These tests pin the gate logic that makes "env-only PASS /\\
ckpt-only PASS != joint proof" mechanically enforceable.
"""

from __future__ import annotations

import pytest

from dicode.simulator_frontier import (
    CROSS_CHECKS,
    REQUIRED_COMPONENTS,
    CombinedRestoreRequest,
    ComponentResult,
    ComponentStatus,
    evaluate_verdict,
    run_combined_restore,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _request(**overrides) -> CombinedRestoreRequest:
    base = dict(
        encoded_bundle_ref="bundle://synthetic/contract-test",
        checkpoint_path="/synthetic/contract_test.pkl",
        expected_candidate_id="SYNTHETIC_CONTRACT_TEST",
        expected_params_sha256=SHA_A,
        expected_file_sha256=SHA_B,
        expected_env_payload_hash=SHA_C,
        expected_global_step=98304,
        expected_memory_spec_hash="d" * 64,
    )
    base.update(overrides)
    return CombinedRestoreRequest(**base)


def _ok(component: str) -> ComponentResult:
    return ComponentResult(component, ComponentStatus.RESTORED_HASH_BOUND, "ok")


def _cross_ok(component: str) -> ComponentResult:
    return ComponentResult(component, ComponentStatus.RESTORED_CROSS_VERIFIED, "ok")


class TestRequestValidation:
    def test_minimal_valid_request_constructs(self):
        request = _request()
        assert request.expected_global_step == 98304

    @pytest.mark.parametrize("field", [
        "encoded_bundle_ref", "checkpoint_path", "expected_candidate_id",
        "expected_params_sha256", "expected_file_sha256",
        "expected_env_payload_hash", "expected_memory_spec_hash"])
    def test_missing_required_field_raises(self, field):
        with pytest.raises(InvalidEvidenceError):
            _request(**{field: ""})

    @pytest.mark.parametrize("field", ["expected_params_sha256", "expected_file_sha256"])
    def test_bad_sha_format_raises(self, field):
        with pytest.raises(InvalidEvidenceError):
            _request(**{field: "not-a-sha"})

    def test_negative_global_step_raises(self):
        with pytest.raises(InvalidEvidenceError):
            _request(expected_global_step=-1)

    def test_bad_optional_optimizer_sha_raises(self):
        with pytest.raises(InvalidEvidenceError):
            _request(expected_optimizer_sha256="zzz")


class TestVerdictEvaluation:
    def _components(self, **status_overrides) -> dict:
        out = {name: _ok(name) for name in REQUIRED_COMPONENTS}
        for name, status in status_overrides.items():
            out[name] = ComponentResult(name, status, "overridden")
        return out

    def test_all_restored_is_combined_pass(self):
        cross = {name: _cross_ok(name) for name in CROSS_CHECKS}
        verdict = evaluate_verdict(self._components(), cross)
        assert verdict.combined_pass is True
        assert verdict.env_only_pass is True
        assert verdict.checkpoint_only_pass is True
        assert "COMBINED_FRESH_PROCESS_RESTORE=true" in verdict.joint_proof_status

    @pytest.mark.parametrize("missing_component", [
        "optimizer", "train_rng", "policy_memory", "history", "env_state"])
    def test_one_missing_component_blocks_joint_pass(self, missing_component):
        components = self._components(
            **{missing_component: ComponentStatus.ABSENT_IN_CHECKPOINT})
        cross = {name: _cross_ok(name) for name in CROSS_CHECKS}
        verdict = evaluate_verdict(components, cross)
        assert verdict.combined_pass is False
        assert "COMBINED_FRESH_PROCESS_RESTORE=false" in verdict.joint_proof_status

    def test_env_only_and_ckpt_only_passes_do_not_compose(self):
        # env side green, checkpoint side green, but policy_memory/history
        # absent -> the honest status is NOT a joint proof.
        components = self._components(
            policy_memory=ComponentStatus.ABSENT_IN_CHECKPOINT,
            history=ComponentStatus.NOT_EXECUTED)
        cross = {name: _cross_ok(name) for name in CROSS_CHECKS}
        verdict = evaluate_verdict(components, cross)
        assert verdict.env_only_pass is True
        assert verdict.checkpoint_only_pass is True
        assert verdict.combined_pass is False

    def test_failed_cross_check_blocks_joint_pass(self):
        components = self._components()
        cross = {name: _cross_ok(name) for name in CROSS_CHECKS}
        cross["policy_step_next_replay"] = ComponentResult(
            "policy_step_next_replay", ComponentStatus.FAILED, "diverged")
        verdict = evaluate_verdict(components, cross)
        assert verdict.combined_pass is False

    def test_missing_component_in_verdict_raises(self):
        components = {n: _ok(n) for n in REQUIRED_COMPONENTS if n != "history"}
        cross = {name: _cross_ok(name) for name in CROSS_CHECKS}
        with pytest.raises(InvalidEvidenceError):
            evaluate_verdict(components, cross)

    def test_component_matrix_reports_statuses(self):
        components = self._components(optimizer=ComponentStatus.ABSENT_IN_CHECKPOINT)
        cross = {name: _cross_ok(name) for name in CROSS_CHECKS}
        verdict = evaluate_verdict(components, cross)
        matrix = verdict.component_matrix()
        assert matrix["optimizer"] == "ABSENT_IN_CHECKPOINT"
        assert matrix["params"] == "RESTORED_HASH_BOUND"


class TestRunCombinedRestoreFailClosed:
    def _all_restorers(self):
        return {name: (lambda req, n=name: _ok(n)) for name in REQUIRED_COMPONENTS}

    def _all_cross(self):
        return {name: (lambda req, comps, n=name: _cross_ok(n)) for name in CROSS_CHECKS}

    def test_full_contract_run_passes(self):
        verdict = run_combined_restore(
            _request(), restorers=self._all_restorers(), cross_checkers=self._all_cross())
        assert verdict.combined_pass is True

    def test_missing_restorer_is_not_executed_and_blocks(self):
        restorers = self._all_restorers()
        del restorers["optimizer"]
        verdict = run_combined_restore(
            _request(), restorers=restorers, cross_checkers=self._all_cross())
        assert verdict.components["optimizer"].status is ComponentStatus.NOT_EXECUTED
        assert verdict.combined_pass is False

    def test_raising_restorer_is_failed_not_fatal(self):
        restorers = self._all_restorers()

        def boom(req):
            raise RuntimeError("synthetic restore crash")

        restorers["env_state"] = boom
        verdict = run_combined_restore(
            _request(), restorers=restorers, cross_checkers=self._all_cross())
        assert verdict.components["env_state"].status is ComponentStatus.FAILED
        assert "synthetic restore crash" in verdict.components["env_state"].detail
        assert verdict.combined_pass is False

    def test_mismatched_component_result_is_failed(self):
        restorers = self._all_restorers()
        restorers["train_rng"] = lambda req: _ok("params")  # wrong component name
        verdict = run_combined_restore(
            _request(), restorers=restorers, cross_checkers=self._all_cross())
        assert verdict.components["train_rng"].status is ComponentStatus.FAILED

    def test_cross_checker_sees_component_map(self):
        seen = {}

        def replay(req, comps):
            seen["names"] = set(comps)
            return _cross_ok("policy_step_next_replay")

        verdict = run_combined_restore(
            _request(), restorers=self._all_restorers(),
            cross_checkers={"policy_step_next_replay": replay})
        assert seen["names"] == set(REQUIRED_COMPONENTS)
        assert verdict.combined_pass is True

    def test_invalid_request_type_raises(self):
        with pytest.raises(InvalidEvidenceError):
            run_combined_restore({"not": "a request"}, restorers={})

    def test_round_honest_default_is_not_combined(self):
        """No restorers at all == this round's situation: contract in place,
        execution pending -> combined_pass must be False."""
        verdict = run_combined_restore(_request(), restorers={})
        assert verdict.combined_pass is False
        assert all(r.status is ComponentStatus.NOT_EXECUTED
                   for r in verdict.components.values())
