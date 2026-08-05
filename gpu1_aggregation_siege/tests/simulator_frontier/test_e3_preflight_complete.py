# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-15): the production preflight is a fail-closed checklist
consuming SIGNED capability descriptors — the spoofable exception-inference
probes are gone, and every production dependency (training surface, restore
bundle, anchor manifest, formal registry, memory artifact, authorized LLM,
bound training runtime, measurable step baseline, taskparam surface) has a
named gate.
"""

import pytest

from dicode.simulator_frontier import e3_window
from dicode.simulator_frontier.e3_window import (
    BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME,
    BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN,
    BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED,
    BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY,
    E3WindowConfig,
    run_e3_preflight,
)
from dicode.simulator_frontier.surface_capability import (
    mint_training_surface_capability,
)
from dicode.simulator_frontier.training_runtime import (
    mint_original_training_runtime,
)
from dicode.student_adapters.fake import FakeStudentAdapter

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _student():
    return FakeStudentAdapter(candidate_id="FAKE_MOUNT_CONTRACT_ONLY")


def _capability(student=None, *, synthetic_signature=False, other_identity=False):
    identity_hash = (("0" * 64) if other_identity
                     else (student or _student()).identity().identity_hash())
    return mint_training_surface_capability(
        descriptor_id="cap-001",
        adapter_identity_hash=identity_hash,
        save_full_state_capable=True,
        restore_full_state_capable=True,
        verifier_id="controller-audit/cc4",
        signature_ref=("SYNTHETIC_SIGNATURE_self-signed" if synthetic_signature
                       else "controller-signature/cap"))


def _original_loss(batch, params):
    """SYNTHETIC original loss (test only)."""
    return 0.5


def _original_update(params, batch):
    """SYNTHETIC original update (test only)."""
    return {"params": params}


class TestSpoofableProbesRemoved:
    def test_exception_inference_probe_no_longer_exists(self):
        assert not hasattr(e3_window, "_probe_training_surface")


class TestFullChecklist:
    def test_default_config_reports_every_named_blocker(self):
        pre = run_e3_preflight(E3WindowConfig())
        assert not pre.ready
        blockers = set(pre.blockers)
        assert BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY in blockers
        assert BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME in blockers
        assert BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED in blockers
        assert BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN in blockers
        assert "BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE" in blockers
        assert "BLOCKED_SHARED_ANCHOR_MANIFEST" in blockers
        assert "BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY" in blockers
        assert "REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT" in blockers

    def test_signed_capability_enables_surface_gates(self):
        student = _student()
        config = E3WindowConfig(
            student=student, student_params=student._params,
            training_surface_capability=_capability(student))
        pre = run_e3_preflight(config)
        assert pre.gates["TRAINING_SURFACE_CAPABILITY_SIGNED"] is True
        assert pre.gates["STUDENT_TRAINING_SURFACE"] is True
        assert pre.gates["CHECKPOINT_ROUND_TRIP_CAPABILITY"] is True

    def test_synthetic_capability_signature_is_rejected(self):
        student = _student()
        config = E3WindowConfig(
            student=student, student_params=student._params,
            training_surface_capability=_capability(student, synthetic_signature=True))
        pre = run_e3_preflight(config)
        assert pre.gates["TRAINING_SURFACE_CAPABILITY_SIGNED"] is False
        assert BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY in pre.blockers

    def test_foreign_identity_capability_is_rejected(self):
        student = _student()
        config = E3WindowConfig(
            student=student, student_params=student._params,
            training_surface_capability=_capability(student, other_identity=True))
        pre = run_e3_preflight(config)
        assert pre.gates["TRAINING_SURFACE_CAPABILITY_SIGNED"] is False

    def test_bound_training_runtime_enables_runtime_gate(self):
        runtime = mint_original_training_runtime(
            loss_fn=_original_loss, optimizer_update_fn=_original_update,
            runtime_id="rt-001", loss_name="PPO_ORIGINAL_VTRACE",
            optimizer_name="ADAMW_ORIGINAL", contract_ref="controller-shared/cc2")
        pre = run_e3_preflight(E3WindowConfig(training_runtime=runtime))
        assert pre.gates["ORIGINAL_TRAINING_RUNTIME_BOUND"] is True
        assert BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME not in pre.blockers

    def test_step_baseline_gate_follows_loaded_state(self):
        ok = run_e3_preflight(E3WindowConfig(loaded_state={"global_step": 12}))
        assert ok.gates["OPTIMIZER_STEP_BASELINE_MEASURABLE"] is True
        bad = run_e3_preflight(E3WindowConfig(loaded_state={"global_step": -1}))
        assert bad.gates["OPTIMIZER_STEP_BASELINE_MEASURABLE"] is False
        assert BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED in bad.blockers

    def test_zero_memory_is_never_a_production_mode(self):
        pre = run_e3_preflight(E3WindowConfig(memory_mode="ZERO_MEMORY"))
        assert "ZERO_MEMORY_NOT_A_PRODUCTION_MODE" in pre.blockers
