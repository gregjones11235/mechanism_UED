# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-15) + BUG-E3-01/02/03/07/08/09: the production preflight
is a fail-closed checklist consuming SIGNED capability descriptors and the
CANONICAL DiCode training chain (CanonicalDiCodeOneUpdateRuntime,
FrontierDistributionEnvironmentAdapter, hydra config, real GenManager,
rl_train_state, RunState checkpoint dir).  The legacy OriginalTrainingRuntime
loss/update path is never consumed by the production pipeline.
"""

import pytest

from dicode.simulator_frontier import e3_window
from dicode.simulator_frontier.e3_window import (
    BLOCKED_DICODE_ANCHOR_SEMANTICS,
    BLOCKED_NO_BOUND_CANONICAL_DICODE_RUNTIME,
    BLOCKED_NO_BOUND_FRONTIER_ENV_ADAPTER,
    BLOCKED_NO_DICODE_CONFIG,
    BLOCKED_NO_DICODE_GEN_MANAGER,
    BLOCKED_NO_DICODE_RL_TRAIN_STATE,
    BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN,
    BLOCKED_NO_RUNSTATE_CHECKPOINT_DIR,
    BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED,
    BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY,
    E3WindowConfig,
    run_e3_preflight,
)
from dicode.simulator_frontier.canonical_dicode_runtime import (
    callable_source_sha256,
    mint_canonical_dicode_one_update_runtime,
    mint_frontier_distribution_environment_adapter,
)
from dicode.simulator_frontier.surface_capability import (
    mint_training_surface_capability,
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


def _canonical_runtime():
    from dicode.simulator_frontier import _dicode_test_runtime as t
    return mint_canonical_dicode_one_update_runtime(
        runtime_id="rt-canon-001",
        selected_candidate_id="PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        run_session_training_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_run_session_training"),
        run_session_implementation_hash=callable_source_sha256(
            "test-session", t.synthetic_run_session_training),
        run_training_session_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_run_training_session"),
        run_training_implementation_hash=callable_source_sha256(
            "test-training", t.synthetic_run_training_session),
        trusted_signer="director/cc4-test")


def _env_adapter():
    from dicode.simulator_frontier import _dicode_test_runtime as t
    return mint_frontier_distribution_environment_adapter(
        adapter_id="adapter-001",
        env_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_env_factory"),
        env_implementation_hash=callable_source_sha256(
            "test-env", t.synthetic_env_factory),
        taskparam_apply_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_taskparam_apply"),
        taskparam_implementation_hash=callable_source_sha256(
            "test-tp", t.synthetic_taskparam_apply),
    )


def _train_state():
    from types import SimpleNamespace
    return SimpleNamespace(params={"w": 1}, opt_state={"m": 1}, step=12)


class TestSpoofableProbesRemoved:
    def test_exception_inference_probe_no_longer_exists(self):
        assert not hasattr(e3_window, "_probe_training_surface")


class TestFullChecklist:
    def test_default_config_reports_every_named_blocker(self):
        pre = run_e3_preflight(E3WindowConfig())
        assert not pre.ready
        blockers = set(pre.blockers)
        assert BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY in blockers
        assert BLOCKED_NO_BOUND_CANONICAL_DICODE_RUNTIME in blockers
        assert BLOCKED_NO_BOUND_FRONTIER_ENV_ADAPTER in blockers
        assert BLOCKED_NO_DICODE_CONFIG in blockers
        assert BLOCKED_NO_DICODE_GEN_MANAGER in blockers
        assert BLOCKED_NO_DICODE_RL_TRAIN_STATE in blockers
        assert BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED in blockers
        assert BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN in blockers
        assert BLOCKED_NO_RUNSTATE_CHECKPOINT_DIR in blockers
        assert BLOCKED_DICODE_ANCHOR_SEMANTICS in blockers
        assert "BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE" in blockers
        assert "BLOCKED_SHARED_ANCHOR_MANIFEST" in blockers
        assert "BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY" in blockers
        assert "REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT" in blockers

    def test_signed_capability_enables_policy_surface(self):
        student = _student()
        config = E3WindowConfig(
            student=student, student_params=student._params,
            training_surface_capability=_capability(student))
        pre = run_e3_preflight(config)
        assert pre.gates["TRAINING_SURFACE_CAPABILITY_SIGNED"] is True
        assert pre.gates["STUDENT_POLICY_SURFACE"] is True
        assert pre.gates["STUDENT_TRAINING_SURFACE"] is True

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

    def test_bound_canonical_runtime_enables_runtime_gate(self):
        pre = run_e3_preflight(E3WindowConfig(
            canonical_dicode_runtime=_canonical_runtime()))
        assert pre.gates["CANONICAL_DICODE_RUNTIME_BOUND"] is True
        assert BLOCKED_NO_BOUND_CANONICAL_DICODE_RUNTIME not in pre.blockers

    def test_canonical_chain_gates_follow_injected_dependencies(self):
        config = E3WindowConfig(
            canonical_dicode_runtime=_canonical_runtime(),
            frontier_env_adapter=_env_adapter(),
            dicode_config=object(),
            gen_manager=type("GM", (), {"archive": object()})(),
            rl_train_state=_train_state(),
            taskparam_apply_fn=lambda params_env, taskparams: params_env,
            runstate_checkpoint_dir="/tmp/e3_rs",
            non_target_anchor_ids=("A", "B", "C"),
            original_task_anchor_id="ORIGINAL",
        )
        pre = run_e3_preflight(config)
        for gate in ("CANONICAL_DICODE_RUNTIME_BOUND",
                     "FRONTIER_ENV_ADAPTER_BOUND",
                     "DICODE_CONFIG_BOUND",
                     "DICODE_GEN_MANAGER_BOUND",
                     "DICODE_RL_TRAIN_STATE_BOUND",
                     "OPTIMIZER_STEP_BASELINE_MEASURABLE",
                     "TASKPARAM_APPLY_FN_INJECTED",
                     "RUNSTATE_CHECKPOINT_BOUND",
                     "DICODE_ANCHOR_SEMANTICS_BOUND"):
            assert pre.gates[gate] is True, gate

    def test_step_baseline_gate_follows_train_state(self):
        ok = run_e3_preflight(E3WindowConfig(rl_train_state=_train_state()))
        assert ok.gates["OPTIMIZER_STEP_BASELINE_MEASURABLE"] is True
        bad = run_e3_preflight(E3WindowConfig(rl_train_state=type(
            "TS", (), {"params": {}, "opt_state": {}, "step": -1})()))
        assert bad.gates["OPTIMIZER_STEP_BASELINE_MEASURABLE"] is False
        assert BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED in bad.blockers

    def test_anchor_semantics_reject_wrong_arity(self):
        pre = run_e3_preflight(E3WindowConfig(
            non_target_anchor_ids=("A", "B", "C", "D"),
            original_task_anchor_id="ORIGINAL"))
        assert pre.gates["DICODE_ANCHOR_SEMANTICS_BOUND"] is False
        assert BLOCKED_DICODE_ANCHOR_SEMANTICS in pre.blockers

    def test_zero_memory_is_never_a_production_mode(self):
        pre = run_e3_preflight(E3WindowConfig(memory_mode="ZERO_MEMORY"))
        assert "ZERO_MEMORY_NOT_A_PRODUCTION_MODE" in pre.blockers
