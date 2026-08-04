"""P0-10 (§19 seam coverage): the two-window smoke executes EXACTLY ONE
optimizer update.

Contract under test:

* window k freezes feedback_k and trains NOTHING (delta=0) — there is no
  prior feedback to consume; window k+1 (plan_{k+1} built from
  feedback_k) executes EXACTLY ONE optimizer update (delta=1) over the
  probe-selected final batch, wrapped in the checkpoint save/load
  round-trip;
* the end-of-run count check fails closed
  (TWO_WINDOW_SMOKE_UPDATE_COUNT_MISMATCH) when a completed run deviates
  from ``RealTwoWindowSmokePolicy.updates_expected_total``;
* without the policy the historical behavior is preserved (every
  training-authorized window updates — the regression contrast that
  motivated P0-10);
* the policy is inert without training authorization and validates its
  fields fail-closed.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: the board runs
on a scripted REAL-kind backend, the EnvCoder is a scripted artifact
factory, the probe runner is a scripted real-kind runner (the grant
ladder requires every lower capability for training), and the training
contract is a scripted checkpoint/optimizer surface — NO real LLM call,
NO simulator episode, NO real optimizer step, and NO passing test flips
a REAL_* flag. Snapshot/restore of a mid-smoke run is covered by the
persistence suite; the policy itself is immutable state.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.feedback_contracts import ProbeMetrics
from d052.feedback_llm_ued.llm_backend import RealBackendAdapter
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
)
from d052.feedback_llm_ued.shared_runtime_binding import (
    TrainingUpdateResult,
)
from d052.feedback_llm_ued.student_binding import (
    EXECUTED_ONE_UPDATE_STATUS,
    RealTwoWindowSmokePolicy,
    sign_full_state_round_trip,
)

from test_feedback_llm_ued_envcoder_sequence import (
    TEST_BACKEND_ID,
    TEST_MODEL_ID,
    scripted_board_transport,
    scripted_real_env_coder,
)

SKIPPED_STATUS = "SKIPPED_SMOKE_POLICY_UPDATE_WINDOW"


class ScriptedTrainingContract:
    """TEST_ONLY / SYNTHETIC shared training surface: records every
    save/load/update call; the optimizer step changes the checkpoint
    hash exactly when an update runs (NO real optimizer involved).
    P0-11: it also plays the director-verifier and signs a PASSING
    full-state round-trip attestation for every reloaded checkpoint."""

    registry_identity = text_sha256("TEST_ONLY_TRAINING_CONTRACT")
    verifier_id = text_sha256("TEST_ONLY_DIRECTOR_VERIFIER_IDENTITY")

    def __init__(self) -> None:
        self.update_calls = []
        self.save_calls = []
        self.load_calls = []
        self.round_trip_verifications = []
        self._version = 0
        self._last_hash = text_sha256("TEST_ONLY_GENESIS_STATE")

    def save_checkpoint(self, *, tag: str) -> str:
        self._version += 1
        checkpoint_hash = text_sha256(
            f"TEST_ONLY_CHECKPOINT_STATE_{self._version}")
        self.save_calls.append((tag, checkpoint_hash))
        self._last_hash = checkpoint_hash
        return checkpoint_hash

    def run_one_optimizer_update(self, *, window: int,
                                 batch_candidate_ids) -> TrainingUpdateResult:
        self.update_calls.append((window, tuple(batch_candidate_ids)))
        return TrainingUpdateResult(
            window=window, optimizer_steps=1,
            env_steps=len(list(batch_candidate_ids)),
            checkpoint_hash_before=self._last_hash,
            checkpoint_hash_after=text_sha256(
                f"TEST_ONLY_POST_UPDATE_{window}"))

    def load_checkpoint(self, *, checkpoint_hash: str) -> None:
        self.load_calls.append(checkpoint_hash)

    def verify_full_state_round_trip(self, *, window: int,
                                     checkpoint_hash: str):
        self.round_trip_verifications.append((window, checkpoint_hash))
        state_hash = text_sha256(
            f"TEST_ONLY_FULL_STATE_{checkpoint_hash}")
        return sign_full_state_round_trip(dict(
            window=window, checkpoint_hash=checkpoint_hash,
            state_hash_before_save=state_hash,
            state_hash_after_reload=state_hash,
            verifier_id=self.verifier_id, verified=True))


class ScriptedRealProbeRunner:
    """TEST_ONLY / SYNTHETIC probe runner on the loop's
    ``SimulatorProbeRunner`` surface: ``real_simulator=True`` (the grant
    ladder requires it for training) and scripted coarse metrics — NO
    simulator runs, NO real episode executes."""

    real_simulator = True
    status = "READY"
    runner_id = "feedback_llm_ued.test_only_scripted_probe_runner.v1"

    def __init__(self) -> None:
        self.probe_calls = 0
        self.total_transitions = 0

    def probe(self, candidate, *, stage, student_episodes,
              reference_episodes) -> ProbeMetrics:
        self.probe_calls += 1
        self.total_transitions += 24
        #: two honesty constraints at once: the Student success rate stays
        #: at the seeded hypotheses' predicted value (the comparator grades
        #: AGREE, never OPPOSITE — opposite matches escalate the critic),
        #: and the Student-vs-Reference gap stays 0.02 < REFERENCE_GAP_LOW
        #: (no high-severity evidence) — so the smoke reaches window k+1
        #: to exercise the single-update contract
        return ProbeMetrics(
            stage=stage, student_success_rate=0.5,
            student_behavior_activation=0.5, student_front_progress=0.5,
            reference_success_rate=0.52, reference_mean_progress=0.55,
            reference_behavior_activation=0.5, global_retention=1.0,
            regret=0.4, learnability=0.4, simulator_transitions=24)


def student_contract():
    return SimpleNamespace(
        candidate_id=C.STRONG_STUDENT_CANDIDATE_ID,
        architecture_family="RMT16",
        memory_family="RMT16_ORIGINAL",
        carry_mode="PERSISTENT",
        parameter_tree_hash=text_sha256("TEST_ONLY_STUDENT_PARAM_TREE"),
        checkpoint_global_step=98304)


def make_controller(*, policy=None, contract=None) -> FeedbackUEDController:
    #: the grant ladder requires EVERY lower capability for training
    #: (real_training -> real_probe -> real_envcoder -> real_llm_backend);
    #: the probe runner is the scripted TEST_ONLY real-kind runner and the
    #: feedback staging stays symbolic — this isolates the update-count
    #: seam under test
    authorization = RealRuntimeAuthorization(
        real_llm_backend=True, real_envcoder=True, real_probe=True,
        real_training=True)
    backend = RealBackendAdapter(scripted_board_transport(),
                                 backend_id=TEST_BACKEND_ID,
                                 model_id=TEST_MODEL_ID,
                                 authorized=True)
    return FeedbackUEDController(
        C.MODE_NORMAL_FEEDBACK, backend=backend,
        probe_runner=ScriptedRealProbeRunner(),
        runtime_authorization=authorization,
        student_init_contract=student_contract(),
        training_contract=contract,
        real_env_coder_callable=scripted_real_env_coder({}),
        two_window_smoke_policy=policy)


class TestExactlyOneUpdate:
    def test_smoke_executes_exactly_one_update_in_window_one(self):
        contract = ScriptedTrainingContract()
        controller = make_controller(policy=RealTwoWindowSmokePolicy(),
                                     contract=contract)
        summary = controller.run(max_windows=2)
        assert summary.n_windows == 2

        #: delta semantics: window 0 trains NOTHING, window 1 updates ONCE
        assert [window for window, _ in contract.update_calls] == [1]
        statuses = [t.status for t in controller.training_log]
        assert statuses.count(EXECUTED_ONE_UPDATE_STATUS) == 1
        assert statuses.count(SKIPPED_STATUS) == 1
        #: phase-D ordering: window 0 skipped, window 1 executed (each
        #: preceded by its REVISION-phase DEFERRED record)
        phase_d = [s for s in statuses
                   if s in (SKIPPED_STATUS, EXECUTED_ONE_UPDATE_STATUS)]
        assert phase_d == [SKIPPED_STATUS, EXECUTED_ONE_UPDATE_STATUS]

    def test_update_consumes_probe_selected_final_batch(self):
        contract = ScriptedTrainingContract()
        controller = make_controller(policy=RealTwoWindowSmokePolicy(),
                                     contract=contract)
        controller.run(max_windows=2)
        window, batch_ids = contract.update_calls[0]
        assert window == 1
        #: the final batch (12 dynamic + 4 anchors), probe-selected after
        #: window 1 consumed feedback_0
        assert len(batch_ids) == C.FINAL_BATCH
        assert len(set(batch_ids)) == C.FINAL_BATCH

    def test_checkpoint_roundtrip_wraps_the_single_update(self):
        contract = ScriptedTrainingContract()
        controller = make_controller(policy=RealTwoWindowSmokePolicy(),
                                     contract=contract)
        controller.run(max_windows=2)
        assert [tag for tag, _ in contract.save_calls] == [
            "window-01-pre-update", "window-01-post-update"]
        hash_before, hash_after = (h for _, h in contract.save_calls)
        assert hash_before != hash_after
        #: the post-update checkpoint reloads (round-trip) and ONLY then
        assert contract.load_calls == [hash_after]

    def test_no_policy_updates_every_training_window(self):
        #: regression contrast: without the smoke policy the historical
        #: behavior updates in BOTH windows — exactly what P0-10 forbids
        #: for the two-window smoke
        contract = ScriptedTrainingContract()
        controller = make_controller(policy=None, contract=contract)
        controller.run(max_windows=2)
        assert [window for window, _ in contract.update_calls] == [0, 1]

    def test_unreachable_update_window_fails_closed(self):
        contract = ScriptedTrainingContract()
        controller = make_controller(
            policy=RealTwoWindowSmokePolicy(update_window_index=5),
            contract=contract)
        with pytest.raises(RuntimeError,
                           match="TWO_WINDOW_SMOKE_UPDATE_COUNT_MISMATCH"):
            controller.run(max_windows=2)
        assert contract.update_calls == []

    def test_zero_expected_and_no_update_window_is_consistent(self):
        contract = ScriptedTrainingContract()
        controller = make_controller(
            policy=RealTwoWindowSmokePolicy(updates_expected_total=0,
                                            update_window_index=5),
            contract=contract)
        summary = controller.run(max_windows=2)
        assert summary.n_windows == 2
        assert contract.update_calls == []

    def test_policy_inert_without_training_authorization(self):
        #: the default mock/symbolic controller never trains; the policy
        #: must not disturb the historical SKIPPED_UNAUTHORIZED record
        controller = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            two_window_smoke_policy=RealTwoWindowSmokePolicy())
        summary = controller.run(max_windows=2)
        assert summary.n_windows == 2
        assert all(t.status == "SKIPPED_UNAUTHORIZED"
                   for t in controller.training_log)


class TestPolicyValidation:
    @pytest.mark.parametrize("bad_count", [-1, True, "1", None])
    def test_illegal_updates_expected_total(self, bad_count):
        with pytest.raises(ValueError,
                           match="ILLEGAL_SMOKE_POLICY_UPDATE_COUNT"):
            RealTwoWindowSmokePolicy(updates_expected_total=bad_count)

    @pytest.mark.parametrize("bad_window", [-1, True, "1", None])
    def test_illegal_update_window_index(self, bad_window):
        with pytest.raises(ValueError,
                           match="ILLEGAL_SMOKE_POLICY_UPDATE_WINDOW"):
            RealTwoWindowSmokePolicy(update_window_index=bad_window)

    def test_defaults(self):
        policy = RealTwoWindowSmokePolicy()
        assert policy.updates_expected_total == 1
        assert policy.update_window_index == 1


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        contract = ScriptedTrainingContract()
        controller = make_controller(policy=RealTwoWindowSmokePolicy(),
                                     contract=contract)
        controller.run(max_windows=2)
        #: the update EXECUTED inside the scripted contract, yet no
        #: capability constant may flip
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.E2_PILOT_AUTHORIZED is False
