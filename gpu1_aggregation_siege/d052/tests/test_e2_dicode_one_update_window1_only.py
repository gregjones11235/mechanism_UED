"""§八 (director smoke handoff): the two-window smoke executes EXACTLY ONE
CanonicalDiCode optimizer update — in window k+1 only, over the 15
curriculum task ids (window0 delta=0, window1 delta=1, total=1).

Contract under test:

* with the director-declared DiCode batch binding injected, window k+1's
  single update is executed through the director-shared
  CanonicalDiCodeOneUpdateRuntime over the 15 curriculum ids — the
  OriginalTask is NEVER in batch_candidate_ids and is appended internally
  once;
* window k trains NOTHING (the smoke policy) — total updates = 1;
* a training contract WITHOUT the DiCode surface fails closed
  (REAL_DICODE_RUNTIME_MISSING — direction two never implements a second
  optimizer).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: the DiCode
runtime is a scripted fake-real stand-in that records the plan; NO real
optimizer runs, and no REAL_* flag flips.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.director_runtime_bundle import (
    DiCodeBatchBindingData,
)
from d052.feedback_llm_ued.llm_backend import RealBackendAdapter
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
)
from d052.feedback_llm_ued.student_binding import (
    EXECUTED_ONE_UPDATE_STATUS,
    RealTwoWindowSmokePolicy,
)

from test_feedback_llm_ued_envcoder_sequence import (
    TEST_BACKEND_ID,
    TEST_MODEL_ID,
    scripted_board_transport,
    scripted_real_env_coder,
)
from test_feedback_llm_ued_two_window_update_count import (
    ScriptedRealProbeRunner,
    student_contract,
)

SKIPPED_STATUS = "SKIPPED_SMOKE_POLICY_UPDATE_WINDOW"
DICODE_RUNTIME_ID = text_sha256("TEST_ONLY_DICODE_ONE_UPDATE_RUNTIME")
RUNTIME_BUNDLE_HASH = text_sha256("TEST_ONLY_RUNTIME_BUNDLE")


class ScriptedDiCodeOneUpdateRuntime:
    """TEST_ONLY / SYNTHETIC stand-in for the director-shared
    CanonicalDiCodeOneUpdateRuntime: records every plan, reports one
    optimizer update and signs a director-verified round-trip (NO real
    optimizer involved)."""

    registry_identity = DICODE_RUNTIME_ID
    verifier_id = text_sha256("TEST_ONLY_DIRECTOR_VERIFIER_IDENTITY")

    def __init__(self):
        self.plans = []

    def run_one_dicode_update(self, *, batch_plan):
        self.plans.append(batch_plan)
        return SimpleNamespace(
            window=batch_plan.window, optimizer_steps=1, env_steps=8,
            checkpoint_hash_after=text_sha256(
                f"TEST_ONLY_DICODE_CKPT_{batch_plan.window}"))

    def verify_director_round_trip(self, *, window, checkpoint_hash):
        from e2_test_sign_helpers import (
            director_round_trip_payload,
            sign_director_verified_round_trip,
        )
        return sign_director_verified_round_trip(
            director_round_trip_payload(
                window, checkpoint_hash, RUNTIME_BUNDLE_HASH,
                verifier_id=self.verifier_id))


def binding(original_task_id="DICODE_ORIGINAL_TASK_V1"):
    return DiCodeBatchBindingData(
        dynamic_task_count=C.DICODE_CURRICULUM_DYNAMIC,
        non_target_anchor_count=C.DICODE_CURRICULUM_NON_TARGET_ANCHORS,
        curriculum_task_count=C.DICODE_CURRICULUM_TASK_COUNT,
        non_target_anchor_ids=list(C.GLOBAL_CANONICAL_ANCHOR_IDS[:3]),
        original_task_id=original_task_id,
        original_task_proportion=C.DICODE_ORIGINAL_TASK_PROPORTION,
        total_task_count=C.DICODE_BATCH_TOTAL_TASKS)


def make_controller(runtime=None, *,
                    batch_binding=None) -> FeedbackUEDController:
    authorization = RealRuntimeAuthorization(
        real_llm_backend=True, real_envcoder=True, real_probe=True,
        real_training=True)
    backend = RealBackendAdapter(scripted_board_transport(),
                                 backend_id=TEST_BACKEND_ID,
                                 model_id=TEST_MODEL_ID, authorized=True)
    return FeedbackUEDController(
        C.MODE_NORMAL_FEEDBACK, backend=backend,
        probe_runner=ScriptedRealProbeRunner(),
        runtime_authorization=authorization,
        student_init_contract=student_contract(),
        training_contract=runtime,
        real_env_coder_callable=scripted_real_env_coder({}),
        two_window_smoke_policy=RealTwoWindowSmokePolicy(),
        dicode_batch_binding=batch_binding,
        dicode_runtime_identity=DICODE_RUNTIME_ID,
        runtime_bundle_hash=RUNTIME_BUNDLE_HASH,
        director_selected_candidate_id=C.STRONG_STUDENT_CANDIDATE_ID)


class TestOneDiCodeUpdateWindow1Only:
    def test_exactly_one_update_in_window_one(self):
        runtime = ScriptedDiCodeOneUpdateRuntime()
        controller = make_controller(runtime, batch_binding=binding())
        summary = controller.run(max_windows=2)
        assert summary.n_windows == 2
        assert summary.request_control_stopped is False

        #: exactly ONE plan consumed, in window k+1
        assert len(runtime.plans) == 1
        plan = runtime.plans[0]
        assert plan.window == 1
        #: window k+1 consumes the 15 curriculum ids — the OriginalTask is
        #: NOT in batch_candidate_ids
        assert len(plan.batch_candidate_ids) == 15
        assert plan.original_task_id not in plan.batch_candidate_ids
        assert plan.total_task_count == 16

        statuses = [t.status for t in controller.training_log]
        assert statuses.count(EXECUTED_ONE_UPDATE_STATUS) == 1
        assert statuses.count(SKIPPED_STATUS) == 1

    def test_window0_update_delta_is_zero(self):
        runtime = ScriptedDiCodeOneUpdateRuntime()
        controller = make_controller(runtime, batch_binding=binding())
        controller.run(max_windows=2)
        statuses = [t.status for t in controller.training_log]
        phase_d = [s for s in statuses
                   if s in (SKIPPED_STATUS, EXECUTED_ONE_UPDATE_STATUS)]
        assert phase_d == [SKIPPED_STATUS, EXECUTED_ONE_UPDATE_STATUS]

    def test_contract_without_dicode_surface_fails_closed(self):
        #: a legacy SharedTrainingContract-shaped object (no
        #: run_one_dicode_update) cannot serve the DiCode plan — direction
        #: two never implements a second optimizer
        legacy = SimpleNamespace(
            run_one_optimizer_update=lambda **kw: None,
            save_checkpoint=lambda **kw: "hash",
            load_checkpoint=lambda **kw: None,
            verify_director_round_trip=lambda **kw: None)
        controller = make_controller(legacy, batch_binding=binding())
        with pytest.raises(Exception,
                           match="REAL_DICODE_RUNTIME_MISSING"):
            controller.run(max_windows=2)

    def test_no_binding_production_fails_closed(self):
        #: the PRODUCTION controller never auto-selects the TEST_ONLY path:
        #: without the director DiCode batch binding the update is refused
        legacy = SimpleNamespace(
            run_one_optimizer_update=lambda **kw: None,
            save_checkpoint=lambda **kw: "hash",
            load_checkpoint=lambda **kw: None,
            verify_director_round_trip=lambda **kw: None)
        controller = make_controller(legacy, batch_binding=None)
        with pytest.raises(RuntimeError,
                           match="REAL_DICODE_BATCH_PLAN_REQUIRED"):
            controller.run(max_windows=2)


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        runtime = ScriptedDiCodeOneUpdateRuntime()
        controller = make_controller(runtime, batch_binding=binding())
        controller.run(max_windows=2)
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.E2_REAL_SMOKE_AUTHORIZED is False
        assert C.FORMAL_EXPERIMENT_AUTHORIZED is False
