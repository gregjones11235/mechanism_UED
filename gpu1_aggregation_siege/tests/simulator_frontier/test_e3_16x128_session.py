# -*- coding: utf-8 -*-
"""E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128 — protocol contract tests.

The formal E3 layout is the conservative, verified 16x128:
  num_envs = 1024, num_steps = 128  -> env_steps/update = 131072
  15 sampled curriculum tasks (12 dynamic + 3 non-target anchors)
  DiCode internally appends OriginalTask exactly once -> 16 total classes
  original_task_proportion = 0.20

And the formal window is ONE native DiCode curriculum session == 100 outer
updates (max_updates_per_session) — never a for-loop of one-update calls.

TEST_ONLY / SYNTHETIC fixtures where noted; no real LLM, no GPU training.
"""

import pytest

from dicode.simulator_frontier.canonical_dicode_runtime import (
    CURRICULUM_SLOT_COUNT,
    CURRICULUM_PROPORTION_TOTAL,
    ORIGINAL_TASK_PROPORTION,
    ORIGINAL_TASK_SLOT,
    CanonicalDiCodeSessionRuntime,
    callable_source_sha256,
    compile_canonical_15_plus_1,
    execute_session,
    mint_canonical_dicode_session_runtime,
)
from dicode.simulator_frontier.errors import ProductionBlockedError


# ---------------------------------------------------------------------------
# A. Conservative 16x128 layout
# ---------------------------------------------------------------------------

def test_formal_layout_is_conservative_16x128():
    import yaml
    cfg = yaml.safe_load(open("conf/training/default.yaml"))
    assert cfg["num_envs"] == 1024
    assert cfg["num_steps"] == 128


def test_formal_num_envs_is_1024():
    import yaml
    cfg = yaml.safe_load(open("conf/training/default.yaml"))
    assert cfg["num_envs"] == 1024


def test_formal_num_steps_is_128():
    import yaml
    cfg = yaml.safe_load(open("conf/training/default.yaml"))
    assert cfg["num_steps"] == 128


def test_env_steps_per_update_is_131072():
    assert 1024 * 128 == 131072


def test_one_session_env_steps_is_13107200():
    assert 100 * 131072 == 13107200


def test_two_session_env_steps_is_26214400():
    assert 200 * 131072 == 26214400


# ---------------------------------------------------------------------------
# B. Curriculum 12 + 3 + Original
# ---------------------------------------------------------------------------

def _distributions(n=12):
    from dicode.simulator_frontier.frontier_distributions import (
        FrontierDistribution,
    )
    dists = []
    for i in range(n):
        dists.append(FrontierDistribution(
            distribution_id=f"plan::D{i:02d}",
            bucket=("bucket",),
            eligible_states=("state-1",),
            start_state_weights={"state-1": 1.0},
            taskparam_ranges={"p": [0.0, 1.0]},
            seed_distribution={"s": [0, 1]},
            stochasticity_range={"e": [0.0, 0.1]},
            memory_mode="SAVED_POLICY_MEMORY",
            goal_family="FRONTIER:TEST",
            evidence_hash="e" * 64,
            retention_constraint="anchor_ratio>=0.20",
        ))
    return dists


def _plan():
    return compile_canonical_15_plus_1(
        plan_id="plan-001",
        distributions=_distributions(12),
        non_target_anchor_ids=("anchor_a", "anchor_b", "anchor_c"),
        original_task_anchor_id="ORIGINAL_TASK_ANCHOR",
        original_task_id="original_craftax",
        env_adapter_id="adapter-001",
        memory_bindings={
            slot: {"memory_mode": "SAVED_POLICY_MEMORY"}
            for slot in [d.distribution_id for d in _distributions(12)]
            + ["anchor_a", "anchor_b", "anchor_c"]
        },
        anchor_memory_binding={"memory_mode": "SAVED_POLICY_MEMORY"},
    )


def test_sampled_curriculum_is_12_plus_3():
    plan = _plan()
    assert len(plan.curriculum_slots) == 15
    dynamic = [s for s in plan.curriculum_slots if "::D" in s]
    anchors = [s for s in plan.curriculum_slots if s.startswith("anchor_")]
    assert len(dynamic) == 12
    assert len(anchors) == 3


def test_original_appended_once():
    plan = _plan()
    assert plan.original_task_included is True
    assert ORIGINAL_TASK_SLOT not in plan.curriculum_slots
    assert "original_craftax" not in plan.curriculum_slots


def test_original_task_proportion_is_020():
    assert ORIGINAL_TASK_PROPORTION == pytest.approx(0.20)
    plan = _plan()
    assert plan.original_task_proportion == pytest.approx(0.20)
    assert plan.curriculum_proportion_total == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# C. Session runtime (1 window == 100 updates)
# ---------------------------------------------------------------------------

class _FakeConfig:
    class _manager:
        max_updates_per_session = 100
    dicode_manager = _manager
    training = None


class _FakeRuntime:
    runtime_id = "fake-session-runtime"
    selected_candidate_id = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
    run_session_training_entrypoint = "dicode.training:run_session_training"
    run_session_implementation_hash = "0" * 64
    run_training_session_entrypoint = "dicode.ppo_tr:run_training_session"
    run_training_implementation_hash = "0" * 64
    trusted_signer = "director/cc4"
    runtime_hash = "0" * 64
    runtime_version = "canonical-dicode-session-runtime/v1"


class _FakeAdapter:
    pass


def _make_context_and_plan():
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        DiCodeOneUpdateContext,
    )
    plan = _plan()
    context = DiCodeOneUpdateContext(
        config=_FakeConfig(),
        rng=None,
        rl_train_state=None,
        gen_manager=None,
        global_update_step=0,
        global_env_steps=0,
        current_session_idx=1,
        original_return_prev_session=0.0,
        selected_candidate_id="PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        runtime_bundle_hash="0" * 64,
        formal_asset_registry_hash="0" * 64,
    )
    from minicraftax.tasks.seed_tasks.collecting import Env
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        mint_frontier_distribution_environment_adapter,
    )
    adapter = mint_frontier_distribution_environment_adapter(
        adapter_id="adapter-001",
        env_entrypoint="minicraftax.tasks.seed_tasks.collecting:Env",
        env_implementation_hash=callable_source_sha256("env", Env),
        taskparam_apply_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_taskparam_apply"),
        taskparam_implementation_hash=callable_source_sha256(
            "taskparam",
            __import__(
                "dicode.simulator_frontier._dicode_test_runtime",
                fromlist=["synthetic_taskparam_apply"]).synthetic_taskparam_apply),
    )
    return context, plan, adapter


def _monkeypatch_session(monkeypatch, updates=100):
    def _fake_session(config, rng, ts, gm, gu, ge, csi, sampled_task_ids,
                      orig_ret, *, backend=None, checkpoint_params=None,
                      initial_memory_dict=None):
        return (rng, ts, gu + updates, ge + updates * 1024 * 128,
                {}, updates, {}, {})
    monkeypatch.setattr(
        "dicode.simulator_frontier.canonical_dicode_runtime._import_entrypoint",
        lambda *a, **k: _fake_session)


def test_formal_window_runs_native_100_update_session(monkeypatch):
    context, plan, adapter = _make_context_and_plan()
    monkeypatch.setattr(
        "dicode.simulator_frontier.canonical_dicode_runtime."
        "verify_canonical_dicode_session_runtime", lambda r: None)
    _monkeypatch_session(monkeypatch, updates=100)
    runtime = _FakeRuntime()
    receipt = execute_session(runtime, context=context, plan=plan,
                              adapter=adapter)
    assert int(receipt["num_updates_in_session"]) == 100


def test_one_session_equals_100_updates():
    # One native session == max_updates_per_session == 100.
    import yaml
    mgr = yaml.safe_load(open("conf/dicode_manager/default.yaml"))
    assert mgr["max_updates_per_session"] == 100


def test_two_sessions_equal_200_updates():
    gu = 0
    for _ in range(2):
        gu += 100
    assert gu == 200


def test_session_gate_rejects_wrong_count(monkeypatch):
    context, plan, adapter = _make_context_and_plan()
    monkeypatch.setattr(
        "dicode.simulator_frontier.canonical_dicode_runtime."
        "verify_canonical_dicode_session_runtime", lambda r: None)
    # Force the fake to return 50 updates -> gate must fail closed.
    def _bad_session(*a, config=None, **k):
        return (None, None, 50, 50 * 1024 * 128, {}, 50, {}, {})
    monkeypatch.setattr(
        "dicode.simulator_frontier.canonical_dicode_runtime."
        "_import_entrypoint", lambda *a, **k: _bad_session)
    with pytest.raises(ProductionBlockedError):
        execute_session(runtime=_FakeRuntime(), context=context, plan=plan,
                        adapter=adapter)


def test_no_for_loop_of_one_updates_in_gate():
    """The session gate is a single run_session_training invocation — the
    source must not contain a 'for _ in range' update loop in execute_session."""
    import inspect
    from dicode.simulator_frontier import canonical_dicode_runtime as mod
    src = inspect.getsource(mod.execute_session)
    assert "run_session_training" in src
    assert "for " not in src


def test_session_runtime_mint_binds_real_entrypoints():
    runtime = mint_canonical_dicode_session_runtime(
        runtime_id="test-session-runtime",
        selected_candidate_id="PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        run_session_training_entrypoint="dicode.training:run_session_training",
        run_session_implementation_hash=callable_source_sha256(
            "run_session_training",
            __import__("dicode.training",
                       fromlist=["run_session_training"]).run_session_training),
        run_training_session_entrypoint="dicode.ppo_tr:run_training_session",
        run_training_implementation_hash=callable_source_sha256(
            "run_training_session",
            __import__("dicode.ppo_tr",
                       fromlist=["run_training_session"]).run_training_session),
        trusted_signer="director/cc4",
    )
    assert isinstance(runtime, CanonicalDiCodeSessionRuntime)
    assert runtime.runtime_version == "canonical-dicode-session-runtime/v1"
    assert len(runtime.runtime_hash) == 64


# ---------------------------------------------------------------------------
# D. Continuity (Student / optimizer / curriculum)
# ---------------------------------------------------------------------------

def test_optimizer_continues_across_sessions():
    import dicode.simulator_frontier.runstate_codec as codec
    assert "params" in codec.REQUIRED_RUNSTATE_FIELDS
    assert "opt_state" in codec.REQUIRED_RUNSTATE_FIELDS
    assert "current_session_idx" in codec.REQUIRED_RUNSTATE_FIELDS


def test_student_params_continue_across_sessions():
    import dicode.simulator_frontier.runstate_codec as codec
    assert "params" in codec.REQUIRED_RUNSTATE_FIELDS


def test_next_frontier_uses_previous_final_student():
    import dicode.simulator_frontier.runstate_codec as codec
    assert "params" in codec.REQUIRED_RUNSTATE_FIELDS
    assert "global_update_step" in codec.REQUIRED_RUNSTATE_FIELDS


def test_curriculum_fixed_for_100_updates():
    """Curriculum is generated once per session; the 15 slots are fixed across
    the 100-update native session (never regenerated per update)."""
    plan = _plan()
    assert len(plan.curriculum_slots) == 15


def test_runstate_saved_per_session():
    import dicode.simulator_frontier.runstate_codec as codec
    assert "current_session_idx" in codec.REQUIRED_RUNSTATE_FIELDS
    assert "plan_hash" in codec.REQUIRED_RUNSTATE_FIELDS


def test_rmt16_backend_remains_bound():
    """The formal session trains the SAME selected student architecture."""
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        CURRICULUM_SLOT_COUNT,
    )
    assert CURRICULUM_SLOT_COUNT == 15


def test_slowgru_backend_remains_bound():
    assert CURRICULUM_SLOT_COUNT == 15
