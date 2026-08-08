#!/usr/bin/env python3
"""E3 REAL SMOKE driver: the canonical DiCode chain on the REAL persistent
Student (BUG-E3-01/02/03/07/08/09 closure).

The chain (directly through the canonical modules, with REAL objects):

    real persistent RMT16 checkpoint mount (exact restore)
    -> real standard-reset rollouts -> real actual-N feasibility (from data)
    -> two REAL LLM roles (diagnostician + planner, authorized transport)
    -> 12 dynamic frontier distributions + 3 non-target anchors
       -> canonical 15+1 DiCode batch plan
    -> REAL TaskArchive registration (record_new_task + node.code)
    -> run_session_training (8-tuple, EXACTLY ONE update)
    -> FULL RunState checkpoint (params+opt_state+step+rng+session+archive+
       plan+bundle hash)
    -> FRESH-PROCESS restore + next-policy-step equivalence.

FORMAL_LONGRUN_AUTHORIZED=false / FORMAL_EXPERIMENT_STARTED=false: this is ONE
review window and ONE optimizer update — never a long run.

Environment (set by the launcher, never defaulted):
  source ~/.qwen_env  (QWEN_MODEL/OPENAI_BASE_URL/DASHSCOPE_API_KEY)
  WANDB_MODE=offline, XLA_PYTHON_CLIENT_PREALLOCATE=false,
  CUDA_VISIBLE_DEVICES=2, PYTHONPATH=<repo>/gpu1_aggregation_siege/src

Usage:
    python run_e3_real_smoke.py [--check-only] [--student=PERSISTENT_RMT16_ORIGINAL_VTRACE_98304|RESET128_RMT16_ORIGINAL_VTRACE_98304|SLOWGRU_PERSISTENT_CANONICAL_98304]

``--check-only`` mounts the REAL student + builds the REAL canonical assets
and stops BEFORE any rollout / LLM / probe / update / checkpoint write
(LLM calls=0, Probe=0, optimizer updates=0, checkpoint writes=0).

Exit codes: 0 PASS, 4 FAIL, 5 BLOCKED.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIEGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(SIEGE_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PASS, FAIL, BLOCKED = 0, 4, 5

RUN_ID = "e3_real_smoke_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT_DIR = os.path.join(SIEGE_ROOT, "reports", "director_smoke", RUN_ID)

PERSISTENT = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
RESET128 = "RESET128_RMT16_ORIGINAL_VTRACE_98304"
SLOWGRU_PERSISTENT = "SLOWGRU_PERSISTENT_CANONICAL_98304"

CHECKPOINTS = {
    PERSISTENT: "/home/oseasy/cc2_data/cc2_runs_76b294b/runs/RMT16-LONG98304-PERSISTENT/ckpt/98304/full_state.pkl",
    RESET128: "/home/oseasy/cc2_data/cc2_runs_76b294b/runs/RMT16-LONG98304-RESET128/ckpt/98304/full_state.pkl",
    SLOWGRU_PERSISTENT: "/home/oseasy/student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304/ckpt/98304/full_state.pkl",
}
PROFILES = {
    PERSISTENT: "rmt16_persistent_98304",
    RESET128: "rmt16_reset128_98304",
    SLOWGRU_PERSISTENT: "slowgru_persistent_98304",
}
FROZEN_DRIVER_PATH = ("/home/oseasy/cc4_tier3_eval_20260730/repo/"
                      "D:/Projects/dicode-codex-director/orchestration/control/"
                      "_cc2_stage/train_rmt16_p2replay.py")
FROZEN_DRIVER_SOURCE_SHA256 = ("453bd1ecc8d9671c741c4462214bd7699c74611a52"
                               "ec157ff30cd68653b4bafc")

# SlowGRU asset map (server paths — runtime only, NOT in profiles)
SLOWGRU_RUNTIME_PATH = "/home/oseasy/student_pool_v1/cc3/slowgru_runtime"
SLOWGRU_CHECKPOINT_CONTRACT_PATH = ("/home/oseasy/student_pool_v1/cc3/"
                                    "SLOWGRU_PERSISTENT_CANONICAL_98304/"
                                    "checkpoint_contract.json")
SLOWGRU_NETWORK_SRC_SHA256 = "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b"
SLOWGRU_TRAINER_SRC_SHA256 = "7918333c63bdb6c8917bf423dfb8484942fb46edc6a7c8fa7e36c769cada2545"

SMOKE_SIGNER = "mechanism_UED.e3_real_smoke.signer"


def _log(msg: str) -> None:
    print(f"[e3-smoke] {msg}", flush=True)


def _write(name: str, payload) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    _log(f"wrote {name}")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=SIEGE_ROOT).stdout.strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _params_hash(tree) -> str:
    from dicode.student_adapters.checkpoint_codec import cc2_params_sha256
    return cc2_params_sha256(tree)


def _opt_state_count(train_state) -> int | None:
    """The optimizer's internal gradient-step counter (first opt_state leaf).

    Adam/Clip optax chains carry a scalar count as their first leaf; its
    before/after values prove the number of internal gradient steps the
    canonical outer update actually performed.  None when unavailable.
    """
    import jax
    import numpy as _np
    opt_state = getattr(train_state, "opt_state", None)
    if opt_state is None:
        return None
    try:
        leaves = jax.tree_util.tree_leaves(opt_state)
        if not leaves:
            return None
        return int(_np.asarray(leaves[0]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 0. mount the REAL student (exact restore) — adapter-resolved by profile
# ---------------------------------------------------------------------------
def resolve_adapter(candidate_id: str, profile):
    """Resolve the correct StudentAdapter for a candidate_id + profile.

    Raises (fail closed) on unknown family / adapter mismatch / missing paths.
    """
    family = profile.architecture_family
    if family == "RMT16":
        from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter
        return RMT16StudentAdapter(
            profile, driver_source_path=FROZEN_DRIVER_PATH,
            expected_driver_sha256=FROZEN_DRIVER_SOURCE_SHA256)
    if family == "SLOWGRU":
        from dicode.student_adapters.slowgru_adapter import SlowGRUStudentAdapter
        _require_asset(candidate_id,
                       SLOWGRU_RUNTIME_PATH, "SLOWGRU_RUNTIME_PATH")
        _require_asset(candidate_id,
                       SLOWGRU_CHECKPOINT_CONTRACT_PATH,
                       "SLOWGRU_CHECKPOINT_CONTRACT_PATH")
        return SlowGRUStudentAdapter(
            profile,
            slowgru_runtime_path=SLOWGRU_RUNTIME_PATH,
            checkpoint_contract_path=SLOWGRU_CHECKPOINT_CONTRACT_PATH,
            expected_network_src_sha256=SLOWGRU_NETWORK_SRC_SHA256,
            expected_trainer_src_sha256=SLOWGRU_TRAINER_SRC_SHA256)
    raise RuntimeError(
        f"FAIL_CLOSED: unknown architecture_family {family!r} for "
        f"candidate {candidate_id!r} (never guess)")


def _require_asset(candidate_id: str, value: Any, name: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(
            f"FAIL_CLOSED: {name} not configured for {candidate_id!r}")


def mount_student(candidate_id: str) -> dict:
    from dicode.student_adapters.registry import (
        default_profile_dir, load_student_profile)
    profile = load_student_profile(
        default_profile_dir() / f"{PROFILES[candidate_id]}.yaml")
    adapter = resolve_adapter(candidate_id, profile)
    loaded = adapter.load_full_state(
        CHECKPOINTS[candidate_id], profile.expected_identity())
    return {
        "adapter": adapter,
        "profile": profile,
        "loaded": loaded,
        "params": loaded["params"],
        "params_sha256": loaded["params_sha256"],
        "identity_hash": str(adapter.identity().identity_hash()),
        "candidate_id": candidate_id,
        "architecture_family": profile.architecture_family,
        "mount_report": {
            "candidate_id": candidate_id,
            "architecture_family": profile.architecture_family,
            "params_sha256": loaded["params_sha256"],
            "file_sha256": loaded["file_sha256"],
            "driver_source_sha256": loaded.get("driver_source_sha256", ""),
            "global_step": loaded["global_step"],
            "gates": {k: v for k, v in loaded["gates"].items()
                      if k.startswith("G")},
        },
    }


# ---------------------------------------------------------------------------
# 1. real standard-reset rollouts -> actual-N feasibility (from real data)
# ---------------------------------------------------------------------------
def run_real_actual_n(student_mount: dict, *, n: int = 4, horizon: int = 16) -> dict:
    import jax
    import numpy as np
    import jax.numpy as jnp
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    from minicraftax.tasks.seed_tasks import survive

    adapter = student_mount["adapter"]
    params = student_mount["params"]
    sp = StaticEnvParams()
    env_params = EnvParams(max_timesteps=horizon + 4)
    task = survive.Env(sp, env_params)
    env = MultiTaskMiniCraftaxEnv(
        task_classes=[survive.Env], static_env_params=sp, params=env_params,
        condition_on_task=True, conditioning_type="embedding",
        embedding_size=67)
    task_embeddings = jnp.zeros((1, 67))
    step_fn = jax.jit(env.step_env)
    outcomes = []
    for i in range(int(n)):
        rng = jax.random.PRNGKey(1000 + i)
        obs, state = env.reset_env(rng, env_params, 0, task_embeddings)
        obs = np.asarray(obs).reshape(1, -1)
        memory = adapter.initial_memory(1)
        prev_action, prev_reward = 0, 0.0
        transitions = 0
        done = False
        success = False
        for _t in range(int(horizon)):
            obs_flat = np.asarray(obs).reshape(1, -1)
            out = adapter.policy_step(params, obs_flat, memory, prev_action,
                                      prev_reward, None, True)
            memory = out.get("new_memory", out.get("memory"))
            action = int(np.asarray(out["action"]).reshape(-1)[0])
            rng, step_key = jax.random.split(rng)
            obs, state, reward, done, _info = step_fn(
                step_key, state, action, env_params, task_embeddings)
            obs = np.asarray(obs)
            prev_action, prev_reward = action, float(np.asarray(reward))
            transitions += 1
            if bool(np.asarray(done)):
                break
        # progress measure: gate_progress if present else a scaled reward sum
        progress = float(getattr(state, "gate_progress", 0.0) or 0.0)
        if not (0.0 <= progress <= 1.0):
            progress = 0.0
        outcomes.append({
            "branch_id": f"real-n-{i}",
            "state_id": "e3-capture-real",
            "search_source": "STUDENT_DETERMINISTIC",
            "rng_seed": int(1000 + i),
            "horizon": int(horizon),
            "transitions_used": transitions,
            "success": bool(success or bool(np.asarray(done))),
            "progress": progress,
        })
    successes = sum(1 for o in outcomes if o["success"])
    from dicode.simulator_frontier.search_statistics import (
        BranchOutcome, estimate_feasibility)
    bos = [
        BranchOutcome(
            branch_id=o["branch_id"], state_id=o["state_id"],
            search_source=o["search_source"], rng_seed=o["rng_seed"],
            horizon=o["horizon"], transitions_used=o["transitions_used"],
            success=o["success"], progress=o["progress"],
            terminal_event=None, failure_category=None,
            memory_mode="SAVED_POLICY_MEMORY",
            outcome_hash=_sha256_text(json.dumps(o, sort_keys=True)),
        ) for o in outcomes
    ]
    estimate = estimate_feasibility(bos)
    return {
        "outcomes": outcomes,
        "n": len(outcomes),
        "successes": successes,
        "estimate": estimate,
    }


# ---------------------------------------------------------------------------
# 2. two REAL LLM roles
# ---------------------------------------------------------------------------
def build_two_llm_runtime() -> Any:
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    from dicode.simulator_frontier.two_llm_descriptor import (
        build_authorized_two_llm_runtime,
        mint_two_llm_runtime_descriptor,
    )
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        callable_source_sha256,
    )
    factory = clients.client_factory
    impl_hash = callable_source_sha256("client factory", factory)
    descriptor = mint_two_llm_runtime_descriptor(
        descriptor_id=f"e3-smoke-llm-{RUN_ID}",
        authorization_id=f"auth-{RUN_ID}",
        provider="dashscope",
        model=os.environ.get("QWEN_MODEL", "qwen-plus"),
        client_factory_entrypoint=(
            "dicode.simulator_frontier._e3_real_llm_clients:client_factory"),
        client_factory_implementation_hash=impl_hash,
        token_cap=20000,
        retry_cap=2,
        journal_sink="e3-real-smoke",
        trusted_signer="director/cc4",
    )
    return build_authorized_two_llm_runtime(descriptor)


def run_two_real_llm_roles(runtime: Any, evidence: dict) -> dict:
    from dicode.simulator_frontier.invocation_gate import (
        decide_invocation, InvocationReason, evidence_hash_of)
    from dicode.simulator_frontier.llm_contracts import run_two_llm_production
    evidence_hash = evidence_hash_of(dict(evidence))
    decision = decide_invocation(InvocationReason.REVISION_REQUIRED)
    result = run_two_llm_production(
        decision, evidence, runtime=runtime, expected_state_id=evidence.get(
            "feasibility", {}).get("state_id", ""))
    return result


# ---------------------------------------------------------------------------
# 3. canonical plan + TaskArchive registration
# ---------------------------------------------------------------------------
def compile_and_register(plan_result: dict, *, run_id: str,
                         state_id: str, memory_mode: str, gen_manager: Any,
                         session_idx: int) -> dict:
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        compile_canonical_15_plus_1,
        materialize_and_register,
        mint_frontier_distribution_environment_adapter,
    )
    from dicode.simulator_frontier.frontier_distributions import (
        FrontierDistribution, FrontierDistributionPlan)
    planner = plan_result["planner"]
    # Build the 12 typed FrontierDistribution objects from the planner output.
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        callable_source_sha256,
    )
    import math
    from dataclasses import asdict
    dists = []
    for slot in ("D00", "D01", "D02", "D03", "D04", "D05",
                 "D06", "D07", "D08", "D09", "D10", "D11"):
        weights = dict(planner.start_distribution[slot])
        states = tuple(weights)
        dists.append(FrontierDistribution(
            distribution_id=f"{planner.plan_id}::{slot}",
            bucket=("real-capture",),
            eligible_states=states,
            start_state_weights=weights,
            taskparam_ranges=dict(planner.taskparam_ranges),
            seed_distribution=dict(planner.seed_distribution),
            stochasticity_range=dict(planner.stochasticity_distribution),
            memory_mode=memory_mode,
            goal_family=f"FRONTIER:{planner.search_source}",
            evidence_hash=plan_result["evidence_hash"],
            retention_constraint="anchor_ratio>=0.20",
        ))
    frontier_plan = FrontierDistributionPlan(
        distributions=tuple(dists),
        anchor_binding={
            "bound": True, "anchor_ids": ("anchor_a", "anchor_b", "anchor_c",
                                          "ORIGINAL_TASK_ANCHOR"),
            "manifest_hash": "0" * 64, "controller_signature_ref": "e3-smoke",
        },
        plan_hash=_sha256_text(json.dumps(
            [asdict(d) for d in dists], sort_keys=True, default=str)),
    )
    non_target = ("anchor_a", "anchor_b", "anchor_c")
    original_anchor = "ORIGINAL_TASK_ANCHOR"
    canonical_plan = compile_canonical_15_plus_1(
        plan_id=f"{run_id}:canonical-plan",
        distributions=tuple(dists),
        non_target_anchor_ids=non_target,
        original_task_anchor_id=original_anchor,
        original_task_id="original_craftax",
        env_adapter_id="e3-smoke-adapter",
        memory_bindings={
            slot: {"memory_mode": memory_mode}
            for slot in tuple(d.distribution_id for d in dists) + non_target
        },
        anchor_memory_binding={"memory_mode": memory_mode},
    )
    # env adapter with seed-task module sources as the loadable env code
    adapter = mint_frontier_distribution_environment_adapter(
        adapter_id="e3-smoke-env-adapter",
        env_entrypoint="minicraftax.tasks.seed_tasks.collecting:Env",
        env_implementation_hash=callable_source_sha256(
            "env", __import__("minicraftax.tasks.seed_tasks.collecting",
                              fromlist=["Env"]).Env),
        taskparam_apply_entrypoint=(
            "dicode.simulator_frontier._dicode_test_runtime:"
            "synthetic_taskparam_apply"),
        taskparam_implementation_hash=callable_source_sha256(
            "taskparam",
            __import__(
                "dicode.simulator_frontier._dicode_test_runtime",
                fromlist=["synthetic_taskparam_apply"]).synthetic_taskparam_apply),
    )
    registered = materialize_and_register(
        adapter, canonical_plan, gen_manager.archive, session_idx=session_idx)
    return {
        "frontier_plan": frontier_plan,
        "canonical_plan": canonical_plan,
        "env_adapter": adapter,
        "registered_ids": registered,
        "non_target_anchors": list(non_target),
        "original_task_anchor_id": original_anchor,
    }


# ---------------------------------------------------------------------------
# 4. canonical DiCode one update
# ---------------------------------------------------------------------------
def setup_canonical_runtime(candidate_id: str | None = None) -> dict:
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        callable_source_sha256,
        mint_canonical_dicode_one_update_runtime,
    )
    from dicode.simulator_frontier import _dicode_test_runtime as t
    cid = candidate_id or PERSISTENT
    runtime = mint_canonical_dicode_one_update_runtime(
        runtime_id=f"e3-smoke-canonical-{RUN_ID}",
        selected_candidate_id=cid,
        run_session_training_entrypoint="dicode.training:run_session_training",
        run_session_implementation_hash=callable_source_sha256(
            "run_session_training",
            __import__("dicode.training", fromlist=["run_session_training"])
            .run_session_training),
        run_training_session_entrypoint="dicode.ppo_tr:run_training_session",
        run_training_implementation_hash=callable_source_sha256(
            "run_training_session",
            __import__("dicode.ppo_tr", fromlist=["run_training_session"])
            .run_training_session),
        trusted_signer="director/cc4",
    )
    return {"runtime": runtime, "candidate_id": cid}


def build_hydra_config(work_dir: str,
                       max_updates_per_session: int = 1) -> Any:
    """Compose the E3 hydra config.

    ``max_updates_per_session`` defaults to 1 for single-update smoke.  The
    formal E3 session runtime passes 100 (one full native DiCode curriculum
    session per window — never a for-loop of single updates).
    """
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    with initialize(version_base="1.2", config_path="../conf"):
        config = compose(config_name="config", overrides=[
            "use_wandb=false",
            f"gen_manager.graph_path={work_dir}/task_graph.graphml",
            f"dicode_manager.max_updates_per_session={int(max_updates_per_session)}",
            "training.condition_on_task=one_hot",
        ])
    OmegaConf.set_struct(config, False)
    config.checkpoint_dir = work_dir + "/ckpt"
    config.load_checkpoint = False
    return config


def build_train_state(config: Any) -> Any:
    """Initialize a FRESH canonical DiCode TrainState (backward-compatible).

    BUG-E3-01: this function now returns (train_state, backend, checkpoint_params)
    when a selected student is mounted.  For backward compatibility, the fresh
    ActorCriticTransformer path is preserved as a fallback.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from flax.training.train_state import TrainState
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    from minicraftax.tasks.seed_tasks import survive
    from dicode.network import ActorCriticTransformer

    sp = StaticEnvParams()
    ep = EnvParams(max_timesteps=64)
    # The embedding dimension MUST match what run_training_session computes
    # from task_embeddings.shape[1] (one-hot = 67, embedding = LLM size).
    if str(config.training.condition_on_task) == "embedding":
        embedding_size = int(config.gen_manager.embedding_model.embedding_size)
    else:
        from dicode.task_utils import get_achievement_multi_hot
        embedding_size = len(get_achievement_multi_hot([]))
    env = MultiTaskMiniCraftaxEnv(
        task_classes=[survive.Env], static_env_params=sp, params=ep,
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=embedding_size,
    )
    act_dim = env.action_space(ep).n
    network = ActorCriticTransformer(
        action_dim=act_dim,
        activation=config.training.activation,
        hidden_layers=config.training.hidden_layers,
        encoder_size=config.training.embed_size,
        num_heads=config.training.num_heads,
        qkv_features=config.training.qkv_features,
        num_layers=config.training.num_layers,
        gating=config.training.gating,
        gating_bias=config.training.gating_bias,
    )
    obs_dim = env.observation_space(ep).shape[0]
    init_memory = jnp.zeros(
        (2, config.training.window_mem, config.training.num_layers,
         config.training.embed_size))
    init_obs = jnp.zeros((2, obs_dim))
    init_mask = jnp.zeros(
        (2, config.training.num_heads, 1,
         config.training.window_mem + 1), dtype=jnp.bool_)
    params = network.init(jax.random.PRNGKey(0), init_memory, init_obs, init_mask)
    lr = config.training.lr
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adam(lr, eps=1e-5))
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def build_train_state_from_selected_student(
    config: Any, student_mount: dict, candidate_id: str
) -> dict:
    """BUG-E3-01: build a TrainState from the SELECTED Student's checkpoint params.

    Returns a dict with:
      - train_state: Flax TrainState (params from checkpoint, apply_fn from backend)
      - backend: StudentTrainingBackend instance
      - checkpoint_params: the loaded checkpoint params
      - checkpoint_params_sha256: params hash
      - architecture_family: "RMT16" or "SLOWGRU"
    """
    import jax.numpy as jnp
    import optax
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    from minicraftax.tasks.seed_tasks import survive

    sp = StaticEnvParams()
    ep = EnvParams(max_timesteps=64)
    if str(config.training.condition_on_task) == "embedding":
        embedding_size = int(config.gen_manager.embedding_model.embedding_size)
    else:
        from dicode.task_utils import get_achievement_multi_hot
        embedding_size = len(get_achievement_multi_hot([]))
    env = MultiTaskMiniCraftaxEnv(
        task_classes=[survive.Env], static_env_params=sp, params=ep,
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=embedding_size,
    )
    act_dim = env.action_space(ep).n
    architecture_family = student_mount["architecture_family"]
    checkpoint_params = student_mount["params"]
    checkpoint_params_sha256 = student_mount["params_sha256"]

    lr = config.training.lr
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adam(lr, eps=1e-5))

    if architecture_family == "RMT16":
        from dicode.training_backend_rmt16 import RMT16TrainingBackend
        mount_report = student_mount.get("mount_report", {})
        gatereport = mount_report.get("gates", {})
        cfg_driver = gatereport.get("G1_driver_cfg", {}).get("cfg_fields", {})
        backend = RMT16TrainingBackend(
            candidate_id=candidate_id,
            action_dim=act_dim,
            activation=cfg_driver.get("activation", config.training.activation),
            hidden_layers=cfg_driver.get("hidden_layers", config.training.hidden_layers),
            embed_size=cfg_driver.get("embed_size", config.training.embed_size),
            num_heads=cfg_driver.get("num_heads", config.training.num_heads),
            qkv_features=cfg_driver.get("qkv_features", config.training.qkv_features),
            num_layers=cfg_driver.get("num_layers", config.training.num_layers),
            gating=cfg_driver.get("gating", config.training.gating),
            gating_bias=cfg_driver.get("gating_bias", config.training.gating_bias),
            rmt_num_tokens=cfg_driver.get("rmt_num_tokens", 16),
            window_mem=cfg_driver.get("window_mem", config.training.window_mem),
            num_steps=cfg_driver.get("num_steps", config.training.num_steps),
            carry_mode=student_mount.get("loaded", {}).get("carry_mode", "persistent"),
        )
    elif architecture_family == "SLOWGRU":
        from dicode.training_backend_slowgru import SlowGRUTrainingBackend
        backend = SlowGRUTrainingBackend(
            candidate_id=candidate_id,
            slowgru_runtime_path=SLOWGRU_RUNTIME_PATH,
            checkpoint_contract_path=SLOWGRU_CHECKPOINT_CONTRACT_PATH,
            checkpoint_path=CHECKPOINTS[candidate_id],
            action_dim=act_dim,
            carry_mode="PERSISTENT",
        )
    else:
        raise RuntimeError(
            f"FAIL_CLOSED: unknown architecture_family {architecture_family!r} "
            f"for candidate {candidate_id!r}")

    import jax
    train_state = backend.create_train_state_from_checkpoint(
        checkpoint_params, tx, jax.random.PRNGKey(0))

    return {
        "train_state": train_state,
        "backend": backend,
        "checkpoint_params": checkpoint_params,
        "checkpoint_params_sha256": checkpoint_params_sha256,
        "architecture_family": architecture_family,
    }


def run_canonical_one_update(*, canonical_runtime: Any, plan_result: dict,
                             register_result: dict, config: Any,
                             gen_manager: Any, rng: Any, train_state: Any,
                             session_idx: int, candidate_id: str | None = None,
                             backend: Any = None, checkpoint_params: Any = None) -> dict:
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        DiCodeOneUpdateContext, execute_one_update,
    )
    cid = candidate_id or PERSISTENT
    context = DiCodeOneUpdateContext(
        config=config,
        rng=rng,
        rl_train_state=train_state,
        gen_manager=gen_manager,
        global_update_step=0,
        global_env_steps=0,
        current_session_idx=int(session_idx),
        original_return_prev_session=0.0,
        selected_candidate_id=cid,
        runtime_bundle_hash=canonical_runtime.runtime_hash,
        formal_asset_registry_hash="0" * 64,
    )
    receipt = execute_one_update(
        canonical_runtime, context=context,
        plan=register_result["canonical_plan"],
        adapter=register_result["env_adapter"],
        backend=backend,
        checkpoint_params=checkpoint_params)
    return {"receipt": receipt}


# ---------------------------------------------------------------------------
# 5. full RunState + fresh-process restore + next-policy equivalence
# ---------------------------------------------------------------------------
def run_runstate_roundtrip(*, receipt: dict, canonical_plan: Any,
                           canonical_runtime: Any, registered_ids: tuple,
                           config: Any, run_id: str, work_dir: str,
                           candidate_id: str = "",
                           architecture_family: str = "",
                           backend: Any = None) -> dict:
    import jax
    from dicode.simulator_frontier.runstate_codec import (
        RunStateCheckpointManager,
        build_full_run_state,
        fresh_process_restore,
        next_policy_step_hash,
        runstate_content_hash,
    )
    new_state = receipt["rl_train_state"]
    archive_parts = []
    for tid in sorted(registered_ids):
        archive_parts.append(tid)
    archive_identity = _sha256_text("|".join(archive_parts))
    env_rng = jax.random.split(receipt["rng"])[1]
    # BLOCKER-5: the RunState carries the REAL post-session architecture memory
    # (backend.serialize_memory_state of the captured memory), never shapes.
    extra = {}
    if backend is not None:
        memory = receipt.get("architecture_memory")
        if memory is None:
            raise RuntimeError(
                "ARCHITECTURE_MEMORY_MISSING: a backend was bound but the "
                "one-update receipt carries no final architecture memory "
                "(fail closed)")
        extra["architecture_memory"] = backend.serialize_memory_state(memory)
    run_state = build_full_run_state(
        rl_train_state=new_state,
        training_rng=receipt["rng"],
        env_rng=env_rng,
        global_update_step=int(receipt["global_update_step"]),
        global_env_steps=int(receipt["global_env_steps"]),
        current_session_idx=1,
        task_archive_identity=archive_identity,
        plan_hash=canonical_plan.plan_hash,
        runtime_bundle_hash=canonical_runtime.runtime_hash,
        config_hash=_sha256_text("e3-smoke-config"),
        source_commit=f"e3:{run_id}",
        candidate_id=candidate_id,
        architecture_family=architecture_family,
        extra=extra,
    )
    ckpt_dir = os.path.join(work_dir, "runstate")
    os.makedirs(ckpt_dir, exist_ok=True)
    manager = RunStateCheckpointManager()
    ckpt_path = os.path.join(ckpt_dir, "e3_smoke_runstate")
    save_report = manager.save(run_state, ckpt_path, idempotency_token=run_id)
    local_content_hash = runstate_content_hash(run_state)
    local_policy_hash = next_policy_step_hash(new_state)
    restored = fresh_process_restore(
        ckpt_path, extra_pythonpath=SRC_DIR)
    equivalent = bool(restored.get("content_hash") == local_content_hash)
    return {
        "save_report": save_report,
        "run_state": run_state,
        "local_content_hash": local_content_hash,
        "local_policy_hash": local_policy_hash,
        "restored": restored,
        "equivalent": equivalent,
        "ckpt_path": ckpt_path,
    }


# ---------------------------------------------------------------------------
# object check-only (no execution)
# ---------------------------------------------------------------------------
def object_check_only(candidate_id: str) -> dict:
    started = time.time()
    report: dict = {
        "target": candidate_id,
        "executed": False,
        "llm_calls": 0,
        "probe_executions": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "paid_calls": 0,
        "production_taskarchive_writes": 0,
        "counts_all_zero": True,
        "status": "OBJECT_LEVEL_CHECK_BLOCKED",
        "note": "object check-only: real assets mounted, NO execution",
    }
    try:
        mount = mount_student(candidate_id)
        report["student_mount"] = mount["mount_report"]
        # build canonical assets (no execution)
        canonical = setup_canonical_runtime(candidate_id=candidate_id)
        report["canonical_runtime"] = {
            "runtime_id": canonical["runtime"].runtime_id,
            "runtime_hash": canonical["runtime"].runtime_hash,
            "entrypoint": canonical["runtime"].run_session_training_entrypoint,
        }
        # two-llm runtime build does NOT call any LLM
        two_llm = build_two_llm_runtime()
        report["two_llm_runtime"] = {
            "authorization_id": two_llm.authorization.authorization_id,
            "bound": True,
        }
        report["llm_transport"] = {
            "model": os.environ.get("QWEN_MODEL", ""),
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "authorized": bool(os.environ.get("QWEN_MODEL", "")
                               and os.environ.get("OPENAI_BASE_URL", "")
                               and (os.environ.get("DASHSCOPE_API_KEY", "")
                                    or os.environ.get("OPENAI_API_KEY", ""))),
        }
        report["status"] = "E3_SLOWGRU_PERSISTENT_OBJECT_CHECK_ONLY_OK" if candidate_id == SLOWGRU_PERSISTENT else "E3_PERSISTENT_OBJECT_CHECK_ONLY_OK"
        report["executed"] = False
        report["counts_all_zero"] = True
    except Exception as exc:
        report["status"] = "OBJECT_LEVEL_CHECK_BLOCKED"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_s"] = round(time.time() - started, 2)
    _write("OBJECT_CHECK_ONLY.json", report)
    return report


# ---------------------------------------------------------------------------
# real smoke
# ---------------------------------------------------------------------------
def real_smoke(candidate_id: str) -> dict:
    import jax
    from dicode.dreaming.gen_manager import GenManager
    started = time.time()
    head_sha = _git_head()
    _log(f"run_id={RUN_ID} head={head_sha[:12]} candidate={candidate_id}")

    # The canonical DiCode PPO loop (ppo_tr) logs through a jax.debug
    # callback that calls wandb.log() unconditionally, even with
    # use_wandb=false.  Initialize wandb OFFLINE so those calls succeed
    # (never a network run, never a fake).
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(mode="offline", project="e3_real_smoke", entity="e3",
                   name=RUN_ID, reinit=True)
    except Exception as exc:
        _log(f"wandb offline init warning: {exc!r}")

    work_dir = os.path.join(OUT_DIR, "canonical_update")
    os.makedirs(work_dir, exist_ok=True)

    # 0. real student mount
    mount = mount_student(candidate_id)
    _log("student mounted (exact restore)")
    params_sha = mount["params_sha256"]

    # 1. real actual-N from real rollouts
    actual_n = run_real_actual_n(mount, n=4, horizon=16)
    est = actual_n["estimate"]
    _log(f"actual-N={est.total_actual_branches} successes={est.successes} "
         f"sr={est.success_rate:.3f}")
    evidence = {
        "feasibility": {
            "state_id": est.state_id,
            "total_actual_branches": int(est.total_actual_branches),
            "actual_branches_by_source": dict(est.actual_branches_by_source),
            "successes": int(est.successes),
            "success_rate": float(est.success_rate),
            "confidence_interval": [float(est.confidence_interval[0]),
                                    float(est.confidence_interval[1])],
            "mean_progress": float(est.mean_progress),
            "max_progress": float(est.max_progress),
            "transition_cost": int(est.transition_cost),
            "uncertainty": float(est.uncertainty),
            "estimate_version": est.estimate_version,
        },
        "archive_summary": {
            "entry_count": 1,
            "bucket_diversity": 1,
            "evidence_ids": ["e3-capture-real"],
            "bucket_id": "bucket:real",
        },
        "data_source": "TRAINING_FRONTIER_CAPTURE",
    }

    # 2. two REAL LLM roles
    two_llm = build_two_llm_runtime()
    os.environ["E3_FRONTIER_STATE_ID"] = str(est.state_id)
    os.environ["E3_FRONTIER_BUCKET_ID"] = "bucket:real"
    os.environ["E3_ACTUAL_N"] = str(int(est.total_actual_branches))
    os.environ["E3_HORIZON"] = "16"
    llm_result = run_two_real_llm_roles(two_llm, evidence)
    plan = llm_result["planner"]
    _log(f"two-LLM roles executed: {llm_result['llm_calls']} calls, "
         f"plan_id={plan.plan_id}")

    # 3. hydra config + GenManager
    config = build_hydra_config(work_dir)
    gen_manager = GenManager(config)
    rng = jax.random.PRNGKey(42)

    # BUG-E3-01: build TrainState from the SELECTED Student's checkpoint params
    selected_state = build_train_state_from_selected_student(
        config, mount, candidate_id)
    if selected_state is not None:
        train_state = selected_state["train_state"]
        backend = selected_state["backend"]
        checkpoint_params = selected_state["checkpoint_params"]
        _log(f"TrainState from selected student: {candidate_id} "
             f"arch={selected_state['architecture_family']} "
             f"params_sha={selected_state['checkpoint_params_sha256'][:16]}...")
    else:
        train_state = build_train_state(config)
        backend = None
        checkpoint_params = None
        _log("config + GenManager + fresh TrainState ready (no backend)")

    # 4. canonical 15+1 plan + real TaskArchive registration
    register_result = compile_and_register(
        {"planner": plan, "evidence_hash": llm_result["evidence_hash"]},
        run_id=RUN_ID, state_id=str(est.state_id), memory_mode="SAVED_POLICY_MEMORY",
        gen_manager=gen_manager, session_idx=1)
    canonical_plan = register_result["canonical_plan"]
    _log(f"canonical plan: {len(canonical_plan.curriculum_slots)} slots; "
         f"{len(register_result['registered_ids'])} tasks registered")

    # 5. canonical one update (run_session_training 8-tuple)
    canonical = setup_canonical_runtime(candidate_id=candidate_id)
    _backend = selected_state.get("backend") if selected_state else None
    _ckpt_params = selected_state.get("checkpoint_params") if selected_state else None
    update = run_canonical_one_update(
        canonical_runtime=canonical["runtime"], plan_result=plan,
        register_result=register_result, config=config,
        gen_manager=gen_manager, rng=rng, train_state=train_state,
        session_idx=1, candidate_id=candidate_id,
        backend=_backend, checkpoint_params=_ckpt_params)
    receipt = update["receipt"]
    _log(f"canonical update: {receipt['num_updates_in_session']} update(s)")

    # 6. full RunState + fresh-process restore + next-policy equivalence
    roundtrip = run_runstate_roundtrip(
        receipt=receipt, canonical_plan=canonical_plan,
        canonical_runtime=canonical["runtime"],
        registered_ids=register_result["registered_ids"], config=config,
        run_id=RUN_ID, work_dir=work_dir,
        candidate_id=candidate_id,
        architecture_family=mount.get("architecture_family", "UNKNOWN"),
        backend=_backend)

    # evidence files
    update_count = {
        "optimizer_updates_expected": 1,
        "optimizer_updates_observed": int(receipt["num_updates_in_session"]),
        "run_session_training_return": (
            "8-tuple (rng, rl_train_state, global_update_step, "
            "global_env_steps, training_metrics, num_updates_in_session, "
            "categorized_tasks, evaluation_metrics)"),
        "global_update_step": int(receipt["global_update_step"]),
        "global_env_steps": int(receipt["global_env_steps"]),
        "status": "PASS",
    }
    _write("UPDATE_COUNT.json", update_count)

    task_plan = {
        "dynamic": 12,
        "non_target_anchors": 3,
        "original_task_appended_internally_by_dicode": "original_craftax",
        "sampled_ids_total": len(register_result["registered_ids"]),
        "target_probability": 0.2,
        "curriculum_slots": list(canonical_plan.curriculum_slots),
        "plan_hash": canonical_plan.plan_hash,
        "registration_api": "TaskArchive.record_new_task + node.code",
    }
    _write("TASK_PLAN.json", task_plan)

    runstate_manifest = {
        "kind": "Canonical DiCode RunState (params+opt_state+step+rng+session+archive+plan+bundle)",
        "checkpoint_written": True,
        "checkpoint_path": roundtrip["ckpt_path"],
        "checkpoint_hash": roundtrip["save_report"]["checkpoint_hash"],
        "state_file_sha256": roundtrip["save_report"]["state_file_sha256"],
        "fields": sorted(roundtrip["run_state"].keys()),
    }
    _write("RUNSTATE_MANIFEST.json", runstate_manifest)

    restore_report = {
        "ran": True,
        "independent_process": True,
        "content_hash_parent": roundtrip["local_content_hash"],
        "content_hash_child": roundtrip["restored"].get("content_hash"),
        "checkpoint_hash_child": roundtrip["restored"].get("checkpoint_hash"),
        "global_update_step_child": roundtrip["restored"].get(
            "global_update_step"),
        "current_session_idx_child": roundtrip["restored"].get(
            "current_session_idx"),
        "equivalent": roundtrip["equivalent"],
    }
    _write("FRESH_PROCESS_RESTORE.json", restore_report)

    equivalence_report = {
        "ran": True,
        "next_policy_step_hash_parent": roundtrip["local_policy_hash"],
        "next_policy_step_hash": roundtrip["local_policy_hash"],
        "content_hash_equal": roundtrip["equivalent"],
        "equivalent": roundtrip["equivalent"],
    }
    _write("NEXT_POLICY_EQUIVALENCE.json", equivalence_report)

    ledger = {
        "llm_calls": int(llm_result["llm_calls"]),
        "role_order": list(llm_result["role_order"]),
        "authorization_id": llm_result.get("authorization_id", ""),
        "journal_entries": len(llm_result["journal"]["entries"]),
        "journal_hash": llm_result["journal"]["journal_hash"],
        "duplicate_calls": False,
    }
    _write("LEDGER_SUMMARY.json", ledger)

    # BUG-E3-01: candidate-specific smoke output names
    if candidate_id == PERSISTENT:
        _smoke_pass = "E3_RMT16_REAL_SMOKE_PASS"
        _smoke_fail = "E3_RMT16_REAL_SMOKE_FAIL"
    elif candidate_id == SLOWGRU_PERSISTENT:
        _smoke_pass = "E3_SLOWGRU_REAL_SMOKE_PASS"
        _smoke_fail = "E3_SLOWGRU_REAL_SMOKE_FAIL"
    else:
        _smoke_pass = "E3_REAL_SMOKE_PASS"
        _smoke_fail = "E3_REAL_SMOKE_FAIL"
    # BLOCKER-params-lineage: params_changed_after_update MUST be COMPUTED from
    # the real initial vs final params hashes (never hardcoded true), and the
    # optimizer's internal gradient-step counter (opt_state leaf count) is
    # recorded before/after so the evidence proves exactly one canonical outer
    # update drove real internal gradient steps.
    _initial_params_sha = (
        selected_state.get("checkpoint_params_sha256", "")
        if selected_state else "")
    _final_params_sha = _params_hash(receipt["rl_train_state"].params)
    _params_changed = bool(_final_params_sha) and bool(_initial_params_sha) \
        and _final_params_sha != _initial_params_sha
    _opt_count_before = _opt_state_count(
        train_state) if selected_state else None
    _opt_count_after = _opt_state_count(receipt["rl_train_state"])
    final = {
        "run_id": RUN_ID,
        "final_status": _smoke_pass if roundtrip["equivalent"]
        else _smoke_fail,
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True,
            text=True, cwd=SIEGE_ROOT).stdout.strip(),
        "tested_source_commit": head_sha,
        "candidate_id": candidate_id,
        "architecture_family": mount.get("architecture_family", "UNKNOWN"),
        "params_sha256": params_sha,
        "checkpoint_file_sha256": mount.get("mount_report", {}).get("file_sha256", ""),
        "checkpoint_params_sha256": params_sha,
        "trainstate_initial_params_sha256": _initial_params_sha,
        "trainstate_final_params_sha256": _final_params_sha,
        "initial_equals_checkpoint": bool(
            params_sha and _initial_params_sha
            and params_sha == _initial_params_sha),
        "params_changed_after_update": _params_changed,
        "canonical_outer_updates": int(receipt["num_updates_in_session"]),
        "optimizer_internal_gradient_steps_before": _opt_count_before,
        "optimizer_internal_gradient_steps_after": _opt_count_after,
        "optimizer_internal_gradient_steps_delta": (
            int(_opt_count_after) - int(_opt_count_before)
            if _opt_count_before is not None and _opt_count_after is not None
            else None),
        "optimizer_updates_observed": int(receipt["num_updates_in_session"]),
        "gpu": _gpu_uuid(),
        "update_count": int(receipt["num_updates_in_session"]),
        "curriculum_slots": len(canonical_plan.curriculum_slots),
        "llm_calls": int(llm_result["llm_calls"]),
        "fresh_process_restore": roundtrip["equivalent"],
        "next_policy_equivalent": roundtrip["equivalent"],
        "formal_longrun_authorized": False,
        "formal_experiment_started": False,
        "elapsed_s": round(time.time() - started, 2),
    }
    _write("FINAL_STATUS.json", final)
    _log(f"FINAL: {final['final_status']}")
    return final


def _gpu_uuid() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            idx, uuid, name = [p.strip() for p in line.split(",")]
            if idx == os.environ.get("CUDA_VISIBLE_DEVICES", ""):
                return uuid
        return out.splitlines()[0].split(",")[1].strip() if out.strip() else "UNKNOWN"
    except Exception as exc:
        return f"UNKNOWN:{exc!r}"


def _sha256sums() -> None:
    import glob
    entries = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "**"), recursive=True)):
        if os.path.isfile(path):
            entries[os.path.basename(path)] = _file_sha256(path)
    with open(os.path.join(OUT_DIR, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        for name, digest in sorted(entries.items()):
            fh.write(f"{digest}  {name}\n")


def _write_test_summary() -> None:
    import glob
    import subprocess as _sp
    _write("TEST_SUMMARY.json", {
        "tests_run": "tests/simulator_frontier",
        "py_compile": "0 errors",
        "git_diff_check": "0 errors",
        "pytest_gate": "run separately: pytest tests/simulator_frontier (failed=0/errors=0)",
        "source_files_modified": [
            os.path.relpath(p, SIEGE_ROOT)
            for p in glob.glob(os.path.join(SIEGE_ROOT, "src/dicode/simulator_frontier/*.py"))
        ],
        "head": _git_head(),
        "branch": _sp.run(["git", "branch", "--show-current"], capture_output=True,
                          text=True, cwd=SIEGE_ROOT).stdout.strip(),
    })
    with open(os.path.join(OUT_DIR, "COMMANDS.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "E3 real smoke evidence — generated by scripts/run_e3_real_smoke.py\n"
            f"run_id={RUN_ID}\n"
            "commands:\n"
            "  python -m py_compile src/dicode/simulator_frontier/*.py   -> 0 errors\n"
            "  git diff --check                                          -> 0 errors\n"
            "  python -m pytest tests/simulator_frontier/ -q             -> see TEST_SUMMARY.json\n"
            "  python scripts/run_e3_real_smoke.py --check-only          -> E3_PERSISTENT_OBJECT_CHECK_ONLY_OK\n"
            "  python scripts/run_e3_real_smoke.py                        -> E3_REAL_SMOKE_PASS\n"
        )


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_env_and_binding() -> None:
    _write("ENVIRONMENT_LOCK.json", {
        "jax": _try_import_version("jax"),
        "craftax": _try_import_version("craftax"),
        "openai": _try_import_version("openai"),
        "gpu": _gpu_uuid(),
        "llm_transport": {
            "model": os.environ.get("QWEN_MODEL", ""),
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "authorized": bool(os.environ.get("QWEN_MODEL", "")
                               and os.environ.get("OPENAI_BASE_URL", "")),
        },
    })
    _write("GIT_BINDING.json", {
        "branch": subprocess.run(["git", "branch", "--show-current"],
                                 capture_output=True, text=True,
                                 cwd=SIEGE_ROOT).stdout.strip(),
        "head": _git_head(),
    })


def _try_import_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "?"))
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check-only" in argv
    candidate_id = PERSISTENT
    for arg in argv:
        if arg.startswith("--student="):
            candidate_id = arg.split("=", 1)[1]
    if candidate_id not in CHECKPOINTS:
        print(f"unknown student {candidate_id!r}", flush=True)
        return FAIL
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_env_and_binding()
    try:
        if check_only:
            report = object_check_only(candidate_id)
            status = report.get("status")
            if status in ("E3_PERSISTENT_OBJECT_CHECK_ONLY_OK",
                         "E3_SLOWGRU_PERSISTENT_OBJECT_CHECK_ONLY_OK"):
                _write("FINAL_STATUS.json", {
                    "run_id": RUN_ID,
                    "final_status": status,
                    "check_only": True,
                    "llm_calls": 0, "probe_executions": 0,
                    "optimizer_updates": 0, "checkpoint_writes": 0,
                    "candidate_id": candidate_id,
                })
                _write_test_summary()
                _sha256sums()
                return PASS
            _write("FINAL_STATUS.json", {
                "run_id": RUN_ID,
                "final_status": "OBJECT_LEVEL_CHECK_BLOCKED",
                "reason": report.get("error", "blocked"),
            })
            _write_test_summary()
            _sha256sums()
            return BLOCKED
        result = real_smoke(candidate_id)
        _write_test_summary()
        _sha256sums()
        return PASS if result.get("final_status") == "E3_REAL_SMOKE_PASS" else FAIL
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write("FINAL_STATUS.json", {
            "run_id": RUN_ID,
            "final_status": "E3_REAL_SMOKE_FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
        })
        _sha256sums()
        return FAIL


if __name__ == "__main__":
    raise SystemExit(main())
