#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 SlowGRU candidate interface smoke (server-side, real GPU forward).

Checks (task sections 六 + 更正轮):
  1. checkpoint load via THIN_GTRXL128_SLOWGRU_RUNTIME (fail-closed file/params SHA);
  2. file SHA + params SHA recomputed (inside load_candidate, re-stated here);
  3. params finite;
  4. real policy forward inside a genuine S4_dark (DEFEAT_KOBOLD) Craftax rollout,
     env construction asserts obs_dim==8335 / action_dim==43 / emb==67 (canonical);
  5. >=32-step rollout (runs 160 to also cross one 128-step segment boundary);
  6. memory state actually changes from init;
  7. segment-boundary behavior matches the carry_mode contract:
       RESET128   -> slow longstate == init-hash after boundary, carry-in non-trivial;
       PERSISTENT -> longstate fully carried (unchanged) across boundary, non-initial;
  8. all actions legal in [0, 43);
  9. params SHA identical before/after the rollout;
 10. determinism gate: fresh reload + identical seeds reproduces the first 32 actions
     and the step-32 memory hash bit-exactly.

HARD RULES: reward numbers are recorded as REFERENCE_ONLY_NOT_PERFORMANCE_JUDGMENT —
no performance claim, no early-stop decision is derived here. Formal ranking is
CC4's common evaluator (formal_eval_binding=WAITING_CC4_COMMON_CONTRACT).
Any failed check -> status=FAIL; the caller must stop and report (no auto-retry).
"""
import argparse
import json
import os
import sys


def _parse_gpu():
    # env vars must be set BEFORE any jax import (runtime imports are lazy)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gpu-uuid", required=True)
    known, _ = ap.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = known.gpu_uuid
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    return known.gpu_uuid


GPU_UUID = _parse_gpu()

HERE = os.path.dirname(os.path.abspath(__file__))
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for _p in (HERE, os.path.join(os.path.dirname(HERE), "cc3_common"), V7 + "/src", V7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
S4_TASK_SHA = "45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d"

import slowgru_runtime as rt


def build_s4_dark_env():
    """Replicates the trainer env construction verbatim (driver lines 176-191)."""
    import jax
    import jax.numpy as jnp
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],
                          dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}
    exec(s4_code, ns)
    Task = ns["Env"]
    static_env_params = StaticEnvParams()
    env_params_ctor = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params_ctor, True,
        conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=0.0,
        completion_bonus_min=0.0, bonus_type="none", dynamic_bonus_k=0.0)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), 16, 1, 16, jnp.ones(1), ach_table)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    return env, env_params, dict(obs_dim=obs_dim, action_dim=action_dim, emb=EMB)


def mem_snapshot(handle, ms):
    import jax
    flat = jax.tree_util.tree_leaves(
        [ms["memories"], ms["memories_mask"], ms["memories_mask_idx"],
         ms["longstate"]["h"], ms["longstate"]["buf"], ms["longstate"]["count"]])
    return dict(
        memories_hash=rt.leaf_hash_pytree([ms["memories"]]),
        mask_hash=rt.leaf_hash_pytree([ms["memories_mask"], ms["memories_mask_idx"]]),
        longstate_hash=rt.longstate_leaf_hash(ms["longstate"]),
        all_hash=rt.leaf_hash_pytree(flat),
    )


def run_rollout(handle, env, env_params, n_steps, env_seed, boundary_at=128):
    import jax
    import jax.numpy as jnp
    import numpy as np

    rt.seed_policy_rng(handle, handle["contract"]["smoke_seed"])
    rng = jax.random.PRNGKey(env_seed)
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)
    ms = rt.init_memory(handle, 16)
    init_snap = mem_snapshot(handle, ms)
    init_ls_hash = rt.longstate_leaf_hash(handle["init_longstate"](16))

    actions = []
    snaps = {}
    rewards_total = np.zeros((16,), dtype=np.float64)
    boundary = None
    true_done = None
    for t in range(n_steps):
        if boundary_at and t == boundary_at:
            ls_hash_pre = rt.longstate_leaf_hash(ms["longstate"])
            ms, info_b = rt.on_segment_boundary(handle, ms)
            ls_hash_post = rt.longstate_leaf_hash(ms["longstate"])
            boundary = dict(at_step=t, longstate_hash_carry_in=ls_hash_pre,
                            longstate_hash_after=ls_hash_post, info=info_b)
        action, ms, extras = rt.policy_step(handle, obs, ms, done_mask=last_done
                                            if t > 0 else jnp.zeros((16,), jnp.bool_),
                                            true_done=true_done)
        a = np.asarray(action)
        actions.append(a)
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, last_done, info = env.step(_rng, env_state, action, env_params)
        true_done = info["returned_episode"]
        rewards_total += np.asarray(reward, dtype=np.float64)
        if (t + 1) in (32, 64, 96, 128, 160):
            snaps[t + 1] = mem_snapshot(handle, ms)
    return dict(actions=np.stack(actions), snaps=snaps, init_snap=init_snap,
                init_ls_hash=init_ls_hash, boundary=boundary,
                rewards_total=rewards_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=160)
    args, _ = ap.parse_known_args()

    import jax
    import numpy as np

    with open(args.contract, encoding="utf-8") as f:
        contract = json.load(f)
    checks = []

    def ck(name, ok, detail=""):
        checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
        return ok

    devs = jax.local_devices()
    ck("single_visible_device_gpu_allowed", len(devs) == 1 and GPU_UUID in (
        "GPU-8df11537-ab79-722d-606f-411966196c4c",
        "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"), str(devs))

    # 1-3. load + SHA + finiteness (fail closed inside load_candidate)
    handle = rt.load_candidate(contract)
    ck("load_candidate_sha_verified", True,
       "file=%s params=%s" % (handle["file_sha256"][:16], handle["params_sha256"][:16]))
    ck("params_sha_matches_contract", handle["params_sha256"] == contract["params_sha256"])
    params_sha_before = rt.params_sha(handle)
    ck("params_finite_before", True, "checked in load_candidate")

    # 4. canonical env construction asserts
    env, env_params, dims = build_s4_dark_env()
    ck("obs_dim_8335", dims["obs_dim"] == 8335, dims["obs_dim"])
    ck("action_dim_43", dims["action_dim"] == 43, dims["action_dim"])
    ck("conditioning_emb_67", dims["emb"] == 67, dims["emb"])
    ck("s4_task_sha", rt.sha_file(S4_TASK_PATH) == S4_TASK_SHA)

    # 5-8. rollout A
    A = run_rollout(handle, env, env_params, args.steps, env_seed=424242)
    actions = A["actions"]
    ck("rollout_steps_executed", actions.shape[0] == args.steps, actions.shape)
    ck("actions_legal", bool(np.all((actions >= 0) & (actions < 43))),
       "min=%d max=%d" % (actions.min(), actions.max()))
    s32 = A["snaps"][32]
    init = A["init_snap"]
    ck("memory_state_changed_by_step32",
       s32["all_hash"] != init["all_hash"] and s32["memories_hash"] != init["memories_hash"],
       "mem32=%s init=%s" % (s32["memories_hash"][:16], init["memories_hash"][:16]))

    b = A["boundary"]
    ck("segment_boundary_crossed", b is not None and b["at_step"] == 128)
    if handle["carry_mode"] == "RESET128":
        ck("reset128_boundary_longstate_eq_init_contract",
           b["longstate_hash_after"] == A["init_ls_hash"],
           "after=%s init=%s" % (b["longstate_hash_after"][:16], A["init_ls_hash"][:16]))
        ck("reset128_carry_in_nontrivial",
           b["longstate_hash_carry_in"] != A["init_ls_hash"],
           "carry_in=%s" % b["longstate_hash_carry_in"][:16])
        ck("reset128_fast_memories_carried_across_boundary",
           A["snaps"][160]["memories_hash"] != init["memories_hash"],
           "mem160=%s init=%s" % (A["snaps"][160]["memories_hash"][:16],
                                  init["memories_hash"][:16]))
    elif handle["carry_mode"] == "PERSISTENT":
        ck("persistent_boundary_full_carry",
           b["longstate_hash_after"] == b["longstate_hash_carry_in"],
           "after=%s in=%s" % (b["longstate_hash_after"][:16],
                               b["longstate_hash_carry_in"][:16]))
        ck("persistent_longstate_non_initial",
           b["longstate_hash_after"] != A["init_ls_hash"])
    else:
        ck("carry_mode_known", False, handle["carry_mode"])

    # 9. params immutability across the whole rollout
    params_sha_after = rt.params_sha(handle)
    ck("params_sha_unchanged_by_inference", params_sha_after == params_sha_before,
       "before=%s after=%s" % (params_sha_before[:16], params_sha_after[:16]))

    # 10. determinism gate: fresh reload + identical seeds -> identical first 32 actions
    handle2 = rt.load_candidate(contract)
    B = run_rollout(handle2, env, env_params, 32, env_seed=424242)
    ck("determinism_first32_actions_bitexact",
       bool(np.array_equal(A["actions"][:32], B["actions"])),
       "n_equal=%d/32" % int(np.sum(A["actions"][:32] == B["actions"])))
    ck("determinism_step32_memory_hash_bitexact",
       A["snaps"][32]["all_hash"] == B["snaps"][32]["all_hash"])

    status = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    result = dict(
        candidate_id=contract["candidate_id"],
        runtime=rt.RUNTIME_NAME,
        abi_version=rt.ABI_VERSION,
        carry_mode=handle["carry_mode"],
        gpu_uuid=GPU_UUID,
        devices=str(devs),
        smoke_status=status,
        steps_executed=int(actions.shape[0]),
        checks=dict(total=len(checks), passed=sum(1 for c in checks if c["passed"]),
                    failed=sum(1 for c in checks if not c["passed"])),
        check_details=checks,
        boundary_event=A["boundary"],
        init_ls_hash=A["init_ls_hash"],
        memory_snapshots={"step32": A["snaps"][32], "step160": A["snaps"][160],
                          "init": init},
        params_sha256_before=params_sha_before,
        params_sha256_after=params_sha_after,
        checkpoint_file_sha256=handle["file_sha256"],
        params_sha256=handle["params_sha256"],
        metadata=rt.candidate_metadata(handle),
        reward_reference_only=dict(
            per_env_total=[float(x) for x in A["rewards_total"]],
            NOTE="REFERENCE_ONLY_NOT_PERFORMANCE_JUDGMENT — 短 smoke reward 不得用于性能判断; "
                 "正式排名由 CC4 公共 evaluator 固定"),
        formal_eval_binding="WAITING_CC4_COMMON_CONTRACT",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    failed = [c["check"] for c in checks if not c["passed"]]
    print("SMOKE candidate=%s carry_mode=%s status=%s checks=%d/%d gpu=%s" % (
        contract["candidate_id"], handle["carry_mode"], status,
        result["checks"]["passed"], result["checks"]["total"], GPU_UUID))
    if failed:
        print("FAILED_CHECKS=%s" % ",".join(failed))
    print("OUT=%s" % args.out)


if __name__ == "__main__":
    main()
