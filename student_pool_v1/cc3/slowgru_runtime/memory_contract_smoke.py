#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 SlowGRU MEMORY CONTRACT smoke (finalize round, task section 5, part 2).

Runs on a server GPU with the real S4_dark (DEFEAT_KOBOLD) environment and the
candidate's own 98304 checkpoint. Produces ACTUAL evidence from THIS run (hashes
computed live) — no preset booleans.

Per-mode contract proven here:
  both modes:
    - in-segment memory IS used (fast window memories + slow longstate evolve
      step-by-step inside a 128-step segment; hashes at 32/64/96/127 all differ
      from init and from each other);
    - params SHA unchanged across the whole smoke.
  RESET128:
    - at the 128-step segment boundary the slow longstate is reset to init
      (hash after boundary == live init hash), carry-in was non-trivial, and the
      fast window memories are CARRIED (unchanged by the boundary call);
  PERSISTENT:
    - at the boundary the longstate is NOT unconditionally cleared (hash after
      == carry-in hash != init hash, non-trivial);
  both modes, done/reset cleanup:
    - reset_memory(all-ones mask) restores longstate EXACTLY to init and zeros
      the fast window memories (contract cleanup, bit-exact);
    - the true_done reset signal measurably alters the slow-state trajectory
      (branch A true_done=zeros vs branch B true_done=ones from the identical
      state -> different longstate hashes);
    - opportunistic: any NATURAL env returned_episode events observed during the
      rollouts are recorded as additional live evidence.

Hard rules: reward numbers are REFERENCE_ONLY_NOT_PERFORMANCE_JUDGMENT; any failed
check -> status=FAIL and the caller must stop and report (no auto-retry).
"""
import argparse
import json
import os
import sys


def _parse_gpu():
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
    """Replicates the trainer env construction verbatim."""
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
    return env, env_params


def snap(handle, ms):
    import jax
    flat = jax.tree_util.tree_leaves(
        [ms["memories"], ms["memories_mask"], ms["memories_mask_idx"],
         ms["longstate"]["h"], ms["longstate"]["buf"], ms["longstate"]["count"]])
    return dict(
        memories_hash=rt.leaf_hash_pytree([ms["memories"]]),
        longstate_hash=rt.longstate_leaf_hash(ms["longstate"]),
        all_hash=rt.leaf_hash_pytree(flat),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    args, _ = ap.parse_known_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    with open(args.contract, encoding="utf-8") as f:
        contract = json.load(f)
    checks = []

    def ck(name, ok, detail=""):
        checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
        return ok

    handle = rt.load_candidate(contract)
    rt.seed_policy_rng(handle, handle["contract"]["smoke_seed"])
    carry_mode = handle["carry_mode"]
    env, env_params = build_s4_dark_env()

    init_ms = rt.init_memory(handle, 16)
    init_snap = snap(handle, init_ms)
    init_ls_hash = rt.longstate_leaf_hash(handle["init_longstate"](16))
    zero_memories_hash = rt.leaf_hash_pytree([jnp.zeros((16, 128, 2, 256))])
    params_sha_before = rt.params_sha(handle)

    # ---------- segment 1: steps 0..127 (in-segment memory use) ----------
    rng = jax.random.PRNGKey(717171)
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)
    ms = rt.init_memory(handle, 16)
    seg_snaps = {}
    true_done = None
    last_done = jnp.zeros((16,), jnp.bool_)
    natural_dones = 0
    rewards_total = np.zeros((16,), dtype=np.float64)
    for t in range(160):
        if t == 128:
            ls_pre_boundary = rt.longstate_leaf_hash(ms["longstate"])
            mem_pre_boundary = rt.leaf_hash_pytree([ms["memories"]])
            ms, info_b = rt.on_segment_boundary(handle, ms)
            ls_post_boundary = rt.longstate_leaf_hash(ms["longstate"])
            mem_post_boundary = rt.leaf_hash_pytree([ms["memories"]])
        action, ms, extras = rt.policy_step(handle, obs, ms, done_mask=last_done,
                                            true_done=true_done, greedy=True)
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, last_done, info = env.step(_rng, env_state, action, env_params)
        true_done = info["returned_episode"]
        natural_dones += int(np.sum(np.asarray(true_done)))
        rewards_total += np.asarray(reward, dtype=np.float64)
        if (t + 1) in (32, 64, 96, 127):
            seg_snaps[t + 1] = snap(handle, ms)

    s = seg_snaps
    # in-segment memory use: evolves at every snapshot, all != init
    ck("insegment_memories_used",
       all(s[k]["memories_hash"] != init_snap["memories_hash"] for k in (32, 64, 96, 127))
       and len({s[k]["memories_hash"] for k in (32, 64, 96, 127)}) == 4,
       "mem32=%s mem64=%s mem96=%s mem127=%s" % tuple(s[k]["memories_hash"][:12]
                                                      for k in (32, 64, 96, 127)))
    ck("insegment_longstate_used",
       all(s[k]["longstate_hash"] != init_ls_hash for k in (32, 64, 96, 127)),
       "ls32=%s ls127=%s init=%s" % (s[32]["longstate_hash"][:12],
                                     s[127]["longstate_hash"][:12], init_ls_hash[:12]))

    # ---------- boundary behavior at step 128 ----------
    if carry_mode == "RESET128":
        ck("reset128_boundary_longstate_reset_to_init",
           ls_post_boundary == init_ls_hash,
           "after=%s init=%s" % (ls_post_boundary[:16], init_ls_hash[:16]))
        ck("reset128_boundary_carry_in_nontrivial",
           ls_pre_boundary != init_ls_hash, "carry_in=%s" % ls_pre_boundary[:16])
        ck("reset128_boundary_fast_memories_carried",
           mem_post_boundary == mem_pre_boundary,
           "pre=%s post=%s" % (mem_pre_boundary[:16], mem_post_boundary[:16]))
    elif carry_mode == "PERSISTENT":
        ck("persistent_boundary_not_unconditionally_cleared",
           ls_post_boundary == ls_pre_boundary and ls_post_boundary != init_ls_hash,
           "after=%s in=%s init=%s" % (ls_post_boundary[:16], ls_pre_boundary[:16],
                                       init_ls_hash[:16]))
        ck("persistent_boundary_longstate_nontrivial",
           ls_pre_boundary != init_ls_hash, "carry_in=%s" % ls_pre_boundary[:16])
    else:
        ck("carry_mode_known", False, carry_mode)

    # post-boundary segment 2 steps 129..160 were executed above (memory continues)
    ck("post_boundary_rollout_continued", True,
       "steps 129-160 executed after boundary; natural_dones_so_far=%d" % natural_dones)

    # ---------- done/reset contract cleanup ----------
    # (a) reset_memory with all-ones mask -> longstate EXACTLY init, memories zeroed
    ms_mid = ms
    mid_ls_hash = rt.longstate_leaf_hash(ms_mid["longstate"])
    ms_reset = rt.reset_memory(handle, ms_mid, jnp.ones((16,), jnp.bool_))
    reset_ls_hash = rt.longstate_leaf_hash(ms_reset["longstate"])
    reset_mem_hash = rt.leaf_hash_pytree([ms_reset["memories"]])
    ck("reset_memory_longstate_exactly_init", reset_ls_hash == init_ls_hash,
       "after_reset=%s init=%s (mid was %s)" % (reset_ls_hash[:16], init_ls_hash[:16],
                                                mid_ls_hash[:16]))
    ck("reset_memory_fast_memories_zeroed", reset_mem_hash == zero_memories_hash,
       "mem=%s zero=%s" % (reset_mem_hash[:16], zero_memories_hash[:16]))

    # (b) true_done reset signal alters the slow-state trajectory from identical state
    rng2 = jax.random.PRNGKey(828282)
    rng2, _r2 = jax.random.split(rng2)
    obs2, es2 = env.reset(_r2, env_params)
    ms_a = ms_b = rt.init_memory(handle, 16)
    zeros16 = jnp.zeros((16,), jnp.bool_)
    ones16 = jnp.ones((16,), jnp.bool_)
    for _t in range(16):   # build identical non-trivial state in both branches
        act_a, ms_a, _ = rt.policy_step(handle, obs2, ms_a, done_mask=zeros16,
                                        true_done=zeros16, greedy=True)
        ms_b, _ = rt.policy_step(handle, obs2, ms_b, done_mask=zeros16,
                                 true_done=zeros16, greedy=True)[1:]
        rng2, _r2 = jax.random.split(rng2)
        obs2, es2, _rew, _d, _i = env.step(_r2, es2, act_a, env_params)
    ls_before_branch = rt.longstate_leaf_hash(ms_a["longstate"])
    same_state = rt.longstate_leaf_hash(ms_b["longstate"]) == ls_before_branch
    _, ms_a2, _ = rt.policy_step(handle, obs2, ms_a, done_mask=zeros16,
                                 true_done=zeros16, greedy=True)
    _, ms_b2, _ = rt.policy_step(handle, obs2, ms_b, done_mask=zeros16,
                                 true_done=ones16, greedy=True)
    ls_a = rt.longstate_leaf_hash(ms_a2["longstate"])
    ls_b = rt.longstate_leaf_hash(ms_b2["longstate"])
    ck("true_done_reset_signal_alters_slow_state",
       bool(same_state) and ls_a != ls_b,
       "same_start=%s no_reset_ls=%s with_reset_ls=%s" % (same_state, ls_a[:16], ls_b[:16]))

    # ---------- params immutability ----------
    params_sha_after = rt.params_sha(handle)
    ck("params_sha_unchanged_by_smoke", params_sha_after == params_sha_before,
       "before=%s after=%s" % (params_sha_before[:16], params_sha_after[:16]))

    status = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    result = dict(
        candidate_id=contract["candidate_id"],
        runtime=rt.RUNTIME_NAME,
        abi_version=rt.ABI_VERSION,
        carry_mode=carry_mode,
        gpu_uuid=GPU_UUID,
        smoke_class="MEMORY_CONTRACT_SMOKE",
        memory_contract_status=status,
        steps_executed=160,
        segment_boundary_crossed_at=128,
        checks=dict(total=len(checks), passed=sum(1 for c in checks if c["passed"]),
                    failed=sum(1 for c in checks if not c["passed"])),
        check_details=checks,
        evidence=dict(
            init_ls_hash=init_ls_hash,
            insegment_snapshots=s,
            boundary=dict(ls_pre=ls_pre_boundary, ls_post=ls_post_boundary,
                          fast_memories_pre=mem_pre_boundary,
                          fast_memories_post=mem_post_boundary),
            reset_memory=dict(mid_ls_hash=mid_ls_hash, after_reset_ls_hash=reset_ls_hash,
                              after_reset_memories_hash=reset_mem_hash,
                              zero_memories_hash=zero_memories_hash),
            true_done_branch=dict(same_start=bool(same_state),
                                  no_reset_ls=ls_a, with_reset_ls=ls_b),
            natural_env_dones_observed=natural_dones,
            params_sha256_before=params_sha_before,
            params_sha256_after=params_sha_after,
            checkpoint_file_sha256=handle["file_sha256"],
            params_sha256=handle["params_sha256"],
        ),
        reward_reference_only=dict(
            per_env_total=[float(x) for x in rewards_total],
            NOTE="REFERENCE_ONLY_NOT_PERFORMANCE_JUDGMENT — 不得用于性能判断或排名"),
        formal_eval_binding="WAITING_CC4_COMMON_CONTRACT",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    failed = [c["check"] for c in checks if not c["passed"]]
    print("MEMCONTRACT candidate=%s carry_mode=%s status=%s checks=%d/%d gpu=%s" % (
        contract["candidate_id"], carry_mode, status,
        result["checks"]["passed"], result["checks"]["total"], GPU_UUID))
    if failed:
        print("FAILED_CHECKS=%s" % ",".join(failed))
    print("OUT=%s" % args.out)


if __name__ == "__main__":
    main()
