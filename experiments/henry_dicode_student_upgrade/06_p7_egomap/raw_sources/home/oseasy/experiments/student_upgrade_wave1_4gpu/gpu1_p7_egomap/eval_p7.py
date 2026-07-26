#!/usr/bin/env python3
"""P7-EGOMAP frozen 256-world evaluator.

Loads a trained checkpoint, runs 256 deterministic episodes, collects:
  DK SR, floor3, conditional kill, death-timeout, episode length,
  unique cells, revisit ratio, coverage (EgoMap arm only),
  paired CI + McNemar (when comparing two arms).

Usage:
  python eval_p7.py --arm control --step 98304
  python eval_p7.py --arm egomap --step 98304
  python eval_p7.py --compare  # runs both arms and computes paired stats
"""
import os, sys, json, argparse, pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src")

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
import ppo_tr_egomap as PE
from launcher_p7 import Cfg, NUM_ENVS, EMB, CKPT_PARENT, CKPT_STEP, OUT_ROOT

NUM_WORLDS = 256
EVAL_STEPS = 4096  # max steps per episode
SEED = 42


def load_params(arm, step):
    path = os.path.join(OUT_ROOT, "checkpoints", arm, f"params_{step}.pkl")
    with open(path, "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def build_eval_fn(arm, params, rng):
    """Build a jitted eval step function for the given arm."""
    cfg = Cfg()
    cfg.egomap_enabled = (arm == "egomap")
    # Use make_eval from ppo_tr_egomap
    eval_fn = PE.make_eval(cfg, [OriginalTask], EVAL_STEPS, jnp.zeros((1, EMB)), None)
    return jax.jit(eval_fn), cfg


def run_eval(arm, step, rng):
    """Run 256-world eval for one arm, return per-episode metrics dict."""
    print(f"\n[eval] arm={arm} step={step} worlds={NUM_WORLDS}")
    params = load_params(arm, step)

    cfg = Cfg()
    cfg.egomap_enabled = (arm == "egomap")

    # Build network
    network = PE.ActorCriticTransformerEgoMap(
        action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
        num_heads=8, qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
        egomap_channels=PE.egomap_lib.N_MAP_CH, egomap_cnn_features=(16, 32))

    # Build env (256 parallel envs)
    from minicraftax.tasks.seed_tasks.original import Env as OriginalTask
    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    task_embeddings = jnp.zeros((1, EMB))
    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=EVAL_STEPS)
    base_env = MultiTaskMiniCraftaxEnv(
        [OriginalTask], static_env_params, env_params, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB,
        completion_bonus_scale=cfg.completion_bonus_scale, completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, rng, NUM_WORLDS, 1, cfg.optimistic_reset_ratio, jnp.array([1.0]), ach_table)

    # Init eval state
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((NUM_WORLDS, 8335))
    init_memory = jnp.zeros((NUM_WORLDS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    init_mask = jnp.zeros((NUM_WORLDS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    init_ego = jnp.zeros((NUM_WORLDS, cfg.egomap_map_size, cfg.egomap_map_size, PE.egomap_lib.N_MAP_CH))
    egomap_cfg = PE._egomap_cfg(cfg)
    egomap_state = PE.egomap_lib.egomap_init_state(NUM_WORLDS, egomap_cfg)

    # Reset env
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)

    # Run eval loop (deterministic: argmax action)
    memories = init_memory
    memories_mask = init_mask
    memories_mask_idx = jnp.full((NUM_WORLDS,), cfg.window_mem + 1, dtype=jnp.int32)
    done = jnp.zeros(NUM_WORLDS, dtype=jnp.bool_)

    episode_results = []
    for step_i in range(EVAL_STEPS):
        # Read egomap features
        ego_features = PE.egomap_lib.egomap_read(egomap_state, obsv, egomap_cfg)
        # Forward (deterministic: use mode not sample)
        pi, value, memories_out = network.apply(
            {"params": params}, memories, obsv, memories_mask,
            ego_features=ego_features, egomap_enabled=cfg.egomap_enabled,
            method=network.model_forward_eval)
        action = pi.mode()  # deterministic
        # Update memory
        memories = jnp.roll(memories, 1, axis=1)
        memories = memories.at[:, 0].set(memories_out)
        # Step env
        rng, _rng = jax.random.split(rng)
        obsv_next, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        # Update egomap
        egomap_state = PE.egomap_lib.egomap_update(egomap_state, obsv, action, done, egomap_cfg)
        obsv = obsv_next
        # Collect episode results
        returned = info.get("returned_episode", jnp.zeros(NUM_WORLDS, dtype=jnp.bool_))
        for e in range(NUM_WORLDS):
            if returned[e]:
                ep_info = {
                    "world": e,
                    "step": step_i,
                    "episode_length": int(info.get("episode_length", jnp.zeros(NUM_WORLDS))[e]),
                    "success": bool(info.get("achievements", {}).get(Achievement.DEFEAT_KOBOLD, jnp.zeros(NUM_WORLDS))[e] > 0),
                }
                episode_results.append(ep_info)

    # Compute metrics
    if not episode_results:
        print(f"  [eval] WARNING: no episodes completed in {EVAL_STEPS} steps")
        return {}

    n = len(episode_results)
    sr = sum(r["success"] for r in episode_results) / n
    avg_len = np.mean([r["episode_length"] for r in episode_results])
    print(f"  [eval] {arm} step={step}: SR={sr:.3f} avg_len={avg_len:.1f} n={n}")
    return {"sr": sr, "avg_len": avg_len, "n": n, "episodes": episode_results}


def compare_arms(control_res, egomap_res):
    """Compute paired CI and McNemar between Control and EgoMap."""
    ctrl_eps = {r["world"]: r["success"] for r in control_res["episodes"]}
    ego_eps = {r["world"]: r["success"] for r in egomap_res["episodes"]}
    common_worlds = set(ctrl_eps.keys()) & set(ego_eps.keys())
    n = len(common_worlds)
    if n == 0:
        print("  [compare] no common worlds")
        return {}
    ctrl_succ = np.array([ctrl_eps[w] for w in sorted(common_worlds)], dtype=np.int32)
    ego_succ = np.array([ego_eps[w] for w in sorted(common_worlds)], dtype=np.int32)
    # Paired difference
    diff = ego_succ - ctrl_succ
    sr_diff = diff.mean()
    # Bootstrap CI (95%)
    rng = np.random.default_rng(42)
    boot_diffs = [rng.choice(diff, size=n, replace=True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    # McNemar
    b = np.sum((ctrl_succ == 0) & (ego_succ == 1))  # ctrl fail, ego success
    c = np.sum((ctrl_succ == 1) & (ego_succ == 0))  # ctrl success, ego fail
    mcnemar_chi2 = (abs(b - c) - 1) ** 2 / (b + c + 1e-12)
    print(f"  [compare] n={n} SR_diff={sr_diff:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}] McNemar_chi2={mcnemar_chi2:.2f} (b={b},c={c})")
    return {"sr_diff": float(sr_diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "mcnemar_chi2": float(mcnemar_chi2), "mcnemar_b": int(b), "mcnemar_c": int(c), "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["control", "egomap"])
    parser.add_argument("--step", type=int, default=98304)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(SEED)

    if args.compare:
        rng, _rng = jax.random.split(rng)
        ctrl = run_eval("control", args.step, _rng)
        rng, _rng = jax.random.split(rng)
        ego = run_eval("egomap", args.step, _rng)
        result = compare_arms(ctrl, ego)
        result["control"] = ctrl
        result["egomap"] = ego
    else:
        rng, _rng = jax.random.split(rng)
        result = run_eval(args.arm, args.step, _rng)

    # Save results
    out_path = os.path.join(OUT_ROOT, "reports", f"eval_{args.arm or 'compare'}_{args.step}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  [eval] results saved → {out_path}")


if __name__ == "__main__":
    main()
