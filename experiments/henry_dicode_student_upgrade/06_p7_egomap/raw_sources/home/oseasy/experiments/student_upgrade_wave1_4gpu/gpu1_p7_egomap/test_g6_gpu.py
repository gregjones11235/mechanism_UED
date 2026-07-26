#!/usr/bin/env python3
"""G6 gate: 64-world SR with initialized EgoMap model vs Baseline (ckpt17500).

Since ego_gate is zero-initialized, the EgoMap model output is bit-exact to the
base model (proven in test_network_g1.py). This test confirms at the env level:
  |SR_egomap_init - SR_baseline| ≤ 5pp  (expected: 0pp)

Run on GPU1:
  CUDA_VISIBLE_DEVICES=1 python test_g6_gpu.py
"""
import os, sys, json
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src")

from launcher_p7 import (
    Cfg, GPU1_UUID, CKPT_PARENT, CKPT_STEP, OUT_ROOT, NUM_ENVS, EMB, SEED,
    verify_gpu, load_base_params, init_egomap_network, merge_ego_params,
    build_train_state, build_env
)
import ppo_tr_egomap as PE
from minicraftax.tasks.seed_tasks.original import Env as OriginalTask
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper

NUM_WORLDS = 64
EVAL_STEPS = 4096


def run_64world(params, egomap_enabled, rng):
    """Run 64 episodes, return SR (DEFEAT_KOBOLD success rate)."""
    cfg = Cfg()
    cfg.egomap_enabled = egomap_enabled
    network = PE.ActorCriticTransformerEgoMap(
        action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
        num_heads=8, qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
        egomap_channels=PE.egomap_lib.N_MAP_CH, egomap_cnn_features=(16, 32))
    egomap_cfg = PE._egomap_cfg(cfg)

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

    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)
    memories = jnp.zeros((NUM_WORLDS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    memories_mask = jnp.zeros((NUM_WORLDS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    egomap_state = PE.egomap_lib.egomap_init_state(NUM_WORLDS, egomap_cfg)
    done = jnp.zeros(NUM_WORLDS, dtype=jnp.bool_)

    successes = []
    for step_i in range(EVAL_STEPS):
        ego_features = PE.egomap_lib.egomap_read(egomap_state, obsv, egomap_cfg)
        pi, value, memories_out = network.apply(
            {"params": params}, memories, obsv, memories_mask,
            ego_features=ego_features, egomap_enabled=egomap_enabled,
            method=network.model_forward_eval)
        action = pi.mode()
        memories = jnp.roll(memories, 1, axis=1).at[:, 0].set(memories_out)
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        egomap_state = PE.egomap_lib.egomap_update(egomap_state, obsv, action, done, egomap_cfg)
        returned = info.get("returned_episode", jnp.zeros(NUM_WORLDS, dtype=jnp.bool_))
        for e in range(NUM_WORLDS):
            if returned[e]:
                ach = info.get("achievements", {})
                dk = ach.get(Achievement.DEFEAT_KOBOLD, jnp.zeros(NUM_WORLDS))
                successes.append(bool(dk[e] > 0))

    sr = sum(successes) / len(successes) if successes else 0.0
    return sr, len(successes)


def main():
    print("G6 GATE: 64-world SR (initialized EgoMap vs Baseline)")
    verify_gpu()

    rng = jax.random.PRNGKey(SEED)

    # Load ckpt17500 base params
    rng, _rng = jax.random.split(rng)
    base_params = load_base_params(_rng)

    # Init EgoMap network (zero ego params)
    rng, _rng = jax.random.split(rng)
    network, full_params = init_egomap_network(_rng, egomap_enabled=True)
    merged_params = merge_ego_params(base_params, full_params)

    # Run Baseline (egomap_enabled=False, merged params — ego keys present but unused
    # since egomap_enabled=False skips the ego path; avoids double {"params":} wrapping)
    print("\n[Baseline] 64-world eval (egomap_enabled=False) ...")
    rng, _rng = jax.random.split(rng)
    sr_base, n_base = run_64world(merged_params, egomap_enabled=False, rng=_rng)
    print(f"  Baseline SR={sr_base:.3f} (n={n_base})")

    # Run EgoMap-init (egomap_enabled=True, zero gate → bit-exact to base)
    print("\n[EgoMap-init] 64-world eval (egomap_enabled=True, zero gate) ...")
    rng, _rng = jax.random.split(rng)
    sr_ego, n_ego = run_64world(merged_params, egomap_enabled=True, rng=_rng)
    print(f"  EgoMap-init SR={sr_ego:.3f} (n={n_ego})")

    diff_pp = abs(sr_ego - sr_base) * 100
    print(f"\n  SR difference: {diff_pp:.2f}pp  (gate: ≤5pp)")
    passed = diff_pp <= 5.0
    print(f"  G6 {'PASS' if passed else 'FAIL'}")

    result = {"sr_base": sr_base, "sr_ego": sr_ego, "diff_pp": diff_pp,
              "n_base": n_base, "n_ego": n_ego, "passed": passed}
    out_path = os.path.join(OUT_ROOT, "reports", "g6_gate.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved → {out_path}")

    if not passed:
        print("  G6 FAILED — STOP long training")
        sys.exit(1)
    print("  G6 PASSED — proceed to long training")


if __name__ == "__main__":
    main()
