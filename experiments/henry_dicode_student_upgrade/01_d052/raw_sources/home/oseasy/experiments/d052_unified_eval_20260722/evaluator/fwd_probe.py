#!/usr/bin/env python3
"""Inspect restored ts.params tree structure per training type and test which
param subset feeds network.apply(model_forward_eval).  GPU, no rollout.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PROBE_GPU", "1")
import sys, json
import jax, jax.numpy as jnp

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
sys.path.insert(0, V7)
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.network import ActorCriticTransformer

R = "/home/oseasy/experiments/mechanism_UED_continuation_20260715/shared_r0/runs/d052_dynamic"
CELLS = {
    "original": f"{R}/original_x_original/seed0_1784441252/checkpoints/98304",
    "tser_ppo": f"{R}/original_x_tser_ppo/seed0_1784451914/checkpoints/98304",
    "lpg_hrl":  f"{R}/original_x_lpg_hrl/seed0_1784448381/checkpoints/98304",
}


def build_cfg():
    base = dict(num_envs=16, num_steps=128, num_minibatches=2, update_epochs=2,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, lr=3e-4, anneal_lr=False, min_lr=3e-6, activation="relu",
        hidden_layers=64, embed_size=32, num_heads=4, qkv_features=128, num_layers=1,
        window_mem=8, window_grad=4, gating=True, gating_bias=1.0,
        total_timesteps=24576, max_updates_per_session=12)
    return type("C", (), base)()


S4 = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2,8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s=b.build(rng); up=b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2,up[0],up[1]].set(ItemType.NONE.value))
'''
ns = {}; exec(S4, ns); EnvCls = ns["Env"]
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
base_env = MultiTaskMiniCraftaxEnv([EnvCls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)
cfg = build_cfg()
network = ActorCriticTransformer(action_dim=base_env.action_space(ctor).n, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
obs_dim = base_env.observation_space(ctor).shape[0]
B = 2
mem = jnp.zeros((B, cfg.window_mem, cfg.num_layers, cfg.embed_size))
mask = jnp.zeros((B, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)


def tree_keys(p, depth=0, prefix=""):
    out = []
    if hasattr(p, "items"):
        for k, v in p.items():
            path = f"{prefix}/{k}"
            out.append(("  " * depth) + f"{k}  (leaves={len(jax.tree_util.tree_leaves(v))})")
            if depth < 2:
                out.extend(tree_keys(v, depth + 1, path))
    return out


report = {}
for training, ckpt in CELLS.items():
    print(f"\n===== {training} =====", flush=True)
    ts = load_weights_only(ckpt, base_env, ctor, cfg, load_opt_state=False)
    print(f"ts.params total leaves = {len(jax.tree_util.tree_leaves(ts.params))}", flush=True)
    print("--- ts.params structure (top 2 levels) ---", flush=True)
    for line in tree_keys(ts.params):
        print(line, flush=True)
    # obs for apply: use a realistic obs dim
    obs = jnp.zeros((B, obs_dim))
    entry = {"obs_dim": int(obs_dim)}
    # Test apply variants
    variants = {}
    for vname, vp in [("ts.params", ts.params), ("ts.params['params']", ts.params.get("params") if hasattr(ts.params, "get") else None)]:
        if vp is None:
            variants[vname] = "N/A"
            continue
        try:
            pi, v, mo = network.apply(vp, mem, obs, mask, method=network.model_forward_eval)
            variants[vname] = f"OK action_dim={pi.logits.shape[-1]} v_shape={v.shape} mem_out_shape={mo.shape}"
        except Exception as e:
            variants[vname] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    for k, v in variants.items():
        print(f"apply[{k}]: {v}", flush=True)
    entry["apply"] = variants
    entry["total_leaves"] = int(len(jax.tree_util.tree_leaves(ts.params)))
    report[training] = entry

out = "/home/oseasy/experiments/d052_unified_eval_20260722/evaluator/fwd_probe_report.json"
with open(out, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nSaved {out}", flush=True)
