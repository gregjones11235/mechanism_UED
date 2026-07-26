#!/usr/bin/env python3
"""Decisive probe: does base-only load_weights_only restore EACH training-type
checkpoint cleanly, or do tser_ppo/lpg_hrl checkpoints carry extra treatment-head
params that break the base template?  CPU only, no rollout.
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
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.utils.general.train_state_utils import load_weights_only

R = "/home/oseasy/experiments/mechanism_UED_continuation_20260715/shared_r0/runs/d052_dynamic"
CELLS = {
    "original": f"{R}/original_x_original/seed0_1784441252/checkpoints/98304",
    "tser_ppo": f"{R}/original_x_tser_ppo/seed0_1784451914/checkpoints/98304",
    "lpg_hrl":  f"{R}/original_x_lpg_hrl/seed0_1784448381/checkpoints/98304",
    "lpac":     f"{R}/original_x_lpac/seed0_1784453656/checkpoints/98304",
    "clpa":     f"{R}/original_x_clpa/seed0_1784455651/checkpoints/98304",
}


def build_cfg(training):
    base = dict(
        num_envs=16, num_steps=128, num_minibatches=2, update_epochs=2,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, lr=3e-4, anneal_lr=False, min_lr=3e-6,
        activation="relu", hidden_layers=64, embed_size=32, num_heads=4,
        qkv_features=128, num_layers=1, window_mem=8, window_grad=4,
        gating=True, gating_bias=1.0, total_timesteps=24576, max_updates_per_session=12,
    )
    return type("C", (), base)()


# Build the Stage4 env (same construction as evaluator)
S4 = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements = [Achievement.DEFEAT_KOBOLD]
        self.completed_achievements = []
        self.label = "DEFEAT_KOBOLD"
    def get_task_params(self):
        return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng, _rng = jax.random.split(rng)
        b = WorldBuilder(_rng, self.static_params, self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2, 8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s = b.build(rng)
        up = b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2, up[0], up[1]].set(ItemType.NONE.value))
'''
ns = {}
exec(S4, ns)
EnvCls = ns["Env"]
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
base_env = MultiTaskMiniCraftaxEnv([EnvCls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)

results = {}
for training, ckpt in CELLS.items():
    cfg = build_cfg(training)
    try:
        ts = load_weights_only(ckpt, base_env, ctor, cfg, load_opt_state=False)
        leaves = len(jax.tree_util.tree_leaves(ts.params))
        results[training] = {"loaded": True, "param_leaves": leaves, "error": None}
    except Exception as e:
        import traceback
        results[training] = {"loaded": False, "param_leaves": None,
                             "error": f"{type(e).__name__}: {str(e)[:400]}"}
    print(f"\n>>> {training}: {json.dumps(results[training])}", flush=True)

out = "/home/oseasy/experiments/d052_unified_eval_20260722/evaluator/load_probe_report.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out}", flush=True)
print("\nSUMMARY:", json.dumps(results, indent=2), flush=True)
