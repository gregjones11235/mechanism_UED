#!/usr/bin/env python3
"""Read-only probe: can Henry `load_weights_only` correctly extract params from
the P2-v1 Level2 orbax self-checkpoint (checkpoints/4096)?

Compares params loaded two ways, leaf-wise bit-exact:
  A. P2-v1 production primitive  restore_p2_v1_checkpoint (proven in Level1 verify)
  B. Henry load_weights_only on the same checkpoint path (what the session175
     pilot evaluator s175_dual_caliber_pilot.py SHA 06221187 uses)

If A == B leaf-wise (and both give param_leaves=80, encoder kernel (8335,256)),
then the 64-ep Stage4 regression evaluator can reuse load_weights_only unchanged
(just point CKPT at checkpoints/4096) -> maximal fidelity to SHA 06221187.
Otherwise the derived evaluator must source params via restore_p2_v1_checkpoint.

Runs on GPU0 (orbax checkpoints saved on GPU cannot restore under JAX cpu).
Read-only w.r.t. the checkpoint.
"""
import os, sys, json
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

import jax, jax.numpy as jnp

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
P2_SRC = "/home/oseasy/experiments/p2_v1_20260722/src"
for p in (V7_SRC, P2_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.constants import Achievement
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

import stage4_continue_launcher as L   # P2-v1 launcher (restore_p2_v1_checkpoint)

CKPT_ROOT = "/home/oseasy/experiments/p2_v1_20260722/checkpoints"
STEP = 4096
CKPT_PATH = os.path.join(CKPT_ROOT, str(STEP))   # full path for load_weights_only

# BIG network config identical to the session175 pilot evaluator (SHA 06221187)
_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6,
            max_grad_norm=1.0, total_timesteps=2005401600, num_envs=1024, num_steps=128,
            update_epochs=4, num_minibatches=8, max_updates_per_session=500)
cfg_big = type("C", (), _cfg)()

NUM_STEPS = 4096
ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])

S4_TASK_CODE = '''
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
ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])
print(f"obs_dim={OBS_DIM} action_dim={ACTION_DIM} emb={EMB}", flush=True)

# ---- A. P2-v1 production restore ----
cfg_p2 = L.Cfg()
network = L.ActorCriticTransformer(
    action_dim=ACTION_DIM, activation=cfg_p2.activation, hidden_layers=cfg_p2.hidden_layers,
    encoder_size=cfg_p2.embed_size, num_heads=cfg_p2.num_heads, qkv_features=cfg_p2.qkv_features,
    num_layers=cfg_p2.num_layers, gating=cfg_p2.gating, gating_bias=cfg_p2.gating_bias)
r = L.restore_p2_v1_checkpoint(CKPT_ROOT, STEP, network, cfg_p2, OBS_DIM)
params_A = r["train_state"].params if "train_state" in r else r["params"]
leaves_A = jax.tree_util.tree_leaves_with_path(params_A)
enc_A = None
for pth, v in leaves_A:
    ks = jax.tree_util.keystr(pth)
    if "encoder" in ks.lower() and "kernel" in ks.lower():
        enc_A = tuple(__import__("numpy").asarray(v).shape)
print(f"[A] restore_p2_v1_checkpoint: param_leaves={len(leaves_A)} encoder_kernel={enc_A}", flush=True)

# ---- B. Henry load_weights_only on the P2-v1 checkpoint path ----
ok_B = True
err_B = None
leaves_B = None
enc_B = None
try:
    ts_B = load_weights_only(CKPT_PATH, s4_base, ctor, cfg_big, load_opt_state=False)
    params_B = ts_B.params
    leaves_B = jax.tree_util.tree_leaves_with_path(params_B)
    for pth, v in leaves_B:
        ks = jax.tree_util.keystr(pth)
        if "encoder" in ks.lower() and "kernel" in ks.lower():
            enc_B = tuple(__import__("numpy").asarray(v).shape)
    print(f"[B] load_weights_only: param_leaves={len(leaves_B)} encoder_kernel={enc_B}", flush=True)
except Exception as e:
    ok_B = False
    err_B = f"{type(e).__name__}: {e}"
    print(f"[B] load_weights_only FAILED: {err_B}", flush=True)

# ---- compare A vs B leaf-wise (only if B succeeded) ----
match = False
detail = {}
if ok_B and leaves_B is not None:
    dictB = dict((p, v) for p, v in leaves_B)
    same_count = 0; total = 0; mismatch_examples = []
    for pth, av in leaves_A:
        total += 1
        if pth in dictB:
            bv = dictB[pth]
            eq = bool(jnp.array_equal(jnp.asarray(av), jnp.asarray(bv)))
            if eq:
                same_count += 1
            elif len(mismatch_examples) < 5:
                mismatch_examples.append(jax.tree_util.keystr(pth))
        else:
            if len(mismatch_examples) < 5:
                mismatch_examples.append("MISSING_IN_B:" + jax.tree_util.keystr(pth))
    match = (same_count == total == len(leaves_B))
    detail = {"leaves_A": len(leaves_A), "leaves_B": len(leaves_B),
              "identical_leaves": same_count, "total_compared": total,
              "mismatch_examples": mismatch_examples}
    print(f"[compare] identical={same_count}/{total} (B leaves={len(leaves_B)}) match={match}", flush=True)
    if mismatch_examples:
        print(f"          mismatches: {mismatch_examples}", flush=True)

verdict = "LOAD_WEIGHTS_ONLY_COMPATIBLE" if (ok_B and match) else "USE_RESTORE_P2V1"
print(f"\nVERDICT: {verdict}", flush=True)

out = {"directive": "P2-v1 方案2 Level2 weight-load compatibility probe (#57 五/#61 步骤7)",
       "checkpoint": CKPT_PATH, "obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "emb": EMB,
       "A_restore_p2v1": {"param_leaves": len(leaves_A), "encoder_kernel": list(enc_A) if enc_A else None},
       "B_load_weights_only": {"ok": bool(ok_B), "error": err_B,
                               "param_leaves": (len(leaves_B) if leaves_B else None),
                               "encoder_kernel": (list(enc_B) if enc_B else None)},
       "compare": detail, "leafwise_bit_exact_match": bool(match), "verdict": verdict}
with open("/home/oseasy/experiments/single_director_20260722/evidence/p2_v1_level2_weight_load_probe.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True, default=str); f.write("\n")
print("evidence: evidence/p2_v1_level2_weight_load_probe.json", flush=True)
sys.exit(0 if (ok_B and match) else 2)
