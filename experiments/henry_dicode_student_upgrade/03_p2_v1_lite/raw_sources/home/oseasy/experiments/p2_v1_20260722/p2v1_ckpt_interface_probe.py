#!/usr/bin/env python3
"""P2-v1 STATIC check 8 — checkpoint interface (GPU, READ-ONLY).

Verifies load_weights_only loads the HEALTHY session175 start checkpoint
(base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500) with the P2-v1 BIG config +
embedding-67 Stage4 env, and that the encoder kernel is (8335, 256) with a working
forward. No training, no checkpoint write. Confirms the P2-v1 start point (NOT the
forbidden P2-v0 98304/122880) is loadable for the later GPU smoke.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"  # GPU0
import json, sys, time
import numpy as np
_HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
sys.path.insert(0, _HENRY_SRC)
import jax, jax.numpy as jnp
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

CKPT = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"
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
ns = {}; exec(S4, ns); S4Cls = ns["Env"]
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                              conditioning_type="embedding", embedding_size=EMB)
obs_dim = int(env.observation_space(ctor).shape[0])
act = int(env.action_space(ctor).n)

cfg = type("C", (), dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
    qkv_features=256, num_layers=2, gating=True, gating_bias=2.0, window_mem=128,
    window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6, max_grad_norm=1.0,
    total_timesteps=2005401600, num_envs=1024, num_steps=128, update_epochs=4,
    num_minibatches=8, max_updates_per_session=500))()

report = {"ckpt": CKPT, "devices": [str(d) for d in jax.devices()],
          "obs_dim": obs_dim, "action_dim": act, "embedding_size": EMB}
t0 = time.time()
try:
    ts = load_weights_only(CKPT, env, ctor, cfg, load_opt_state=False)
    leaves = len(jax.tree_util.tree_leaves(ts.params))
    def enc(p, pre=""):
        out = {}
        if hasattr(p, "items"):
            for k, v in p.items(): out.update(enc(v, f"{pre}/{k}"))
        elif "encoder" in pre and "kernel" in pre:
            out[pre] = tuple(int(x) for x in np.asarray(p).shape)
        return out
    kernel = enc(ts.params)
    network = ActorCriticTransformer(action_dim=act, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
    mem = jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mask = jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    obs = jnp.zeros((2, obs_dim))
    pi, v, mo = network.apply(ts.params, mem, obs, mask, method=network.model_forward_eval)
    finite = bool(np.all(np.isfinite(np.asarray(pi.logits))) and np.all(np.isfinite(np.asarray(v))))
    report["load"] = {"ok": True, "param_leaves": leaves, "encoder_kernel": kernel,
                      "forward_logits": tuple(pi.logits.shape), "forward_value": tuple(v.shape),
                      "forward_finite": finite, "load_time_s": round(time.time() - t0, 1),
                      "kernel_matches_8335_256": kernel.get("/transformer/encoder/kernel") == (8335, 256)}
except Exception as e:
    import traceback; traceback.print_exc()
    report["load"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
print(json.dumps(report, indent=2, default=str))
with open("/home/oseasy/experiments/p2_v1_20260722/ckpt_interface_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
