#!/usr/bin/env python3
"""Verify the session175 / healthy-v7 checkpoint loads with the BIG network
config (embed 256) + embedding-67 Stage4 env, and probe the natural FULL env
(default world-gen spawn floor + obs_dim). GPU. No rollout, no training.

session175 ckpt = base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500 (per directive
section 八 P2-v1 smoke start point).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PROBE_GPU", "0")
import sys, json
import jax, jax.numpy as jnp
import numpy as np

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
sys.path.insert(0, V7)
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from minicraftax.tasks.base_task import BaseTask
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.network import ActorCriticTransformer

CKPT = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"


def big_cfg():
    d = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
             qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
             window_mem=128, window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6,
             max_grad_norm=1.0, total_timesteps=2005401600, num_envs=1024, num_steps=128,
             update_epochs=4, num_minibatches=8, max_updates_per_session=500)
    return type("C", (), d)()


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

# FULL natural task: default world-gen, condition on DK achievement (no scaffold)
FULL = '''
import jax
from craftax.craftax.constants import Achievement
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="FULL_DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams()
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        return b.build(rng)
'''
nf = {}; exec(FULL, nf); FullCls = nf["Env"]

ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
report = {"embedding_size": EMB, "DK_idx": int(Achievement.DEFEAT_KOBOLD.value)}

# ---- Stage4 env ----
s4_env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                 conditioning_type="embedding", embedding_size=EMB)
s4_obs = int(s4_env.observation_space(ctor).shape[0])
s4_act = int(s4_env.action_space(ctor).n)
report["stage4_obs_dim"] = s4_obs
report["stage4_action_dim"] = s4_act
print(f"Stage4 env: obs_dim={s4_obs} action_dim={s4_act}", flush=True)

# ---- FULL env ----
full_env = MultiTaskMiniCraftaxEnv([FullCls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)
full_obs = int(full_env.observation_space(ctor).shape[0])
report["full_obs_dim"] = full_obs
print(f"FULL env: obs_dim={full_obs}", flush=True)

# ---- FULL natural spawn floor (reset introspection) ----
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
fwrap = DistributedMultiTaskOptimisticLogWrapper(full_env, jax.random.PRNGKey(0), 64, 1, 16, jnp.array([1.0]), table)
rng = jax.random.PRNGKey(42); rng, rr = jax.random.split(rng)
fobs, fstate = fwrap.reset(rr, ctor)
fes = fstate.env_state
fpl = np.asarray(fes.player_level)
fach = np.asarray(fes.achievements)
report["full_at_reset"] = {
    "player_level_counts": {int(k): int(v) for k, v in zip(*np.unique(fpl, return_counts=True))},
    "DK_preset_count": int(fach[:, int(Achievement.DEFEAT_KOBOLD.value)].sum()),
}
print(f"FULL reset player_level counts: {report['full_at_reset']['player_level_counts']}", flush=True)
print(f"FULL reset DK preset count: {report['full_at_reset']['DK_preset_count']}", flush=True)

# ---- load session175 checkpoint with BIG config + Stage4 env ----
cfg = big_cfg()
try:
    ts = load_weights_only(CKPT, s4_env, ctor, cfg, load_opt_state=False)
    nleaves = len(jax.tree_util.tree_leaves(ts.params))
    # encoder kernel shape
    def find_enc(p, pre=""):
        out = {}
        if hasattr(p, "items"):
            for k, v in p.items():
                out.update(find_enc(v, f"{pre}/{k}"))
        else:
            if "encoder" in pre and "kernel" in pre:
                out[pre] = tuple(int(x) for x in np.asarray(p).shape)
        return out
    enc = find_enc(ts.params)
    report["session175_load"] = {"ok": True, "param_leaves": nleaves, "encoder_kernel": enc}
    print(f"session175 load OK: leaves={nleaves} encoder_kernel={enc}", flush=True)
    # forward sanity on Stage4 obs
    network = ActorCriticTransformer(action_dim=s4_act, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
    B = 2
    mem = jnp.zeros((B, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mask = jnp.zeros((B, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    obs = jnp.zeros((B, s4_obs))
    psub = ts.params.get("params", ts.params) if hasattr(ts.params, "get") else ts.params
    for vname, vp in [("ts.params", ts.params), ("ts.params['params']", psub)]:
        try:
            pi, v, mo = network.apply(vp, mem, obs, mask, method=network.model_forward_eval)
            report.setdefault("session175_forward", {})[vname] = f"OK logits={pi.logits.shape} v={v.shape}"
            print(f"forward[{vname}]: OK logits={pi.logits.shape}", flush=True)
        except Exception as e:
            report.setdefault("session175_forward", {})[vname] = f"ERR {type(e).__name__}: {str(e)[:200]}"
            print(f"forward[{vname}]: ERR {type(e).__name__}: {str(e)[:200]}", flush=True)
except Exception as e:
    import traceback
    report["session175_load"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    print(f"session175 load ERR: {type(e).__name__}: {str(e)[:300]}", flush=True)
    traceback.print_exc()

out = "/home/oseasy/experiments/d052_unified_eval_20260722/evaluator/s175_verify_report.json"
with open(out, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nSaved {out}", flush=True)
print("\nFULL:", json.dumps(report, indent=2, default=str), flush=True)
