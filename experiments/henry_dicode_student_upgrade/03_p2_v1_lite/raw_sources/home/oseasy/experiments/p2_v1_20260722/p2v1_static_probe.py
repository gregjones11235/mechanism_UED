#!/usr/bin/env python3
"""P2-v1 STATIC runtime checks (directive section 五, items 4-9). CPU only, READ-ONLY.

Verifies against the REAL v7 runtime — does NOT assume the goal embedding lives in
the last 67 obs dims. Instead it localizes the goal slice empirically by resetting
the env under two different goal-embedding tables and diffing the observations.

Checks:
  5. dicode.__file__ (which codebase is actually imported)
  6. real observation schema: obs_dim, action_dim, base symbolic dim
  7. real goal-conditioning position: which obs dims carry the goal embedding,
     embedding_size, one-hot/embedding validity (diff-based, not assumed)
  9. memory API: memory/mask shapes the network expects
  (8. checkpoint interface is verified separately on GPU — load fails on CPU.)
No checkpoint is loaded, no file is written outside the report path.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys

_HENRY_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
              "Henry_work/code/dicode_v7fix58_armB/src")
sys.path.insert(0, _HENRY_SRC)

import jax
import jax.numpy as jnp
import numpy as np

report = {"platform": [str(d) for d in jax.devices()]}

import dicode
report["dicode_file"] = dicode.__file__
import dicode.network as dnet
report["dicode_network_file"] = dnet.__file__

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.network import ActorCriticTransformer
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from minicraftax.tasks.base_task import BaseTask

DK = int(Achievement.DEFEAT_KOBOLD.value)
report["DK_idx"] = DK

# ---- Stage4 task (floor-2 spawn scaffold) ----
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
ns = {}
exec(S4, ns)
S4Cls = ns["Env"]

ctor = EnvParams(max_timesteps=4096)

# goal-embedding table for DEFEAT_KOBOLD (size 67)
table_dk = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table_dk.shape[1])
report["embedding_size"] = EMB
report["dk_embedding_vector_sum"] = float(table_dk.sum())
report["dk_embedding_nonzero_idx"] = [int(i) for i in np.flatnonzero(np.asarray(table_dk[0]))]

# ---- env WITH embedding-67 conditioning ----
env67 = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                conditioning_type="embedding", embedding_size=EMB)
obs67 = int(env67.observation_space(ctor).shape[0])
act = int(env67.action_space(ctor).n)
report["obs_dim_embedding67"] = obs67
report["action_dim"] = act

# ---- env WITHOUT task conditioning (base symbolic dim) ----
try:
    env_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                       conditioning_type="embedding", embedding_size=0)
    obs_base = int(env_base.observation_space(ctor).shape[0])
except Exception as e:
    # fallback: probe base by constructing with no conditioning flag if supported
    obs_base = None
    report["base_obs_probe_error"] = f"{type(e).__name__}: {str(e)[:150]}"
report["base_symbolic_obs_dim"] = obs_base
if obs_base is not None:
    report["conditioning_block_dim"] = obs67 - obs_base

# ---- EMPIRICAL goal-slice localization: reset under two different goal tables ----
# Table A = DEFEAT_KOBOLD multi-hot; Table B = a DIFFERENT single achievement (index 0).
# The obs dims that differ between the two resets are exactly the goal-conditioning slice.
alt_vec = np.zeros((1, EMB), dtype=np.float32)
alt_vec[0, 0] = 1.0  # one-hot on achievement index 0 (NOT dk) -> different conditioning
table_alt = jnp.array(alt_vec)

from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper

def reset_obs(table):
    base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)
    w = DistributedMultiTaskOptimisticLogWrapper(base, jax.random.PRNGKey(0), 16, 1, 16,
                                                 jnp.array([1.0]), table)
    rng = jax.random.PRNGKey(123)
    rng, rr = jax.random.split(rng)
    obs, _ = w.reset(rr, ctor)
    return np.asarray(obs)  # [16, obs_dim]

obs_A = reset_obs(table_dk)
obs_B = reset_obs(table_alt)
report["reset_obs_shape_A"] = list(obs_A.shape)
# per-dim: does it differ across the two goal tables (averaged over envs)?
diff = np.abs(obs_A - obs_B).max(axis=0)  # [obs_dim]
goal_dims = [int(i) for i in np.flatnonzero(diff > 1e-8)]
report["empirical_goal_conditioning_dims"] = goal_dims
report["empirical_goal_slice_count"] = len(goal_dims)
if goal_dims:
    report["empirical_goal_slice_min"] = goal_dims[0]
    report["empirical_goal_slice_max"] = goal_dims[-1]
    report["empirical_goal_slice_is_last_block"] = (goal_dims[-1] == obs67 - 1 and
                                                    len(goal_dims) == EMB and
                                                    goal_dims[0] == obs67 - EMB)
    report["empirical_goal_slice_contiguous"] = (goal_dims == list(range(goal_dims[0], goal_dims[0] + len(goal_dims))))

# ---- memory API: shapes the network expects (from launcher cfg) ----
cfg_mem = dict(window_mem=128, num_layers=2, embed_size=256, num_heads=8)
report["memory_api"] = {
    "memory_shape": ["B", cfg_mem["window_mem"], cfg_mem["num_layers"], cfg_mem["embed_size"]],
    "mask_shape": ["B", cfg_mem["num_heads"], 1, cfg_mem["window_mem"] + 1],
    "obs_shape": ["B", obs67],
    **cfg_mem,
}

# ---- network encoder input dim (build network, init on obs67) ----
cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
           qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
           window_mem=128, window_grad=64)
network = ActorCriticTransformer(action_dim=act, activation=cfg["activation"],
    encoder_size=cfg["embed_size"], hidden_layers=cfg["hidden_layers"],
    num_heads=cfg["num_heads"], qkv_features=cfg["qkv_features"],
    num_layers=cfg["num_layers"], gating=cfg["gating"], gating_bias=cfg["gating_bias"])
B = 2
mem = jnp.zeros((B, cfg["window_mem"], cfg["num_layers"], cfg["embed_size"]))
mask = jnp.zeros((B, cfg["num_heads"], 1, cfg["window_mem"] + 1), dtype=jnp.bool_)
obs_in = jnp.zeros((B, obs67))


def _enc_kernel_shapes(params, pre=""):
    out = {}
    if hasattr(params, "items"):
        for k, v in params.items():
            out.update(_enc_kernel_shapes(v, f"{pre}/{k}"))
    else:
        if "encoder" in pre and "kernel" in pre:
            out[pre] = tuple(int(x) for x in np.asarray(params).shape)
    return out


try:
    init_vars = network.init(jax.random.PRNGKey(0), mem, obs_in, mask,
                             method=network.model_forward_eval)
    params = init_vars.get("params", init_vars) if hasattr(init_vars, "get") else init_vars
    enc = _enc_kernel_shapes(params)
    report["network_init"] = {"ok": True, "encoder_kernel": enc}
    # forward sanity
    pi, v, mo = network.apply(init_vars, mem, obs_in, mask, method=network.model_forward_eval)
    report["network_forward"] = f"OK logits={tuple(pi.logits.shape)} value={tuple(v.shape)} mem_out={tuple(mo.shape)}"
    report["param_leaves"] = len(jax.tree_util.tree_leaves(params))
except Exception as e:
    import traceback
    report["network_init"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    traceback.print_exc()

out = "/home/oseasy/experiments/p2_v1_20260722/static_check_report.json"
with open(out, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
print(f"\nSaved {out}", flush=True)
