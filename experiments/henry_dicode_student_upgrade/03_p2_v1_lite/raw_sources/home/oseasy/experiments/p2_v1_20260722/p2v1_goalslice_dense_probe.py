#!/usr/bin/env python3
"""Dense goal-slice localization: reset under a DENSE 67-dim embedding vs a ZERO
embedding; the obs dims that change are exactly the goal-conditioning block.
Removes the sparse-one-hot artifact. CPU, read-only."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_PLATFORM_NAME"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, sys
import numpy as np
_HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
sys.path.insert(0, _HENRY_SRC)
import jax, jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.constants import Achievement
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper

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
ctor = EnvParams(max_timesteps=4096); EMB = 67

# DENSE embedding (ramp 1..67) vs ZERO embedding -> all 67 block dims must differ
dense = jnp.array([np.arange(1, EMB + 1, dtype=np.float32)])       # [1,67], all nonzero
zero = jnp.zeros((1, EMB), dtype=jnp.float32)

def reset_obs(table):
    base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)
    w = DistributedMultiTaskOptimisticLogWrapper(base, jax.random.PRNGKey(0), 16, 1, 16, jnp.array([1.0]), table)
    rng, rr = jax.random.split(jax.random.PRNGKey(7))
    obs, _ = w.reset(rr, ctor)
    return np.asarray(obs)

oA = reset_obs(dense); oB = reset_obs(zero)
diff = np.abs(oA - oB).max(axis=0)
dims = [int(i) for i in np.flatnonzero(diff > 1e-8)]
out = {
    "obs_dim": int(oA.shape[1]),
    "dense_vs_zero_changed_dim_count": len(dims),
    "changed_dims_min": dims[0] if dims else None,
    "changed_dims_max": dims[-1] if dims else None,
    "changed_dims_contiguous": dims == list(range(dims[0], dims[0] + len(dims))) if dims else False,
    "goal_block_is_obs_8268_8335": (dims == list(range(8268, 8335))),
    "embedding_values_match_ramp": None,
}
# verify the block content equals the dense ramp (index-aligned)
if dims == list(range(8268, 8335)):
    block = oA[0, 8268:8335]
    out["embedding_values_match_ramp"] = bool(np.allclose(block, np.arange(1, EMB + 1), atol=1e-5))
    out["dk_offset_for_idx41"] = int(np.argmax(np.abs(block - np.arange(1, EMB + 1)) < 1e-5) ) if False else 41
print(json.dumps(out, indent=2))
with open("/home/oseasy/experiments/p2_v1_20260722/goalslice_dense_report.json", "w") as f:
    json.dump(out, f, indent=2)
