import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in [V7 + "/src", V7]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax, jax.numpy as jnp
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

KOBOLD = int(Achievement.DEFEAT_KOBOLD.value)
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
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
NUM_ENVS = 16

base_env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=int(table.shape[1]))
env = DistributedMultiTaskOptimisticLogWrapper(
    base_env, jax.random.PRNGKey(0), NUM_ENVS, 1, 16, jnp.array([1.0]), table, probe_term=True)
ep = env.default_params

print("=== candidate max_timesteps sources ===")
print("ctor.max_timesteps          =", ctor.max_timesteps)
print("env.default_params.max_ts   =", getattr(ep, "max_timesteps", "<none>"))
print("base_env default_params.max =", getattr(getattr(base_env, "default_params", None), "max_timesteps", "<none>"))
print("base_env has .tasks         =", hasattr(base_env, "tasks"))
if hasattr(base_env, "tasks") and base_env.tasks:
    t0 = base_env.tasks[0]
    print("base.tasks[0].params.max_ts =", getattr(getattr(t0, "params", None), "max_timesteps", "<none>"))
    print("base.tasks[0] static .max   =", getattr(getattr(t0, "static_params", None), "max_timesteps", "<none>"))
print("base_env attrs w/ 'param'   =", [a for a in dir(base_env) if "param" in a.lower()])
# the wrapper's self._env is the base env (the object actually used for reset/step)
print("wrapper._env is base_env    =", env._env is base_env)

# env_state fields
rng, rk = jax.random.split(jax.random.PRNGKey(1))
obs, st = env.reset(rk, ep)
inner = st.env_state
fields = [a for a in dir(inner) if not a.startswith("_")]
print("=== env_state ===")
print("env_state has max_timesteps =", "max_timesteps" in fields)
print("env_state has timestep      =", "timestep" in fields)
print("env_state.timestep[0]       =", float(np.asarray(inner.timestep)[0]))
if "max_timesteps" in fields:
    print("env_state.max_timesteps[0]  =", float(np.asarray(inner.max_timesteps)[0]))

# --- empirically determine the env's TRUE time-limit threshold ---
# set env0 timestep to 4095, step once -> if done fires, threshold == 4096 (done_steps).
print("=== empirical time-limit threshold ===")
ts = np.asarray(inner.timestep).copy(); ts[0] = 4095
st2 = st.replace(env_state=inner.replace(timestep=jnp.asarray(ts)))
rng, sk = jax.random.split(rng)
_, log_st, rew, done, info = env.step(sk, st2, jnp.zeros(NUM_ENVS, jnp.int32), ep)
done_np = np.asarray(done).astype(bool)
term_ts = int(np.asarray(info["_term_timestep"])[0])
term_ds = bool(np.asarray(info["_term_done_steps"])[0])
ret_ts = int(np.asarray(log_st.env_state.timestep)[0])
print("after forcing env0 4095->step: done[0]=%s, info _term_timestep=%d, _term_done_steps=%s, returned_ts=%d"
      % (bool(done_np[0]), term_ts, term_ds, ret_ts))
# what is_success / is_dead say at that step (to confirm done came from done_steps not death)
print("is_success[0]=%s  _term_is_dead[0]=%s" % (
    bool(np.asarray(info["is_success"])[0]), bool(np.asarray(info["_term_is_dead"])[0])))
print("CONCLUSION: env true done at timestep==%d -> threshold is %d; "
      "_term_done_steps used %s (wrapper step `params.max_timesteps`)" % (
          term_ts, 4096 if bool(done_np[0]) else ">4095",
          getattr(ep, "max_timesteps", "<none>")))
