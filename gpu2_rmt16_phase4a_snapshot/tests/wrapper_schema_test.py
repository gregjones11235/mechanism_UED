import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np

SRC = "/home/oseasy/experiments/rmt16_replay_phase4a/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in [SRC, V7 + "/src", V7]:
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
EXPECTED_TERM = ["_term_done_steps", "_term_is_dead", "_term_player_health",
                 "_term_player_level", "_term_timestep"]


def build(probe_term):
    base_env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                       conditioning_type="embedding", embedding_size=int(table.shape[1]))
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), NUM_ENVS, 1, 16, jnp.array([1.0]), table,
        probe_term=probe_term)
    return env, env.default_params


def map_reason(done_steps, is_dead, is_success):
    # EXACT replica of the collector's no-inference rule (rmt_collect.py).
    c = []
    if done_steps:
        c.append("time_limit")
    if is_success:
        c.append("task_success")
    if is_dead:
        c.append("player_death")
    return (c[0] if len(c) == 1 else "unknown"), c


def keys_of(info):
    return sorted(info.keys())


def check_schema(info, label):
    ks = keys_of(info)
    term = sorted(k for k in ks if k.startswith("_term_"))
    assert term == EXPECTED_TERM, f"[{label}] _term_* set {term} != {EXPECTED_TERM}"
    for k in EXPECTED_TERM:
        v = np.asarray(info[k])
        assert v.shape == (NUM_ENVS,), f"[{label}] {k} shape {v.shape}"
        assert not (np.issubdtype(v.dtype, np.str_) or v.dtype == object), f"[{label}] {k} string dtype"
        assert bool(np.all(np.isfinite(v.astype(float)))), f"[{label}] {k} non-finite"
    return ks


print("=" * 70)
print("GATE 4: wrapper termination-schema test (CPU)")
print("=" * 70)

# ---------- Part A: probe_term OFF -> info identical to original (no _term_*) ----------
env_off, p_off = build(False)
rng = jax.random.PRNGKey(7)
rng, rk = jax.random.split(rng)
obs, st = env_off.reset(rk, p_off)
rng, sk = jax.random.split(rng)
_, _, _, _, info_off = env_off.step(sk, st, jnp.zeros(NUM_ENVS, jnp.int32), p_off)
assert not any(k.startswith("_term_") for k in info_off), "probe_term OFF must NOT emit _term_*"
print("[A] probe_term=OFF: n_keys=%d, _term_* present=%s  (bit-exact original info)"
      % (len(info_off), any(k.startswith('_term_') for k in info_off)))

# ---------- Part B: probe_term ON -> fixed 5 _term_* every step; key set invariant ----------
env_on, p_on = build(True)
rng = jax.random.PRNGKey(7)
rng, rk = jax.random.split(rng)
obs, st = env_on.reset(rk, p_on)
keysets = []
for i in range(3):
    rng, sk = jax.random.split(rng)
    obs, st, _, _, info = env_on.step(sk, st, jnp.zeros(NUM_ENVS, jnp.int32), p_on)
    keysets.append(tuple(check_schema(info, f"normal_step{i}")))
assert len(set(keysets)) == 1, f"[B] key set changed across normal steps: {set(keysets)}"
added = sorted(set(keysets[0]) - set(info_off.keys()))
removed = sorted(set(info_off.keys()) - set(keysets[0]))
assert added == EXPECTED_TERM, f"[B] added != expected: {added}"
assert removed == [], f"[B] keys removed: {removed}"
print("[B] probe_term=ON: 3 normal steps -> key set INVARIANT; exactly 5 _term_* added; "
      "shape (%d,) fixed; no strings" % NUM_ENVS)

# ---------- Part C+D: four scenarios forced on env0 (sole done -> reset target) ----------
def force_scenario(name, mutate):
    rng = jax.random.PRNGKey(99)
    rng, rk = jax.random.split(rng)
    obs, st = env_on.reset(rk, p_on)
    inner = st.env_state
    st2 = st.replace(env_state=mutate(inner)) if mutate else st   # wrap modified inner back into LogEnvState
    rng, sk = jax.random.split(rng)
    final_obs, log_st, rew, done, info = env_on.step(sk, st2, jnp.zeros(NUM_ENVS, jnp.int32), p_on)
    done_np = np.asarray(done).astype(bool)
    check_schema(info, name)               # schema identical even on a done step
    return inner, log_st.env_state, done_np, info


def get(inner, field):
    return np.asarray(getattr(inner, field))


results = {}

# --- normal (no forcing) ---
_, _, d_norm, info_norm = force_scenario("normal", None)
print("[C-normal] done_count=%d (env0 done=%s); _term_* schema identical to normal steps"
      % (int(d_norm.sum()), bool(d_norm[0])))
results["normal"] = dict(done_count=int(d_norm.sum()), env0_done=bool(d_norm[0]),
                         term_is_dead=bool(np.asarray(info_norm["_term_is_dead"])[0]),
                         term_done_steps=bool(np.asarray(info_norm["_term_done_steps"])[0]),
                         is_success=bool(np.asarray(info_norm["is_success"])[0]))

# --- time_limit: env0 timestep -> 4095, after one step -> 4096 = max ---
def mut_time(inner):
    ts = get(inner, "timestep").copy(); ts[0] = 4095
    return inner.replace(timestep=jnp.asarray(ts)) if hasattr(inner, "replace") else \
        inner.replace(**{"timestep": jnp.asarray(ts)})
pre_inner, ret_inner, d_tl, info_tl = force_scenario("time_limit", mut_time)
assert bool(d_tl[0]), "[time_limit] env0 not done"
assert int(d_tl[1:].sum()) == 0, "[time_limit] other envs done -> env0 not guaranteed reset target"
term_ts = int(np.asarray(info_tl["_term_timestep"])[0])
term_ds = bool(np.asarray(info_tl["_term_done_steps"])[0])
term_dead = bool(np.asarray(info_tl["_term_is_dead"])[0])
iss = bool(np.asarray(info_tl["is_success"])[0])
ret_ts = int(get(ret_inner, "timestep")[0])          # POST-reset returned state
reason, cands = map_reason(term_ds, term_dead, iss)
print("[D-time_limit] done[0]=True; info _term_timestep=%d (PRE-reset) vs returned_state timestep=%d "
      "(POST-reset); _term_done_steps=%s; provenance_differs=%s; done_reason=%s"
      % (term_ts, ret_ts, term_ds, term_ts != ret_ts, reason))
assert term_ds and term_ts >= 4096, "[time_limit] terminal timestep not at max"
assert term_ts != ret_ts, "[time_limit] PROVENANCE FAIL: info terminal == returned reset state"
assert reason == "time_limit", f"[time_limit] reason {reason} cands {cands}"
results["time_limit"] = dict(term_timestep=term_ts, returned_timestep=ret_ts,
                             term_done_steps=term_ds, term_is_dead=term_dead, is_success=iss,
                             done_reason=reason, provenance_proven=bool(term_ts != ret_ts))

# --- player_death: env0 player_health -> 0 ---
def mut_death(inner):
    hl = get(inner, "player_health").copy(); hl[0] = 0.0
    return inner.replace(player_health=jnp.asarray(hl))
_, ret_inner_d, d_d, info_d = force_scenario("player_death", mut_death)
assert bool(d_d[0]), "[player_death] env0 not done"
assert int(d_d[1:].sum()) == 0, "[player_death] other envs done"
term_hl = float(np.asarray(info_d["_term_player_health"])[0])
term_dead = bool(np.asarray(info_d["_term_is_dead"])[0])
term_ds = bool(np.asarray(info_d["_term_done_steps"])[0])
iss = bool(np.asarray(info_d["is_success"])[0])
ret_hl = float(get(ret_inner_d, "player_health")[0])     # POST-reset fresh health
reason, cands = map_reason(term_ds, term_dead, iss)
print("[D-player_death] done[0]=True; info _term_player_health=%.3f (PRE-reset) vs returned_state "
      "health=%.3f (POST-reset); _term_is_dead=%s; provenance_differs=%s; done_reason=%s"
      % (term_hl, ret_hl, term_dead, abs(term_hl - ret_hl) > 1e-6, reason))
assert term_dead and term_hl <= 0.0, "[player_death] terminal health not <=0"
assert abs(term_hl - ret_hl) > 1e-6, "[player_death] PROVENANCE FAIL: info health == returned reset health"
assert reason == "player_death", f"[player_death] reason {reason} cands {cands}"
results["player_death"] = dict(term_health=term_hl, returned_health=ret_hl, term_is_dead=term_dead,
                               term_done_steps=term_ds, is_success=iss, done_reason=reason,
                               provenance_proven=bool(abs(term_hl - ret_hl) > 1e-6))

# --- task_success: env0 achievements[KOBOLD] -> 1 (monotonic; survives the step) ---
def mut_success(inner):
    ach = get(inner, "achievements").copy(); ach[0, KOBOLD] = 1
    return inner.replace(achievements=jnp.asarray(ach))
_, ret_inner_s, d_s, info_s = force_scenario("task_success", mut_success)
assert bool(d_s[0]), "[task_success] env0 not done"
assert int(d_s[1:].sum()) == 0, "[task_success] other envs done"
term_dead = bool(np.asarray(info_s["_term_is_dead"])[0])
term_ds = bool(np.asarray(info_s["_term_done_steps"])[0])
iss = bool(np.asarray(info_s["is_success"])[0])
ret_ach = float(get(ret_inner_s, "achievements")[0, KOBOLD])   # POST-reset (should be cleared)
term_ach_info = float(np.asarray(info_s["Achievements/defeat_kobold"])[0])
reason, cands = map_reason(term_ds, term_dead, iss)
print("[D-task_success] done[0]=True; info is_success=%s & Achievements/defeat_kobold=%.1f (PRE-reset) "
      "vs returned_state achievements[kobold]=%.1f (POST-reset); done_reason=%s"
      % (iss, term_ach_info, ret_ach, reason))
assert iss, "[task_success] is_success not True"
assert ret_ach == 0.0, "[task_success] PROVENANCE FAIL: returned achievements not reset"
assert reason == "task_success", f"[task_success] reason {reason} cands {cands}"
results["task_success"] = dict(is_success=iss, term_ach_info=term_ach_info, returned_ach=ret_ach,
                               term_is_dead=term_dead, term_done_steps=term_ds, done_reason=reason,
                               provenance_proven=bool(ret_ach == 0.0 and iss))

print("=" * 70)
import json
print("GATE4_RESULTS=" + json.dumps(results, default=str))
print("GATE4_PASS=True")
