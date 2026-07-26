#!/usr/bin/env python3
"""DECISIVE probe: reconcile the D052 checkpoint's expected obs dim (encoder
kernel input) with the env obs dim built the runner's exact way.

Replicates launch_d052_pure_dynamic_enhanced.py::make_task_classes and the
make_train env construction (MultiTaskMiniCraftaxEnv + onehot condition_on_task
+ conditioning_type='embedding' + embedding_size=len(task_classes) +
completion_bonus_* + bonus_type + dynamic_bonus_k).  Measures obs_dim for a
range of task counts N, and reads the checkpoint encoder kernel input dim K.
Then num_tasks = K - base_symbolic where base_symbolic = obs(N=1) - 1.

GPU (checkpoint restore needs GPU sharding). No rollout, no training.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PROBE_GPU", "1")
import sys, json, hashlib
import jax, jax.numpy as jnp
import numpy as np

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
sys.path.insert(0, V7)
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.utils.general.train_state_utils import load_weights_only

_sp, _ep = StaticEnvParams(), EnvParams()
_all_a = list(Achievement)


def sha(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


# ---- faithful copy of runner make_task_classes ----
def make_task_classes(cands):
    classes = []
    for spc in cands:
        cid = spc["task_id"]
        ch = spc.get("chash", sha(cid))
        tp = spc.get("task_params", {}) if isinstance(spc.get("task_params"), dict) else json.loads(spc.get("task_params", "{}"))
        ta_names = spc.get("target_achievements", ["COLLECT_WOOD"])[:4]
        ta = [_all_a[hash(f"{cid}_{a}") % len(_all_a)] for a in ta_names]
        _cid, _ch, _tp, _ta = cid, ch, tp, ta

        class _CT(BaseTask):
            def __init__(self, sp=_sp, ep=_ep, __cid=_cid, __ch=_ch, __tp=_tp, __ta=_ta):
                super().__init__(sp, ep)
                self._cid, self._ch = __cid, __ch
                self.relevant_achievements = list(__ta)
                self.completed_achievements = []
                self.label = f"d052_{__cid}"
                self._sm = float(__tp.get("passive_spawn_multiplier", 1.0))
                self._mm = float(__tp.get("melee_spawn_multiplier", self._sm * 0.8))
                self._hm = float(__tp.get("mob_health_multiplier", 2.0))
                self._dm = float(__tp.get("mob_damage_multiplier", 1.0))

            @property
            def candidate_hash(self): return self._ch

            @property
            def candidate_id(self): return self._cid

            def get_task_params(self):
                from minicraftax.craftax_state import TaskParams
                return TaskParams(passive_spawn_multiplier=self._sm, melee_spawn_multiplier=self._mm,
                                  mob_health_multiplier=self._hm, mob_damage_multiplier=self._dm)

            def generate_world(self, rng):
                rng, _r = jax.random.split(rng)
                from minicraftax.world_builder import WorldBuilder
                return WorldBuilder(_r, self.static_params, self.params).build(rng)

        _CT.__name__ = f"C_{_cid}"; _CT.__qualname__ = f"C_{_cid}"
        classes.append(_CT)
    return classes


def synth_cands(n):
    return [{"task_id": f"synth_{i:03d}", "chash": sha(f"synth_{i:03d}"),
             "task_params": {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 0.8,
                             "mob_health_multiplier": 2.0, "mob_damage_multiplier": 1.0},
             "target_achievements": ["COLLECT_WOOD", "DEFEAT_ZOMBIE"]} for i in range(n)]


def build_env_obsdim(n):
    cands = synth_cands(n)
    task_classes = make_task_classes(cands)
    te = jnp.eye(len(task_classes))
    emb = int(te.shape[1])
    base_env = MultiTaskMiniCraftaxEnv(
        task_classes, StaticEnvParams(), EnvParams(max_timesteps=4096), "onehot",
        conditioning_type="embedding", embedding_size=emb,
        completion_bonus_scale=0.1, completion_bonus_min=0.0,
        bonus_type="none", dynamic_bonus_k=0)
    od = int(base_env.observation_space(EnvParams(max_timesteps=4096)).shape[0])
    ad = int(base_env.action_space(EnvParams(max_timesteps=4096)).n)
    return od, ad, emb


report = {"obs_dim_by_N": {}, "action_dim": None}
for n in [1, 8, 32, 102]:
    try:
        od, ad, emb = build_env_obsdim(n)
        report["obs_dim_by_N"][str(n)] = {"obs_dim": od, "embedding_size": emb}
        report["action_dim"] = ad
        print(f"N={n:3d}: obs_dim={od}  embedding_size={emb}  action_dim={ad}", flush=True)
    except Exception as e:
        report["obs_dim_by_N"][str(n)] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"N={n:3d}: ERR {type(e).__name__}: {str(e)[:200]}", flush=True)

# base_symbolic from N=1 (onehot adds exactly 1 dim)
od1 = report["obs_dim_by_N"].get("1", {}).get("obs_dim")
if od1:
    report["base_symbolic_inferred"] = od1 - 1
    print(f"\nbase_symbolic (inferred from N=1) = {od1 - 1}", flush=True)

# ---- read checkpoint encoder kernel input dim ----
CKPT = "/home/oseasy/experiments/mechanism_UED_continuation_20260715/shared_r0/runs/d052_dynamic/original_x_original/seed0_1784441252/checkpoints/98304"


def cfg_obj():
    base = dict(num_envs=128, num_steps=32, num_minibatches=2, update_epochs=2,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, lr=3e-4, anneal_lr=False, min_lr=3e-6, activation="relu",
        hidden_layers=64, embed_size=32, num_heads=4, qkv_features=128, num_layers=1,
        window_mem=8, window_grad=4, gating=True, gating_bias=1.0,
        total_timesteps=24576, max_updates_per_session=6)
    return type("C", (), base)()


# build a dummy env for the loader template (obs dim does not matter; loader
# has raw-extraction fallback that recovers saved params regardless).
dummy_tc = make_task_classes(synth_cands(1))
dummy_env = MultiTaskMiniCraftaxEnv(dummy_tc, StaticEnvParams(), EnvParams(max_timesteps=4096), "onehot",
    conditioning_type="embedding", embedding_size=1,
    completion_bonus_scale=0.1, completion_bonus_min=0.0, bonus_type="none", dynamic_bonus_k=0)
ts = load_weights_only(CKPT, dummy_env, EnvParams(max_timesteps=4096), cfg_obj(), load_opt_state=False)


def leaf_shapes(p, prefix=""):
    out = {}
    if hasattr(p, "items"):
        for k, v in p.items():
            out.update(leaf_shapes(v, f"{prefix}/{k}"))
    else:
        try:
            out[prefix] = tuple(int(x) for x in np.asarray(p).shape)
        except Exception:
            out[prefix] = str(type(p))
    return out


shapes = leaf_shapes(ts.params)
# find encoder kernel
enc = {k: v for k, v in shapes.items() if "encoder" in k and "kernel" in k}
report["encoder_kernel_shapes"] = enc
report["total_leaves"] = len(shapes)
print(f"\nencoder kernel shapes: {json.dumps(enc, default=str)}", flush=True)
# infer num_tasks from K
for k, v in enc.items():
    if isinstance(v, (list, tuple)) and len(v) == 2:
        K = v[0]
        report["encoder_input_dim_K"] = K
        if od1:
            report["num_tasks_inferred"] = K - (od1 - 1)
        print(f"encoder input dim K = {K}; num_tasks_inferred = {K - (od1 - 1) if od1 else '?'}", flush=True)
        break

# dump a few representative leaf shapes
rep = {k: v for k, v in list(shapes.items())[:8]}
report["sample_leaf_shapes"] = rep
print(f"sample leaves: {json.dumps(rep, default=str)}", flush=True)

out = "/home/oseasy/experiments/d052_unified_eval_20260722/evaluator/obsdim_probe_report.json"
with open(out, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nSaved {out}", flush=True)
print("\nFULL REPORT:", json.dumps(report, indent=2, default=str), flush=True)
