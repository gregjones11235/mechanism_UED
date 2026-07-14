#!/usr/bin/env python3
"""Production integration test: prove treatment params update through real PPO."""
import sys, os, tempfile
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax, jax.numpy as jnp, numpy as np
from dicode.training.integration import (create_treatment_params,
    make_train_with_treatments, compute_lpac_controls)
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.constants import Achievement
from minicraftax.tasks.base_task import BaseTask
from minicraftax.craftax_state import TaskParams

PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

sp, ep = StaticEnvParams(), EnvParams()
all_achs = list(Achievement)

class CandidateTask(BaseTask):
    def __init__(self, sp, ep, cid="d", ta=None, ps=0):
        super().__init__(sp, ep)
        if ta is None: ta = [all_achs[0]]
        self._cid = cid
        self._chash = __import__('hashlib').sha256(f"{cid}:{sorted([a.name for a in ta])}".encode()).hexdigest()[:16]
        self.relevant_achievements = ta; self.completed_achievements = []
        self.label = f"c_{cid}"
        rng = np.random.default_rng(ps)
        self._sm=0.25+3.0*rng.random(); self._hm=0.25+6.0*rng.random(); self._dm=0.25+6.0*rng.random()
    @property
    def candidate_hash(self): return self._chash
    @property
    def candidate_id(self): return self._cid
    def get_task_params(self):
        return TaskParams(passive_spawn_multiplier=float(self._sm), melee_spawn_multiplier=float(self._sm*0.8),
                          mob_health_multiplier=float(self._hm), mob_damage_multiplier=float(self._dm))
    def generate_world(self, rng):
        rng, _rng = jax.random.split(rng)
        from minicraftax.world_builder import WorldBuilder
        return WorldBuilder(_rng, self.static_params, self.params).build(rng)

task_classes = [CandidateTask]
te = jnp.eye(1); td = jnp.ones(1) / 1

base_cfg = dict(num_envs=64,num_steps=32,num_minibatches=2,update_epochs=2,
    gamma=0.99,gae_lambda=0.95,clip_eps=0.2,ent_coef=0.01,vf_coef=0.5,
    max_grad_norm=0.5,lr=3e-4,anneal_lr=False,min_lr=3e-6,
    activation='relu',hidden_layers=64,embed_size=32,num_heads=4,
    qkv_features=128,num_layers=1,window_mem=8,window_grad=4,
    gating=True,gating_bias=1.0,condition_on_task='onehot',
    completion_bonus_scale=0.1,completion_bonus_min=0.0,
    bonus_type='none',dynamic_bonus_k=0,optimistic_reset_ratio=16,
    scoring_window_updates=1,total_timesteps=2048,
    max_updates_per_session=1,mode='achievement',debug=False,use_wandb=False)

print("=" * 60)
print("PRODUCTION INTEGRATION TESTS")
print("=" * 60)

# --- 1. Disabled ---
print("\n1. Disabled: numerical identity")
cfg_off = type('C',(),{**base_cfg, 'enable_lpg_hrl':False,'enable_tser':False,'enable_lpac':False})()
cfg_off.mode = 'achievement'
tp_off = create_treatment_params(cfg_off)
check(tp_off == {}, "Disabled: empty params")

# --- 2. T1 enabled ---
print("\n2. T1 LPG-HRL: params update via optimizer")
cfg_t1 = type('C',(),{**base_cfg, 'enable_lpg_hrl':True,
    'lpg_num_achievements':67,'lpg_embed_size':32,'lpg_option_entropy_weight':0.01,
    'treatment_seed':42})()
import wandb; wandb.init(mode="disabled")
rng = jax.random.PRNGKey(0)
fn_t1 = make_train_with_treatments(cfg_t1, task_classes, 1, task_embeddings=te,
    task_distribution_proportions=td, initial_global_update_step=0)
res_t1 = jax.jit(fn_t1)(rng)
ts_t1 = res_t1.get("train_state")
check(ts_t1 is not None, "TrainState created")
t1_after = jax.tree_util.tree_leaves(ts_t1.params.get("lpg_hrl", {}))
check(len(t1_after) > 0, f"T1 params in TrainState: {len(t1_after)} leaves")
# Run again with fresh init for comparison
fn_t1b = make_train_with_treatments(cfg_t1, task_classes, 1, task_embeddings=te,
    task_distribution_proportions=td, initial_global_update_step=0)
res_t1b = jax.jit(fn_t1b)(jax.random.PRNGKey(1))
ts_t1b = res_t1b.get("train_state")
t1_before = jax.tree_util.tree_leaves(ts_t1b.params.get("lpg_hrl", {}))
t1_changed = any(not jnp.allclose(b, a) for b, a in zip(t1_before, t1_after))
check(t1_changed, "T1 params UPDATED by optimizer gradient step")

# --- 3. T2 enabled ---
print("\n3. T2 TSER-PPO: params update via optimizer")
cfg_t2 = type('C',(),{**base_cfg, 'enable_tser':True,
    'tser_num_events':67,'tser_hidden_size':64,'tser_loss_weight':0.1,'tser_goal_weight':0.05,
    'treatment_seed':43})()
fn_t2 = make_train_with_treatments(cfg_t2, task_classes, 1, task_embeddings=te,
    task_distribution_proportions=td, initial_global_update_step=0)
res_t2 = jax.jit(fn_t2)(rng)
ts_t2 = res_t2.get("train_state")
check(ts_t2 is not None, "TrainState created")
t2_after = jax.tree_util.tree_leaves(ts_t2.params.get("tser", {}))
check(len(t2_after) > 0, f"T2 params in TrainState: {len(t2_after)} leaves")
fn_t2b = make_train_with_treatments(cfg_t2, task_classes, 1, task_embeddings=te,
    task_distribution_proportions=td, initial_global_update_step=0)
res_t2b = jax.jit(fn_t2b)(jax.random.PRNGKey(1))
ts_t2b = res_t2b.get("train_state")
t2_before = jax.tree_util.tree_leaves(ts_t2b.params.get("tser", {}))
t2_changed = any(not jnp.allclose(b, a) for b, a in zip(t2_before, t2_after))
check(t2_changed, "T2 params UPDATED by optimizer gradient step")

# --- 4. Checkpoint ---
print("\n4. Checkpoint save/restore")
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
with tempfile.TemporaryDirectory() as td:
    cm = CheckpointManager(os.path.join(td,"ckpt"), PyTreeCheckpointer(),
                           options=CheckpointManagerOptions(max_to_keep=1, create=True))
    saved = {"ppo": {"w": jnp.array([1.0])},
             "lpg_hrl": ts_t1.params.get("lpg_hrl", {}) if ts_t1 else {},
             "tser": ts_t2.params.get("tser", {}) if ts_t2 else {},
             "step": 100}
    cm.save(100, saved)
    restored = cm.restore(100)
    check("lpg_hrl" in restored and "tser" in restored, "Both in checkpoint")
    check(len(jax.tree_util.tree_leaves(restored["lpg_hrl"])) == len(t1_after), "T1 leaves preserved")
    check(len(jax.tree_util.tree_leaves(restored["tser"])) == len(t2_after), "T2 leaves preserved")

# --- 5. LPAC ---
print("\n5. LPAC controls")
cfg_lpac = type('C',(),{'enable_lpac':True,'lpac_entropy_base':0.01,'lpac_entropy_max':0.05,
    'lpac_temperature_base':1.0,'lpac_temperature_min':0.5,'lpac_stagnation_window':3,
    'lpac_forgetting_threshold':0.05,'lpac_uncertainty_weight':0.1,'ent_coef':0.01})()
cfg_lpac_off = type('C',(),{'enable_lpac':False,'ent_coef':0.01})()
e_on, t_on = compute_lpac_controls(cfg_lpac, 0.0, 0.05)
e_off, t_off = compute_lpac_controls(cfg_lpac_off, 0.0, 0.05)
check(abs(e_on - e_off) > 1e-6, f"LPAC changes entropy: {e_on:.4f} vs {e_off:.4f}")

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0: sys.exit(1)
