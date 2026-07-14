#!/usr/bin/env python3
"""Minimal end-to-end test: prove params update through make_train_with_treatments."""
import sys, os
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax, jax.numpy as jnp, numpy as np
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.constants import Achievement
from minicraftax.tasks.base_task import BaseTask
from minicraftax.craftax_state import TaskParams
from dicode.training.integration import make_train_with_treatments

sp,ep=StaticEnvParams(),EnvParams()
achs=list(Achievement)
class CT(BaseTask):
    def __init__(self,sp,ep,cid="d",ta=None,ps=0):
        super().__init__(sp,ep)
        if ta is None: ta=[achs[0]]
        self._cid=cid;self.relevant_achievements=ta;self.completed_achievements=[];self.label="ct"
        rng=np.random.default_rng(ps)
        self._sm=0.25+3.0*rng.random();self._hm=0.25+6.0*rng.random();self._dm=0.25+6.0*rng.random()
    @property
    def candidate_hash(self):
        import hashlib
        return hashlib.sha256(f"{self._cid}:{sorted([a.name for a in self.relevant_achievements])}".encode()).hexdigest()[:16]
    @property
    def candidate_id(self): return self._cid
    def get_task_params(self):
        return TaskParams(passive_spawn_multiplier=float(self._sm),melee_spawn_multiplier=float(self._sm*0.8),mob_health_multiplier=float(self._hm),mob_damage_multiplier=float(self._dm))
    def generate_world(self,rng):
        rng,_rng=jax.random.split(rng)
        from minicraftax.world_builder import WorldBuilder
        return WorldBuilder(_rng,self.static_params,self.params).build(rng)

cfg=type("C",(),{"num_envs":64,"num_steps":32,"num_minibatches":2,"update_epochs":2,
    "gamma":0.99,"gae_lambda":0.95,"clip_eps":0.2,"ent_coef":0.01,"vf_coef":0.5,
    "max_grad_norm":0.5,"lr":3e-4,"anneal_lr":False,"min_lr":3e-6,
    "activation":"relu","hidden_layers":64,"embed_size":32,"num_heads":4,
    "qkv_features":128,"num_layers":1,"window_mem":8,"window_grad":4,
    "gating":True,"gating_bias":1.0,"condition_on_task":"onehot",
    "completion_bonus_scale":0.1,"completion_bonus_min":0.0,
    "bonus_type":"none","dynamic_bonus_k":0,"optimistic_reset_ratio":16,
    "scoring_window_updates":1,"total_timesteps":2048,
    "max_updates_per_session":1,"mode":"achievement","debug":False,"use_wandb":False,
    "enable_lpg_hrl":True,"enable_tser":True,"lpg_num_achievements":67,"lpg_embed_size":32,"tser_num_events":67,"tser_hidden_size":64,"tser_loss_weight":0.1,"tser_goal_weight":0.05,
    "lpg_option_entropy_weight":0.01,"treatment_seed":42})()
cfg.mode="achievement"
te=jnp.eye(1);td=jnp.ones(1)
import wandb;wandb.init(mode="disabled")
fn=make_train_with_treatments(cfg,[CT],1,task_embeddings=te,task_distribution_proportions=td,initial_global_update_step=0)
r=jax.jit(fn)(jax.random.PRNGKey(0))
ts=r.get("train_state")
lp=ts.params.get("lpg_hrl",{})
print(f"LPG leaves: {len(jax.tree_util.tree_leaves(lp))}")
key="lpg_hrl"
before=jax.tree_util.tree_leaves(lp)
fn2=make_train_with_treatments(cfg,[CT],1,task_embeddings=te,task_distribution_proportions=td,initial_global_update_step=0)
r2=jax.jit(fn2)(jax.random.PRNGKey(1))
ts2=r2.get("train_state")
after=jax.tree_util.tree_leaves(ts2.params.get("lpg_hrl",{}))
changed=any(not jnp.allclose(b,a) for b,a in zip(before,after))
print(f"Params changed: {changed}")
