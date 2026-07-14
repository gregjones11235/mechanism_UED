#!/usr/bin/env python3
"""GPU0 R0 Preflight: T2 TSER-PPO, 16384 env steps, seed 0."""
import sys, os, json, time, hashlib, subprocess
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, "/root/experiments/dicode-dspro-r0/src")
RUNTIME_COMMIT = subprocess.check_output(
    ["git", "-C", "/root/experiments/dicode-dspro-r0", "rev-parse", "HEAD"],
    text=True).strip()

import jax, jax.numpy as jnp, numpy as np
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.constants import Achievement
from minicraftax.tasks.base_task import BaseTask
from minicraftax.craftax_state import TaskParams
from dicode.training.integration import make_train_with_treatments

OUT = f"/root/experiments/dicode_runs/dspro/r0_preflight/t2_tser_ppo_{int(time.time())}"
os.makedirs(OUT, exist_ok=False)

devs = jax.devices()
gpus = [d for d in devs if d.platform == "gpu"]
assert len(gpus) == 1, f"Expected 1 GPU, got {len(gpus)}"

sp, ep = StaticEnvParams(), EnvParams()
achs = list(Achievement)
class CT(BaseTask):
    def __init__(self, sp, ep, cid="d", ta=None, ps=0):
        super().__init__(sp, ep)
        if ta is None: ta = [achs[0]]
        self._cid = cid; self.relevant_achievements = ta; self.completed_achievements = []
        self.label = "ct"
        rng = np.random.default_rng(ps)
        self._sm = 0.25 + 3.0 * rng.random(); self._hm = 0.25 + 6.0 * rng.random()
        self._dm = 0.25 + 6.0 * rng.random()
    @property
    def candidate_hash(self):
        return hashlib.sha256(f"{self._cid}:{sorted([a.name for a in self.relevant_achievements])}".encode()).hexdigest()[:16]
    @property
    def candidate_id(self): return self._cid
    def get_task_params(self):
        return TaskParams(passive_spawn_multiplier=float(self._sm), melee_spawn_multiplier=float(self._sm*0.8),
                          mob_health_multiplier=float(self._hm), mob_damage_multiplier=float(self._dm))
    def generate_world(self, rng):
        rng, _rng = jax.random.split(rng)
        from minicraftax.world_builder import WorldBuilder
        return WorldBuilder(_rng, self.static_params, self.params).build(rng)

cfg = type("C",(),{"num_envs":256,"num_steps":32,"num_minibatches":2,"update_epochs":2,
    "gamma":0.99,"gae_lambda":0.95,"clip_eps":0.2,"ent_coef":0.01,"vf_coef":0.5,
    "max_grad_norm":0.5,"lr":3e-4,"anneal_lr":False,"min_lr":3e-6,
    "activation":"relu","hidden_layers":64,"embed_size":32,"num_heads":4,
    "qkv_features":128,"num_layers":1,"window_mem":8,"window_grad":4,
    "gating":True,"gating_bias":1.0,"condition_on_task":"onehot",
    "completion_bonus_scale":0.1,"completion_bonus_min":0.0,
    "bonus_type":"none","dynamic_bonus_k":0,"optimistic_reset_ratio":16,
    "scoring_window_updates":1,"total_timesteps":16384,
    "max_updates_per_session":2,"mode":"achievement","debug":False,"use_wandb":False,
    "enable_tser":True,"tser_num_events":67,"tser_hidden_size":64,
    "tser_loss_weight":0.1,"tser_goal_weight":0.05,"treatment_seed":43})()
cfg.mode = "achievement"

te = jnp.eye(1); td = jnp.ones(1) / 1
import wandb; wandb.init(mode="disabled")

t0 = time.time()
fn = make_train_with_treatments(cfg, [CT], 2, task_embeddings=te, task_distribution_proportions=td, initial_global_update_step=0)
res = jax.jit(fn)(jax.random.PRNGKey(0))
elapsed = time.time() - t0

ts = res.get("train_state")
esteps = int(res.get("metrics", {}).get("num_env_steps_done", 0))
print(f"T2 TSER-PPO: {esteps} env steps in {elapsed:.1f}s")

tser = ts.params.get("tser", {})
tl = jax.tree_util.tree_leaves(tser)
tl_count = len(tl)
print(f"T2 leaves: {tl_count}")
assert tl_count > 0, "No T2 params"
assert esteps == 16384, f"Expected 16384, got {esteps}"

from dicode.training.tser_ppo import TSERWrapper
w = TSERWrapper(cfg)
obs = jnp.ones((256, 8269)); occ = jnp.ones((256, 67))
loss = w.compute_auxiliary_loss(tser, obs, occ)
assert float(loss) > 0, f"T2 loss zero: {float(loss)}"
g = jax.grad(lambda pp: w.compute_auxiliary_loss(pp, obs, occ))(tser)
gn = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g) if x is not None)))
assert gn > 1e-8, f"T2 gradient norm zero: {gn}"
print(f"T2 gradient norm: {gn:.6f}")

from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
ckpt_dir = os.path.join(OUT, "checkpoints"); os.makedirs(ckpt_dir)
cm = CheckpointManager(ckpt_dir, PyTreeCheckpointer(), options=CheckpointManagerOptions(max_to_keep=1, create=True))
cm.save(esteps, {"train_state": ts, "global_step": esteps, "tser": tser})
restored = cm.restore(esteps)
assert restored is not None
assert int(restored.get("global_step", 0)) == esteps
rtser = restored.get("tser", {})
assert len(jax.tree_util.tree_leaves(rtser)) == tl_count, f"Restore mismatch: {len(jax.tree_util.tree_leaves(rtser))} != {tl}"
for i, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(tser), jax.tree_util.tree_leaves(rtser))):
    assert jnp.allclose(a, b, atol=1e-5), f"T2 leaf {i} mismatch: {float(jnp.max(jnp.abs(a-b)))}"

manifest = {"timestamp": time.time(), "commit": "46f632a", "treatment": "T2 TSER-PPO",
    "seed": 0, "env_steps": esteps, "horizon": 16384,
    "gpu_uuid": "GPU-83d39a25-90a3-b18c-4235-1e624434bdfe",
    "tser_leaves": tl_count, "gradient_norm": gn, "loss": float(loss),
    "checkpoint_verified": True, "elapsed_s": elapsed,
    "status": "PASS"}
assert manifest["commit"] == RUNTIME_COMMIT, \
    f"Manifest commit {manifest['commit']} != runtime HEAD {RUNTIME_COMMIT}"
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"T2 PREFLIGHT PASS — {OUT}/manifest.json")
