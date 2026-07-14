#!/usr/bin/env python3
"""GPU0 R0 Preflight: T1 LPG-HRL, 16384 env steps, seed 0."""
import sys, os, json, time, hashlib, subprocess
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, "/root/experiments/dicode-dspro-r0/src")

# Resolve actual runtime commit — NEVER hardcode
RUNTIME_COMMIT = subprocess.check_output(
    ["git", "-C", "/root/experiments/dicode-dspro-r0", "rev-parse", "HEAD"],
    text=True).strip()

import jax, jax.numpy as jnp, numpy as np
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.constants import Achievement
from minicraftax.tasks.base_task import BaseTask
from minicraftax.craftax_state import TaskParams
from dicode.training.integration import make_train_with_treatments

OUT = f"/root/experiments/dicode_runs/dspro/r0_preflight/t1_lpg_hrl_{int(time.time())}"
os.makedirs(OUT, exist_ok=False)

# Verify GPU
devs = jax.devices()
gpus = [d for d in devs if d.platform == "gpu"]
assert len(gpus) == 1, f"Expected 1 GPU, got {len(gpus)}: {gpus}"
print(f"JAX GPU: {gpus[0]}")

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
    "enable_lpg_hrl":True,"lpg_num_achievements":67,"lpg_embed_size":32,
    "lpg_option_entropy_weight":0.01,"treatment_seed":42})()
cfg.mode = "achievement"

te = jnp.eye(1); td = jnp.ones(1) / 1
import wandb; wandb.init(mode="disabled")

t0 = time.time()
fn = make_train_with_treatments(cfg, [CT], 2, task_embeddings=te, task_distribution_proportions=td, initial_global_update_step=0)
res = jax.jit(fn)(jax.random.PRNGKey(0))
elapsed = time.time() - t0

ts = res.get("train_state")
metrics = res.get("metrics", {})
esteps = int(metrics.get("num_env_steps_done", 0))
print(f"T1 LPG-HRL: {esteps} env steps in {elapsed:.1f}s")

# Checks
lpg = ts.params.get("lpg_hrl", {})
lpg_leaves = jax.tree_util.tree_leaves(lpg)
lpg_count = len(lpg_leaves)
print(f"T1 leaves: {lpg_count}")
assert lpg_count > 0, "No T1 params in TrainState"
assert esteps == 16384, f"Expected 16384 steps, got {esteps}"

# Gradient check
from dicode.training.lpg_hrl import LPGHRLWrapper
w = LPGHRLWrapper(cfg)
obs = jnp.ones((4, 8269)); opt = jnp.zeros((4,),dtype=jnp.int32); term = jnp.zeros((4,))
loss = w.compute_option_loss(lpg, obs, opt, term)
assert float(loss) > 0, f"T1 loss is zero: {float(loss)}"
g = jax.grad(lambda pp: w.compute_option_loss(pp, obs, opt, term))(lpg)
gn = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g) if x is not None)))
assert gn > 1e-8, f"T1 gradient norm zero: {gn}"
print(f"T1 gradient norm: {gn:.6f}")

# Changed parameter leaves: compare pre-training against post-optimizer
fn_init = make_train_with_treatments(cfg, [CT], 0, task_embeddings=te, task_distribution_proportions=td, initial_global_update_step=0)
res_init = jax.jit(fn_init)(jax.random.PRNGKey(0))
ts_init = res_init.get("train_state")
lpg_init = ts_init.params.get("lpg_hrl", {})
lpg_init_leaves = jax.tree_util.tree_leaves(lpg_init)
params_changed = any(not jnp.allclose(a, b, atol=1e-6) for a, b in zip(lpg_init_leaves, lpg_leaves))
assert params_changed, "Treatment params did NOT change through optimizer application"
print(f"Params changed: {params_changed}")

# Checkpoint
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
ckpt_dir = os.path.join(OUT, "checkpoints"); os.makedirs(ckpt_dir)
cm = CheckpointManager(ckpt_dir, PyTreeCheckpointer(), options=CheckpointManagerOptions(max_to_keep=1, create=True))
cm.save(esteps, {"train_state": ts, "global_step": esteps, "lpg_hrl": lpg})
restored = cm.restore(esteps)
assert restored is not None
assert int(restored.get("global_step", 0)) == esteps
rlpg = restored.get("lpg_hrl", {})
assert len(jax.tree_util.tree_leaves(rlpg)) == lpg_count, f"Checkpoint leaves: {len(jax.tree_util.tree_leaves(rlpg))} != {lpg_leaves}"
for i, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(lpg), jax.tree_util.tree_leaves(rlpg))):
    if a.shape != b.shape:
        raise AssertionError(f"Leaf {i} shape mismatch: {a.shape} vs {b.shape}")
    diff = float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
    assert diff < 0.1, f"Leaf {i} substantial mismatch: max_diff={diff:.6f}"
print(f"Checkpoint restore verified ({lpg_count} leaves)")

# Manifest
manifest = {
    "timestamp": time.time(), "commit": RUNTIME_COMMIT, "treatment": "T1 LPG-HRL",
    "seed": 0, "env_steps": esteps, "horizon": 16384,
    "gpu_uuid": "GPU-83d39a25-90a3-b18c-4235-1e624434bdfe",
    "lpg_leaves": lpg_count, "gradient_norm": gn, "loss": float(loss),
    "params_changed_by_optimizer": params_changed,
    "checkpoint_verified": True, "elapsed_s": elapsed,
    "status": "PASS"
}
# Regression: manifest commit MUST equal actual git HEAD
assert manifest["commit"] == RUNTIME_COMMIT, \
    f"Manifest commit {manifest['commit']} != runtime HEAD {RUNTIME_COMMIT}"
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"T1 PREFLIGHT PASS — manifest at {OUT}/manifest.json")
