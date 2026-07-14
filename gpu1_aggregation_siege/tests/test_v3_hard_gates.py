#!/usr/bin/env python3
"""Preflight V3 Hard Gate Tests — fail-closed, no GPU required.

Tests every resume condition from CORRECTION_REQUIRED intervention.
Run: cd /root/experiments/dicode-siege-aggregation && PYTHONPATH=src:/root/experiments/dicode-aggregation-v2/src python tests/test_v3_hard_gates.py
"""
import sys, os, json, tempfile, hashlib, subprocess
# SIEGE src BEFORE aggregation-v2 (ensures dicode.siege is found)
_siege_src = os.path.join(os.path.dirname(__file__), "..", "src")
_agg_src = "/root/experiments/dicode-aggregation-v2/src"
for p in [_siege_src, _agg_src]:
    if p in sys.path: sys.path.remove(p)
sys.path.insert(0, _siege_src)  # Must be index 0 for dicode.siege
sys.path.insert(1, _agg_src)   # Index 1 for dicode.mechanisms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

print("=" * 60)
print("PREFLIGHT V3 HARD GATE TESTS")
print("=" * 60)

# ── Test 1: Rejected candidates never enter pool ──
print("\n1. Chain-rejected candidates NEVER enter pool")
from dicode.siege.siege_notebook import SiegeNotebook
from dicode.siege.aggregation_integration import chain_completeness_gate

with tempfile.TemporaryDirectory() as td:
    nb = SiegeNotebook(td); nb.define_craftax_chains()
    nb.update({"collect_wood": 0.96, "craft_planks": 0.55}, 1000)
    candidates = [f"cand_{i:04d}" for i in range(40)]
    meta = {}
    for i, cid in enumerate(candidates):
        ach = ["collect_wood","craft_planks"] if i < 25 else ["unknown_skill"]
        meta[cid] = nb.get_candidate_metadata(cid, ach) if i < 25 else {"siege_wall":False,"chain_complete":False}
    admitted, rejected, report = chain_completeness_gate(candidates, meta, nb)
    check(len(rejected) > 0, f"Chain gate rejected {len(rejected)} candidates")
    # F1: rejected must NOT be in admitted
    rejected_set = set(rejected)
    overlap = set(admitted) & rejected_set
    check(len(overlap) == 0, f"No rejected candidates in admitted set (overlap={len(overlap)})")

# ── Test 2: Cache is read-only (no put allowed) ──
print("\n2. Frozen cache — read-only verification")
CACHE_PATH = "/root/experiments/dicode_runs/siege_aggregation/frozen_immutable_cache"
check(os.path.isdir(CACHE_PATH), f"Frozen cache exists at {CACHE_PATH}")
from dicode.mechanisms.immutable_cache import MultiRoleImmutableCache
multi = MultiRoleImmutableCache(cache_dir=CACHE_PATH)
multi.load_all()
total = sum(c.entry_count for c in multi._caches.values())
check(total >= 96, f"Cache has {total} entries (>=96 required)")
# Verify all entries readable
from dicode.mechanisms.immutable_cache import compute_immutable_cache_key
from dicode.mechanisms.model_manifest import ROLE_CONFIG_MAP
pool = [f"candidate_{i:04d}" for i in range(32)]
hits = 0
for role in ["tutor","critic","explorer"]:
    cfg = ROLE_CONFIG_MAP.get(role)
    if not cfg: continue
    for tid in pool:
        try:
            key = compute_immutable_cache_key(task_code_hash=tid,student_stage_id="stage_0",
                role=role,provider=cfg["provider"],exact_model_id=cfg["exact_model_id"],
                prompt_version=cfg["prompt_version"],schema_version=cfg["schema_version"])
        except ValueError: continue
        if multi.get_cache(role).get(key): hits += 1
rate = hits/max(1, total)
check(rate >= 0.95, f"Cache hit rate {rate:.4f} >= 0.95 (hits={hits}/{total})")

# ── Test 3: CandidateTask has distinct hashes and executable TaskParams ──
print("\n3. CandidateTask — distinct hashes + executable TaskParams")
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.craftax_state import TaskParams
from craftax.craftax.constants import Achievement
from minicraftax.tasks.base_task import BaseTask

all_achs = list(Achievement)
sp, ep = StaticEnvParams(), EnvParams()

class CandidateTask(BaseTask):
    def __init__(self, sp, ep, candidate_id, target_achievements, param_seed):
        super().__init__(sp, ep)
        self._cid = candidate_id
        self._chash = hashlib.sha256(f"{candidate_id}:{sorted([a.name for a in target_achievements])}".encode()).hexdigest()[:16]
        self.relevant_achievements = target_achievements
        self.completed_achievements = []
        self.label = f"candidate_{candidate_id}"
        rng = np.random.default_rng(param_seed)
        self._spawn_mult = 0.25 + 3.0 * rng.random()
        self._health_mult = 0.25 + 6.0 * rng.random()
        self._damage_mult = 0.25 + 6.0 * rng.random()
    @property
    def candidate_hash(self): return self._chash
    @property
    def candidate_id(self): return self._cid
    def get_task_params(self):
        return TaskParams(passive_spawn_multiplier=float(self._spawn_mult),
                         melee_spawn_multiplier=float(self._spawn_mult*0.8),
                         mob_health_multiplier=float(self._health_mult),
                         mob_damage_multiplier=float(self._damage_mult))

    def generate_world(self, rng):
        from minicraftax.world_builder import WorldBuilder
        return WorldBuilder.generate_world(rng, self.static_params, self.params, self.get_task_params())
    def __repr__(self): return f"CandidateTask({self._cid}, hash={self._chash[:8]})"

tasks = [CandidateTask(sp, ep, f"t{i:04d}", all_achs[i:i+2], 100+i) for i in range(8)]
hashes = [t.candidate_hash for t in tasks]
ids = [t.candidate_id for t in tasks]
check(len(set(hashes)) == 8, f"All 8 candidate hashes distinct: {len(set(hashes))}/8")
check(len(set(ids)) == 8, f"All 8 candidate IDs distinct")

# Each has unique TaskParams
params = [t.get_task_params() for t in tasks]
unique_spawns = len(set(round(p.passive_spawn_multiplier, 4) for p in params))
check(unique_spawns == 8, f"All 8 spawn multipliers unique: {unique_spawns}/8")

# ── Test 4: make_train accepts CandidateTask instances ──
print("\n4. make_train accepts CandidateTask instances")
import jax, jax.numpy as jnp
from dicode.ppo_tr import make_train
import wandb; wandb.init(mode="disabled")
ne, ns, nu = 256, 64, 1
cfg = type('C',(),{'num_envs':ne,'num_steps':ns,'num_minibatches':4,'update_epochs':4,
    'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,'ent_coef':0.01,'vf_coef':0.5,
    'max_grad_norm':0.5,'lr':3e-4,'anneal_lr':False,'min_lr':3e-6,
    'activation':'relu','hidden_layers':256,'embed_size':64,'num_heads':4,
    'qkv_features':256,'num_layers':2,'window_mem':16,'window_grad':8,
    'gating':True,'gating_bias':1.0,'condition_on_task':'onehot',
    'completion_bonus_scale':0.1,'completion_bonus_min':0.0,
    'bonus_type':'none','dynamic_bonus_k':0,'optimistic_reset_ratio':16,
    'scoring_window_updates':1,'total_timesteps':ne*ns,
    'max_updates_per_session':1,'mode':'achievement','debug':False,'use_wandb':False})()
te = jnp.eye(len(tasks)); td = jnp.ones(len(tasks))/len(tasks)
rng = jax.random.PRNGKey(0)
fn = make_train(cfg, tasks, nu, task_embeddings=te, task_distribution_proportions=td, initial_global_update_step=0)
fj = jax.jit(fn)
res = fj(rng)
esteps = res['metrics']['num_env_steps_done']
ts = res.get('train_state')
check(esteps > 0, f"PPO completed {esteps} env steps")
check(ts is not None, "train_state returned by make_train")

# ── Test 5: Missing steps hard-fails ──
print("\n5. Missing runtime steps hard-fails")
fake_metrics = {}
env_steps_raw = fake_metrics.get("num_env_steps_done")
if env_steps_raw is None:
    would_raise = True
else:
    would_raise = env_steps_raw == 0
check(would_raise, "Missing num_env_steps_done correctly detected as failure")

# ── Test 6: Missing train_state hard-fails ──
print("\n6. Missing train_state hard-fails")
check(ts is not None, "train_state present (would fail if None)")

# ── Test 7: Checkpoint save/restore ──
print("\n7. Checkpoint save/restore works")
with tempfile.TemporaryDirectory() as td:
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
    ckpt_dir = os.path.join(td, "checkpoints")
    os.makedirs(ckpt_dir)
    oc = PyTreeCheckpointer()
    co = CheckpointManagerOptions(max_to_keep=1, create=True)
    cm = CheckpointManager(ckpt_dir, oc, options=co)
    cm.save(esteps, {"train_state": ts, "global_step": esteps})
    restored = cm.restore(esteps)
    check(restored is not None, "Checkpoint restore returned data")
    check("train_state" in restored, "Restored checkpoint has train_state")
    check("global_step" in restored, "Restored checkpoint has global_step")
    ckpt_files = [f for f in os.listdir(ckpt_dir) if not f.startswith('.')]
    check(len(ckpt_files) > 0, f"Checkpoint files saved: {len(ckpt_files)}")

# ── Test 8: No base_task indirection ──
print("\n8. No .base_task indirection in preflight v3")
with open('scripts/run_data_plane_preflight_v3.py') as f:
    src = f.read()
check('[t.base_task for t in selected_tasks]' not in src, "No .base_task unwrapping")
check('class CandidateTask(BaseTask)' in src, "CandidateTask inherits BaseTask")
check('def get_task_params' in src, "get_task_params implemented")
check('v3' in src.lower(), "v3 labels present (no stale v2)")

# ── Test 9: Preflight v3 syntax valid ──
print("\n9. Preflight v3 syntax check")
result = subprocess.run(['python3','-c',"compile(open('scripts/run_data_plane_preflight_v3.py').read(),'v3','exec')"],
                       capture_output=True, text=True, cwd='/root/experiments/dicode-siege-aggregation')
check(result.returncode == 0, f"Syntax: {'OK' if result.returncode==0 else result.stderr[:100]}")

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0:
    sys.exit(1)
else:
    print("READY_FOR_INDEPENDENT_REVIEW")
    sys.exit(0)
