#!/usr/bin/env python3
"""T2 TSER-PPO real module tests — actual JAX/Flax parameters, gradients, checkpoint."""
import sys, os, tempfile
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax, jax.numpy as jnp
from dicode.training.tser_ppo import TSERWrapper

PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

print("=" * 60)
print("T2 TSER-PPO REAL MODULE TESTS")
print("=" * 60)

# --- 1. Disabled path: zero loss, no gradients ---
print("\n1. Disabled path: zero loss, no gradients, no params")
cfg_off = type('C',(),{'enable_tser':False})()
tser_off = TSERWrapper(cfg_off)
check(not tser_off.enabled, "Disabled flag respected")
p = tser_off.init_params(jax.random.PRNGKey(0))
check(p == {}, "No params when disabled")
obs = jnp.ones((4, 256)); occ = jnp.ones((4, 67))
check(float(tser_off.compute_auxiliary_loss({}, obs, occ)) == 0.0, "Zero loss when disabled")
check(not tser_off.has_gradient({}, obs, occ), "No gradient when disabled")

# --- 2. Enabled: real Flax parameters ---
print("\n2. Enabled: real Flax parameters initialized")
cfg_on = type('C',(),{'enable_tser':True,'tser_num_events':67,'tser_hidden_size':128,'tser_loss_weight':0.1,'tser_goal_weight':0.05})()
tser_on = TSERWrapper(cfg_on)
check(tser_on.enabled, "Enabled flag active")
real_p = tser_on.init_params(jax.random.PRNGKey(42))
leaf_count = len(jax.tree_util.tree_leaves(real_p))
check(leaf_count > 0, f"Real trainable parameters: {leaf_count} leaves")

# --- 3. Forward pass produces nonzero loss ---
print("\n3. Forward pass: nonzero loss")
obs_batch = jnp.ones((4, 256))
occ_batch = jnp.ones((4, 67))
goals = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
loss_on = tser_on.compute_auxiliary_loss(real_p, obs_batch, occ_batch)
check(float(loss_on) > 0.0, f"Nonzero loss: {float(loss_on):.6f}")
loss_goal = tser_on.compute_auxiliary_loss(real_p, obs_batch, occ_batch, active_goals=goals)
check(float(loss_goal) != float(loss_on), "Goal-reachability loss changes total")

# --- 4. Real nonzero gradients ---
print("\n4. Real nonzero gradients")
check(tser_on.has_gradient(real_p, obs_batch, occ_batch), "Gradient norm > 1e-8")

# --- 5. Parameter update changes params ---
print("\n5. Parameter update changes parameters")
def loss_fn(p):
    return tser_on.compute_auxiliary_loss(p, obs_batch, occ_batch)
grad = jax.grad(loss_fn)(real_p)
lr = 0.01
updated = jax.tree_util.tree_map(lambda p, g: p - lr * g, real_p, grad)
before = jax.tree_util.tree_leaves(real_p)
after = jax.tree_util.tree_leaves(updated)
changed = any(not jnp.allclose(b, a) for b, a in zip(before, after))
check(changed, "Parameter update changes at least one leaf")

# --- 6. Checkpoint save/restore with real TSER params ---
print("\n6. Checkpoint save/restore with real TSER params")
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
with tempfile.TemporaryDirectory() as td:
    ckpt_dir = os.path.join(td, "ckpt")
    os.makedirs(ckpt_dir)
    cm = CheckpointManager(ckpt_dir, PyTreeCheckpointer(),
                           options=CheckpointManagerOptions(max_to_keep=1, create=True))
    step = 200
    saved = {"ppo": {"w": jnp.array([1.0, 2.0])}, "tser": real_p, "step": step}
    cm.save(step, saved)
    restored = cm.restore(step)
    check(restored is not None, "Restored checkpoint exists")
    check("tser" in restored, "TSER params in restored checkpoint")
    check("ppo" in restored, "PPO params preserved alongside TSER")
    rl = jax.tree_util.tree_leaves(restored["tser"])
    sl = jax.tree_util.tree_leaves(real_p)
    check(len(rl) == len(sl), f"Leaf count preserved: {len(rl)} == {len(sl)}")
    check(all(jnp.allclose(a, b) for a, b in zip(rl, sl)), "All restored TSER leaves equal original")

# --- 7. Both modules coexist in checkpoint ---
print("\n7. T1 + T2 coexist in same checkpoint")
from dicode.training.lpg_hrl import LPGHRLWrapper
cfg_both_lpg = type('C',(),{'enable_lpg_hrl':True,'lpg_num_achievements':67,'lpg_embed_size':64,'lpg_option_entropy_weight':0.01})()
cfg_both_tser = type('C',(),{'enable_tser':True,'tser_num_events':67,'tser_hidden_size':128,'tser_loss_weight':0.1,'tser_goal_weight':0.05})()
lpg_p = LPGHRLWrapper(cfg_both_lpg).init_params(jax.random.PRNGKey(0))
tser_p2 = TSERWrapper(cfg_both_tser).init_params(jax.random.PRNGKey(1))
with tempfile.TemporaryDirectory() as td:
    cm = CheckpointManager(os.path.join(td, "ckpt"), PyTreeCheckpointer(),
                           options=CheckpointManagerOptions(max_to_keep=1, create=True))
    combined = {"ppo": {"w": jnp.array([1.0])}, "lpg_hrl": lpg_p, "tser": tser_p2, "step": 300}
    cm.save(300, combined)
    r = cm.restore(300)
    check("lpg_hrl" in r and "tser" in r, "Both T1 and T2 in same checkpoint")
    check(len(jax.tree_util.tree_leaves(r["lpg_hrl"])) == len(jax.tree_util.tree_leaves(lpg_p)),
          "T1 leaf count preserved")
    check(len(jax.tree_util.tree_leaves(r["tser"])) == len(jax.tree_util.tree_leaves(tser_p2)),
          "T2 leaf count preserved")

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0: sys.exit(1)
