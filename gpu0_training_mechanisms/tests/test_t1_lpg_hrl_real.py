#!/usr/bin/env python3
"""T1 LPG-HRL real module tests — actual JAX/Flax parameters, gradients, checkpoint."""
import sys, os, tempfile
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax, jax.numpy as jnp
import numpy as np
from dicode.training.lpg_hrl import LPGHRLWrapper

PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

print("=" * 60)
print("T1 LPG-HRL REAL MODULE TESTS")
print("=" * 60)

# --- 1. Disabled path: numerical identity ---
print("\n1. Disabled path: zero loss, no gradients, no params")
cfg_disabled = type('C',(),{'enable_lpg_hrl':False})()
lpg_off = LPGHRLWrapper(cfg_disabled)
check(not lpg_off.enabled, "Disabled flag respected")
params = lpg_off.init_params(jax.random.PRNGKey(0))
check(params == {}, "No params when disabled")
loss = lpg_off.compute_option_loss({}, jnp.ones((4,64)), jnp.zeros((4,),dtype=jnp.int32), jnp.zeros((4,)))
check(float(loss) == 0.0, "Zero loss when disabled")
check(not lpg_off.has_gradient({}, jnp.ones((4,64)), jnp.zeros((4,),dtype=jnp.int32), jnp.zeros((4,))),
      "No gradient when disabled")

# --- 2. Enabled: real params initialized ---
print("\n2. Enabled: real Flax parameters initialized")
cfg_on = type('C',(),{'enable_lpg_hrl':True,'lpg_num_achievements':67,'lpg_embed_size':64,'lpg_option_entropy_weight':0.01})()
lpg_on = LPGHRLWrapper(cfg_on)
check(lpg_on.enabled, "Enabled flag active")
rng = jax.random.PRNGKey(42)
real_params = lpg_on.init_params(rng)
check("graph_encoder" in real_params, "Graph encoder params present")
check("option_policy" in real_params, "Option policy params present")
check("termination_gate" in real_params, "Termination gate params present")
leaf_count = len(jax.tree_util.tree_leaves(real_params))
check(leaf_count > 0, f"Real trainable parameters: {leaf_count} leaves")

# --- 3. Forward pass produces nonzero loss ---
print("\n3. Forward pass: nonzero loss")
obs = jnp.ones((4, 64))
opt = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
term = jnp.array([0.0, 0.0, 1.0, 0.0])
loss_on = lpg_on.compute_option_loss(real_params, obs, opt, term)
check(float(loss_on) > 0.0, f"Nonzero loss when enabled: {float(loss_on):.6f}")
check(float(loss_on) != float(lpg_off.compute_option_loss({}, obs, opt, term)),
      "Enabled loss differs from disabled (zero)")

# --- 4. Real nonzero gradients ---
print("\n4. Real nonzero gradients")
check(lpg_on.has_gradient(real_params, obs, opt, term),
      "Gradient norm > 1e-8 (real backward pass)")

# --- 5. Parameter update changes params ---
print("\n5. Parameter update changes parameters")
def loss_fn(p):
    return lpg_on.compute_option_loss(p, obs, opt, term)
grad = jax.grad(loss_fn)(real_params)
lr = 0.001
updated = jax.tree_util.tree_map(lambda p, g: p - lr * g, real_params, grad)
leaves_before = jax.tree_util.tree_leaves(real_params)
leaves_after = jax.tree_util.tree_leaves(updated)
any_changed = any(not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after))
check(any_changed, "Parameter update changes at least one leaf")

# --- 6. Checkpoint save/restore with real params ---
print("\n6. Checkpoint save/restore with real LPG-HRL params")
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
with tempfile.TemporaryDirectory() as td:
    ckpt_dir = os.path.join(td, "ckpt")
    os.makedirs(ckpt_dir)
    oc = PyTreeCheckpointer()
    cm = CheckpointManager(ckpt_dir, oc, options=CheckpointManagerOptions(max_to_keep=1, create=True))
    step = 100
    saved = {"ppo": {"w": jnp.array([1.0, 2.0])}, "lpg_hrl": real_params, "step": step}
    cm.save(step, saved)
    restored = cm.restore(step)
    check(restored is not None, "Restored checkpoint exists")
    check("lpg_hrl" in restored, "LPG-HRL params in restored checkpoint")
    check("ppo" in restored, "PPO params preserved alongside LPG-HRL")
    restored_leaves = jax.tree_util.tree_leaves(restored["lpg_hrl"])
    real_leaves = jax.tree_util.tree_leaves(real_params)
    check(len(restored_leaves) == len(real_leaves),
          f"Leaf count preserved: {len(restored_leaves)} == {len(real_leaves)}")
    all_close = all(jnp.allclose(rl, r) for rl, r in zip(restored_leaves, real_leaves))
    check(all_close, "All restored LPG-HRL leaves equal original")

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0: sys.exit(1)
