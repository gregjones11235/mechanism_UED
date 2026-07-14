#!/usr/bin/env python3
"""Regression tests: LPG-HRL OptionTerminationGate with non-128 embed sizes."""
import sys, os
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax, jax.numpy as jnp

PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

print("=" * 60)
print("LPG-HRL EMBED SIZE REGRESSION TESTS")
print("=" * 60)

from dicode.training.lpg_hrl import LPGHRLWrapper

# Test 1: Default embed (should work with any non-128 embed)
print("\n1. Default embed_size=64 — init and forward pass")
cfg64 = type('C',(),{'enable_lpg_hrl':True,'lpg_num_achievements':67,'lpg_embed_size':64,
    'lpg_option_entropy_weight':0.01})()
w64 = LPGHRLWrapper(cfg64)
p64 = w64.init_params(jax.random.PRNGKey(0), obs_feature_dim=64)
check(len(jax.tree_util.tree_leaves(p64)) > 0, "Params created with embed_size=64")
obs64 = jnp.ones((4, 64)); opt64 = jnp.zeros((4,), dtype=jnp.int32); term64 = jnp.zeros((4,))
loss64 = w64.compute_option_loss(p64, obs64, opt64, term64)
check(float(loss64) > 0, f"Nonzero loss with embed=64: {float(loss64):.6f}")

# Test 2: Explicit embed_size=32
print("\n2. embed_size=32 — init and forward pass")
cfg32 = type('C',(),{'enable_lpg_hrl':True,'lpg_num_achievements':67,'lpg_embed_size':32,
    'lpg_option_entropy_weight':0.01})()
w32 = LPGHRLWrapper(cfg32)
p32 = w32.init_params(jax.random.PRNGKey(1), obs_feature_dim=32)
check(len(jax.tree_util.tree_leaves(p32)) > 0, "Params created with embed_size=32")
obs32 = jnp.ones((4, 32)); opt32 = jnp.zeros((4,), dtype=jnp.int32); term32 = jnp.zeros((4,))
loss32 = w32.compute_option_loss(p32, obs32, opt32, term32)
check(float(loss32) > 0, f"Nonzero loss with embed=32: {float(loss32):.6f}")

# Test 3: Explicit embed_size=256
print("\n3. embed_size=256 — init and forward pass")
cfg256 = type('C',(),{'enable_lpg_hrl':True,'lpg_num_achievements':67,'lpg_embed_size':256,
    'lpg_option_entropy_weight':0.01})()
w256 = LPGHRLWrapper(cfg256)
p256 = w256.init_params(jax.random.PRNGKey(2), obs_feature_dim=256)
check(len(jax.tree_util.tree_leaves(p256)) > 0, "Params created with embed_size=256")
obs256 = jnp.ones((4, 256)); opt256 = jnp.zeros((4,), dtype=jnp.int32); term256 = jnp.zeros((4,))
loss256 = w256.compute_option_loss(p256, obs256, opt256, term256)
check(float(loss256) > 0, f"Nonzero loss with embed=256: {float(loss256):.6f}")

# Test 4: Gradient flows at embed_size=64
print("\n4. Gradient flows at embed_size=64")
g64 = jax.grad(lambda pp: w64.compute_option_loss(pp, obs64, opt64, term64))(p64)
gn64 = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g64) if x is not None)))
check(gn64 > 1e-8, f"Nonzero gradient at embed=64: {gn64:.6f}")

# Test 5: Gradient flows at embed_size=32
print("\n5. Gradient flows at embed_size=32")
g32 = jax.grad(lambda pp: w32.compute_option_loss(pp, obs32, opt32, term32))(p32)
gn32 = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g32) if x is not None)))
check(gn32 > 1e-8, f"Nonzero gradient at embed=32: {gn32:.6f}")

# Test 6: OptionTerminationGate uses num_options not hardcoded 128
print("\n6. OptionTerminationGate uses num_options, not hardcoded 128")
from dicode.training.lpg_hrl import OptionTerminationGate
import inspect
src = inspect.getsource(OptionTerminationGate.__call__)
check("128" not in src.split("jax.nn.one_hot")[1].split(",")[1][:10],
      "No hardcoded 128 in one_hot — uses self.num_options")
check("self.num_options" in src, "Uses self.num_options for one-hot depth")

# Test 7: Checkpoint round-trip at embed=64
print("\n7. Checkpoint at embed=64")
import tempfile
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
with tempfile.TemporaryDirectory() as td:
    cm = CheckpointManager(os.path.join(td,"ckpt"), PyTreeCheckpointer(),
                           options=CheckpointManagerOptions(max_to_keep=1, create=True))
    saved = {"lpg_hrl": p64, "step": 100}
    cm.save(100, saved)
    restored = cm.restore(100)
    rp = restored.get("lpg_hrl", {})
    check(len(jax.tree_util.tree_leaves(rp)) == len(jax.tree_util.tree_leaves(p64)),
          f"Checkpoint preserves leaf count: {len(jax.tree_util.tree_leaves(rp))}")

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0: sys.exit(1)
