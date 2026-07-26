"""Compatible-init fingerprint verification from Henry base ckpt17500 (GPU0, orbax).

  CI.load   weights-only load from ckpt17500 succeeds on GPU0 via the proven loader
  CI.fprint param_count==4,906,028; content SHA256 == 5dfe67dd...; zero-obs batch2
            value~=3.5761, entropy~=0.9791, top action==12 (prob~=0.718)
  CI.schema encoder kernel == (8335,256); P2-Full-A network accepts the loaded params
            (EMA target initialised to the loaded online params)

Run on GPU0 ONLY (orbax GPU checkpoint cannot be restored on CPU).
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID   # MUST be set before jax import
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax

import compat_init as CI


def test_compatible_init_fingerprint():
    res = CI.compatible_init(strict=True)     # raises on any fingerprint mismatch
    fp = res["fingerprint"]
    assert fp["ok"], fp["problems"]
    assert fp["param_count"] == CI.EXPECTED_PARAM_COUNT
    assert fp["params_sha256"] == CI.EXPECTED_PARAMS_SHA256
    assert fp["top_action"] == CI.EXPECTED_TOP_ACTION
    assert abs(fp["value"] - CI.EXPECTED_ZERO_OBS_VALUE) < 1e-2
    assert abs(fp["entropy"] - CI.EXPECTED_ZERO_OBS_ENTROPY) < 1e-2
    assert abs(fp["top_prob"] - CI.EXPECTED_TOP_PROB) < 1e-2
    # schema
    assert res["obs_dim"] == CI.OBS_DIM, res["obs_dim"]
    assert res["action_dim"] == CI.NET_DIMS["action_dim"]
    assert res["kernels"]["encoder"] == (CI.OBS_DIM, CI.NET_DIMS["encoder_size"])
    # EMA target initialised == online params (bit-exact copy)
    la = jax.tree_util.tree_leaves(res["params"])
    lb = jax.tree_util.tree_leaves(res["target_params"])
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))
    # callables present
    assert res["apply_eval_recon"] is not None and res["scan_fn"] is not None
    print("PASS CI.load + CI.fprint + CI.schema")
    print("  params_sha256=%s" % fp["params_sha256"])
    print("  param_count=%d value=%.4f entropy=%.4f top=%d@%.3f" % (
        fp["param_count"], fp["value"], fp["entropy"], fp["top_action"], fp["top_prob"]))
    print("  obs_dim=%d action_dim=%d encoder_kernel=%s" % (
        res["obs_dim"], res["action_dim"], res["kernels"]["encoder"]))


if __name__ == "__main__":
    test_compatible_init_fingerprint()
    print("ALL_COMPAT_INIT_TESTS_PASS")
