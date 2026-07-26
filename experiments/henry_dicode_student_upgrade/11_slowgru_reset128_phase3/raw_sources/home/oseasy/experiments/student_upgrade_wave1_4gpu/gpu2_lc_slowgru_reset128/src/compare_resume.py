#!/usr/bin/env python3
"""Compare EventMemory32 continuous vs resumed checkpoints (exact-resume gate 6 + determinism).
Reads full_state.pkl params, hashes tree leaves in order (same convention as the trainer _params_sha)."""
import os, sys, pickle, hashlib
import numpy as np, jax, jax.numpy as jnp

def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()

def load_params(path):
    d = pickle.load(open(os.path.join(path, "full_state.pkl"), "rb"))
    leaves, td = d["params"]
    return _params_sha(jax.tree_util.tree_unflatten(td, [jnp.asarray(l) for l in leaves])), d

base = sys.argv[1]
A4096, _ = load_params(f"{base}/smoke_A/ckpt/4096")
B1_4096, _ = load_params(f"{base}/smoke_B/ckpt/4096")
A8192, dA = load_params(f"{base}/smoke_A/ckpt/8192")
B2_8192, dB = load_params(f"{base}/smoke_B2/ckpt/8192")
det_4096 = (A4096 == B1_4096)
resume_8192 = (A8192 == B2_8192)
print("A@4096      ", A4096[:16])
print("B1@4096     ", B1_4096[:16], "  deterministic_4096 =", det_4096)
print("A@8192      ", A8192[:16])
print("B2@8192     ", B2_8192[:16], "  EXACT_RESUME_8192 =", resume_8192)
print("A@8192 update_count =", dA["update_count"], " B2@8192 update_count =", dB["update_count"])
print("RESULT:", "EXACT_RESUME_PASS" if (det_4096 and resume_8192) else "EXACT_RESUME_FAIL")
