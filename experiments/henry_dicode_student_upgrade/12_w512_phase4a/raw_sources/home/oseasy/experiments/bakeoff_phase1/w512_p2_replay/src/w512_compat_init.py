"""W512 × P2 compatible init: load ckpt17500 into W512 network.

Loads base GTrXL params from ckpt17500 (via Henry's load_weights_only),
then initializes the W512 network with those base params + zero-init W512
additions (cross-attention, gate, posenc). The w512_gate is zero-init so
at init z_t == h_t (bit-exact with ckpt17500).

Builds W512-adapted apply callables for the P2 replay update.
"""
import os
import sys
import hashlib

import numpy as np
import jax
import jax.numpy as jnp

# Paths
BAKE_SRC = "/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/src"
HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
P2_SRC = "/home/oseasy/experiments/p2_full_20260723/src"
THIS_SRC = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

for p in [THIS_SRC, BAKE_SRC, P2_SRC]:
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, THIS_SRC)
sys.path.insert(1, BAKE_SRC)
sys.path.insert(2, P2_SRC)
if HENRY_SRC not in sys.path:
    sys.path.insert(3, HENRY_SRC)

CKPT17500 = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
             "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")

# Frozen W512 architecture dims (from bakeoff)
NET_DIMS = dict(action_dim=43, activation="relu", hidden_layers=256,
                encoder_size=256, num_heads=8, qkv_features=256,
                num_layers=2, gating=True, gating_bias=2.0)
W512_LONG_SIZE = 384
OBS_DIM = 8335


def build_w512_network():
    from network_w512 import ActorCriticTransformerW512
    return ActorCriticTransformerW512(long_size=W512_LONG_SIZE, **NET_DIMS)


def load_w512_params(ckpt_path=CKPT17500):
    """Load ckpt17500 base params and init W512 network.
    Returns (params, network) where params includes W512 additions."""
    network = build_w512_network()

    # Init W512 network with dummy input to get full param structure
    rng = jax.random.PRNGKey(0)
    init_mem = jnp.zeros((2, 128, 2, 256))
    init_obs = jnp.zeros((2, OBS_DIM))
    init_mask = jnp.zeros((2, 8, 1, 129), dtype=jnp.bool_)
    init_lbuf = jnp.zeros((2, W512_LONG_SIZE, 256))
    init_lmsk = jnp.zeros((2, W512_LONG_SIZE), dtype=jnp.bool_)
    full_params = network.init(rng, init_mem, init_obs, init_mask,
                               long_buf=init_lbuf, long_mask=init_lmsk)
    if isinstance(full_params, dict) and "params" in full_params:
        full_params = full_params["params"]

    # Load base params from ckpt17500
    from dicode.utils.general.train_state_utils import load_weights_only

    class _Space:
        def __init__(self, n=None, shape=None):
            self._n = n; self._shape = shape
        @property
        def n(self): return self._n
        @property
        def shape(self): return self._shape

    class _StubEnv:
        def action_space(self, ep=None): return _Space(n=43)
        def observation_space(self, ep=None): return _Space(shape=(OBS_DIM,))

    class _Cfg:
        activation = "relu"; embed_size = 256; hidden_layers = 256
        num_heads = 8; qkv_features = 256; num_layers = 2
        gating = True; gating_bias = 2.0; window_mem = 128
        anneal_lr = False; lr = 2e-4; min_lr = 2e-6; max_grad_norm = 1.0

    ts = load_weights_only(ckpt_path, _StubEnv(), None, _Cfg(),
                           load_opt_state=False)
    base_params = ts.params
    if isinstance(base_params, dict) and "params" in base_params:
        base_params = base_params["params"]

    # Merge: base params overwrite W512 init; W512-only keys keep their init
    def _merge(base, full):
        if isinstance(base, dict) and isinstance(full, dict):
            merged = dict(full)
            for k, v in base.items():
                if k in merged:
                    merged[k] = _merge(v, merged[k])
                else:
                    merged[k] = v
            return merged
        else:
            # Leaf: use base value (ckpt17500)
            return base

    merged = _merge(base_params, full_params)

    # Verify base params match
    base_sha = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(base_params):
        base_sha.update(np.asarray(leaf).tobytes())
    print(f"[w512_compat_init] base params sha={base_sha.hexdigest()[:16]}",
          flush=True)

    return merged, network


def build_w512_callables(network, w5_cfg):
    """Build W512-adapted apply callables for P2 replay update."""
    import w512_memory as w5m
    from full_p2_learner import FullP2Config
    cfg = FullP2Config()

    def apply_eval_w512_raw(params, memories, obs, mask, long_buf, long_mask):
        pi, value, mem_out, h_t = network.apply(
            {"params": params}, memories, obs, mask,
            long_buf=long_buf, long_mask=long_mask,
            method=network.model_forward_eval)
        return pi.logits, value, mem_out, h_t

    # Padded version for batch==1 reconstruction
    def apply_eval_w512_recon(params, memories, obs, mask, long_buf, long_mask):
        if memories.shape[0] == 1:
            memories2 = jnp.concatenate([memories, memories], axis=0)
            obs2 = jnp.concatenate([obs, obs], axis=0)
            mask2 = jnp.concatenate([mask, mask], axis=0)
            lbuf2 = jnp.concatenate([long_buf, long_buf], axis=0)
            lmsk2 = jnp.concatenate([long_mask, long_mask], axis=0)
            lg, vl, mo, ht = apply_eval_w512_raw(
                params, memories2, obs2, mask2, lbuf2, lmsk2)
            return lg[:1], vl[:1], mo[:1], ht[:1]
        return apply_eval_w512_raw(params, memories, obs, mask,
                                    long_buf, long_mask)

    apply_eval_w512_recon_jit = jax.jit(apply_eval_w512_recon)

    # Scan function for loss windows (B>=2)
    from w512_p2_learner import _scan_lax_w512
    def scan_fn_w512(params, memories, mem_mask, mem_idx,
                     long_buf, long_mask, obs_seq, w5_cfg_inner, delay_state):
        return _scan_lax_w512(
            apply_eval_w512_raw, params, memories, mem_mask, mem_idx,
            long_buf, long_mask, obs_seq, cfg, w5_cfg_inner, delay_state)

    scan_fn_w512_jit = jax.jit(scan_fn_w512, static_argnums=(7,))

    return apply_eval_w512_recon_jit, apply_eval_w512_raw, scan_fn_w512_jit


def compatible_init_w512(strict=True):
    """Full W512 compatible init: load params, build network + callables.
    Returns dict with params, target_params, network, callables, fingerprint."""
    import w512_memory as w5m

    params, network = load_w512_params()
    target_params = jax.tree_util.tree_map(jnp.array, params)

    w5_cfg = w5m.W512Config(long_size=W512_LONG_SIZE, delay_size=128,
                             encoder_size=256)

    a_rec, a_raw, scan_fn = build_w512_callables(network, w5_cfg)

    # Fingerprint
    param_count = sum(int(np.asarray(l).size)
                      for l in jax.tree_util.tree_leaves(params))
    params_sha = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        params_sha.update(np.asarray(leaf).tobytes())

    # Zero-obs fingerprint
    obs = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    mem = jnp.zeros((2, 128, 2, 256), dtype=jnp.float32)
    mask = jnp.ones((2, 8, 1, 129), dtype=jnp.bool_)
    lbuf = jnp.zeros((2, W512_LONG_SIZE, 256), dtype=jnp.float32)
    lmsk = jnp.zeros((2, W512_LONG_SIZE), dtype=jnp.bool_)
    pi, value, _, _ = network.apply(
        {"params": params}, mem, obs, mask,
        long_buf=lbuf, long_mask=lmsk,
        method=network.model_forward_eval)
    probs = np.asarray(jax.nn.softmax(pi.logits, axis=-1)[0])

    fingerprint = {
        "params_sha256": params_sha.hexdigest(),
        "param_count": param_count,
        "value": float(np.asarray(value[0])),
        "top_action": int(np.argmax(probs)),
        "top_prob": float(probs[np.argmax(probs)]),
    }

    return {
        "params": params,
        "target_params": target_params,
        "network": network,
        "w5_cfg": w5_cfg,
        "apply_eval_recon": a_rec,
        "apply_eval_raw": a_raw,
        "scan_fn": scan_fn,
        "fingerprint": fingerprint,
        "source_checkpoint": CKPT17500,
    }
