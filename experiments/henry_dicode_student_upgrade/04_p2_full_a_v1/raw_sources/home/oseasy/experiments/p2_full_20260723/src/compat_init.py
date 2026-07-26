"""P2-Full-A compatible init from the Henry base ckpt17500 (GPU0, orbax).

Loads ONLY model weights from the frozen healthy Henry Student base checkpoint
(ckpt17500), builds the P2-Full-A ActorCriticTransformer (frozen dims), verifies a
BIT-EXACT fingerprint, and returns params + EMA-target init + the apply callables the
combined update needs.

Why orbax here: ckpt17500 is an orbax ocdbt checkpoint written on GPU; it can only be
restored on GPU0 (orbax GPU<->CPU restore is incompatible). P2-Full-A's OWN checkpoints
are pure pickle (see checkpointing.py) and CPU-portable. So the base is read once via
orbax on GPU0, then P2-Full-A trains/checkpoints in its own portable format.

The load reuses Henry's proven `load_weights_only` (the exact path the read-only
architecture inspection used), so the restored weights are bit-identical to the
audited base. The fingerprint (param count, content SHA256, zero-obs policy) is
ASSERTED fail-closed against the audited reference values; a mismatch raises.
"""
import os
import sys

# GPU0 must be selected BEFORE jax is imported by the caller; the test/launcher also
# sets this at its top. Setting here is defensive for direct import.
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_UUID)

import numpy as np
import jax
import jax.numpy as jnp

HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
BASE_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
# NOTE: deliberately NO P2-v1 src on the path. The P2-v1 launcher defines same-named
# top-level modules (checkpointing/hindsight/pending_episodes/rng_utils) that would
# collide with P2-Full-A's. compat_init depends ONLY on the Henry base (dicode.*) for
# the weight load, so the two never share a process namespace.
# BASE must win P2-Full-A module names -> force to front unconditionally.
if BASE_SRC in sys.path:
    sys.path.remove(BASE_SRC)
sys.path.insert(0, BASE_SRC)
if HENRY_SRC not in sys.path:
    sys.path.insert(1, HENRY_SRC)

from checkpointing import params_content_sha256, param_count

CKPT17500 = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
             "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")

# ---- audited reference fingerprint (read-only base; must match bit-exact) ----
EXPECTED_PARAM_COUNT = 4_906_028
EXPECTED_PARAMS_SHA256 = "5dfe67dda87ef15aa716276730de7685d73ac4096761abbc405fc7198cc6cd61"
EXPECTED_ZERO_OBS_VALUE = 3.5761
EXPECTED_ZERO_OBS_ENTROPY = 0.9791
EXPECTED_TOP_ACTION = 12
EXPECTED_TOP_PROB = 0.718

# ---- frozen P2-Full-A architecture dims (from the architecture audit) ----
NET_DIMS = dict(action_dim=43, activation="relu", hidden_layers=256,
                encoder_size=256, num_heads=8, qkv_features=256,
                num_layers=2, gating=True, gating_bias=2.0)
OBS_DIM = 8335
EMB = 67


class _Space:
    def __init__(self, n=None, shape=None):
        self._n = n
        self._shape = shape
    @property
    def n(self):
        return self._n
    @property
    def shape(self):
        return self._shape


class _StubEnv:
    """Minimal env exposing only what Henry's load_weights_only reads: action_space.n
    and observation_space.shape. Avoids importing the full Craftax env stack (and the
    P2-v1 launcher) — the dims are the audited frozen P2-Full-A values."""
    def __init__(self, action_dim, obs_dim):
        self._ad = action_dim
        self._od = obs_dim
    def action_space(self, env_params=None):
        return _Space(n=self._ad)
    def observation_space(self, env_params=None):
        return _Space(shape=(self._od,))


class _CompatCfg:
    """Self-contained config for Henry's load_weights_only. anneal_lr=False matches the
    exact path the read-only architecture inspection used (P2-v1 Cfg.anneal_lr=False),
    which restored ckpt17500 successfully. Only the dims + a plain Adam tx are needed
    when anneal_lr=False; the schedule fields are unused on this branch."""
    activation = "relu"
    embed_size = 256
    hidden_layers = 256
    num_heads = 8
    qkv_features = 256
    num_layers = 2
    gating = True
    gating_bias = 2.0
    window_mem = 128
    anneal_lr = False
    lr = 2e-4
    min_lr = 2e-6
    max_grad_norm = 1.0


def load_base_params(ckpt_path=CKPT17500):
    """Load weights-only from ckpt17500 via Henry's proven loader (orbax, GPU0), using a
    stub env + self-contained config so no P2-v1 launcher is imported. Returns
    (params, obs_dim, action_dim, emb, cfg)."""
    from dicode.utils.general.train_state_utils import load_weights_only
    cfg = _CompatCfg()
    stub_env = _StubEnv(NET_DIMS["action_dim"], OBS_DIM)
    ts = load_weights_only(ckpt_path, stub_env, None, cfg, load_opt_state=False)
    params = ts.params
    # Henry's loader returns the flax VARIABLES dict {"params": {...}} (its callers pass
    # it straight to network.apply). P2-Full-A convention is the INNER params dict (the
    # apply callables wrap with {"params": params}). Unwrap one level to match.
    if isinstance(params, dict) and "params" in params and isinstance(params["params"], dict):
        params = params["params"]
    return params, OBS_DIM, NET_DIMS["action_dim"], EMB, cfg


def build_network():
    """Build the frozen P2-Full-A ActorCriticTransformer (no params)."""
    from dicode.network import ActorCriticTransformer
    return ActorCriticTransformer(**NET_DIMS)


def build_callables(network, cfg=None):
    """Construct the three apply callables the combined update needs (mirrors
    fputil.build_net's closures, independent of param init)."""
    import memory_anchor as MA
    import full_p2_learner as FL
    from full_p2_learner import FullP2Config
    cfg = cfg or FullP2Config()

    def apply_eval_raw(params, memories, obs, mask):
        pi, value, mem_out = network.apply(
            {"params": params}, memories, obs, mask, method=network.model_forward_eval)
        return pi.logits, value, mem_out

    apply_eval_recon = jax.jit(MA.make_apply_eval(network))

    def _scan(params, memories, mem_mask, mem_idx, obs_seq):
        return FL._scan_lax(apply_eval_raw, params, memories, mem_mask, mem_idx,
                            obs_seq, cfg)
    scan_fn = jax.jit(_scan)
    return apply_eval_recon, apply_eval_raw, scan_fn


def zero_obs_fingerprint(network, params, obs_dim=OBS_DIM, batch=2):
    """Forward a zero-obs batch (mask all-ones, zero memory) and report value,
    entropy, top action + prob — the audited base fingerprint."""
    from full_p2_learner import FullP2Config
    cfg = FullP2Config()
    obs = jnp.zeros((batch, obs_dim), dtype=jnp.float32)
    mem = jnp.zeros((batch, cfg.window_mem, cfg.num_layers, cfg.embed), dtype=jnp.float32)
    mask = jnp.ones((batch, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    pi, value, _ = network.apply(
        {"params": params}, mem, obs, mask, method=network.model_forward_eval)
    probs = np.asarray(jax.nn.softmax(pi.logits, axis=-1)[0])
    top = int(np.argmax(probs))
    return {
        "value": float(np.asarray(value[0])),
        "entropy": float(np.asarray(pi.entropy()[0])),
        "top_action": top,
        "top_prob": float(probs[top]),
    }


def verify_fingerprint(network, params, obs_dim=OBS_DIM, strict=True):
    """Assert the loaded params match the audited base fingerprint (fail-closed)."""
    n = param_count(params)
    sha = params_content_sha256(params)
    fp = zero_obs_fingerprint(network, params, obs_dim=obs_dim)
    report = {"param_count": n, "params_sha256": sha, **fp}

    problems = []
    if n != EXPECTED_PARAM_COUNT:
        problems.append(f"param_count {n} != expected {EXPECTED_PARAM_COUNT}")
    if sha != EXPECTED_PARAMS_SHA256:
        problems.append(f"params_sha256 {sha} != expected {EXPECTED_PARAMS_SHA256}")
    if fp["top_action"] != EXPECTED_TOP_ACTION:
        problems.append(f"top_action {fp['top_action']} != {EXPECTED_TOP_ACTION}")
    if abs(fp["value"] - EXPECTED_ZERO_OBS_VALUE) > 1e-2:
        problems.append(f"value {fp['value']} != {EXPECTED_ZERO_OBS_VALUE} (±1e-2)")
    if abs(fp["entropy"] - EXPECTED_ZERO_OBS_ENTROPY) > 1e-2:
        problems.append(f"entropy {fp['entropy']} != {EXPECTED_ZERO_OBS_ENTROPY} (±1e-2)")
    if abs(fp["top_prob"] - EXPECTED_TOP_PROB) > 1e-2:
        problems.append(f"top_prob {fp['top_prob']} != {EXPECTED_TOP_PROB} (±1e-2)")

    report["ok"] = not problems
    report["problems"] = problems
    if problems and strict:
        raise RuntimeError("ckpt17500 fingerprint MISMATCH:\n  " + "\n  ".join(problems))
    return report


def _find_encoder_kernel(params):
    found = {}
    def walk(node, prefix):
        if hasattr(node, "items"):
            for k, v in node.items():
                walk(v, f"{prefix}/{k}")
        elif hasattr(node, "shape"):
            if "encoder" in prefix and prefix.endswith("kernel"):
                found["encoder"] = tuple(int(x) for x in node.shape)
            if prefix.endswith("actor_out/kernel"):
                found["actor_out"] = tuple(int(x) for x in node.shape)
    walk(params, "")
    return found


def compatible_init(ckpt_path=CKPT17500, strict=True):
    """Full compatible-init: load base weights, build P2-Full-A network + callables,
    verify the fingerprint, return everything the smoke launcher needs.

    Returns dict: network, params, target_params (EMA init == params copy),
    apply_eval_recon, apply_eval_raw, scan_fn, fingerprint, obs_dim, action_dim, emb,
    base_cfg, kernels.
    """
    params, obs_dim, action_dim, emb, base_cfg = load_base_params(ckpt_path)

    kernels = _find_encoder_kernel(params)
    enc = kernels.get("encoder")
    if enc != (int(obs_dim), NET_DIMS["encoder_size"]):
        raise RuntimeError(
            f"REFUSED: encoder kernel {enc} != expected "
            f"{(int(obs_dim), NET_DIMS['encoder_size'])}. Fail closed on schema mismatch.")
    ao = kernels.get("actor_out")
    if ao is not None and ao != (NET_DIMS["encoder_size"], NET_DIMS["action_dim"]):
        raise RuntimeError(f"REFUSED: actor_out kernel {ao} unexpected.")

    network = build_network()
    fp = verify_fingerprint(network, params, obs_dim=obs_dim, strict=strict)

    # EMA target initialised to the loaded online params (tau-stepped thereafter).
    target_params = jax.tree_util.tree_map(lambda x: x, params)

    apply_eval_recon, apply_eval_raw, scan_fn = build_callables(network)

    return {
        "network": network,
        "params": params,
        "target_params": target_params,
        "apply_eval_recon": apply_eval_recon,
        "apply_eval_raw": apply_eval_raw,
        "scan_fn": scan_fn,
        "fingerprint": fp,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "emb": emb,
        "base_cfg": base_cfg,
        "kernels": kernels,
        "source_checkpoint": ckpt_path,
    }
