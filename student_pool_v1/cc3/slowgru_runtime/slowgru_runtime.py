#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THIN_GTRXL128_SLOWGRU_RUNTIME — CC3 shared SlowGRU candidate runtime.

Task: CC3_CORRECTED_MATCHED_SLOWGRU_STUDENTS (+ inherited ABI from
CC3_OWN_SLOWGRU_AND_EVENTMEM_STUDENTS_END_TO_END section 五).

One SlowGRU core (server-side slowgru_network.py, SHA-pinned per contract);
the ONLY behavioral difference between candidates is the carry/reset policy:

  carry_mode=RESET128   -> at every 128-step segment boundary the SLOW longstate is
                           reset to init (fast window memories carry — exactly the
                           trainer's boundary-clear block semantics);
  carry_mode=PERSISTENT -> memory carries across segment boundaries; clearing happens
                           only on true episode done/reset (inside policy_step via the
                           true_done reset signal and via explicit reset_memory).

The two modes are NOT unified into the same reset behavior for interface convenience.

Unified ABI:
  load_candidate(checkpoint_contract) -> handle
  init_memory(handle, batch_size) -> memory_state
  policy_step(handle, observation, memory_state, done_mask, true_done=None)
      -> (action, memory_state_new, extras)
  reset_memory(handle, memory_state, reset_mask) -> memory_state_new
  on_segment_boundary(handle, memory_state) -> memory_state_new   # mode-dependent
  candidate_metadata(handle) -> dict

Fail-closed identity: load_candidate recomputes the file SHA and params SHA and
refuses to serve on any mismatch / non-finite params / wrong source SHA.
All network memory mechanics replicate the trainer driver `_env_step` verbatim
(mask_idx/mask update, forward_eval, memories roll).
"""
import hashlib
import os
import sys

import numpy as np

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
V7_SRC = V7 + "/src"

RUNTIME_NAME = "THIN_GTRXL128_SLOWGRU_RUNTIME"
ABI_VERSION = "cc3_runtime_abi/v1"

WINDOW_MEM = 128
NUM_LAYERS = 2
EMBED_SIZE = 256
NUM_HEADS = 8
OBS_DIM = 8335
ACTION_DIM = 43


# ----------------------------------------------------------------- hashing (driver-exact)
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_sha_packed(packed):
    leaves, _treedef = packed
    h = hashlib.sha256()
    for v in leaves:
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def leaf_hash_pytree(leaves):
    """Trainer `_leaf_hash` over a sequence of arrays (shape-prefixed)."""
    h = hashlib.sha256()
    for l in leaves:
        a = np.asarray(l)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------- ABI
def load_candidate(checkpoint_contract):
    """Load + identity-verify a SlowGRU checkpoint per its contract. Fail closed."""
    import pickle
    c = checkpoint_contract

    # 1. file identity
    pkl = c["checkpoint_path"]
    if not os.path.isfile(pkl):
        raise RuntimeError("CHECKPOINT_NOT_FOUND: %s" % pkl)
    file_sha = sha_file(pkl)
    if file_sha != c["checkpoint_file_sha256"]:
        raise RuntimeError("FILE_SHA_MISMATCH recomputed=%s contract=%s"
                           % (file_sha, c["checkpoint_file_sha256"]))

    # 2. family loader: trainer-identical sys.path, then unpickle
    arm_src = c["arm_src"]
    for p in (arm_src, V7_SRC, V7):
        if p not in sys.path:
            sys.path.insert(0, p)
    net_path = os.path.join(arm_src, c.get("network_module", "slowgru_network") + ".py")
    net_sha = sha_file(net_path)
    if net_sha != c["network_src_sha256"]:
        raise RuntimeError("NETWORK_SRC_SHA_MISMATCH disk=%s contract=%s"
                           % (net_sha, c["network_src_sha256"]))
    with open(pkl, "rb") as f:
        rd = pickle.load(f)

    # 3. params identity + finiteness
    params_sha = params_sha_packed(rd["params"])
    if params_sha != c["params_sha256"]:
        raise RuntimeError("PARAMS_SHA_MISMATCH recomputed=%s contract=%s"
                           % (params_sha, c["params_sha256"]))
    leaves, treedef = rd["params"]
    if not all(np.all(np.isfinite(np.asarray(v))) for v in leaves
               if np.asarray(v).dtype.kind in "fi"):
        raise RuntimeError("PARAMS_NON_FINITE")

    # 4. construction (frozen ctor from contract; action_dim asserted)
    import jax
    import jax.numpy as jnp
    from slowgru_network import ActorCriticSlowGRU, init_longstate, SLOW_INTERVAL, SLOW_DIM
    if SLOW_INTERVAL != 32 or SLOW_DIM != 256:
        raise RuntimeError("SLOW_CONSTANTS_DRIFT interval=%s dim=%s" % (SLOW_INTERVAL, SLOW_DIM))
    ctor = c["constructor"]
    assert int(ctor["action_dim"]) == ACTION_DIM, "action_dim must be 43"
    network = ActorCriticSlowGRU(
        action_dim=int(ctor["action_dim"]), activation=ctor["activation"],
        hidden_layers=int(ctor["hidden_layers"]), encoder_size=int(ctor["encoder_size"]),
        num_heads=int(ctor["num_heads"]), qkv_features=int(ctor["qkv_features"]),
        num_layers=int(ctor["num_layers"]), gating=bool(ctor["gating"]),
        gating_bias=float(ctor["gating_bias"]), use_longmem=bool(ctor["use_longmem"]))
    params = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(l) for l in leaves])

    forward_eval = jax.jit(
        lambda params, memories, obs, mask, ls, reset: network.apply(
            params, memories, obs, mask, ls, reset, method=network.forward_eval))

    return dict(
        runtime=RUNTIME_NAME,
        candidate_id=c["candidate_id"],
        carry_mode=c["carry_mode"],
        contract=c,
        network=network,
        forward_eval=forward_eval,
        init_longstate=init_longstate,
        params=params,
        params_sha256=params_sha,
        file_sha256=file_sha,
        rng=None,   # seeded at first policy_step / by seed_policy_rng
    )


def seed_policy_rng(handle, seed):
    import jax
    handle["rng"] = jax.random.PRNGKey(int(seed))
    return handle


def init_memory(handle, batch_size):
    import jax.numpy as jnp
    B = int(batch_size)
    return dict(
        memories=jnp.zeros((B, WINDOW_MEM, NUM_LAYERS, EMBED_SIZE)),
        memories_mask=jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), dtype=jnp.bool_),
        memories_mask_idx=jnp.zeros((B,), dtype=jnp.int32) + (WINDOW_MEM + 1),
        longstate=handle["init_longstate"](B),
        true_done=jnp.zeros((B,), dtype=jnp.bool_),
        step_idx=0,
    )


def _longstate_leaves(ls):
    return [ls["h"], ls["buf"], ls["count"]]


def longstate_leaf_hash(ls):
    return leaf_hash_pytree(_longstate_leaves(ls))


def policy_step(handle, observation, memory_state, done_mask, true_done=None):
    """One env step, replicating the trainer `_env_step` memory mechanics verbatim.

    done_mask : (B,) bool — wrapper-level done used for the fast-memory mask logic.
    true_done : (B,) bool — returned_episode of THIS step; stored and used as the slow
                reset signal from the NEXT step on (trainer carry semantics). If None,
                the carried true_done is reused as the reset signal and kept as-is.
    """
    import jax
    import jax.numpy as jnp
    ms = memory_state
    B = ms["memories_mask_idx"].shape[0]
    if handle["rng"] is None:
        raise RuntimeError("policy rng not seeded: call seed_policy_rng(handle, seed)")

    # mask mechanics (driver lines 235-241)
    done = jnp.asarray(done_mask)
    mask_idx = jnp.where(done, WINDOW_MEM, jnp.clip(ms["memories_mask_idx"] - 1, 0, WINDOW_MEM))
    mask = jnp.where(done[:, None, None, None],
                     jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), dtype=jnp.bool_),
                     ms["memories_mask"])
    ohot = jax.nn.one_hot(mask_idx, WINDOW_MEM + 1)[:, None, None, :].repeat(NUM_HEADS, 1)
    mask = jnp.logical_or(mask, ohot)

    # trainer carry semantics: the reset signal of this step is the PREVIOUS step's
    # returned_episode; the caller passes it via true_done. If omitted, reuse carried.
    reset_signal = (jnp.asarray(true_done) if true_done is not None else ms["true_done"])
    pi, value, mem_out, ls_new = handle["forward_eval"](
        handle["params"], ms["memories"], jnp.asarray(observation), mask,
        ms["longstate"], reset_signal)

    handle["rng"], _rng = jax.random.split(handle["rng"])
    action = pi.sample(seed=_rng)
    logits = pi.logits
    memories = jnp.roll(ms["memories"], -1, axis=1).at[:, -1].set(mem_out)

    ms_new = dict(
        memories=memories,
        memories_mask=mask,
        memories_mask_idx=mask_idx,
        longstate=ls_new,
        true_done=reset_signal,
        step_idx=ms["step_idx"] + 1,
    )
    extras = dict(logits=logits, value=value, action=action)
    return action, ms_new, extras


def reset_memory(handle, memory_state, reset_mask):
    """Selective per-env reset (env-level reset semantics). Honored in BOTH modes."""
    import jax.numpy as jnp
    ms = memory_state
    r = jnp.asarray(reset_mask)
    B = r.shape[0]
    init_ls = handle["init_longstate"](B)
    return dict(
        memories=jnp.where(r[:, None, None, None],
                           jnp.zeros_like(ms["memories"]), ms["memories"]),
        memories_mask=jnp.where(r[:, None, None, None],
                                jnp.zeros_like(ms["memories_mask"]), ms["memories_mask"]),
        memories_mask_idx=jnp.where(r, jnp.full_like(ms["memories_mask_idx"], WINDOW_MEM + 1),
                                    ms["memories_mask_idx"]),
        longstate={k: jnp.where(r[:, None] if v.ndim == 2 else (r[:, None, None] if v.ndim == 3
                                 else r), jnp.asarray(iv), v)
                   for (k, v), iv in zip(ms["longstate"].items(), _longstate_leaves(init_ls))},
        true_done=jnp.zeros_like(ms["true_done"]),
        step_idx=ms["step_idx"],
    )


def on_segment_boundary(handle, memory_state):
    """Mode-dependent 128-step segment boundary behavior (the ONLY intentional
    behavioral divergence between the two candidates).

    RESET128   : slow longstate -> init (fast window memories carry; matches the
                 trainer boundary-clear block exactly).
    PERSISTENT : full carry; nothing is forced to zero at the boundary.
    """
    ms = memory_state
    if handle["carry_mode"] == "RESET128":
        B = ms["memories_mask_idx"].shape[0]
        ms = dict(ms)
        ms["longstate"] = handle["init_longstate"](B)
        return ms, dict(boundary_action="LONGSTATE_RESET_TO_INIT", fast_memories="CARRIED")
    if handle["carry_mode"] == "PERSISTENT":
        return ms, dict(boundary_action="FULL_CARRY_NO_CLEAR")
    raise RuntimeError("UNKNOWN_CARRY_MODE %s" % handle["carry_mode"])


def params_sha(handle):
    import jax
    import jax.numpy as jnp
    leaves = jax.tree_util.tree_leaves(handle["params"])
    h = hashlib.sha256()
    for v in leaves:
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def candidate_metadata(handle):
    c = handle["contract"]
    return dict(
        runtime=RUNTIME_NAME,
        abi_version=ABI_VERSION,
        candidate_id=handle["candidate_id"],
        network_family="SlowGRU (GTrXL128 + zero-init additive slow-GRU actor channel)",
        carry_mode=handle["carry_mode"],
        network_src_sha256=c["network_src_sha256"],
        params_sha256=handle["params_sha256"],
        checkpoint_file_sha256=handle["file_sha256"],
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        window_mem=WINDOW_MEM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        slow_interval=32,
        slow_dim=256,
        constructor=c["constructor"],
        boundary_semantics=("RESET128: slow longstate cleared to init at every 128-step "
                            "segment boundary (fast memories carry)"
                            if handle["carry_mode"] == "RESET128" else
                            "PERSISTENT: full cross-segment carry; clears only on true "
                            "episode done/reset"),
    )
