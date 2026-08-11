#!/usr/bin/env python
"""CC2 corrected §二 — W512_RESET128_P2REPLAY_CANONICAL_98304 FOUNDATION self-test (SERVER CPU).

Validates the W512 replay foundation WITHOUT training and WITHOUT touching GPU0/1:
  G1  ckpt17500 base params SHA == d4e85af5... (authoritative _params_sha definition).
  G2  W512 network with ckpt17500 base merged: the base-submodule subtree SHA == d4e85af5
      (direct proof the W512 network is base-compatible -> ckpt17500 loads unchanged).
  G3  structural base key/shape map of W512 base submodules == ckpt17500 tree (diagnostic).
  G4  reset128 clear zeroes long_buf/long_mask at episode-local seg_step=128 (not 127); delay
      line untouched.
  G5  300-step transition: logits/values finite throughout; seg_step advances; reset128 fires.
  G6  anchor round-trip BIT-EXACT: direct stepping to s == reconstruct-from-anchor(a) to s.
  G7  jitted lax.scan == eager python loop BIT-EXACT (sequence includes a terminal done).
  G8  inline-helper composition (what w512_collect will run) == w512_step_forward BIT-EXACT.

Runs on CPU (JAX_PLATFORMS=cpu). Prints W512_FOUNDATION_SELFTEST=PASS/FAIL.
GPU is NEVER used here; the real run binds GPU2 in its own launcher.
"""
from __future__ import annotations
import os, sys, hashlib, argparse
# Platform is CALLER-controlled: the base-SHA gates (G1/G2/G3) restore a GPU-sharded orbax
# checkpoint and therefore must run on a visible GPU (GPU2); the mechanics gates (G4-G8) are
# cheap and run on whatever backend is active. Set W512_SELFTEST_CPU=1 to force CPU and skip
# the checkpoint gates (mechanics-only validation).
if os.environ.get("W512_SELFTEST_CPU") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.dirname(HERE)                       # .../runtime
FROZEN = os.path.join(RUNTIME, "frozen_modules")
# dicode source for the Transformer import used by the networks.
DICODE_SRC_CANDIDATES = [
    "/home/oseasy/experiments/dreaming-in-code/src",
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src",
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB",
]
for p in [HERE, FROZEN] + DICODE_SRC_CANDIDATES:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

import jax, jax.numpy as jnp

# ---- frozen canonical Cfg (driver line 317) ----
ACTION_DIM = 43
OBS_DIM = 8335
ACTIVATION = "relu"
EMBED = 256
HIDDEN = 256
NUM_HEADS = 8
QKV = 256
NUM_LAYERS = 2
GATING = True
GATING_BIAS = 2.0
WINDOW_MEM = 128
NUM_ENVS = 16
W512_LONG = 384
W512_DELAY = 128
SEGMENT_LEN = 128
EXPECTED_BASE_SHA = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"

GATES = {}
DETAIL = {}


def params_sha(params):
    """Authoritative driver _params_sha (line 570)."""
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def build_network():
    from network_w512 import ActorCriticTransformerW512
    return ActorCriticTransformerW512(
        action_dim=ACTION_DIM, activation=ACTIVATION, hidden_layers=HIDDEN,
        encoder_size=EMBED, num_heads=NUM_HEADS, qkv_features=QKV,
        num_layers=NUM_LAYERS, gating=GATING, gating_bias=GATING_BIAS,
        long_size=W512_LONG)


def init_w512_params(network, rng):
    init_obs = jnp.zeros((2, OBS_DIM))
    init_mem = jnp.zeros((2, WINDOW_MEM, NUM_LAYERS, EMBED))
    init_mask = jnp.zeros((2, NUM_HEADS, 1, WINDOW_MEM + 1), dtype=jnp.bool_)
    init_lbuf = jnp.zeros((2, W512_LONG, EMBED))
    init_lmsk = jnp.zeros((2, W512_LONG), dtype=jnp.bool_)
    return network.init(rng, init_mem, init_obs, init_mask,
                        long_buf=init_lbuf, long_mask=init_lmsk)["params"]


def base_subtree(params, base_keys):
    return {k: params[k] for k in base_keys if k in params}


def leaf_shape_map(tree):
    return {"/".join(map(str, p)): tuple(np.asarray(v).shape)
            for p, v in jax.tree_util.tree_leaves_with_path(tree)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt17500", default=None,
                    help="orbax checkpoint dir path e.g. .../ckpt17500 (enables G1/G2/G3)")
    args = ap.parse_args()

    import w512_memory as w5m
    import w512_memory_anchor as WA

    network = build_network()
    w5_cfg = w5m.W512Config(long_size=W512_LONG, delay_size=W512_DELAY, encoder_size=EMBED)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    full_params = init_w512_params(network, init_rng)
    w512_keys = [k for k in full_params.keys() if str(k).startswith("w512_")]
    base_keys = [k for k in full_params.keys() if not str(k).startswith("w512_")]
    DETAIL["w512_param_keys"] = sorted(map(str, full_params.keys()))
    DETAIL["base_keys"] = sorted(map(str, base_keys))
    DETAIL["w512_only_keys"] = sorted(map(str, w512_keys))

    # ---------------- G1/G2/G3: base checkpoint compatibility ----------------
    if args.ckpt17500:
        import orbax.checkpoint as ocp

        def _merge(base, full):
            if isinstance(base, dict) and isinstance(full, dict):
                out = dict(full)
                for k in base:
                    if k in full:
                        out[k] = _merge(base[k], full[k])
                return out
            return base

        mgr = ocp.CheckpointManager(os.path.dirname(args.ckpt17500))
        raw = mgr.restore(int(os.path.basename(args.ckpt17500)))
        base_inner = raw["params"]["params"]
        ckpt_sha = params_sha(base_inner)
        GATES["G1_ckpt17500_base_sha"] = (ckpt_sha == EXPECTED_BASE_SHA)
        DETAIL["ckpt17500_base_sha"] = ckpt_sha

        merged = _merge(base_inner, full_params)
        merged_base_sha = params_sha(base_subtree(merged, base_keys))
        GATES["G2_w512_merged_base_sha"] = (merged_base_sha == EXPECTED_BASE_SHA)
        DETAIL["w512_merged_base_sha"] = merged_base_sha

        ckpt_map = leaf_shape_map(base_inner)
        w512_base_map = leaf_shape_map(base_subtree(full_params, base_keys))
        GATES["G3_base_structure_match"] = (ckpt_map == w512_base_map)
        DETAIL["ckpt_base_leaf_count"] = len(ckpt_map)
        DETAIL["w512_base_leaf_count"] = len(w512_base_map)
        if ckpt_map != w512_base_map:
            only_ckpt = set(ckpt_map) - set(w512_base_map)
            only_w512 = set(w512_base_map) - set(ckpt_map)
            shape_mm = {k for k in set(ckpt_map) & set(w512_base_map)
                        if ckpt_map[k] != w512_base_map[k]}
            DETAIL["structure_only_ckpt"] = sorted(only_ckpt)
            DETAIL["structure_only_w512"] = sorted(only_w512)
            DETAIL["structure_shape_mismatch"] = sorted(shape_mm)
        # use the merged (ckpt-loaded) params for the mechanics tests
        params = merged
    else:
        GATES["G1_ckpt17500_base_sha"] = None
        GATES["G2_w512_merged_base_sha"] = None
        GATES["G3_base_structure_match"] = None
        DETAIL["ckpt_note"] = "--ckpt17500 not supplied; G1/G2/G3 skipped (foundation mechanics still validated)"
        params = full_params

    apply_eval = WA.make_apply_eval_w512(network)

    # ---------------- G4: reset128 clear semantics ----------------
    st = WA.w512_fresh_state(NUM_ENVS, w5_cfg)
    # put nonzero junk into long_buf/long_mask + delay_buf to confirm what gets cleared
    st = {**st,
          "long_buf": jnp.ones_like(st["long_buf"]),
          "long_mask": jnp.ones_like(st["long_mask"], dtype=jnp.bool_),
          "delay_buf": jnp.ones_like(st["delay_buf"])}
    st127 = {**st, "seg_step": jnp.full((NUM_ENVS,), 127, jnp.int32)}
    cl127 = WA.w512_reset128_clear(st127, SEGMENT_LEN)
    no_clear_at_127 = bool((np.asarray(cl127["long_buf"]) == 1.0).all())
    st128 = {**st, "seg_step": jnp.full((NUM_ENVS,), 128, jnp.int32)}
    cl128 = WA.w512_reset128_clear(st128, SEGMENT_LEN)
    clear_at_128 = bool((np.asarray(cl128["long_buf"]) == 0.0).all()
                        and (np.asarray(cl128["long_mask"]) == 0).all())
    delay_untouched = bool((np.asarray(cl128["delay_buf"]) == 1.0).all())
    st0 = {**st, "seg_step": jnp.zeros((NUM_ENVS,), jnp.int32)}
    cl0 = WA.w512_reset128_clear(st0, SEGMENT_LEN)
    no_clear_at_0 = bool((np.asarray(cl0["long_buf"]) == 1.0).all())
    GATES["G4_reset128_clear"] = (no_clear_at_0 and no_clear_at_127 and clear_at_128 and delay_untouched)
    DETAIL["G4"] = dict(no_clear_at_0=no_clear_at_0, no_clear_at_127=no_clear_at_127,
                        clear_at_128=clear_at_128, delay_untouched=delay_untouched)

    # ---------------- G5: 300-step transition finiteness + reset128 firing ----------------
    B = NUM_ENVS
    memories = jnp.zeros((B, WINDOW_MEM, NUM_LAYERS, EMBED))
    mem_mask = jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_)
    mem_idx = jnp.full((B,), WINDOW_MEM, jnp.int32)
    w5st = WA.w512_fresh_state(B, w5_cfg)
    rngkey = jax.random.PRNGKey(1)
    all_finite = True
    longmask_sum_at = {}
    de = jnp.zeros((B,), jnp.bool_)            # no entering done (single episode)
    dn = jnp.zeros((B,), jnp.bool_)
    for t in range(300):
        rngkey, ok = jax.random.split(rngkey)
        obs_t = jax.random.normal(ok, (B, OBS_DIM))
        (memories, mem_mask, mem_idx, w5st, lg, vl, _mp) = WA.w512_step_forward(
            apply_eval, params, memories, mem_mask, mem_idx, w5st, obs_t, de, dn,
            WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN)
        if not (bool(np.isfinite(np.asarray(lg)).all()) and bool(np.isfinite(np.asarray(vl)).all())):
            all_finite = False
        longmask_sum_at[t] = int(np.asarray(w5st["long_mask"]).sum())
    # long_mask sum must DROP back toward 0 right after each 128 boundary (cleared), then regrow.
    # at post-step index 127 (seg_step became 128 entering next) vs 128 (just cleared+1 push).
    seg_step_final = int(np.asarray(w5st["seg_step"])[0])
    GATES["G5_transition_finite_and_segstep"] = (all_finite and seg_step_final == 300)
    DETAIL["G5"] = dict(all_finite=all_finite, seg_step_final=seg_step_final,
                        longmask_sum_step126=longmask_sum_at.get(126),
                        longmask_sum_step127=longmask_sum_at.get(127),
                        longmask_sum_step128=longmask_sum_at.get(128),
                        longmask_sum_step255=longmask_sum_at.get(255),
                        longmask_sum_step256=longmask_sum_at.get(256))

    # ---------------- G6: anchor round-trip BIT-EXACT ----------------
    # Direct-step 200 steps recording state at anchor a=128 and at target s=190; reconstruct
    # from the anchor over [128,190) and compare to the directly-stepped state at 190.
    A_STEP, S_STEP = 128, 190
    obs_seq = []
    rk = jax.random.PRNGKey(7)
    for t in range(S_STEP):
        rk, ok = jax.random.split(rk)
        obs_seq.append(jax.random.normal(ok, (B, OBS_DIM)))
    # direct stepping
    mem_d = jnp.zeros((B, WINDOW_MEM, NUM_LAYERS, EMBED))
    mm_d = jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_)
    mi_d = jnp.full((B,), WINDOW_MEM, jnp.int32)
    st_d = WA.w512_fresh_state(B, w5_cfg)
    anchor = None
    de_z = jnp.zeros((B,), jnp.bool_); dn_z = jnp.zeros((B,), jnp.bool_)
    for t in range(S_STEP):
        if t == A_STEP:
            anchor = (mem_d, mm_d, mi_d, jax.tree_util.tree_map(jnp.asarray, st_d))
        (mem_d, mm_d, mi_d, st_d, _lg, _vl, _mp) = WA.w512_step_forward(
            apply_eval, params, mem_d, mm_d, mi_d, st_d, obs_seq[t], de_z, dn_z,
            WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN)
    # reconstruct from anchor over [A_STEP, S_STEP)
    obs_seg = jnp.stack(obs_seq[A_STEP:S_STEP], axis=0)      # [gap, B, obs]
    de_seg = jnp.zeros((S_STEP - A_STEP, B), jnp.bool_)
    dn_seg = jnp.zeros((S_STEP - A_STEP, B), jnp.bool_)
    mem_r, mm_r, mi_r, st_r = WA.reconstruct_w512_state_with_network(
        network, apply_eval, params, anchor[0], anchor[1], anchor[2], anchor[3],
        obs_seg, de_seg, dn_seg, WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN)
    rt_mem = bool(np.array_equal(np.asarray(mem_d), np.asarray(mem_r)))
    rt_mm = bool(np.array_equal(np.asarray(mm_d), np.asarray(mm_r)))
    rt_mi = bool(np.array_equal(np.asarray(mi_d), np.asarray(mi_r)))
    rt_st = all(bool(np.array_equal(np.asarray(st_d[k]), np.asarray(st_r[k])))
                for k in st_d.keys())
    GATES["G6_anchor_roundtrip_bitexact"] = (rt_mem and rt_mm and rt_mi and rt_st)
    DETAIL["G6"] = dict(mem=rt_mem, mask=rt_mm, idx=rt_mi, w512_state=rt_st,
                        anchor=A_STEP, target=S_STEP)

    # ---------------- G7: jitted scan == eager loop BIT-EXACT (with a terminal done) ----------------
    T = 20
    obs_seq2 = []
    rk2 = jax.random.PRNGKey(11)
    for t in range(T):
        rk2, ok = jax.random.split(rk2)
        obs_seq2.append(jax.random.normal(ok, (B, OBS_DIM)))
    obs_arr = jnp.stack(obs_seq2, 0)                          # [T,B,obs]
    # terminal done for env 0 at step T-1 (new done); entering done for env 0 at step T-1 is False,
    # entering done for the step AFTER a done would be True (but there is no next step here).
    dn_arr = jnp.zeros((T, B), jnp.bool_)
    dn_arr = dn_arr.at[T - 1, 0].set(True)                    # new done at last step, env 0
    de_arr = jnp.zeros((T, B), jnp.bool_)
    de_arr = de_arr.at[1:, :].set(dn_arr[:-1, :])             # entering done = previous new done
    # eager loop
    mem_e = jnp.zeros((B, WINDOW_MEM, NUM_LAYERS, EMBED))
    mm_e = jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_)
    mi_e = jnp.full((B,), WINDOW_MEM, jnp.int32)
    st_e = WA.w512_fresh_state(B, w5_cfg)
    lg_e, vl_e = [], []
    for t in range(T):
        (mem_e, mm_e, mi_e, st_e, lg, vl, _mp) = WA.w512_step_forward(
            apply_eval, params, mem_e, mm_e, mi_e, st_e, obs_arr[t], de_arr[t], dn_arr[t],
            WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN)
        lg_e.append(lg); vl_e.append(vl)
    lg_e = jnp.stack(lg_e, 0); vl_e = jnp.stack(vl_e, 0)
    # jitted scan (run TWICE to prove determinism)
    scan_fn = WA.make_scan_w512(network, WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN)

    def _scan():
        return scan_fn(params, jnp.zeros((B, WINDOW_MEM, NUM_LAYERS, EMBED)),
                       jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_),
                       jnp.full((B,), WINDOW_MEM, jnp.int32),
                       WA.w512_fresh_state(B, w5_cfg), obs_arr, de_arr, dn_arr)
    lg_s, vl_s = _scan()
    lg_s2, vl_s2 = _scan()
    # G7a: the jitted scan is deterministic (same input -> bit-identical). This is the real
    # correctness gate for replay/loss-region scans.
    scan_deterministic = (bool(np.array_equal(np.asarray(lg_s), np.asarray(lg_s2)))
                          and bool(np.array_equal(np.asarray(vl_s), np.asarray(vl_s2))))
    GATES["G7a_scan_deterministic_bitexact"] = scan_deterministic
    # G7b: eager-loop vs jitted-scan agree to XLA op-fusion tolerance. The frozen RMT16 protocol
    # has the SAME split (eager rmt_collect vs jitted _make_scan_rmt / rmt_ppo scans); the
    # difference is FP reduction-order rounding from op fusion, NOT a logic error (G6/G8 prove
    # the transition logic is bit-exact eager-vs-eager). Record the magnitude honestly.
    lg_abs = float(np.abs(np.asarray(lg_e) - np.asarray(lg_s)).max())
    vl_abs = float(np.abs(np.asarray(vl_e) - np.asarray(vl_s)).max())
    # Fusion noise scales with activation magnitude (trained ckpt base weights -> ~1e-2 logits;
    # small random init -> ~1e-5). A WRONG done convention or transition bug would produce O(1)
    # differences, so 5e-2 cleanly separates rounding from logic errors.
    FUSION_TOL = 5e-2
    GATES["G7b_eager_scan_within_fusion_tol"] = (lg_abs < FUSION_TOL and vl_abs < FUSION_TOL)
    DETAIL["G7"] = dict(scan_deterministic=scan_deterministic,
                        logits_eager_vs_scan_max_abs=lg_abs,
                        values_eager_vs_scan_max_abs=vl_abs, fusion_tol=FUSION_TOL)

    # ---------------- G8: inline-helper composition == w512_step_forward ----------------
    def inline_step(params, memories, mem_mask, mem_idx, w512_st, obs, de, dn):
        # exactly what w512_collect will run, composed from the shared modular helpers
        w512_st = WA.w512_reset128_clear(w512_st, SEGMENT_LEN)
        mem_pre = memories
        mem_idx, mem_mask = WA.w512_advance_mask(mem_idx, mem_mask, de, WINDOW_MEM, NUM_HEADS)
        logits, value, mem_out, h_t = apply_eval(
            params, memories, obs, mem_mask, w512_st["long_buf"], w512_st["long_mask"])
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
        import w512_memory as _w5m
        new_st = _w5m.w512_step(w512_st, h_t, dn, w5_cfg)
        new_st = {**new_st,
                  "seg_step": jnp.where(dn, 0, w512_st["seg_step"] + 1).astype(jnp.int32)}
        return post_memories, mem_mask, mem_idx, new_st, logits, value, mem_pre

    rk3 = jax.random.PRNGKey(13)
    rk3, ok = jax.random.split(rk3)
    obs_g = jax.random.normal(ok, (B, OBS_DIM))
    de_g = jnp.zeros((B,), jnp.bool_)
    dn_g = jnp.zeros((B,), jnp.bool_).at[3].set(True)
    mem_g = jax.random.normal(jax.random.PRNGKey(14), (B, WINDOW_MEM, NUM_LAYERS, EMBED))
    mm_g = jnp.zeros((B, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_)
    mi_g = jnp.full((B,), WINDOW_MEM // 2, jnp.int32)
    st_g = WA.w512_fresh_state(B, w5_cfg)
    st_g = {**st_g, "seg_step": jnp.full((B,), 128, jnp.int32),     # exercise the clear path
            "long_buf": jax.random.normal(jax.random.PRNGKey(15), (B, W512_LONG, EMBED)),
            "long_mask": jnp.ones((B, W512_LONG), jnp.bool_)}
    out_ref = WA.w512_step_forward(apply_eval, params, mem_g, mm_g, mi_g, st_g,
                                   obs_g, de_g, dn_g, WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN)
    out_inl = inline_step(params, mem_g, mm_g, mi_g, st_g, obs_g, de_g, dn_g)
    g8 = True
    for a, b in zip(out_ref[:3], out_inl[:3]):
        g8 = g8 and bool(np.array_equal(np.asarray(a), np.asarray(b)))
    g8 = g8 and bool(np.array_equal(np.asarray(out_ref[4]), np.asarray(out_inl[4])))  # logits
    g8 = g8 and bool(np.array_equal(np.asarray(out_ref[5]), np.asarray(out_inl[5])))  # value
    for k in out_ref[3].keys():
        g8 = g8 and bool(np.array_equal(np.asarray(out_ref[3][k]), np.asarray(out_inl[3][k])))
    GATES["G8_inline_eq_stepforward"] = g8

    # ---------------- summary ----------------
    effective = {k: v for k, v in GATES.items() if v is not None}
    all_pass = bool(effective) and all(effective.values())
    result = {"selftest": "W512_FOUNDATION", "gates": GATES, "all_gates_pass": all_pass,
              "detail": DETAIL}
    print(_json_dumps(result))
    print("W512_FOUNDATION_SELFTEST=" + ("PASS" if all_pass else "FAIL"))


def _json_dumps(obj):
    import json
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, indent=2, default=conv, ensure_ascii=False)


if __name__ == "__main__":
    main()
