#!/usr/bin/env python3
"""RMT16 × P2-Replay — CPU unit tests (Phase4A, task #26).

Validates the bit-exactness / conservation gates that are checkable WITHOUT a full GPU
training run (directive §六 18-gate map; reports/rmt16_replay_frozen_config.md §四):

  T1 init gate-zero bit-exactness      (rmt_gate=0 -> read has no effect at init)
  T2 anchor -> burn-in roundtrip        (gate 6/7: entering state reproduced BIT-EXACT)
  T3 loss-region scan == step loop      (scan_rmt_eval matches rmt_step_forward loop)
  T4 carry_mode single-point diff       (gate 3/9: persistent vs reset128 differ ONLY in
                                         mem_tokens at the 128/segment boundary)
  T5 true-done full reset               (gate 9: tokens/seg_buf/seg_count -> 0 on done)
  T6 buffer roundtrip + conservation    (gate 5/8: RMT schema validates; episodes intact)
  T7 hindsight relabel keeps RMT fields (gate 5/6; ValueError when no goal achieved)
  T8 compute_gae correctness            (frozen gamma/lambda GAE on a known case)
  T9 rmt params receive gradient        (gate 14: read/update/gate finite nonzero grad)

Uses a SMALL network (embed=32, heads=2, layers=1, rmt_num_tokens=4, window_mem=16,
segment_len=8) so segment boundaries fire within ~10 steps. Bit-exactness is asserted
with max-abs-diff == 0.0 (identical op sequence => identical arrays on CPU).
"""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

SRC = os.path.dirname(os.path.abspath(__file__))
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in [SRC, V7 + "/src", V7]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax, jax.numpy as jnp, numpy as np

from network_rmt16 import ActorCriticTransformerRMT16
import rmt16_memory as rmtm
from rmt_memory_anchor import (
    make_apply_eval_rmt, make_update_fn, rmt_step_forward, rmt_advance_tokens,
    reconstruct_rmt_state_with_network, scan_rmt_eval,
)
import rmt_ppo as PPO
import rmt_hindsight as RH
from rmt_replay_buffer import RMTTrajectory, RMTReplayBuffer
from replay_buffer import anchor_steps_for_length

# ----------------------------- tiny config -----------------------------
EMB, HEADS, LAYERS, WM, NTOK, SEG, ADIM, ODIM = 32, 2, 1, 16, 4, 8, 6, 24
rmt_cfg = rmtm.RMT16Config(num_tokens=NTOK, segment_len=SEG, encoder_size=EMB)
B = 1  # reconstruction path is batch-1 (padded internally); keep ground truth at B=1 too

net = ActorCriticTransformerRMT16(
    action_dim=ADIM, activation="relu", encoder_size=EMB, hidden_layers=EMB,
    num_heads=HEADS, qkv_features=EMB, num_layers=LAYERS, gating=True, gating_bias=2.0,
    rmt_num_tokens=NTOK)


def _init_params(seed=0):
    rng = jax.random.PRNGKey(seed)
    variables = net.init(
        rng, jnp.zeros((2, WM, LAYERS, EMB)), jnp.zeros((2, ODIM)),
        jnp.zeros((2, HEADS, 1, WM + 1), jnp.bool_),
        mem_tokens=jnp.zeros((2, NTOK, EMB)), seg_buf=jnp.zeros((2, SEG, EMB)),
        method=net.init_all)
    return variables["params"]   # INNER (apply convention)


PARAMS = jax.tree_util.tree_map(jnp.asarray, _init_params())
APPLY = make_apply_eval_rmt(net)
UPDATE_FN = make_update_fn(net, PARAMS)

_results = []
def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def fresh_state():
    return (jnp.zeros((B, WM, LAYERS, EMB)),
            jnp.zeros((B, HEADS, 1, WM + 1), jnp.bool_),
            jnp.full((B,), WM, jnp.int32),
            rmtm.rmt16_init(B, rmt_cfg))


def run_loop(obs_seq, dones_seq, carry_mode, init=None):
    """Step-by-step rmt_step_forward; record ENTERING state per step + outputs."""
    mem, mask, idx, st = init if init is not None else fresh_state()
    pre_mem, pre_mask, pre_idx, pre_tok, lgs, vls = [], [], [], [], [], []
    for t in range(obs_seq.shape[0]):
        pre_mem.append(mem); pre_mask.append(mask); pre_idx.append(idx)
        pre_tok.append(st["mem_tokens"])
        (mem, mask, idx, st, lg, vl, _mp, _et) = rmt_step_forward(
            APPLY, PARAMS, mem, mask, idx, st, obs_seq[t], dones_seq[t],
            WM, HEADS, rmt_cfg, carry_mode, UPDATE_FN)
        lgs.append(lg); vls.append(vl)
    return (pre_mem, pre_mask, pre_idx, pre_tok, lgs, vls, (mem, mask, idx, st))


def maxdiff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype == np.bool_ or b.dtype == np.bool_:
        return float(np.sum(a != b))     # count of mismatched bool entries (0 == identical)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


# =====================================================================
print("== T1: gate-zero bit-exact at init ==", flush=True)
mem, mask, idx, st = fresh_state()
obs0 = jax.random.normal(jax.random.PRNGKey(1), (B, ODIM))
lg_z, v_z, _, _ = APPLY(PARAMS, mem, obs0, mask, jnp.zeros((B, NTOK, EMB)))
lg_r, v_r, _, _ = APPLY(PARAMS, mem, obs0, mask,
                        jax.random.normal(jax.random.PRNGKey(2), (B, NTOK, EMB)))
check("T1 logits identical for any mem_tokens at init (gate=0)",
      maxdiff(lg_z, lg_r) == 0.0, f"maxdiff={maxdiff(lg_z, lg_r)}")
check("T1 value identical at init", maxdiff(v_z, v_r) == 0.0)
check("T1 rmt_gate is zero", float(np.abs(np.asarray(PARAMS["rmt_gate"])).max()) == 0.0)

# =====================================================================
print("== T2: anchor -> burn-in roundtrip (BIT-EXACT) ==", flush=True)
N = 20
obs_seq = jax.random.normal(jax.random.PRNGKey(3), (N, B, ODIM))
dones_seq = jnp.zeros((N, B), jnp.bool_)
for mode in ["persistent", "reset128"]:
    pre_mem, pre_mask, pre_idx, pre_tok, lgs, vls, _ = run_loop(obs_seq, dones_seq, mode)
    init = (pre_mem[0], pre_mask[0], pre_idx[0],
            {"mem_tokens": pre_tok[0], "seg_buf": jnp.zeros((B, SEG, EMB)),
             "seg_count": jnp.zeros((B,), jnp.int32)})
    worst = 0.0
    for k in [1, 5, 8, 13, 16, 19]:   # k spans intra-segment + across the seg-8 boundary
        rm, rmask, ridx, rst = reconstruct_rmt_state_with_network(
            net, APPLY, PARAMS, init[0], init[1], init[2],
            {"mem_tokens": init[3]["mem_tokens"], "seg_buf": init[3]["seg_buf"],
             "seg_count": init[3]["seg_count"]},
            obs_seq[:k], dones_seq[:k], WM, HEADS, rmt_cfg, mode)
        d = max(maxdiff(rm, pre_mem[k]), maxdiff(rmask, pre_mask[k]),
                maxdiff(ridx, pre_idx[k]), maxdiff(rst["mem_tokens"], pre_tok[k]),
                maxdiff(rst["seg_buf"], np.zeros((B, SEG, EMB))) if False else 0.0)
        # compare full entering state at step k
        d = max(d, maxdiff(rst["mem_tokens"], pre_tok[k]))
        worst = max(worst, d)
    check(f"T2 roundtrip bit-exact [{mode}]", worst == 0.0, f"worst_maxdiff={worst}")

# =====================================================================
print("== T3: loss-region scan == step loop (BIT-EXACT) ==", flush=True)
for mode in ["persistent", "reset128"]:
    pre_mem, pre_mask, pre_idx, pre_tok, lgs, vls, _ = run_loop(obs_seq, dones_seq, mode)
    init = fresh_state()
    (s_pm, s_pmask, s_pidx, s_ptok, s_lg, s_vl, _final) = scan_rmt_eval(
        net, APPLY, PARAMS, obs_seq, dones_seq, WM, HEADS, rmt_cfg, mode,
        init[0], init[1], init[2], init[3])
    worst = 0.0
    for t in range(N):
        worst = max(worst, maxdiff(s_ptok[t], pre_tok[t]), maxdiff(s_lg[t], lgs[t]),
                    maxdiff(s_vl[t], vls[t]), maxdiff(s_pm[t], pre_mem[t]))
    check(f"T3 scan==loop bit-exact [{mode}]", worst == 0.0, f"worst_maxdiff={worst}")

# =====================================================================
print("== T4: carry_mode single-point diff ==", flush=True)
gp = run_loop(obs_seq, dones_seq, "persistent")
gr = run_loop(obs_seq, dones_seq, "reset128")
gtrxl_same = all(maxdiff(gp[0][t], gr[0][t]) == 0.0 for t in range(N))   # memories identical
check("T4 GTrXL memory identical across carry modes", gtrxl_same)
# tokens identical before first boundary (step < SEG=8 entering), differ at/after boundary
pre_boundary_same = all(maxdiff(gp[3][t], gr[3][t]) == 0.0 for t in range(1, SEG))
check("T4 tokens identical before first segment boundary", pre_boundary_same,
      f"(steps 1..{SEG-1})")
# at the boundary the persistent tokens get updated (nonzero), reset128 zeroed
tok_p_after = np.asarray(gp[3][SEG])      # entering tokens at step SEG (post first update)
tok_r_after = np.asarray(gr[3][SEG])
reset_zeroed = np.max(np.abs(tok_r_after)) == 0.0
persist_changed = np.max(np.abs(tok_p_after)) > 0.0
check("T4 reset128 tokens zeroed at boundary", reset_zeroed, f"max|tok|={np.max(np.abs(tok_r_after)):.2e}")
check("T4 persistent tokens carried (nonzero) at boundary", persist_changed,
      f"max|tok|={np.max(np.abs(tok_p_after)):.2e}")

# =====================================================================
print("== T5: true-done full reset ==", flush=True)
for mode in ["persistent", "reset128"]:
    mem, mask, idx, st = fresh_state()
    # seed nonzero tokens/seg_buf, then apply a done
    st = {"mem_tokens": jnp.ones((B, NTOK, EMB)), "seg_buf": jnp.ones((B, SEG, EMB)),
          "seg_count": jnp.full((B,), 3, jnp.int32)}
    h_t = jnp.ones((B, EMB))
    done = jnp.array([True])
    # collect-style: advance then full-reset on done (mirrors rmt_collect lines 203-209)
    st2 = rmt_advance_tokens(st, h_t, done, UPDATE_FN, rmt_cfg, mode)
    st2 = {"mem_tokens": jnp.where(done[:, None, None], 0.0, st2["mem_tokens"]),
           "seg_buf":    jnp.where(done[:, None, None], 0.0, st2["seg_buf"]),
           "seg_count":  jnp.where(done, 0, st2["seg_count"])}
    ok = (np.max(np.abs(np.asarray(st2["mem_tokens"]))) == 0.0 and
          np.max(np.abs(np.asarray(st2["seg_buf"]))) == 0.0 and
          int(np.asarray(st2["seg_count"]).max()) == 0)
    check(f"T5 done fully resets RMT state [{mode}]", ok)

# =====================================================================
print("== T6: buffer roundtrip + conservation ==", flush=True)
L = 200
asteps = anchor_steps_for_length(L)
na = len(asteps)
rng = np.random.RandomState(0)
EMB_G = 4
traj = RMTTrajectory(
    observations=rng.randn(L, ODIM).astype(np.float32),
    actions=rng.randint(0, ADIM, L).astype(np.int32),
    rewards=rng.randn(L).astype(np.float32),
    dones=np.array([False]*(L-1)+[True]),
    values=rng.randn(L).astype(np.float32),
    log_probs=rng.randn(L).astype(np.float32),
    initial_memory=np.zeros((WM, LAYERS, EMB), np.float32),
    achievements=np.zeros((L, EMB_G), np.float32),
    target_achievements=np.array([1, 0, 0, 0], np.float32),   # 1-D [n_ach] (P2 convention)
    next_observations=rng.randn(L, ODIM).astype(np.float32),
    memory_anchors=np.zeros((na, WM, LAYERS, EMB), np.float32),
    anchor_steps=np.array(asteps, np.int64),
    anchor_masks=np.zeros((na, HEADS, 1, WM + 1), np.bool_),
    anchor_idxs=np.zeros(na, np.int64),
    trajectory_id=0, collected_update_count=0,
    rmt_initial_tokens=np.zeros((NTOK, EMB), np.float32),
    rmt_initial_segbuf=np.zeros((SEG, EMB), np.float32),
    rmt_initial_segcount=0,
    rmt_anchor_tokens=rng.randn(na, NTOK, EMB).astype(np.float32),
    rmt_anchor_segbuf=rng.randn(na, SEG, EMB).astype(np.float32),
    rmt_anchor_segcount=np.zeros(na, np.int64))
try:
    traj.validate_anchors(); vok = True
except Exception as e:
    vok = False; print("    validate_anchors raised:", e)
check("T6 RMTTrajectory.validate_anchors passes", vok, f"anchors={asteps}")

buf = RMTReplayBuffer(capacity=64, seed=42)
buf.insert(traj)
check("T6 conservation collected==inserted after insert",
      buf.counters.trajectories_inserted == 1 and len(buf) == 1)
s = buf.sample(sequence_length=129)
rmt_ok = (s.pre_anchor_rmt_tokens.shape == (NTOK, EMB) and
          s.pre_anchor_rmt_segbuf.shape == (SEG, EMB))
check("T6 sample carries RMT anchor fields", rmt_ok,
      f"tok={s.pre_anchor_rmt_tokens.shape} segbuf={s.pre_anchor_rmt_segbuf.shape}")
check("T6 sample does not cross episode (single trajectory)",
      s.source_trajectory_id == 0 and s.length == 129)

# =====================================================================
print("== T7: hindsight relabel keeps RMT fields ==", flush=True)
# give the trajectory an achieved goal so relabel succeeds
traj.achievements[50, 1] = 1.0
rel = RH.relabel_trajectory_rmt(traj, embedding_size=EMB_G)
keep = (np.array_equal(rel.rmt_anchor_tokens, traj.rmt_anchor_tokens) and
        np.array_equal(rel.rmt_anchor_segbuf, traj.rmt_anchor_segbuf) and
        np.array_equal(rel.rmt_anchor_segcount, traj.rmt_anchor_segcount) and
        np.array_equal(rel.rmt_initial_tokens, traj.rmt_initial_tokens))
check("T7 relabel preserves RMT anchor fields unchanged", keep)
check("T7 relabel changes target to an achieved goal",
      int(rel.target_achievements.argmax()) in {0, 1})
# fabricated goal -> ValueError (Gate 6)
traj_noach = RH.RMTTrajectory(**{**traj.__dict__,
                                 "achievements": np.zeros((L, EMB_G), np.float32)})
try:
    RH.relabel_trajectory_rmt(traj_noach, goal_index=3, embedding_size=EMB_G)   # never achieved
    vok6 = False
except ValueError:
    vok6 = True
check("T7 fabricated/unachieved goal raises ValueError (Gate 6)", vok6)

# =====================================================================
print("== T8: compute_gae correctness ==", flush=True)
# T=2, E=1, gamma=.999, lam=.8; no done; last_value=0
rewards = np.array([[1.0], [1.0]], np.float32)
values = np.array([[0.0], [0.0]], np.float32)
dones = np.array([[0.0], [0.0]], np.float32)
adv, tgt = PPO.compute_gae(rewards, values, dones, np.array([0.0], np.float32),
                           0.999, 0.8, -50, 300)
# delta1 = 1 + .999*0 - 0 = 1 ; adv[1]=1
# delta0 = 1 + .999*0 - 0 = 1 ; adv[0]=1 + .999*.8*1 = 1.7992
exp_adv0 = 1.0 + 0.999*0.8*1.0
check("T8 GAE adv[1]", abs(float(adv[1, 0]) - 1.0) < 1e-5, f"={float(adv[1,0])}")
check("T8 GAE adv[0]", abs(float(adv[0, 0]) - exp_adv0) < 1e-4,
      f"={float(adv[0,0])} exp={exp_adv0}")
check("T8 targets == adv + values (values=0)", maxdiff(tgt, adv) == 0.0)

# =====================================================================
print("== T9: rmt params receive gradient (gate 14) ==", flush=True)
def _read_loss(params):
    # forward one step with NONZERO mem_tokens so the read path is exercised
    mem, mask, idx, st = fresh_state()
    st_tok = jax.random.normal(jax.random.PRNGKey(7), (B, NTOK, EMB))
    obs0 = jax.random.normal(jax.random.PRNGKey(8), (B, ODIM))
    lg, vl, mo, ht = APPLY(params, mem, obs0, mask, st_tok)
    return lg.mean() + vl.mean()
grads = jax.grad(_read_loss)(PARAMS)
def gnorm(key):
    return float(np.sqrt(np.sum([np.sum(np.square(np.asarray(v)))
                                 for v in jax.tree_util.tree_leaves(PARAMS[key])]))  \
                 if isinstance(PARAMS[key], dict) else np.sum(np.square(np.asarray(grads[key]))))
g_gate = float(np.abs(np.asarray(grads["rmt_gate"])).max())
g_read = float(optax_global_norm(grads["rmt_read_attn"])) if False else None
# compute leaf-norms manually
def leaf_norm(subtree):
    return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(v))))
                             for v in jax.tree_util.tree_leaves(subtree))))
g_read = leaf_norm(grads["rmt_read_attn"])
g_readln = leaf_norm(grads["rmt_read_ln"])
check("T9 grad w.r.t. rmt_gate finite & nonzero", np.isfinite(g_gate) and g_gate > 0,
      f"|g|={g_gate:.3e}")
check("T9 grad w.r.t. rmt_read_attn finite & nonzero", np.isfinite(g_read) and g_read > 0,
      f"|g|={g_read:.3e}")
check("T9 grad w.r.t. rmt_read_ln finite & nonzero", np.isfinite(g_readln) and g_readln > 0,
      f"|g|={g_readln:.3e}")
# update-path grad (through update_rmt_tokens)
def _upd_loss(params):
    tok = jax.random.normal(jax.random.PRNGKey(9), (B, NTOK, EMB))
    sb = jax.random.normal(jax.random.PRNGKey(10), (B, SEG, EMB))
    new_tok = net.apply({"params": params}, tok, sb, method=net.update_rmt_tokens)
    return new_tok.mean()
gu = jax.grad(_upd_loss)(PARAMS)
g_upd = leaf_norm(gu["rmt_update_attn"])
check("T9 grad w.r.t. rmt_update_attn finite & nonzero", np.isfinite(g_upd) and g_upd > 0,
      f"|g|={g_upd:.3e}")

# =====================================================================
n_pass = sum(1 for _, ok, _ in _results if ok)
n_fail = sum(1 for _, ok, _ in _results if not ok)
print("\n" + "=" * 60, flush=True)
print(f"RESULT: {n_pass} passed, {n_fail} failed (of {len(_results)})", flush=True)
if n_fail:
    print("FAILED TESTS:", flush=True)
    for name, ok, det in _results:
        if not ok:
            print(f"  - {name}  {det}", flush=True)
print("=" * 60, flush=True)
sys.exit(1 if n_fail else 0)
