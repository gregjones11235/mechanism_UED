"""RMT16 × P2-Replay — combined controlled replay update (Phase4A).

ADAPTS the frozen P2-Full-A `full_p2_learner.full_p2_update` to the RMT16 network. ALL
frozen coefficients and gate logic are reused UNCHANGED from P2-Full-A v2.1 (V-trace
rho/c=1.0, AWR beta=1.0/w_max=20/lambda_kl=0.01, w_vtrace=w_awr=0.5, vf_coef=0.5,
ent_coef=0.002, EMA tau=0.995, MAX_POLICY_LAG=16, transactional kl_replay_max=0.05 with
actor step scales {1.0,0.5,0.25,0.125} + rollback, grad_clip=1.0, ent_floor=0.05). The
ONLY adaptation: memory reconstruction and the loss-region scans now carry the RMT16
persistent-token state (mem_tokens/seg_buf/seg_count) in addition to the GTrXL memory,
and evolve tokens at 128-step segment boundaries (Persistent carry vs Reset128 clear).

The RMT16 read/update modules (rmt_read_attn/rmt_read_ln/rmt_update_attn/rmt_update_ln/
rmt_gate) are all in the actor forward, so the critic-only partition (CRITIC_ONLY_TOKENS)
correctly classifies them as POLICY-AFFECTING -> they are covered by the KL gate.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

import awr as A
import vtrace as V
import memory_anchor as MA
import full_p2_learner as FPL
from full_p2_learner import FullP2Config
from rmt_memory_anchor import (
    make_apply_eval_rmt, make_update_fn, rmt_step_forward,
    reconstruct_rmt_state_with_network,
)


# ----------------------------- RMT reconstruction -----------------------------

def _rmt_anchor_state(sample, cfg):
    """Build the ENTERING anchor state (GTrXL + RMT) for one sample."""
    mem_a = jnp.asarray(sample.pre_anchor_memory, dtype=jnp.float32)[None]
    mask_a, idx_a = MA.derive_anchor_entering_state(
        int(sample.pre_anchor_step), cfg.window_mem, cfg.num_heads, 1)
    rmt_st = {
        "mem_tokens": jnp.asarray(sample.pre_anchor_rmt_tokens, jnp.float32)[None],
        "seg_buf":    jnp.asarray(sample.pre_anchor_rmt_segbuf, jnp.float32)[None],
        "seg_count":  jnp.full((1,), int(sample.pre_anchor_rmt_segcount), jnp.int32),
    }
    return mem_a, mask_a, idx_a, rmt_st


def reconstruct_rmt_batch(network, apply_eval_rmt, params, samples, cfg,
                          rmt_cfg, carry_mode):
    """Reconstruct loss-window-start ENTERING state per sample (<=128 burn-in, stop_grad).
    Returns (memories[B], masks[B], idxs[B], rmt_st{[B,...]})."""
    update_fn = make_update_fn(network, params)
    mems, masks, idxs, toks, sbufs, scnts = [], [], [], [], [], []
    for s in samples:
        mem_a, mask_a, idx_a, rmt_st = _rmt_anchor_state(s, cfg)
        burn = np.asarray(s.burn_in_obs)
        if burn.shape[0] > 0:
            burn_b = jnp.asarray(burn, jnp.float32)[:, None, :]
            dones_b = jnp.zeros((burn.shape[0], 1), jnp.bool_)   # burn-in is within-episode
            res = reconstruct_rmt_state_with_network(
                network, apply_eval_rmt, params, mem_a, mask_a, idx_a, rmt_st,
                burn_b, dones_b, cfg.window_mem, cfg.num_heads, rmt_cfg, carry_mode)
            mem_a, mask_a, idx_a, rmt_st = jax.tree_util.tree_map(
                jax.lax.stop_gradient, res)
        mems.append(mem_a[0]); masks.append(mask_a[0]); idxs.append(idx_a[0])
        toks.append(rmt_st["mem_tokens"][0]); sbufs.append(rmt_st["seg_buf"][0])
        scnts.append(rmt_st["seg_count"][0])
    rmt_st_b = {"mem_tokens": jnp.stack(toks), "seg_buf": jnp.stack(sbufs),
                "seg_count": jnp.stack(scnts)}
    return jnp.stack(mems), jnp.stack(masks), jnp.stack(idxs), rmt_st_b


# ----------------------------- RMT loss-region scan -----------------------------

def _make_scan_rmt(network, cfg, rmt_cfg, carry_mode):
    """jitted lax.scan region scanner carrying GTrXL + RMT state (B>=2).
    scan_fn(params, memories, mask, idx, rmt_st, obs_seq[T,B], dones_seq[T,B])
        -> (logits[T,B,A], values[T,B])."""
    apply_eval_rmt = make_apply_eval_rmt(network)

    def scan_fn(params, memories, mem_mask, mem_idx, rmt_st, obs_seq, dones_seq):
        update_fn = make_update_fn(network, params)
        def body(carry, inp):
            mem, mask, idx, st = carry
            obs_t, d_t = inp
            (mem, mask, idx, st, lg, vl, _mp, _et) = rmt_step_forward(
                apply_eval_rmt, params, mem, mask, idx, st, obs_t, d_t,
                cfg.window_mem, cfg.num_heads, rmt_cfg, carry_mode, update_fn)
            return (mem, mask, idx, st), (lg, vl)
        (_fm, _fmask, _fidx, _fst), (logits, values) = jax.lax.scan(
            body, (memories, mem_mask, mem_idx, rmt_st), (obs_seq, dones_seq))
        return logits, values

    return jax.jit(scan_fn)


def _ext_obs_dones(po):
    obs = jnp.transpose(po["observations"], (1, 0, 2))               # [L,B,obs]
    obs_ext = jnp.concatenate([obs, po["next_obs_final"][None]], 0)  # [L+1,B,obs]
    dones = jnp.transpose(po["dones"], (1, 0))                        # [L,B]
    B = dones.shape[1]
    dones_ext = jnp.concatenate([dones, jnp.zeros((1, B), jnp.float32)], 0)  # [L+1,B]
    return obs_ext, dones_ext


# ----------------------------- loss (grad graph = 2 online RMT scans) -----------------------------

def compute_loss_rmt(params, scan_fn, po, pr, obs_o_ext, don_o_ext, obs_r_ext, don_r_ext,
                     target_vals_o, target_vals_r, recon_o, recon_r, cfg, update_count):
    B, L = po["observations"].shape[0], po["observations"].shape[1]
    valid = jnp.ones((B, L))
    actions_t = po["actions"].T                                       # [L,B]

    # ----- ORIGINAL (V-trace) online RMT scan (grad) -----
    lg_o, val_o = scan_fn(params, *recon_o, obs_o_ext, don_o_ext)
    v_online = val_o[:L].T                                            # [B,L]
    log_pi_o, ent_o = FPL._log_pi_and_entropy(lg_o[:L], actions_t)
    log_pi_o, ent_o = log_pi_o.T, ent_o.T

    v_target_tp1 = target_vals_o[:, 1:]
    boot_o = jnp.where(po["terminal"] > 0.5, 0.0, target_vals_o[:, L])
    vt_cfg = V.VtraceConfig(cfg.gamma, cfg.rho_bar, cfg.c_bar, cfg.vt_clip_min, cfg.vt_clip_max)
    vt = V.vtrace_targets(log_pi_o, po["log_probs"], v_online, v_target_tp1,
                          po["rewards"], po["dones"], boot_o, vt_cfg)
    vt_vloss = V.vtrace_value_loss(vt, v_online, valid)
    vt_aloss = V.vtrace_actor_loss(vt, log_pi_o, ent_o, valid, cfg.ent_coef)
    ess = V.ess_fraction(vt.ratio, valid)

    # ----- RELABELED (AWR) online RMT scan (grad) -----
    lg_r, val_r = scan_fn(params, *recon_r, obs_r_ext, don_r_ext)
    logits_rel = jnp.transpose(lg_r[:L], (1, 0, 2))                   # [B,L,A]
    v_rel_online = val_r[:L].T
    logits_before = jax.lax.stop_gradient(logits_rel)
    target_rel = target_vals_r[:, :L]
    boot_r = jnp.where(pr["terminal"] > 0.5, 0.0, target_vals_r[:, L])
    lag = jnp.maximum(update_count - pr["lag"], 0)
    awr_valid = valid * (lag <= cfg.max_policy_lag).astype(jnp.float32)[:, None]
    awr_cfg = A.AWRConfig(cfg.gamma, cfg.beta, cfg.w_max, cfg.lambda_kl,
                          cfg.vt_clip_min, cfg.vt_clip_max)
    awr = A.awr_losses(logits_rel, logits_before, po["actions"], v_rel_online,
                       target_rel, pr["rewards"], pr["dones"], boot_r, awr_valid, awr_cfg)

    loss = cfg.w_vtrace * (vt_aloss + cfg.vf_coef * vt_vloss) \
        + cfg.w_awr * (awr.actor_loss + cfg.vf_coef * awr.value_loss)

    metrics = dict(
        loss=loss, vtrace_actor=vt_aloss, vtrace_value=vt_vloss,
        awr_actor=awr.actor_loss, awr_value=awr.value_loss,
        ess=ess, entropy=ent_o.sum() / valid.sum(),
        awr_w_mean=awr.w_mean, awr_kl=awr.kl_mean,
        awr_valid_frac=awr_valid.sum() / valid.sum(),
        mean_v_online=v_online.sum() / valid.sum(), ratio_max=vt.ratio.max(),
    )
    return loss, metrics


def _target_scan_rmt(scan_fn, target_params, recon, obs_ext, don_ext):
    values = scan_fn(target_params, *recon, obs_ext, don_ext)[1]
    return jax.lax.stop_gradient(values).T                            # [B,L+1]


def _window_log_softmax_rmt(scan_fn, params, recon, obs_ext, don_ext, L):
    lg = scan_fn(params, *recon, obs_ext, don_ext)[0]
    a = jnp.transpose(lg[:L], (1, 0, 2))
    return FPL._log_softmax(a)


# ----------------------------- update (transactional KL gate, reused) -----------------------------

def full_p2_update_rmt(network, params, target_params, opt_state, optimizer,
                       apply_eval_rmt, scan_fn,
                       samples_orig, samples_rel, cfg, rmt_cfg, carry_mode, update_count):
    """RMT16 combined controlled update. Same transactional KL gate / rollback / EMA /
    critic-only partition / policy-lag / ESS as frozen P2-Full-A (full_p2_learner)."""
    po = FPL.pack_batch(samples_orig)
    pr = FPL.pack_batch(samples_rel)
    obs_o_ext, don_o_ext = _ext_obs_dones(po)
    obs_r_ext, don_r_ext = _ext_obs_dones(pr)
    B, L = po["observations"].shape[0], po["observations"].shape[1]

    # reconstruction (stop_gradient, eager) for online + target, orig + relabeled
    recon_o = reconstruct_rmt_batch(network, apply_eval_rmt, params, samples_orig, cfg, rmt_cfg, carry_mode)
    recon_r = reconstruct_rmt_batch(network, apply_eval_rmt, params, samples_rel, cfg, rmt_cfg, carry_mode)
    recon_o_t = reconstruct_rmt_batch(network, apply_eval_rmt, target_params, samples_orig, cfg, rmt_cfg, carry_mode)
    recon_r_t = reconstruct_rmt_batch(network, apply_eval_rmt, target_params, samples_rel, cfg, rmt_cfg, carry_mode)

    # eager target value scans (stop_gradient constants)
    target_vals_o = _target_scan_rmt(scan_fn, target_params, recon_o_t, obs_o_ext, don_o_ext)
    target_vals_r = _target_scan_rmt(scan_fn, target_params, recon_r_t, obs_r_ext, don_r_ext)

    def loss_fn(p):
        return compute_loss_rmt(p, scan_fn, po, pr, obs_o_ext, don_o_ext, obs_r_ext, don_r_ext,
                                target_vals_o, target_vals_r, recon_o, recon_r, cfg, update_count)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    mask = FPL.critic_only_mask(params)
    full_new = optax.apply_updates(params, updates)
    ema_full = FPL.ema_update(full_new, target_params, cfg.ema_tau)

    lp_old = jax.lax.stop_gradient(
        _window_log_softmax_rmt(scan_fn, params, recon_o, obs_o_ext, don_o_ext, L))

    trials = []
    accepted_scale = None; accepted_kl = None; new_params = None
    for scale in cfg.actor_step_scales:
        scaled_updates = jax.tree_util.tree_map(
            lambda m, u, s=scale: jnp.where(m, u, s * u), mask, updates)
        cand = optax.apply_updates(params, scaled_updates)
        lp_new = _window_log_softmax_rmt(scan_fn, cand, recon_o, obs_o_ext, don_o_ext, L)
        kl = float(np.asarray(FPL._kl_mean(lp_new, lp_old)))
        trials.append({"scale": float(scale), "kl": kl})
        if np.isfinite(kl) and kl <= cfg.kl_replay_max:
            accepted_scale = float(scale); accepted_kl = kl; new_params = cand
            break

    if accepted_scale is not None:
        policy_committed = True; kl_rejected = False
        chosen_scale = accepted_scale; policy_kl = accepted_kl
        new_target = FPL.ema_update(new_params, target_params, cfg.ema_tau)
        final_opt_state = new_opt_state
    else:
        policy_committed = False; kl_rejected = True
        chosen_scale = -1.0
        policy_kl = trials[-1]["kl"] if trials else float("nan")
        new_params = FPL._select_where(mask, full_new, params)
        final_opt_state = FPL._select_opt_critic(new_opt_state, opt_state)
        new_target = FPL._select_where(mask, ema_full, target_params)

    metrics.update(dict(
        grad_norm=optax.global_norm(grads),
        policy_kl=policy_kl,
        kl_replay_max=float(cfg.kl_replay_max),
        kl_run_max=float(cfg.kl_run_max),
        kl_gate_pass=bool(policy_committed and policy_kl <= cfg.kl_replay_max),
        kl_rejected_update=bool(kl_rejected),
        policy_committed=bool(policy_committed),
        chosen_actor_step_scale=float(chosen_scale),
        n_actor_scale_trials=len(trials),
        actor_step_scales_tried=trials,
        finite=bool(np.isfinite(float(loss))),
        entropy_floor_pass=(float(metrics["entropy"]) >= cfg.ent_floor),
    ))
    return new_params, new_target, final_opt_state, metrics


# ============================================================================
# Phase4A-v2 (CC2 directive §五/§八): ORIGINAL-GOAL V-trace ONLY replay learner.
# ============================================================================
#
# This is a DEDICATED replay update path for replay_mode=original_vtrace. It is NOT the
# combined full_p2_update_rmt with the relabeled branch "turned off" — it is a separate
# function whose grad graph STRUCTURALLY contains:
#   * the ORIGINAL observations / goal / rewards / dones / behavior log_probs only,
#   * the RMT/GTrXL anchor reconstruction (reconstruct_rmt_batch) of the original sample,
#   * ONE online RMT scan (V-trace log_pi/value) + ONE corresponding target-network scan,
#   * the V-trace value loss + actor loss + entropy,
#   * the SAME transactional KL gate / rollback / EMA / grad-clip as full_p2_update_rmt.
# and STRUCTURALLY does NOT contain (cannot reach, no symbol reference at all):
#   * rmt_hindsight.relabel_sample_rmt / any relabeled observations/goals/rewards,
#   * awr.* / w_awr / any AWR loss,
#   * a SECOND online reconstruction (recon_r) or a SECOND target scan (target_vals_r).
#
# The loss weight is FROZEN at W_ORIGINAL_VTRACE = 1.0 — NOT the combined loss's
# unexplained 0.5 (which only existed to balance the V-trace term against the AWR term;
# with no AWR term there is nothing to balance against).
#
# Hindsight firewall (directive §八): because this function never imports/calls
# rmt_hindsight or awr, monkeypatching RH.relabel_sample_rmt to raise leaves this path
# fully functional — the non-entry is STRUCTURAL, not "temporarily not using the output".
# The launcher additionally hard-asserts the four firewall counters == 0 and the tests
# monkeypatch relabel_sample_rmt to raise.

W_ORIGINAL_VTRACE = 1.0   # FROZEN (directive §五). Deliberately not the combined 0.5.


def compute_loss_original_vtrace_rmt(params, scan_fn, po, obs_o_ext, don_o_ext,
                                     target_vals_o, recon_o, cfg):
    """ORIGINAL-goal V-trace ONLY loss (directive §五/§六).

    Exactly ONE online RMT scan (grad) + the precomputed corresponding target scan. No
    relabeled scan, no AWR. loss = W_ORIGINAL_VTRACE * (vt_aloss + vf_coef * vt_vloss).
    """
    B, L = po["observations"].shape[0], po["observations"].shape[1]
    valid = jnp.ones((B, L))
    actions_t = po["actions"].T                                       # [L,B]

    # ----- ORIGINAL (V-trace) online RMT scan (the ONLY scan in the grad graph) -----
    lg_o, val_o = scan_fn(params, *recon_o, obs_o_ext, don_o_ext)
    v_online = val_o[:L].T                                            # [B,L]
    log_pi_o, ent_o = FPL._log_pi_and_entropy(lg_o[:L], actions_t)
    log_pi_o, ent_o = log_pi_o.T, ent_o.T

    v_target_tp1 = target_vals_o[:, 1:]
    boot_o = jnp.where(po["terminal"] > 0.5, 0.0, target_vals_o[:, L])
    vt_cfg = V.VtraceConfig(cfg.gamma, cfg.rho_bar, cfg.c_bar, cfg.vt_clip_min, cfg.vt_clip_max)
    vt = V.vtrace_targets(log_pi_o, po["log_probs"], v_online, v_target_tp1,
                          po["rewards"], po["dones"], boot_o, vt_cfg)
    vt_vloss = V.vtrace_value_loss(vt, v_online, valid)
    vt_aloss = V.vtrace_actor_loss(vt, log_pi_o, ent_o, valid, cfg.ent_coef)
    ess = V.ess_fraction(vt.ratio, valid)

    loss = W_ORIGINAL_VTRACE * (vt_aloss + cfg.vf_coef * vt_vloss)

    metrics = dict(
        loss=loss, vtrace_actor=vt_aloss, vtrace_value=vt_vloss,
        ess=ess, entropy=ent_o.sum() / valid.sum(),
        mean_v_online=v_online.sum() / valid.sum(), ratio_max=vt.ratio.max(),
        w_original_vtrace=W_ORIGINAL_VTRACE,
    )
    return loss, metrics


def original_vtrace_update_rmt(network, params, target_params, opt_state, optimizer,
                               apply_eval_rmt, scan_fn,
                               samples_orig, cfg, rmt_cfg, carry_mode):
    """ORIGINAL-goal V-trace ONLY controlled replay update (directive §五/§八).

    Reuses the SAME transactional KL gate / actor step-scale retry / rollback / EMA /
    grad-clip / critic-only partition as full_p2_update_rmt, but:
      * takes ONLY samples_orig (no samples_rel argument exists — relabeled samples cannot
        even be passed in),
      * reconstructs ONLY the original sample (recon_o / recon_o_t; no recon_r),
      * computes ONLY target_vals_o (no target_vals_r),
      * the loss is compute_loss_original_vtrace_rmt (one online scan + one target scan).
    V-trace needs NO policy-lag (the stored behavior log_prob IS the true behavior policy),
    so there is no update_count / lag argument here either.
    Returns (new_params, new_target, final_opt_state, metrics)."""
    po = FPL.pack_batch(samples_orig)
    obs_o_ext, don_o_ext = _ext_obs_dones(po)
    L = po["observations"].shape[1]

    # reconstruction (stop_gradient, eager) for online + target — ORIGINAL sample only.
    recon_o = reconstruct_rmt_batch(network, apply_eval_rmt, params, samples_orig, cfg, rmt_cfg, carry_mode)
    recon_o_t = reconstruct_rmt_batch(network, apply_eval_rmt, target_params, samples_orig, cfg, rmt_cfg, carry_mode)

    # eager target value scan (stop_gradient constant) — the ONE corresponding target scan.
    target_vals_o = _target_scan_rmt(scan_fn, target_params, recon_o_t, obs_o_ext, don_o_ext)

    def loss_fn(p):
        return compute_loss_original_vtrace_rmt(p, scan_fn, po, obs_o_ext, don_o_ext,
                                                target_vals_o, recon_o, cfg)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    mask = FPL.critic_only_mask(params)
    full_new = optax.apply_updates(params, updates)
    ema_full = FPL.ema_update(full_new, target_params, cfg.ema_tau)

    lp_old = jax.lax.stop_gradient(
        _window_log_softmax_rmt(scan_fn, params, recon_o, obs_o_ext, don_o_ext, L))

    trials = []
    accepted_scale = None; accepted_kl = None; new_params = None
    for scale in cfg.actor_step_scales:
        scaled_updates = jax.tree_util.tree_map(
            lambda m, u, s=scale: jnp.where(m, u, s * u), mask, updates)
        cand = optax.apply_updates(params, scaled_updates)
        lp_new = _window_log_softmax_rmt(scan_fn, cand, recon_o, obs_o_ext, don_o_ext, L)
        kl = float(np.asarray(FPL._kl_mean(lp_new, lp_old)))
        trials.append({"scale": float(scale), "kl": kl})
        if np.isfinite(kl) and kl <= cfg.kl_replay_max:
            accepted_scale = float(scale); accepted_kl = kl; new_params = cand
            break

    if accepted_scale is not None:
        policy_committed = True; kl_rejected = False
        chosen_scale = accepted_scale; policy_kl = accepted_kl
        new_target = FPL.ema_update(new_params, target_params, cfg.ema_tau)
        final_opt_state = new_opt_state
    else:
        policy_committed = False; kl_rejected = True
        chosen_scale = -1.0
        policy_kl = trials[-1]["kl"] if trials else float("nan")
        new_params = FPL._select_where(mask, full_new, params)
        final_opt_state = FPL._select_opt_critic(new_opt_state, opt_state)
        new_target = FPL._select_where(mask, ema_full, target_params)

    metrics.update(dict(
        grad_norm=optax.global_norm(grads),
        policy_kl=policy_kl,
        kl_replay_max=float(cfg.kl_replay_max),
        kl_run_max=float(cfg.kl_run_max),
        kl_gate_pass=bool(policy_committed and policy_kl <= cfg.kl_replay_max),
        kl_rejected_update=bool(kl_rejected),
        policy_committed=bool(policy_committed),
        chosen_actor_step_scale=float(chosen_scale),
        n_actor_scale_trials=len(trials),
        actor_step_scales_tried=trials,
        finite=bool(np.isfinite(float(loss))),
        entropy_floor_pass=(float(metrics["entropy"]) >= cfg.ent_floor),
        replay_learner="original_vtrace",
    ))
    return new_params, new_target, final_opt_state, metrics
