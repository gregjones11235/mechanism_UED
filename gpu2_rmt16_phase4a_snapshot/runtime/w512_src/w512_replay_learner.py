"""W512 × P2-Replay — ORIGINAL-GOAL V-trace ONLY replay learner (CC2 corrected §二/§五).

This is the W512 analog of the frozen `rmt_replay_learner.original_vtrace_update_rmt`. The V-trace
loss + the transactional KL-gate update are NETWORK-AGNOSTIC: they only call scan_fn + the
reconstructed entering state + the frozen FPL helpers (_log_pi_and_entropy / pack_batch /
critic_only_mask / ema_update / _kl_mean / _select_where / _select_opt_critic) + V.vtrace. So we
REUSE them VERBATIM by import — the ONLY things that change for W512 are:
  * reconstruction (reconstruct_w512_batch, carrying the W512 raw-history state instead of RMT
    tokens), and
  * the loss-region scan (make_scan_w512_loss, carrying (memories,mem_mask,mem_idx,w512_state) and
    deriving done_enter internally from the single done_new stream with done_enter[0]=False — valid
    because a within-trajectory loss window has no entering done, and at an episode-start window the
    memory is all-zero so the entering-done value is a no-op).

Crucially, make_scan_w512_loss has the EXACT RMT signature scan_fn(params, *recon, obs_ext, don_ext)
(recon = 4-tuple), so compute_loss_original_vtrace_rmt / _target_scan_rmt / _window_log_softmax_rmt
are reused UNCHANGED. The loss weight is FROZEN at W_ORIGINAL_VTRACE = 1.0 (reused).

Hindsight/AWR firewall (directive §八): this module never imports/calls rmt_hindsight or awr — it
imports ONLY the original_vtrace path from rmt_replay_learner (which itself never touches awr in
that path). The grad graph structurally contains the original observations/rewards/dones/behavior
log_probs, ONE online W512 scan + ONE target scan, and the V-trace losses — nothing relabeled.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

import vtrace as V                                   # noqa: F401  (documenting the frozen dependency)
import memory_anchor as MA
import full_p2_learner as FPL
from w512_memory_anchor import (
    make_apply_eval_w512, reconstruct_w512_state_with_network, w512_step_forward,
)
# Network-agnostic frozen original_vtrace machinery, reused unchanged.
from rmt_replay_learner import (
    W_ORIGINAL_VTRACE,
    compute_loss_original_vtrace_rmt,
    _ext_obs_dones,
    _target_scan_rmt,
    _window_log_softmax_rmt,
)


# ----------------------------- W512 reconstruction -----------------------------

def _w512_anchor_state(sample, cfg):
    """Build the ENTERING anchor state (GTrXL + W512) for one sample. GTrXL mask/idx are DERIVED
    from the anchor step (frozen MA.derive_anchor_entering_state, same as RMT) so reconstruction
    lies in the scan convention; the W512 raw-history state is taken from the sample's anchor
    fields (the snapshotted ENTERING state, before that step's reset128 clear)."""
    mem_a = jnp.asarray(sample.pre_anchor_memory, dtype=jnp.float32)[None]   # [1, wm, layers, embed]
    mask_a, idx_a = MA.derive_anchor_entering_state(
        int(sample.pre_anchor_step), cfg.window_mem, cfg.num_heads, 1)
    w512_st = {
        "delay_buf":   jnp.asarray(sample.pre_anchor_w512_delay_buf, jnp.float32)[None],
        "delay_idx":   jnp.full((1,), int(sample.pre_anchor_w512_delay_idx), jnp.int32),
        "delay_count": jnp.full((1,), int(sample.pre_anchor_w512_delay_count), jnp.int32),
        "long_buf":    jnp.asarray(sample.pre_anchor_w512_long_buf, jnp.float32)[None],
        "long_mask":   jnp.asarray(sample.pre_anchor_w512_long_mask, jnp.bool_)[None],
        "long_idx":    jnp.full((1,), int(sample.pre_anchor_w512_long_idx), jnp.int32),
        "seg_step":    jnp.full((1,), int(sample.pre_anchor_w512_seg_step), jnp.int32),
    }
    return mem_a, mask_a, idx_a, w512_st


def reconstruct_w512_batch(network, apply_eval_w512, params, samples, cfg,
                           w512_cfg, segment_len=128):
    """Reconstruct loss-window-start ENTERING state per sample (<=128 burn-in, stop_grad).

    The burn-in is within a single episode, so done_enter = done_new = False for every burn-in
    step (a done would have ended the episode before the window). Returns
    (memories[B], masks[B], idxs[B], w512_st{[B,...]})."""
    mems, masks, idxs = [], [], []
    dbs, dis, dcs, lbs, lms, lis, sss = [], [], [], [], [], [], []
    for s in samples:
        mem_a, mask_a, idx_a, w512_st = _w512_anchor_state(s, cfg)
        burn = np.asarray(s.burn_in_obs)
        if burn.shape[0] > 0:
            burn_b = jnp.asarray(burn, jnp.float32)[:, None, :]               # [gap,1,obs]
            z_b = jnp.zeros((burn.shape[0], 1), jnp.bool_)                     # within-episode
            res = reconstruct_w512_state_with_network(
                network, apply_eval_w512, params, mem_a, mask_a, idx_a, w512_st,
                burn_b, z_b, z_b, cfg.window_mem, cfg.num_heads, w512_cfg, segment_len)
            mem_a, mask_a, idx_a, w512_st = jax.tree_util.tree_map(jax.lax.stop_gradient, res)
        mems.append(mem_a[0]); masks.append(mask_a[0]); idxs.append(idx_a[0])
        dbs.append(w512_st["delay_buf"][0]); dis.append(w512_st["delay_idx"][0])
        dcs.append(w512_st["delay_count"][0]); lbs.append(w512_st["long_buf"][0])
        lms.append(w512_st["long_mask"][0]); lis.append(w512_st["long_idx"][0])
        sss.append(w512_st["seg_step"][0])
    w512_st_b = {
        "delay_buf":   jnp.stack(dbs), "delay_idx": jnp.stack(dis), "delay_count": jnp.stack(dcs),
        "long_buf":    jnp.stack(lbs), "long_mask": jnp.stack(lms), "long_idx": jnp.stack(lis),
        "seg_step":    jnp.stack(sss),
    }
    return jnp.stack(mems), jnp.stack(masks), jnp.stack(idxs), w512_st_b


# ----------------------------- W512 loss-region scan -----------------------------

def make_scan_w512_loss(network, cfg, w512_cfg, segment_len=128):
    """jitted lax.scan region scanner carrying GTrXL + W512 state (B>=2).

    Signature MATCHES the frozen RMT loss scan exactly:
        scan_fn(params, memories, mem_mask, mem_idx, w512_st, obs_seq[T,B], dones_new_seq[T,B])
            -> (logits[T,B,A], values[T,B])
    done_enter is derived inside: done_enter[t] = done_new[t-1], done_enter[0] = False. (False is
    correct at a within-episode window start; at an episode-start window the memory is all-zero so
    the value is a no-op.) This lets compute_loss_original_vtrace_rmt / _target_scan_rmt /
    _window_log_softmax_rmt call it UNCHANGED."""
    apply_eval_w512 = make_apply_eval_w512(network)

    def scan_fn(params, memories, mem_mask, mem_idx, w512_st, obs_seq, dones_new_seq):
        dn = dones_new_seq.astype(jnp.bool_)
        B = dn.shape[1]
        de = jnp.concatenate([jnp.zeros((1, B), jnp.bool_), dn[:-1]], axis=0)

        def body(carry, inp):
            mem, mask, idx, st = carry
            obs_t, de_t, dn_t = inp
            (mem, mask, idx, st, lg, vl, _mp) = w512_step_forward(
                apply_eval_w512, params, mem, mask, idx, st, obs_t, de_t, dn_t,
                cfg.window_mem, cfg.num_heads, w512_cfg, segment_len)
            return (mem, mask, idx, st), (lg, vl)

        (_fm, _fmask, _fidx, _fst), (logits, values) = jax.lax.scan(
            body, (memories, mem_mask, mem_idx, w512_st), (obs_seq, de, dn))
        return logits, values

    return jax.jit(scan_fn)


# ----------------------------- update (transactional KL gate, reused) -----------------------------

def original_vtrace_update_w512(network, params, target_params, opt_state, optimizer,
                                apply_eval_w512, scan_fn,
                                samples_orig, cfg, w512_cfg, segment_len=128):
    """ORIGINAL-goal V-trace ONLY controlled replay update for W512 (directive §五/§八).

    Identical structure to rmt_replay_learner.original_vtrace_update_rmt — SAME transactional KL
    gate / actor step-scale retry {1.0,0.5,0.25,0.125} / rollback / EMA tau=0.995 / grad-clip /
    critic-only partition — but reconstructs the W512 raw-history state and forwards via the W512
    loss scan. Takes ONLY samples_orig (no relabeled argument exists). V-trace needs NO policy-lag
    (the stored behavior log_prob IS the true behavior policy), so there is no update_count.
    Returns (new_params, new_target, final_opt_state, metrics)."""
    po = FPL.pack_batch(samples_orig)
    obs_o_ext, don_o_ext = _ext_obs_dones(po)
    L = po["observations"].shape[1]

    # reconstruction (stop_gradient, eager) for online + target — ORIGINAL sample only.
    recon_o = reconstruct_w512_batch(network, apply_eval_w512, params, samples_orig,
                                     cfg, w512_cfg, segment_len)
    recon_o_t = reconstruct_w512_batch(network, apply_eval_w512, target_params, samples_orig,
                                       cfg, w512_cfg, segment_len)

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
        network_family="W512",
    ))
    return new_params, new_target, final_opt_state, metrics
