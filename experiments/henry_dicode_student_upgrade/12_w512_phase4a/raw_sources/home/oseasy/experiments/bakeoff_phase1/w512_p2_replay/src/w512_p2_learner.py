"""W512 × P2 Replay learner: V-trace + AWR combined update adapted for W512 network.

Adapted from p2_full_20260723/src/full_p2_learner.py (frozen P2-Full-A).
Changes:
  - _scan_lax carries W512 long state (long_buf, long_mask) through the scan
  - reconstruct_batch reconstructs W512 long state from anchors + burn-in
  - compute_loss passes W512 long state to the forward function
  - full_p2_update_w512 wraps the full update with W512 state handling

Frozen P2-Full-A hyperparameters (FullP2Config) are UNCHANGED.
"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

import awr as A
import vtrace as V
import w512_memory as w5m


# ---- Re-use FullP2Config from P2-Full-A (frozen) ----
from full_p2_learner import (
    FullP2Config, _log_softmax, _log_pi_and_entropy,
    ema_update, critic_only_mask, is_critic_only_path,
    _select_where, _select_opt_critic, build_optimizer,
    CRITIC_ONLY_TOKENS,
)


# ---- W512-adapted scan ----

def _scan_lax_w512(apply_eval_w512_raw, params, memories, mem_mask, mem_idx,
                   long_buf, long_mask, obs_seq, cfg, w5_cfg, w512_delay_state):
    """lax.scan forward_eval over obs_seq [T,B,obs] carrying GTrXL memory AND
    W512 long state.

    apply_eval_w512_raw: (params, mem, obs, mask, long_buf, long_mask)
                          -> (logits, value, mem_out, h_t)
    w512_delay_state: dict with delay_buf, delay_idx, delay_count, long_idx
                      (the non-long_buf/long_mask parts of W512 state)
    """
    def body(carry, obs_t):
        mem, mask, idx, lbuf, lmsk, delay_st = carry
        # advance GTrXL mask
        idx_new = jnp.clip(idx - 1, 0, cfg.window_mem).astype(jnp.int32)
        ohot = jax.nn.one_hot(idx_new, cfg.window_mem + 1)
        ohot = ohot[:, None, None, :].repeat(cfg.num_heads, axis=1)
        mask_new = jnp.logical_or(mask, ohot)
        # W512 forward
        lg, vl, mem_out, h_t = apply_eval_w512_raw(
            params, mem, obs_t, mask_new, lbuf, lmsk)
        # update GTrXL memory
        mem_new = jnp.roll(mem, -1, axis=1).at[:, -1].set(mem_out)
        # update W512 long state (no done during scan - episodes are contiguous)
        done_false = jnp.zeros(h_t.shape[0], dtype=jnp.bool_)
        w5_full = {**delay_st, "long_buf": lbuf, "long_mask": lmsk}
        w5_new = w5m.w512_step(w5_full, h_t, done_false, w5_cfg)
        new_delay = {k: w5_new[k] for k in ("delay_buf", "delay_idx",
                                             "delay_count", "long_idx")}
        return (mem_new, mask_new, idx_new,
                w5_new["long_buf"], w5_new["long_mask"],
                new_delay), (lg, vl)

    init_carry = (memories, mem_mask, mem_idx, long_buf, long_mask,
                  w512_delay_state)
    (fm, fmask, fidx, flbuf, flmsk, fdelay), (logits, values) = jax.lax.scan(
        body, init_carry, obs_seq)
    return logits, values, fm, fmask, fidx, flbuf, flmsk, fdelay


# ---- W512 reconstruction ----

def _advance_mask_w512(mem_idx, mem_mask, window_mem, num_heads):
    mem_idx = jnp.clip(mem_idx - 1, 0, window_mem).astype(jnp.int32)
    ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
    ohot = ohot[:, None, None, :].repeat(num_heads, axis=1)
    mem_mask = jnp.logical_or(mem_mask, ohot)
    return mem_idx, mem_mask


def reconstruct_w512(apply_eval_w512_recon, params,
                     mem_a, mask_a, idx_a,
                     w512_state_a,
                     burn_in_obs, cfg, w5_cfg):
    """Reconstruct GTrXL memory + W512 long state at loss-window start from
    anchor state + burn-in observations (<=128 steps). All stop_gradient."""
    B = mem_a.shape[0]
    lbuf = w512_state_a["long_buf"]
    lmsk = w512_state_a["long_mask"]
    delay_st = {k: w512_state_a[k] for k in ("delay_buf", "delay_idx",
                                               "delay_count", "long_idx")}
    gap = int(burn_in_obs.shape[0]) if burn_in_obs is not None else 0
    for t in range(gap):
        obs_t = burn_in_obs[t]
        if obs_t.ndim == 1:
            obs_t = obs_t[None]  # add batch dim
        idx_a, mask_a = _advance_mask_w512(idx_a, mask_a, cfg.window_mem,
                                            cfg.num_heads)
        lg, vl, mem_out, h_t = apply_eval_w512_recon(
            params, mem_a, obs_t, mask_a, lbuf, lmsk)
        mem_a = jnp.roll(mem_a, -1, axis=1).at[:, -1].set(mem_out)
        done_false = jnp.zeros(B, dtype=jnp.bool_)
        w5_full = {**delay_st, "long_buf": lbuf, "long_mask": lmsk}
        w5_new = w5m.w512_step(w5_full, h_t, done_false, w5_cfg)
        lbuf = w5_new["long_buf"]
        lmsk = w5_new["long_mask"]
        delay_st = {k: w5_new[k] for k in ("delay_buf", "delay_idx",
                                             "delay_count", "long_idx")}
    result_w512 = {**delay_st, "long_buf": lbuf, "long_mask": lmsk}
    return mem_a, mask_a, idx_a, result_w512


def reconstruct_batch_w512(apply_eval_w512_recon, params, samples, cfg, w5_cfg):
    """Reconstruct per-sample GTrXL + W512 state at loss-window start."""
    from memory_anchor import derive_anchor_entering_state
    memories0, masks0, idxs0 = [], [], []
    w512_states0 = []
    for s in samples:
        mem_a = jnp.asarray(s.pre_anchor_memory, dtype=jnp.float32)[None]
        mask_a, idx_a = derive_anchor_entering_state(
            int(s.pre_anchor_step), cfg.window_mem, cfg.num_heads, 1)
        # W512 anchor state
        if s.pre_anchor_w512_state is not None:
            w5_a = {k: jnp.asarray(v, dtype=jnp.float32 if k != "long_mask"
                                   else jnp.bool_)[None]
                    for k, v in s.pre_anchor_w512_state.items()}
            # Ensure int types for indices
            for ik in ("delay_idx", "delay_count", "long_idx"):
                if ik in w5_a:
                    w5_a[ik] = w5_a[ik].astype(jnp.int32)
            if "long_mask" in w5_a:
                w5_a["long_mask"] = w5_a["long_mask"].astype(jnp.bool_)
        else:
            # Fallback: zero W512 state
            w5_a = w5m.w512_init(1, w5_cfg)
        burn = np.asarray(s.burn_in_obs)
        burn_jnp = jnp.asarray(burn, dtype=jnp.float32) if burn.shape[0] > 0 else None
        mem_r, mask_r, idx_r, w5_r = reconstruct_w512(
            apply_eval_w512_recon, params,
            mem_a, mask_a, idx_a, w5_a, burn_jnp, cfg, w5_cfg)
        mem_r, mask_r, idx_r = jax.tree_util.tree_map(
            jax.lax.stop_gradient, (mem_r, mask_r, idx_r))
        w5_r = jax.tree_util.tree_map(jax.lax.stop_gradient, w5_r)
        memories0.append(mem_r[0])
        masks0.append(mask_r[0])
        idxs0.append(idx_r[0])
        w512_states0.append({k: v[0] for k, v in w5_r.items()})
    # Stack W512 states
    w512_batched = {}
    if w512_states0:
        for k in w512_states0[0]:
            w512_batched[k] = jnp.stack([s[k] for s in w512_states0])
    return (jnp.stack(memories0), jnp.stack(masks0), jnp.stack(idxs0),
            w512_batched)


# ---- W512-adapted loss ----

def _ext_obs(po):
    obs = jnp.transpose(po["observations"], (1, 0, 2))
    return jnp.concatenate([obs, po["next_obs_final"][None]], 0)


def pack_batch(samples):
    return dict(
        observations=jnp.stack([jnp.asarray(s.observations, np.float32) for s in samples]),
        actions=jnp.stack([jnp.asarray(s.actions, np.int32) for s in samples]),
        rewards=jnp.stack([jnp.asarray(s.rewards, np.float32) for s in samples]),
        dones=jnp.stack([jnp.asarray(s.dones, np.float32) for s in samples]),
        log_probs=jnp.stack([jnp.asarray(s.log_probs, np.float32) for s in samples]),
        next_obs_final=jnp.stack(
            [jnp.asarray(np.asarray(s.next_observations)[-1], np.float32) for s in samples]),
        terminal=jnp.stack(
            [jnp.asarray(float(np.asarray(s.dones)[-1]), np.float32) for s in samples]),
        lag=jnp.stack(
            [jnp.asarray(int(s.collected_update_count), np.int32) for s in samples]),
    )


def compute_loss_w512(params, apply_eval_w512_raw, po, pr,
                      obs_o_ext, obs_r_ext,
                      target_vals_o, target_vals_r,
                      recon_o, recon_r, cfg, update_count, w5_cfg):
    """W512-adapted loss: V-trace (original) + AWR (relabeled).
    recon_o/recon_r are (mem, mask, idx, w512_state) tuples."""
    B, L = po["observations"].shape[0], po["observations"].shape[1]
    valid = jnp.ones((B, L))
    actions_t = po["actions"].T

    mem_o, mask_o, idx_o, w5_o = recon_o
    mem_r, mask_r, idx_r, w5_r = recon_r

    # ---- ORIGINAL (V-trace) online scan ----
    delay_o = {k: w5_o[k] for k in ("delay_buf", "delay_idx",
                                     "delay_count", "long_idx")}
    lg_o, val_o, _, _, _, _, _, _ = _scan_lax_w512(
        apply_eval_w512_raw, params, mem_o, mask_o, idx_o,
        w5_o["long_buf"], w5_o["long_mask"], obs_o_ext, cfg, w5_cfg, delay_o)
    v_online = val_o[:L].T
    log_pi_o, ent_o = _log_pi_and_entropy(lg_o[:L], actions_t)
    log_pi_o, ent_o = log_pi_o.T, ent_o.T

    v_target_tp1 = target_vals_o[:, 1:]
    boot_o = jnp.where(po["terminal"] > 0.5, 0.0, target_vals_o[:, L])

    vt_cfg = V.VtraceConfig(cfg.gamma, cfg.rho_bar, cfg.c_bar,
                            cfg.vt_clip_min, cfg.vt_clip_max)
    vt = V.vtrace_targets(log_pi_o, po["log_probs"], v_online, v_target_tp1,
                          po["rewards"], po["dones"], boot_o, vt_cfg)
    vt_vloss = V.vtrace_value_loss(vt, v_online, valid)
    vt_aloss = V.vtrace_actor_loss(vt, log_pi_o, ent_o, valid, cfg.ent_coef)
    ess = V.ess_fraction(vt.ratio, valid)

    # ---- RELABELED (AWR) online scan ----
    delay_r = {k: w5_r[k] for k in ("delay_buf", "delay_idx",
                                     "delay_count", "long_idx")}
    lg_r, val_r, _, _, _, _, _, _ = _scan_lax_w512(
        apply_eval_w512_raw, params, mem_r, mask_r, idx_r,
        w5_r["long_buf"], w5_r["long_mask"], obs_r_ext, cfg, w5_cfg, delay_r)
    logits_rel = jnp.transpose(lg_r[:L], (1, 0, 2))
    v_rel_online = val_r[:L].T
    logits_before = jax.lax.stop_gradient(logits_rel)

    target_rel = target_vals_r[:, :L]
    boot_r = jnp.where(pr["terminal"] > 0.5, 0.0, target_vals_r[:, L])

    lag = jnp.maximum(update_count - pr["lag"], 0)
    awr_valid = valid * (lag <= cfg.max_policy_lag).astype(jnp.float32)[:, None]

    awr_cfg = A.AWRConfig(cfg.gamma, cfg.beta, cfg.w_max, cfg.lambda_kl,
                          cfg.vt_clip_min, cfg.vt_clip_max)
    awr = A.awr_losses(logits_rel, logits_before, po["actions"], v_rel_online,
                       target_rel, pr["rewards"], pr["dones"], boot_r, awr_valid,
                       awr_cfg)

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


# ---- W512 target scan ----

def _target_scan_w512(scan_fn_w512, target_params, recon, obs_ext, w5_cfg):
    mem, mask, idx, w5_st = recon
    delay = {k: w5_st[k] for k in ("delay_buf", "delay_idx",
                                    "delay_count", "long_idx")}
    _, values, _, _, _, _, _, _ = scan_fn_w512(
        target_params, mem, mask, idx,
        w5_st["long_buf"], w5_st["long_mask"], obs_ext, w5_cfg, delay)
    return jax.lax.stop_gradient(values).T


# ---- W512 KL probe ----

def _window_log_softmax_w512(scan_fn_w512, params, recon, obs_ext, L, w5_cfg):
    mem, mask, idx, w5_st = recon
    delay = {k: w5_st[k] for k in ("delay_buf", "delay_idx",
                                    "delay_count", "long_idx")}
    lg, _, _, _, _, _, _, _ = scan_fn_w512(
        params, mem, mask, idx,
        w5_st["long_buf"], w5_st["long_mask"], obs_ext, w5_cfg, delay)
    a = jnp.transpose(lg[:L], (1, 0, 2))
    return _log_softmax(a)


def _kl_mean(lp_new, lp_old_sg):
    p = jax.nn.softmax(lp_new, axis=-1)
    return (p * (lp_new - lp_old_sg)).sum(-1).mean()


# ---- W512 full update ----

def full_p2_update_w512(params, target_params, opt_state, optimizer,
                        apply_eval_w512_recon, apply_eval_w512_raw,
                        scan_fn_w512,
                        samples_orig, samples_rel, cfg, update_count, w5_cfg):
    """W512-adapted combined controlled update with transactional KL gate.
    Identical gate logic to P2-Full-A (frozen §6/§14)."""
    import optax

    po = pack_batch(samples_orig)
    pr = pack_batch(samples_rel)
    obs_o_ext = _ext_obs(po)
    obs_r_ext = _ext_obs(pr)
    B, L = po["observations"].shape[0], po["observations"].shape[1]

    # reconstruction (stop_gradient, eager)
    recon_o = reconstruct_batch_w512(
        apply_eval_w512_recon, params, samples_orig, cfg, w5_cfg)
    recon_r = reconstruct_batch_w512(
        apply_eval_w512_recon, params, samples_rel, cfg, w5_cfg)
    recon_o_t = reconstruct_batch_w512(
        apply_eval_w512_recon, target_params, samples_orig, cfg, w5_cfg)
    recon_r_t = reconstruct_batch_w512(
        apply_eval_w512_recon, target_params, samples_rel, cfg, w5_cfg)

    # eager target value scans
    target_vals_o = _target_scan_w512(
        scan_fn_w512, target_params, recon_o_t, obs_o_ext, w5_cfg)
    target_vals_r = _target_scan_w512(
        scan_fn_w512, target_params, recon_r_t, obs_r_ext, w5_cfg)

    def loss_fn(p):
        return compute_loss_w512(
            p, apply_eval_w512_raw, po, pr, obs_o_ext, obs_r_ext,
            target_vals_o, target_vals_r, recon_o, recon_r, cfg,
            update_count, w5_cfg)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    mask = critic_only_mask(params)
    full_new = optax.apply_updates(params, updates)
    ema_full = ema_update(full_new, target_params, cfg.ema_tau)

    lp_old = jax.lax.stop_gradient(
        _window_log_softmax_w512(scan_fn_w512, params, recon_o, obs_o_ext,
                                  L, w5_cfg))

    trials = []
    accepted_scale = None
    accepted_kl = None
    new_params = None
    for scale in cfg.actor_step_scales:
        scaled_updates = jax.tree_util.tree_map(
            lambda m, u, s=scale: jnp.where(m, u, s * u), mask, updates)
        cand = optax.apply_updates(params, scaled_updates)
        lp_new = _window_log_softmax_w512(
            scan_fn_w512, cand, recon_o, obs_o_ext, L, w5_cfg)
        kl = float(np.asarray(_kl_mean(lp_new, lp_old)))
        trials.append({"scale": float(scale), "kl": kl})
        if np.isfinite(kl) and kl <= cfg.kl_replay_max:
            accepted_scale = float(scale)
            accepted_kl = kl
            new_params = cand
            break

    if accepted_scale is not None:
        policy_committed = True
        kl_rejected = False
        chosen_scale = accepted_scale
        policy_kl = accepted_kl
        new_target = ema_update(new_params, target_params, cfg.ema_tau)
        final_opt_state = new_opt_state
    else:
        policy_committed = False
        kl_rejected = True
        chosen_scale = -1.0
        policy_kl = trials[-1]["kl"] if trials else float("nan")
        new_params = _select_where(mask, full_new, params)
        final_opt_state = _select_opt_critic(new_opt_state, opt_state)
        new_target = _select_where(mask, ema_full, target_params)

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
