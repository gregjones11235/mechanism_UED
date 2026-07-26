"""P2-Full-A combined controlled update (frozen design v2, §3/§5/§6/§9/§14).

One update consumes K sampled complete-episode sequences (B=K) and their hindsight
relabeled twins:

  ORIGINAL-goal path  -> V-trace actor + value   (uses stored behavior log-prob mu)
  RELABELED-goal path -> hindsight AWR actor + relabeled-return value  (NO IS ratio)

Memory at each loss-window start is reconstructed from the nearest sparse anchor by
replaying the current network forward (<=128 burn-in steps) under stop_gradient; the
loss region itself is forwarded with current params via jax.lax.scan (differentiated).
The EMA target network supplies V_target for the TD targets / bootstrap / AWR baseline;
its memory is reconstructed separately with the target params and its loss-region scan
is computed EAGERLY (stop_gradient), outside the gradient graph, so the differentiated
graph contains only the two online lax.scans.

Loss = w_vtrace*(vtrace_actor + vf_coef*vtrace_value)
     + w_awr   *(awr_actor    + vf_coef*awr_value)

Gates (reported, never silently weakened): policy KL, ESS, per-sample policy-lag
(samples staler than MAX_POLICY_LAG are excluded from the AWR path), entropy floor.

Two apply callables are required (built by the caller / fputil):
  apply_eval_recon : jax.jit'd, batch-1 padded  -> (logits, value, mem_out); for the
                     <=128-step reconstruction (per-sequence, possibly batch==1).
  scan_fn          : jax.jit'd lax.scan region scanner over obs_seq [T,B,obs] with
                     B>=2 -> (logits [T,B,A], values [T,B], fm, fmask, fidx); used for
                     eager target scans and policy-KL probes.
  apply_eval_raw   : NON-jitted, non-padded forward (B>=2) used INSIDE the differentiated
                     lax.scan (so it is traced into the grad graph, not re-jitted).
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import awr as A
import memory_anchor as MA
import vtrace as V


@dataclass
class FullP2Config:
    window_mem: int = 128
    num_heads: int = 8
    num_layers: int = 2
    embed: int = 256
    action_dim: int = 43
    obs_dim: int = 8335
    gamma: float = 0.999
    rho_bar: float = 1.0
    c_bar: float = 1.0
    vt_clip_min: float = -50.0
    vt_clip_max: float = 300.0
    beta: float = 1.0
    w_max: float = 20.0
    lambda_kl: float = 0.01
    w_vtrace: float = 0.5
    w_awr: float = 0.5
    vf_coef: float = 0.5
    ent_coef: float = 0.002
    kl_max: float = 0.05             # LEGACY alias of kl_replay_max (kept for back-compat)
    kl_replay_max: float = 0.05      # single replay-update KL gate (TRANSACTIONAL; §6)
    kl_run_max: float = 0.1          # cumulative run-level KL ceiling (frozen §14 KL_MAX_RUN)
    actor_step_scales: tuple = (1.0, 0.5, 0.25, 0.125)   # actor step-scale retry on KL breach
    max_policy_lag: int = 16
    grad_clip: float = 1.0
    adam_eps: float = 1e-5             # frozen: align Full P2 adam eps with Control (1e-5); NOT an algorithm-inherent diff
    ent_floor: float = 0.05
    ema_tau: float = 0.995
    # NOTE — three DISTINCT KL quantities (never conflated; item 5):
    #   kl_replay_max (0.05): gates ONE replay update (transactional rollback below).
    #   kl_run_max    (0.10): gates the CUMULATIVE policy KL over a 24576-step run
    #                         (frozen §14 KL_MAX_RUN); NOT 0.05.
    #   kl_baseline_cumulative: tracked diagnostic = cumulative KL relative to the
    #                         Baseline (ckpt17500) policy; reported, own threshold.


# ----------------------------- helpers -----------------------------

def _log_softmax(logits):
    return logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)


def _log_pi_and_entropy(logits_tba, actions_tb):
    lp = _log_softmax(logits_tba)
    log_pi_taken = jnp.take_along_axis(lp, actions_tb[..., None], axis=-1)[..., 0]
    probs = jax.nn.softmax(logits_tba, axis=-1)
    entropy = -(probs * lp).sum(-1)
    return log_pi_taken, entropy


def _scan_lax(apply_eval_raw, params, memories, mem_mask, mem_idx, obs_seq, cfg):
    """lax.scan forward_eval over obs_seq [T,B,obs] (B>=2) carrying memory."""
    def body(carry, obs_t):
        mem, mask, idx = carry
        mem, mask, idx, lg, vl, _ = MA.step_forward(
            apply_eval_raw, params, mem, mask, idx, obs_t,
            cfg.window_mem, cfg.num_heads)
        return (mem, mask, idx), (lg, vl)
    (fm, fmask, fidx), (logits, values) = jax.lax.scan(
        body, (memories, mem_mask, mem_idx), obs_seq)
    return logits, values, fm, fmask, fidx


def reconstruct_batch(apply_eval_recon, params, samples, cfg):
    """Reconstruct the loss-window-start entering state per sample (<=128 burn-in),
    batched. Burn-in is stop_gradient. Returns (memory, mask, idx)."""
    memories0, masks0, idxs0 = [], [], []
    for s in samples:
        mem_a = jnp.asarray(s.pre_anchor_memory, dtype=jnp.float32)[None]
        mask_a, idx_a = MA.derive_anchor_entering_state(
            int(s.pre_anchor_step), cfg.window_mem, cfg.num_heads, 1)
        burn = np.asarray(s.burn_in_obs)
        if burn.shape[0] > 0:
            burn_b = jnp.asarray(burn, dtype=jnp.float32)[:, None, :]
            res = MA.reconstruct_state(
                apply_eval_recon, params, mem_a, mask_a, idx_a, burn_b,
                cfg.window_mem, cfg.num_heads)
            mem_a, mask_a, idx_a = jax.tree_util.tree_map(
                jax.lax.stop_gradient, res)
        memories0.append(mem_a[0])
        masks0.append(mask_a[0])
        idxs0.append(idx_a[0])
    return jnp.stack(memories0), jnp.stack(masks0), jnp.stack(idxs0)


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


def _ext_obs(po):
    obs = jnp.transpose(po["observations"], (1, 0, 2))               # [L,B,obs]
    return jnp.concatenate([obs, po["next_obs_final"][None]], 0)     # [L+1,B,obs]


# ----------------------------- loss (grad graph = 2 online scans) ----

def compute_loss(params, apply_eval_raw, po, pr, obs_o_ext, obs_r_ext,
                 target_vals_o, target_vals_r, recon_o, recon_r, cfg, update_count):
    B, L = po["observations"].shape[0], po["observations"].shape[1]
    valid = jnp.ones((B, L))
    actions_t = po["actions"].T                                       # [L,B]

    # ----- ORIGINAL (V-trace) online scan (grad) -----
    lg_o, val_o, _, _, _ = _scan_lax(
        apply_eval_raw, params, *recon_o, obs_o_ext, cfg)
    v_online = val_o[:L].T                                            # [B,L]
    log_pi_o, ent_o = _log_pi_and_entropy(lg_o[:L], actions_t)
    log_pi_o, ent_o = log_pi_o.T, ent_o.T

    v_target_tp1 = target_vals_o[:, 1:]                               # [B,L]
    boot_o = jnp.where(po["terminal"] > 0.5, 0.0, target_vals_o[:, L])

    vt_cfg = V.VtraceConfig(cfg.gamma, cfg.rho_bar, cfg.c_bar,
                            cfg.vt_clip_min, cfg.vt_clip_max)
    vt = V.vtrace_targets(log_pi_o, po["log_probs"], v_online, v_target_tp1,
                          po["rewards"], po["dones"], boot_o, vt_cfg)
    vt_vloss = V.vtrace_value_loss(vt, v_online, valid)
    vt_aloss = V.vtrace_actor_loss(vt, log_pi_o, ent_o, valid, cfg.ent_coef)
    ess = V.ess_fraction(vt.ratio, valid)

    # ----- RELABELED (AWR) online scan (grad) -----
    lg_r, val_r, _, _, _ = _scan_lax(
        apply_eval_raw, params, *recon_r, obs_r_ext, cfg)
    logits_rel = jnp.transpose(lg_r[:L], (1, 0, 2))                   # [B,L,A]
    v_rel_online = val_r[:L].T
    logits_before = jax.lax.stop_gradient(logits_rel)

    target_rel = target_vals_r[:, :L]                                 # [B,L]
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


# ----------------------------- diagnostics / gates ------------------

def _target_scan(scan_fn, target_params, recon, obs_ext):
    _, values, _, _, _ = scan_fn(target_params, *recon, obs_ext)
    return jax.lax.stop_gradient(values).T                            # [B,L+1]


def diagnose_vtrace(params, target_params, scan_fn, apply_eval_recon,
                    samples_orig, cfg):
    """V-trace intermediates the network feeds (for Gate 2 reference comparison)."""
    po = pack_batch(samples_orig)
    B, L = po["observations"].shape[0], po["observations"].shape[1]
    recon_o = reconstruct_batch(apply_eval_recon, params, samples_orig, cfg)
    recon_t = reconstruct_batch(apply_eval_recon, target_params, samples_orig, cfg)
    obs_o_ext = _ext_obs(po)
    lg_o, val_o, _, _, _ = scan_fn(params, *recon_o, obs_o_ext)
    v_online = val_o[:L].T
    log_pi_o, _ = _log_pi_and_entropy(lg_o[:L], po["actions"].T)
    log_pi_o = log_pi_o.T
    val_ot = _target_scan(scan_fn, target_params, recon_t, obs_o_ext)
    v_target_tp1 = val_ot[:, 1:]
    boot = jnp.where(po["terminal"] > 0.5, 0.0, val_ot[:, L])
    vt_cfg = V.VtraceConfig(cfg.gamma, cfg.rho_bar, cfg.c_bar,
                            cfg.vt_clip_min, cfg.vt_clip_max)
    vt = V.vtrace_targets(log_pi_o, po["log_probs"], v_online, v_target_tp1,
                          po["rewards"], po["dones"], boot, vt_cfg)
    return dict(
        log_pi=np.asarray(log_pi_o), log_mu=np.asarray(po["log_probs"]),
        v_online=np.asarray(v_online), v_target_tp1=np.asarray(v_target_tp1),
        rewards=np.asarray(po["rewards"]), dones=np.asarray(po["dones"]),
        bootstrap=np.asarray(boot), vs=np.asarray(vt.vs),
        ratio=np.asarray(vt.ratio),
        ess=float(np.asarray(V.ess_fraction(vt.ratio, jnp.ones((B, L))))),
    )


def ema_update(online_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda o, t: tau * t + (1.0 - tau) * o, online_params, target_params)


def _policy_kl_window(scan_fn, params_new, params_old, recon, obs_ext, L):
    lg_new, _, _, _, _ = scan_fn(params_new, *recon, obs_ext)
    lg_old, _, _, _, _ = scan_fn(params_old, *recon, obs_ext)
    a_new = jnp.transpose(lg_new[:L], (1, 0, 2))
    a_old = jnp.transpose(lg_old[:L], (1, 0, 2))
    lp_new = _log_softmax(a_new)
    lp_old = jax.lax.stop_gradient(_log_softmax(a_old))
    p_new = jax.nn.softmax(a_new, axis=-1)
    return (p_new * (lp_new - lp_old)).sum(-1).mean()


# ----------------------------- policy / critic partition -----------------------------
#
# The KL transactional gate must cover EVERY parameter the actor forward depends on:
# the encoder (obs->embed, inside the Transformer), the GTrXL/shared trunk, any
# goal/context params, and the actor head (actor_ln1/ln2/actor_out). The ONLY params
# NOT in the actor forward are the pure critic head (critic_ln1/critic_ln2/critic_out);
# those may commit independently of the KL gate (item 3). We classify by module-name
# path tokens, then prove the partition structurally in test_kl_transactional_gate
# (perturbing a critic-only leaf leaves the policy logits unchanged; perturbing an
# actor/encoder leaf changes them).
CRITIC_ONLY_TOKENS = ("critic_ln1", "critic_ln2", "critic_out")


def _path_str(keypath):
    parts = []
    for p in keypath:
        if hasattr(p, "key"):        # flax DictKey
            parts.append(str(p.key))
        elif hasattr(p, "idx"):      # flax SequenceKey
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts)


def is_critic_only_path(path_str):
    return any(tok in path_str for tok in CRITIC_ONLY_TOKENS)


def critic_only_mask(params):
    """Bool pytree (same structure as params): True for critic-only leaves."""
    return jax.tree_util.tree_map_with_path(
        lambda kp, _: is_critic_only_path(_path_str(kp)), params)


def classify_params(params):
    """Audit helper: split param leaves into critic-only vs policy-affecting."""
    paths_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    critic_only, policy, leaves = [], [], []
    for kp, leaf in paths_leaves:
        ps = _path_str(kp)
        arr = np.asarray(leaf)
        co = is_critic_only_path(ps)
        leaves.append({"path": ps, "shape": tuple(int(x) for x in arr.shape),
                       "size": int(arr.size), "critic_only": co})
        (critic_only if co else policy).append(ps)
    return {
        "critic_only_paths": critic_only,
        "policy_affecting_paths": policy,
        "n_critic_leaves": len(critic_only),
        "n_policy_leaves": len(policy),
        "critic_param_count": sum(l["size"] for l in leaves if l["critic_only"]),
        "policy_param_count": sum(l["size"] for l in leaves if not l["critic_only"]),
        "leaves": leaves,
    }


def _select_where(mask, a, b):
    """Per-leaf: take `a` where mask is True else `b` (mask/a/b same pytree)."""
    return jax.tree_util.tree_map(lambda m, x, y: jnp.where(m, x, y), mask, a, b)


def _select_opt_critic(new_opt, old_opt):
    """Opt-state merge: keep NEW moments for critic-only leaves, OLD for everything
    else (policy-affecting moments rolled back; shared adam count rolled back with the
    policy side on a rejected update — conservative, documented in the update docstring)."""
    return jax.tree_util.tree_map_with_path(
        lambda kp, n, o: jnp.where(is_critic_only_path(_path_str(kp)), n, o),
        new_opt, old_opt)


def _window_log_softmax(scan_fn, params, recon, obs_ext, L):
    lg, _, _, _, _ = scan_fn(params, *recon, obs_ext)
    a = jnp.transpose(lg[:L], (1, 0, 2))                              # [B,L,A]
    return _log_softmax(a)


def _kl_mean(lp_new, lp_old_sg):
    """mean KL(pi_new || pi_old) over the window (pi_old log-probs stop-gradient)."""
    p = jax.nn.softmax(lp_new, axis=-1)
    return (p * (lp_new - lp_old_sg)).sum(-1).mean()


# ----------------------------- update -----------------------------

def full_p2_update(params, target_params, opt_state, optimizer,
                   apply_eval_recon, apply_eval_raw, scan_fn,
                   samples_orig, samples_rel, cfg, update_count):
    """Combined controlled update with a TRANSACTIONAL single-replay KL gate.

    The gate covers ALL policy-affecting params (encoder + GTrXL/shared trunk + actor
    head + any goal/context params). The pure critic-only head commits INDEPENDENTLY.

    Procedure (frozen §6 / §14, v2.1 addendum; directive items 1-5):
      1. one optimizer step on the full gradient (clip_by_global_norm + adam);
      2. probe KL(pi_candidate || pi_current) on the ORIGINAL-goal loss window;
      3. try actor step scales cfg.actor_step_scales = (1.0,0.5,0.25,0.125): the
         policy-affecting part of the update is multiplied by the scale, the critic-only
         part always takes the full step. The first scale with KL <= cfg.kl_replay_max
         (0.05) is ACCEPTED -> commit candidate params + full opt moments + EMA target.
      4. if EVERY scale exceeds the gate -> KL_REJECTED_UPDATE: commit ONLY the
         critic-only head; ROLL BACK the policy-affecting params, their optimizer
         moments, and the EMA-target update derived from the rejected candidate.

    kl_replay_max (0.05) gates a SINGLE replay update only. The cumulative run KL is
    governed separately by kl_run_max (0.1, frozen KL_MAX_RUN); the 0.05 threshold is
    NEVER applied to the 24576-step cumulative KL (directive item 5).
    """
    import optax

    po = pack_batch(samples_orig)
    pr = pack_batch(samples_rel)
    obs_o_ext = _ext_obs(po)
    obs_r_ext = _ext_obs(pr)
    B, L = po["observations"].shape[0], po["observations"].shape[1]

    # reconstruction (stop_gradient, eager) for online + target, orig + relabeled
    recon_o = reconstruct_batch(apply_eval_recon, params, samples_orig, cfg)
    recon_r = reconstruct_batch(apply_eval_recon, params, samples_rel, cfg)
    recon_o_t = reconstruct_batch(apply_eval_recon, target_params, samples_orig, cfg)
    recon_r_t = reconstruct_batch(apply_eval_recon, target_params, samples_rel, cfg)

    # eager target value scans (stop_gradient constants for the grad graph)
    target_vals_o = _target_scan(scan_fn, target_params, recon_o_t, obs_o_ext)
    target_vals_r = _target_scan(scan_fn, target_params, recon_r_t, obs_r_ext)

    def loss_fn(p):
        return compute_loss(p, apply_eval_raw, po, pr, obs_o_ext, obs_r_ext,
                            target_vals_o, target_vals_r, recon_o, recon_r, cfg,
                            update_count)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    mask = critic_only_mask(params)
    full_new = optax.apply_updates(params, updates)               # full-step candidate
    ema_full = ema_update(full_new, target_params, cfg.ema_tau)

    # old policy logits on the probe window, computed once (stop-gradient reference)
    lp_old = jax.lax.stop_gradient(
        _window_log_softmax(scan_fn, params, recon_o, obs_o_ext, L))

    trials = []
    accepted_scale = None
    accepted_kl = None
    new_params = None
    for scale in cfg.actor_step_scales:
        # critic-only leaves keep the full update; policy-affecting leaves are scaled
        scaled_updates = jax.tree_util.tree_map(
            lambda m, u, s=scale: jnp.where(m, u, s * u), mask, updates)
        cand = optax.apply_updates(params, scaled_updates)
        lp_new = _window_log_softmax(scan_fn, cand, recon_o, obs_o_ext, L)
        kl = float(np.asarray(_kl_mean(lp_new, lp_old)))
        trials.append({"scale": float(scale), "kl": kl})
        if np.isfinite(kl) and kl <= cfg.kl_replay_max:
            accepted_scale = float(scale)
            accepted_kl = kl
            new_params = cand
            break

    if accepted_scale is not None:
        # ACCEPTED: commit candidate (policy + critic), full opt moments, EMA target
        policy_committed = True
        kl_rejected = False
        chosen_scale = accepted_scale
        policy_kl = accepted_kl
        new_target = ema_update(new_params, target_params, cfg.ema_tau)
        final_opt_state = new_opt_state
    else:
        # KL_REJECTED_UPDATE: critic-only commits; policy-affecting side rolled back
        policy_committed = False
        kl_rejected = True
        chosen_scale = -1.0
        policy_kl = trials[-1]["kl"] if trials else float("nan")
        new_params = _select_where(mask, full_new, params)          # critic new, policy old
        final_opt_state = _select_opt_critic(new_opt_state, opt_state)   # policy moments back
        new_target = _select_where(mask, ema_full, target_params)   # policy target frozen

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


def build_optimizer(lr, cfg):
    import optax
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                       optax.adam(lr, eps=cfg.adam_eps))
