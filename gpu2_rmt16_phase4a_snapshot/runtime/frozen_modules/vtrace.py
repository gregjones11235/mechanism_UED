"""P2-Full-A V-trace (original-goal trajectories only).

Frozen design §3 (p2_full_frozen_design.md v2). Discrete actions.

V-trace is used ONLY on the ORIGINAL-goal replay trajectories, where the stored
behavior log-prob mu IS the true behavior policy that produced the actions, so
rho = pi/mu is a correct on->off importance correction. Hindsight-relabeled
trajectories MUST NOT use this ratio (cross-goal ratio forbidden) — they use AWR
(see awr.py). This module therefore consumes behavior log-probs; awr.py does not.

All functions operate on batched [B, T] arrays (B>=1). Pure JAX, deterministic.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class VtraceConfig(NamedTuple):
    gamma: float = 0.999
    rho_bar: float = 1.0
    c_bar: float = 1.0
    vt_clip_min: float = -50.0
    vt_clip_max: float = 300.0


class VtraceOutput(NamedTuple):
    vs: jnp.ndarray            # [B,T] v-trace targets (clipped)
    vs_tp1: jnp.ndarray        # [B,T] vs shifted by +1 (last = bootstrap_value)
    rho_bar: jnp.ndarray       # [B,T] clipped importance ratios min(rho_bar, pi/mu)
    c: jnp.ndarray             # [B,T] clipped trace-decay min(c_bar, pi/mu)
    ratio: jnp.ndarray         # [B,T] raw pi/mu
    delta: jnp.ndarray         # [B,T] rho_bar_t * (r_t + gamma*Vtarget_tp1*(1-done) - Vonline_t)
    pg_advantage: jnp.ndarray  # [B,T] rho_bar_t*(r_t + gamma*sg(vs_tp1)*(1-done) - sg(Vonline_t))


def _safe_ratio(log_pi: jnp.ndarray, log_mu: jnp.ndarray) -> jnp.ndarray:
    """pi/mu in log space, exp clipped to a sane finite range before use."""
    log_ratio = log_pi - log_mu
    log_ratio = jnp.clip(log_ratio, -30.0, 30.0)
    return jnp.exp(log_ratio)


def vtrace_targets(
    log_pi: jnp.ndarray,          # [B,T] current log pi(a_t|x_t) (taken action)
    log_mu: jnp.ndarray,          # [B,T] behavior log mu(a_t|x_t)
    values_online: jnp.ndarray,   # [B,T] V_online(x_t)
    target_values_tp1: jnp.ndarray,  # [B,T] V_target(x_{t+1}) (pre-shifted)
    rewards: jnp.ndarray,         # [B,T]
    dones: jnp.ndarray,           # [B,T] 1.0 at terminal step
    bootstrap_value: jnp.ndarray, # [B] V_target at state after last step
    cfg: VtraceConfig = VtraceConfig(),
) -> VtraceOutput:
    """Compute V-trace targets via reverse scan (frozen §3 sum form).

    G_t = delta_t + gamma*c_t*(1-done_t)*G_{t+1},  G_L = 0
    v_t = V_online(x_t) + G_t
    delta_t = rho_bar_t * (r_t + gamma*V_target(x_{t+1})*(1-done_t) - V_online(x_t))
    """
    ratio = _safe_ratio(log_pi, log_mu)
    rho_bar = jnp.minimum(cfg.rho_bar, ratio)
    c = jnp.minimum(cfg.c_bar, ratio)
    not_done = 1.0 - dones

    delta = rho_bar * (rewards + cfg.gamma * target_values_tp1 * not_done - values_online)

    # reverse scan for G_t
    def scan_fn(carry, t):
        g_next = carry
        g_t = delta[:, t] + cfg.gamma * c[:, t] * not_done[:, t] * g_next
        return g_t, g_t

    T = values_online.shape[1]
    init_g = jnp.zeros(values_online.shape[0], dtype=values_online.dtype)
    _, g_rev = jax.lax.scan(scan_fn, init_g, jnp.arange(T - 1, -1, -1))
    g = jnp.flip(g_rev, axis=0).T  # -> [B,T]

    vs = values_online + g
    vs = jnp.clip(vs, cfg.vt_clip_min, cfg.vt_clip_max)

    # vs_tp1: shift left by one; last column = bootstrap_value
    vs_tp1 = jnp.concatenate([vs[:, 1:], bootstrap_value[:, None]], axis=1)

    pg_advantage = rho_bar * (
        rewards + cfg.gamma * jax.lax.stop_gradient(vs_tp1) * not_done
        - jax.lax.stop_gradient(values_online)
    )

    return VtraceOutput(
        vs=vs, vs_tp1=vs_tp1, rho_bar=rho_bar, c=c, ratio=ratio,
        delta=delta, pg_advantage=pg_advantage,
    )


def vtrace_value_loss(out: VtraceOutput, values_online: jnp.ndarray,
                      valid_mask: jnp.ndarray) -> jnp.ndarray:
    """0.5 * mean_valid (V_online(x_t) - sg(v_t))^2."""
    target = jax.lax.stop_gradient(out.vs)
    sq = jnp.square(values_online - target) * valid_mask
    denom = jnp.maximum(valid_mask.sum(), 1.0)
    return 0.5 * sq.sum() / denom


def vtrace_actor_loss(
    out: VtraceOutput,
    log_pi_taken: jnp.ndarray,   # [B,T] log pi(a_t|x_t) (current)
    entropy: jnp.ndarray,        # [B,T] per-step policy entropy
    valid_mask: jnp.ndarray,     # [B,T]
    ent_coef: float = 0.002,
) -> jnp.ndarray:
    """IMPALA V-trace policy gradient (frozen §3).

    L = -mean_valid[ log_pi * sg(pg_advantage) ] - ent_coef*mean_valid[H]
    pg_advantage already carries rho_bar and stop_gradient'd targets.
    """
    pg = log_pi_taken * jax.lax.stop_gradient(out.pg_advantage)
    denom = jnp.maximum(valid_mask.sum(), 1.0)
    policy_term = -(pg * valid_mask).sum() / denom
    entropy_term = -(entropy * valid_mask).sum() / denom
    return policy_term + ent_coef * entropy_term


def ess_fraction(ratio: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    """Effective sample size fraction over valid steps (normalized weights=rho)."""
    w = ratio * valid_mask
    sum_w = w.sum()
    sum_w2 = jnp.square(w).sum()
    n = jnp.maximum(valid_mask.sum(), 1.0)
    ess = jnp.square(sum_w) / jnp.maximum(sum_w2, 1e-12)
    return ess / n
