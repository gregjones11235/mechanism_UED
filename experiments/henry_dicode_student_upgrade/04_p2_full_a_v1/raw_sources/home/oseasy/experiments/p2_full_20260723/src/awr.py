"""P2-Full-A hindsight AWR (relabeled trajectories only).

Frozen design §5 (p2_full_frozen_design.md v2): Hindsight is SEPARATED from
V-trace. A hindsight-relabeled trajectory changes the goal-conditioning (obs tail
67 dims) and the reward. The ORIGINAL-goal behavior log-prob mu is NOT the
behavior policy for the relabeled goal-conditioned policy, so pi_relabeled/mu is a
CROSS-GOAL importance ratio and is FORBIDDEN here.

Therefore the relabeled path uses:
  * Value/Q: relabeled TD target (discounted relabeled return G'), NO IS ratio.
  * Actor: advantage-weighted behavior cloning (AWR) — a weighted MLE on the
    actions actually taken, weights w_t = min(w_max, exp(A'_t/beta)), with a KL
    constraint to the pre-update policy. NO importance ratio, NO behavior log-prob.

This module's functions deliberately take NO behavior log-prob / log_mu argument.
Gate G4.4 verifies (structurally + by perturbation) that no cross-goal ratio is
used. All arrays batched [B, T]. Pure JAX, deterministic.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AWRConfig(NamedTuple):
    gamma: float = 0.999
    beta: float = 1.0       # AWR temperature
    w_max: float = 20.0     # advantage-weight clip
    lambda_kl: float = 0.01  # soft KL penalty to pre-update policy
    vt_clip_min: float = -50.0
    vt_clip_max: float = 300.0


class AWROutput(NamedTuple):
    returns: jnp.ndarray       # [B,T] relabeled discounted return G' (clipped)
    advantage: jnp.ndarray     # [B,T] sg(G' - V_target(x'_t))
    weights: jnp.ndarray       # [B,T] sg(min(w_max, exp(A'/beta)))
    kl: jnp.ndarray            # [B,T] KL(pi_current || pi_before) per step
    actor_loss: jnp.ndarray    # scalar
    value_loss: jnp.ndarray    # scalar
    w_max_actual: jnp.ndarray  # scalar diag
    w_mean: jnp.ndarray        # scalar diag
    kl_mean: jnp.ndarray       # scalar diag


def relabeled_returns(
    rewards: jnp.ndarray,        # [B,T] relabeled reward r'
    dones: jnp.ndarray,          # [B,T]
    bootstrap_value: jnp.ndarray,  # [B] V_target at state after last step
    cfg: AWRConfig = AWRConfig(),
) -> jnp.ndarray:
    """G'_t = r'_t + gamma*(1-done_t)*G'_{t+1}, G'_L = bootstrap_value. Clipped."""
    not_done = 1.0 - dones

    def scan_fn(carry, t):
        g_next = carry
        g_t = rewards[:, t] + cfg.gamma * not_done[:, t] * g_next
        return g_t, g_t

    T = rewards.shape[1]
    _, g_rev = jax.lax.scan(scan_fn, bootstrap_value, jnp.arange(T - 1, -1, -1))
    g = jnp.flip(g_rev, axis=0).T  # [B,T]
    return jnp.clip(g, cfg.vt_clip_min, cfg.vt_clip_max)


def _log_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    return logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)


def awr_losses(
    logits: jnp.ndarray,              # [B,T,A] current policy logits on RELABELED obs
    logits_before: jnp.ndarray,       # [B,T,A] pre-update logits on relabeled obs (sg)
    actions: jnp.ndarray,             # [B,T] int actions actually taken
    values_online: jnp.ndarray,       # [B,T] V_online(x'_t)
    target_values: jnp.ndarray,       # [B,T] V_target(x'_t) (baseline for advantage)
    rewards: jnp.ndarray,             # [B,T] relabeled reward r'
    dones: jnp.ndarray,               # [B,T]
    bootstrap_value: jnp.ndarray,     # [B] V_target after last step
    valid_mask: jnp.ndarray,          # [B,T]
    cfg: AWRConfig = AWRConfig(),
) -> AWROutput:
    """AWR actor (weighted BC + KL) + relabeled-return value loss. No IS ratio.

    NOTE: there is intentionally NO behavior log-prob / log_mu parameter.
    """
    returns = relabeled_returns(rewards, dones, bootstrap_value, cfg)
    advantage = jax.lax.stop_gradient(returns - target_values)
    weights = jax.lax.stop_gradient(
        jnp.minimum(cfg.w_max, jnp.exp(advantage / cfg.beta))
    )

    # taken-action log prob under current policy
    logp = _log_softmax(logits)
    logp_before = jax.lax.stop_gradient(_log_softmax(logits_before))
    A = logits.shape[-1]
    actions_oh = jax.nn.one_hot(actions, A)
    log_pi_taken = (logp * actions_oh).sum(axis=-1)         # [B,T]

    # KL(pi_current || pi_before) per step (full distribution)
    p = jax.nn.softmax(logits, axis=-1)
    kl = (p * (logp - logp_before)).sum(axis=-1)            # [B,T]

    denom = jnp.maximum(valid_mask.sum(), 1.0)

    bc_term = -(weights * log_pi_taken * valid_mask).sum() / denom
    kl_term = (kl * valid_mask).sum() / denom
    actor_loss = bc_term + cfg.lambda_kl * kl_term

    value_target = jax.lax.stop_gradient(returns)
    value_loss = 0.5 * (jnp.square(values_online - value_target) * valid_mask).sum() / denom

    w_valid = weights * valid_mask
    w_max_actual = jnp.max(jnp.where(valid_mask > 0, weights, 0.0))
    w_mean = w_valid.sum() / denom
    kl_mean = kl_term

    return AWROutput(
        returns=returns, advantage=advantage, weights=weights, kl=kl,
        actor_loss=actor_loss, value_loss=value_loss,
        w_max_actual=w_max_actual, w_mean=w_mean, kl_mean=kl_mean,
    )
