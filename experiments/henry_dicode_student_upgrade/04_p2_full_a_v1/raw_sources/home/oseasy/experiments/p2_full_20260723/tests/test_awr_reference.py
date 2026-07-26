"""Gate 4 / 2.8 — hindsight AWR vs independent numpy reference (CPU).

Verifies: relabeled discounted return, advantage weights + clip, weighted-BC
actor loss, KL penalty, relabeled-return value loss. No IS ratio anywhere.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import jax.numpy as jnp

import awr as A


def _log_softmax(x):
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def numpy_awr(logits, logits_before, actions, v_online, v_target, rewards,
              dones, bootstrap, valid, cfg):
    B, T, Ac = logits.shape
    nd = 1.0 - dones
    g = np.zeros((B, T + 1))
    g[:, T] = bootstrap
    for t in range(T - 1, -1, -1):
        g[:, t] = rewards[:, t] + cfg.gamma * nd[:, t] * g[:, t + 1]
    returns = np.clip(g[:, :T], cfg.vt_clip_min, cfg.vt_clip_max)
    adv = returns - v_target
    w = np.minimum(cfg.w_max, np.exp(adv / cfg.beta))
    logp = _log_softmax(logits)
    logp_b = _log_softmax(logits_before)
    oh = np.eye(Ac)[actions]
    log_pi_taken = (logp * oh).sum(-1)
    p = np.exp(logp)
    kl = (p * (logp - logp_b)).sum(-1)
    denom = max(valid.sum(), 1.0)
    bc = -(w * log_pi_taken * valid).sum() / denom
    klt = (kl * valid).sum() / denom
    actor = bc + cfg.lambda_kl * klt
    value = 0.5 * ((v_online - returns) ** 2 * valid).sum() / denom
    return returns, adv, w, kl, actor, value


def _rand(rng, shape):
    return rng.standard_normal(shape).astype(np.float32)


def test_awr_matches_reference():
    rng = np.random.RandomState(0)
    B, T, Ac = 3, 25, 7
    logits = _rand(rng, (B, T, Ac))
    logits_before = logits + _rand(rng, (B, T, Ac)) * 0.1
    actions = rng.randint(0, Ac, (B, T))
    v_online = _rand(rng, (B, T)) * 3
    v_target = _rand(rng, (B, T)) * 3
    rewards = (rng.rand(B, T) > 0.7).astype(np.float32)  # sparse +1
    dones = np.zeros((B, T), np.float32)
    dones[:, -1] = 1.0
    bootstrap = np.zeros((B,), np.float32)
    valid = np.ones((B, T), np.float32)
    cfg = A.AWRConfig()
    out = A.awr_losses(jnp.array(logits), jnp.array(logits_before), jnp.array(actions),
                       jnp.array(v_online), jnp.array(v_target), jnp.array(rewards),
                       jnp.array(dones), jnp.array(bootstrap), jnp.array(valid), cfg)
    ret_ref, adv_ref, w_ref, kl_ref, actor_ref, value_ref = numpy_awr(
        logits, logits_before, actions, v_online, v_target, rewards, dones,
        bootstrap, valid, cfg)
    assert np.allclose(np.asarray(out.returns), ret_ref, atol=1e-4), "returns mismatch"
    assert np.allclose(np.asarray(out.advantage), adv_ref, atol=1e-4)
    assert np.allclose(np.asarray(out.weights), w_ref, atol=1e-4)
    assert np.allclose(np.asarray(out.kl), kl_ref, atol=1e-4)
    assert np.allclose(float(out.actor_loss), actor_ref, atol=1e-4), "actor mismatch"
    assert np.allclose(float(out.value_loss), value_ref, atol=1e-4), "value mismatch"
    print("PASS test_awr_matches_reference")


def test_awr_weight_clip():
    B, T, Ac = 1, 4, 3
    logits = np.zeros((B, T, Ac), np.float32)
    logits_before = logits.copy()
    actions = np.zeros((B, T), np.int32)
    v_online = np.zeros((B, T), np.float32)
    v_target = np.full((B, T), -1000.0, np.float32)  # huge advantage -> huge raw weight
    rewards = np.ones((B, T), np.float32)
    dones = np.zeros((B, T), np.float32)
    bootstrap = np.zeros((B,), np.float32)
    valid = np.ones((B, T), np.float32)
    cfg = A.AWRConfig()
    out = A.awr_losses(jnp.array(logits), jnp.array(logits_before), jnp.array(actions),
                       jnp.array(v_online), jnp.array(v_target), jnp.array(rewards),
                       jnp.array(dones), jnp.array(bootstrap), jnp.array(valid), cfg)
    w = np.asarray(out.weights)
    assert np.all(w <= cfg.w_max + 1e-5), f"weight not clipped: {w.max()}"
    assert float(out.w_max_actual) <= cfg.w_max + 1e-5
    print("PASS test_awr_weight_clip")


def test_awr_kl_zero_when_same_policy():
    B, T, Ac = 2, 10, 5
    rng = np.random.RandomState(3)
    logits = _rand(rng, (B, T, Ac))
    actions = rng.randint(0, Ac, (B, T))
    v = _rand(rng, (B, T))
    rewards = np.zeros((B, T), np.float32)
    dones = np.zeros((B, T), np.float32)
    bootstrap = np.zeros((B,), np.float32)
    valid = np.ones((B, T), np.float32)
    cfg = A.AWRConfig()
    out = A.awr_losses(jnp.array(logits), jnp.array(logits), jnp.array(actions),
                       jnp.array(v), jnp.array(v), jnp.array(rewards),
                       jnp.array(dones), jnp.array(bootstrap), jnp.array(valid), cfg)
    assert np.allclose(np.asarray(out.kl), 0.0, atol=1e-5), "KL must be 0 for same policy"
    print("PASS test_awr_kl_zero_when_same_policy")


def test_awr_no_behavior_logprob_in_signature():
    """Structural: AWR must NOT take a behavior log-prob / log_mu argument."""
    import inspect
    sig = inspect.signature(A.awr_losses)
    params = set(sig.parameters.keys())
    forbidden = {"log_mu", "behavior_log_probs", "log_probs", "behavior_logprob",
                 "log_behavior", "mu"}
    assert not (params & forbidden), f"AWR must not take behavior logprob; found {params & forbidden}"
    sig2 = inspect.signature(A.relabeled_returns)
    assert not (set(sig2.parameters.keys()) & forbidden)
    print("PASS test_awr_no_behavior_logprob_in_signature")


if __name__ == "__main__":
    test_awr_matches_reference()
    test_awr_weight_clip()
    test_awr_kl_zero_when_same_policy()
    test_awr_no_behavior_logprob_in_signature()
    print("ALL_AWR_TESTS_PASS")
