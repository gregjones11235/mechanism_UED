"""W512 × P2-Replay — on-policy PPO MAIN update (CC2 corrected §二).

Faithful reproduction of the frozen RMT16 "Original PPO 主更新" (rmt_ppo.ppo_update_rmt): clipped
ratio actor + clipped value + entropy, GAE with the FROZEN gamma=0.999 / lambda=0.8, lr=2e-5,
Adam eps=1e-5, clip_eps=0.2, vf_coef=0.5, ent_coef=0.002, max_grad_norm=1.0, update_epochs=1,
num_minibatches=2, value-target clip [-50,300]. compute_gae / build_ppo_optimizer are imported
VERBATIM from rmt_ppo (network-agnostic).

The rollout is re-forwarded under the current params via the SAME per-step transition used by
collection (w512_memory_anchor.w512_step_forward), so the read/update/reset128 dynamics are
identical everywhere and collection old_logp == re-forward new_logp (valid PPO ratio). The scan
carries (memories, mem_mask, mem_idx, w512_state) and reconstructs the two-done streams from the
single recorded done_new sequence + the rollout-start entering done (done_enter0): within a
rollout done_enter[t] = done_new[t-1], done_enter[0] = done_enter0. Minibatches are env-permuted
groups (preserving time order) scanned sequentially — mathematically the standard PPO objective.
This module is replay-agnostic: with the replay channel disabled the trainer reduces to a clean
W512 PPO run (the feature-off reference).
"""
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Network-agnostic frozen helpers, reused unchanged.
from rmt_ppo import compute_gae, build_ppo_optimizer   # noqa: F401  (re-exported for the driver)
from w512_memory_anchor import make_apply_eval_w512, w512_step_forward


def _scan_rollout_w512(network, apply_eval_w512, params, start, obs_seq, dones_seq,
                       cfg, w512_cfg, segment_len):
    """lax.scan w512_step_forward over obs_seq [T,Eg] from the (already env-sliced) start state.

    start: {memories[Eg,...], mem_mask[Eg,...], mem_idx[Eg], w512_state{[Eg,...]}, done_enter0[Eg]}.
    dones_seq [T,Eg] = recorded done_new per step. done_enter[t] = done_new[t-1], done_enter[0] =
    done_enter0. Returns (logits[T,Eg,A], values[T,Eg])."""
    done_enter_seq = jnp.concatenate(
        [start["done_enter0"][None, :], dones_seq[:-1, :]], axis=0)   # [T,Eg]

    def body(carry, inp):
        mem, mask, idx, st = carry
        obs_t, de_t, dn_t = inp
        (mem, mask, idx, st, lg, vl, _mp) = w512_step_forward(
            apply_eval_w512, params, mem, mask, idx, st, obs_t, de_t, dn_t,
            cfg["window_mem"], cfg["num_heads"], w512_cfg, segment_len)
        return (mem, mask, idx, st), (lg, vl)

    init = (start["memories"], start["mem_mask"], start["mem_idx"], start["w512_state"])
    _, (logits, values) = jax.lax.scan(
        body, init, (obs_seq, done_enter_seq, dones_seq))
    return logits, values


def ppo_update_w512(network, params, ppo_opt_state, optimizer,
                    rollout, advantages, targets, cfg, w512_cfg, segment_len, rng):
    """One PPO main update over the just-collected rollout.

    rollout: dict with start{memories,mem_mask,mem_idx,w512_state,done_enter0}, obs/actions/values/
             rewards/log_probs/dones [T,E]. advantages/targets [T,E].
    Returns (new_params, new_ppo_opt_state, metrics)."""
    apply_eval_w512 = make_apply_eval_w512(network)
    T, E = rollout["obs"].shape[:2]
    obs = jnp.asarray(rollout["obs"], jnp.float32)            # [T,E,obs]
    dones = jnp.asarray(rollout["dones"], jnp.float32)        # [T,E]
    actions = jnp.asarray(rollout["actions"], jnp.int32)      # [T,E]
    old_logp = jnp.asarray(rollout["log_probs"], jnp.float32)  # [T,E]
    old_values = jnp.asarray(rollout["values"], jnp.float32)   # [T,E]
    adv = jnp.asarray(advantages, jnp.float32)                # [T,E]
    tgt = jnp.asarray(targets, jnp.float32)                   # [T,E]
    start = rollout["start"]

    nmb = cfg["num_minibatches"]
    Eg = E // nmb

    def _loss_fn(params, idx_env):
        st = jax.tree_util.tree_map(lambda x: x[idx_env], start)
        obs_g = obs[:, idx_env]; don_g = dones[:, idx_env]
        act_g = actions[:, idx_env]; olp_g = old_logp[:, idx_env]
        ov_g = old_values[:, idx_env]; adv_g = adv[:, idx_env]; tgt_g = tgt[:, idx_env]
        logits, values = _scan_rollout_w512(network, apply_eval_w512, params, st,
                                            obs_g, don_g, cfg, w512_cfg, segment_len)
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(act_g)
        ent = pi.entropy()
        ratio = jnp.exp(log_prob - olp_g)
        adv_n = (adv_g - adv_g.mean()) / (adv_g.std() + 1e-8)
        la1 = ratio * adv_n
        la2 = jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_n
        actor_loss = -jnp.minimum(la1, la2).mean()
        v_clip = ov_g + jnp.clip(values - ov_g, -cfg["clip_eps"], cfg["clip_eps"])
        value_loss = 0.5 * jnp.maximum(jnp.square(values - tgt_g),
                                       jnp.square(v_clip - tgt_g)).mean()
        ent_mean = ent.mean()
        total = actor_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * ent_mean
        return total, (value_loss, actor_loss, ent_mean)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    ts_params = params
    opt_state = ppo_opt_state
    tot = []; vl = []; al = []; en = []; gn = []
    for _epoch in range(cfg["update_epochs"]):
        rng, _rng = jax.random.split(rng)
        perm = jax.random.permutation(_rng, E)
        for m in range(nmb):
            idx_env = jax.lax.dynamic_slice_in_dim(perm, m * Eg, Eg, axis=0)
            (total, (v_l, a_l, e_m)), grads = grad_fn(ts_params, idx_env)
            gnorm = optax.global_norm(grads)
            updates, opt_state = optimizer.update(grads, opt_state, ts_params)
            ts_params = optax.apply_updates(ts_params, updates)
            tot.append(float(total)); vl.append(float(v_l)); al.append(float(a_l))
            en.append(float(e_m)); gn.append(float(gnorm))

    metrics = dict(ppo_total=float(np.mean(tot)), ppo_value=float(np.mean(vl)),
                   ppo_actor=float(np.mean(al)), ppo_entropy=float(np.mean(en)),
                   ppo_grad_norm=float(np.mean(gn)), ppo_grad_norm_max=float(np.max(gn)),
                   ppo_finite=bool(np.all(np.isfinite(tot))))
    return ts_params, opt_state, metrics
