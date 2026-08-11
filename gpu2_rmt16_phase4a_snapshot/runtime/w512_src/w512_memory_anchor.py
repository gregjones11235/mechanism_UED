"""W512 × P2-Replay — canonical single-step transition + reconstruction (CC2 corrected §二).

This is the W512 analog of the frozen RMT16 `rmt_memory_anchor.py`. It provides the ONE
per-step transition (`w512_step_forward`) used by BOTH collection (w512_collect) and replay
reconstruction / loss-region scan (w512_replay_learner), so anchor round-trip is bit-exact BY
CONSTRUCTION (collect and rebuild run the identical transition).

Network: ActorCriticTransformerW512 (network_w512.py). SAME base submodules as
ActorCriticTransformer (transformer / actor_ln1/2/out / critic_ln1/2/out) so ckpt17500 params
load unchanged and the canonical base params SHA (d4e85af5...) is reproduced. NEW params:
w512_cross_attn / w512_ln / w512_gate (zero-init) / w512_posenc (fixed sinusoidal). With the
gate at 0, z_t == h_t exactly at init (bit-exact gate), so step-0 == the pure GTrXL backbone.

Memory: w512_memory.py — a 512-step raw-history state = 128-step delay line + 384-step ring
buffer of GTrXL h_t. Cross-attention reads the 384 buffer with sinusoidal posenc.

carry_mode = reset128 (fixed by the directive): the 384 long buffer (long_buf/long_mask) is
CLEARED at every 128-step EPISODE-LOCAL segment boundary. This is the replay-compatible analog
of RMT16 reset128 ("clear tokens at the 128-step segment boundary"): it aligns exactly with the
replay anchors (episode steps 0,128,256,...), so reconstruction from an anchor re-applies the
same clear bit-exactly. NOTE: the historical bakeoff W512 reset128 keyed the clear to a GLOBAL
step counter — an on-policy artifact (it never combined with anchored replay). The canonical
replay run uses the episode-local counter, which is behaviorally equivalent within an episode
and is the only form compatible with sparse-anchor reconstruction. The delay line is NOT cleared
at the boundary (matches the historical reset128 which cleared only long_buf/long_mask).

Done convention (mirrors the proven historical ppo_tr_w512_reset128._env_step EXACTLY):
  * done_enter : the done carried INTO this step (= done_new of the previous step). Used for the
                 GTrXL mem_idx/mem_mask reset BEFORE the forward.
  * done_new   : the done produced by THIS step's env.step. Used for the w512 buffer advance
                 (w512_step resets done envs) and the seg_step reset.
Within a single episode (a complete trajectory) done is False everywhere except the terminal
transition, so the two coincide except at the terminal step; the explicit split keeps the
terminal-step semantics identical to _env_step.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import w512_memory as w5m


# ---------------------------------------------------------------------------
# carried W512 state = w512_memory state + episode-local segment counter
# ---------------------------------------------------------------------------

def w512_fresh_state(num_envs, w512_cfg):
    """Fresh per-env W512 carry: zero delay/long buffers + seg_step=0."""
    st = w5m.w512_init(int(num_envs), w512_cfg)
    st["seg_step"] = jnp.zeros((int(num_envs),), dtype=jnp.int32)
    return st


def w512_reset_state_on_done(w512_st, done):
    """Full reset of done envs to fresh (used after a TRUE terminal done in collection)."""
    d2 = done[:, None]
    d3 = done[:, None, None]
    return {
        "delay_buf":   jnp.where(d3, 0.0, w512_st["delay_buf"]),
        "delay_idx":   jnp.where(done, 0, w512_st["delay_idx"]),
        "delay_count": jnp.where(done, 0, w512_st["delay_count"]),
        "long_buf":    jnp.where(d3, 0.0, w512_st["long_buf"]),
        "long_mask":   jnp.where(d2, False, w512_st["long_mask"]),
        "long_idx":    jnp.where(done, 0, w512_st["long_idx"]),
        "seg_step":    jnp.where(done, 0, w512_st["seg_step"]),
    }


# ---------------------------------------------------------------------------
# eval apply (batch-1 -> 2 pad to dodge forward_eval squeeze; bit-exact slice back)
# ---------------------------------------------------------------------------

def make_apply_eval_w512(network):
    """apply_eval_w512(params, memories, obs, mask, long_buf, long_mask)
         -> (logits, value, mem_out, h_t).
    Wraps network.model_forward_eval. long_buf/long_mask of None => read path skipped
    (pure GTrXL backbone); the canonical reset128 run always passes real buffers."""
    def _raw(params, memories, obs, mask, long_buf, long_mask):
        pi, value, mem_out, h_t = network.apply(
            {"params": params}, memories, obs, mask,
            long_buf=long_buf, long_mask=long_mask,
            method=network.model_forward_eval)
        return pi.logits, value, mem_out, h_t

    def apply_eval_w512(params, memories, obs, mask, long_buf, long_mask):
        if memories.shape[0] == 1:
            memories2 = jnp.concatenate([memories, memories], 0)
            obs2 = jnp.concatenate([obs, obs], 0)
            mask2 = jnp.concatenate([mask, mask], 0)
            lb2 = None if long_buf is None else jnp.concatenate([long_buf, long_buf], 0)
            lm2 = None if long_mask is None else jnp.concatenate([long_mask, long_mask], 0)
            lg, vl, mo, ht = _raw(params, memories2, obs2, mask2, lb2, lm2)
            return lg[:1], vl[:1], mo[:1], ht[:1]
        return _raw(params, memories, obs, mask, long_buf, long_mask)

    return apply_eval_w512


# ---------------------------------------------------------------------------
# modular per-step pieces (shared by collect + scan + reconstruct)
# ---------------------------------------------------------------------------

def w512_reset128_clear(w512_st, segment_len):
    """RESET128: clear long_buf/long_mask when the episode-local seg_step is a positive
    multiple of segment_len (fires at the START of episode steps 128,256,...). Delay line is
    NOT cleared (matches historical reset128). Pure: returns a new state dict."""
    seg_step = w512_st["seg_step"]
    at_boundary = jnp.logical_and(seg_step > 0, (seg_step % segment_len) == 0)  # (N,)
    return {
        **w512_st,
        "long_buf":  jnp.where(at_boundary[:, None, None],
                               jnp.zeros_like(w512_st["long_buf"]), w512_st["long_buf"]),
        "long_mask": jnp.where(at_boundary[:, None],
                               jnp.zeros_like(w512_st["long_mask"]), w512_st["long_mask"]),
    }


def w512_advance_mask(mem_idx, mem_mask, done_enter, window_mem, num_heads):
    """GTrXL mem_idx decrement + one_hot OR into the mask; done_enter resets idx->window_mem
    and clears the mask (mirrors _env_step lines 147-153 exactly)."""
    mem_idx = jnp.where(done_enter, window_mem,
                        jnp.clip(mem_idx - 1, 0, window_mem)).astype(jnp.int32)
    mem_mask = jnp.where(done_enter[:, None, None, None],
                         jnp.zeros_like(mem_mask), mem_mask)
    ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
    ohot = ohot[:, None, None, :].repeat(num_heads, 1)
    mem_mask = jnp.logical_or(mem_mask, ohot)
    return mem_idx, mem_mask


def w512_step_forward(apply_eval_w512, params, memories, mem_mask, mem_idx,
                      w512_st, obs, done_enter, done_new,
                      window_mem, num_heads, w512_cfg, segment_len=128):
    """ONE W512 rollout iteration (the single source of truth).

    Order (== historical _env_step):
      1. reset128 clear of long_buf/long_mask at the episode-local 128 boundary;
      2. GTrXL mem_idx/mem_mask advance on done_enter;
      3. read the (post-clear) long buffer + forward_eval -> (logits, value, mem_out, h_t);
      4. roll memories, write mem_out;
      5. w512 buffer advance with pre-fusion h_t on done_new (w512_step resets done envs);
      6. seg_step advance (reset on done_new, else +1).

    Returns (post_memories, new_mask, new_idx, new_w512_st, logits, value, mem_pre)
    where mem_pre is the ENTERING memory (pre-roll) the loss reads.
    """
    w512_st = w512_reset128_clear(w512_st, segment_len)
    mem_pre = memories
    mem_idx, mem_mask = w512_advance_mask(mem_idx, mem_mask, done_enter,
                                          window_mem, num_heads)
    logits, value, mem_out, h_t = apply_eval_w512(
        params, memories, obs, mem_mask, w512_st["long_buf"], w512_st["long_mask"])
    post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
    new_w512_st = w5m.w512_step(w512_st, h_t, done_new, w512_cfg)
    new_w512_st = {
        **new_w512_st,
        "seg_step": jnp.where(done_new, 0, w512_st["seg_step"] + 1).astype(jnp.int32),
    }
    return (post_memories, mem_mask, mem_idx, new_w512_st,
            logits, value, mem_pre)


# ---------------------------------------------------------------------------
# reconstruction (burn-in from a sparse anchor; no grad)
# ---------------------------------------------------------------------------

def reconstruct_w512_state_with_network(network, apply_eval_w512, params,
                                        memories, mem_mask, mem_idx, w512_st,
                                        obs_segment, dones_enter_segment, dones_new_segment,
                                        window_mem, num_heads, w512_cfg, segment_len=128):
    """Bit-exact burn-in replay from a W512 anchor (<=128 steps). dones_enter_segment[t] is the
    entering done for step t; dones_new_segment[t] the new done. Within an episode both are
    False except possibly at a terminal step."""
    gap = int(obs_segment.shape[0])
    for t in range(gap):
        de = dones_enter_segment[t] if dones_enter_segment is not None else jnp.zeros(
            (memories.shape[0],), jnp.bool_)
        dn = dones_new_segment[t] if dones_new_segment is not None else jnp.zeros(
            (memories.shape[0],), jnp.bool_)
        (memories, mem_mask, mem_idx, w512_st,
         _lg, _vl, _mp) = w512_step_forward(
            apply_eval_w512, params, memories, mem_mask, mem_idx, w512_st,
            obs_segment[t], de, dn, window_mem, num_heads, w512_cfg, segment_len)
    return memories, mem_mask, mem_idx, w512_st


# ---------------------------------------------------------------------------
# loss-region / ground-truth scan (jitted lax.scan; B>=2 so no batch pad)
# ---------------------------------------------------------------------------

def make_scan_w512(network, window_mem, num_heads, w512_cfg, segment_len=128):
    """Return a jitted scan_fn(params, memories, mem_mask, mem_idx, w512_st,
                               obs_seq[T,B], dones_enter_seq[T,B], dones_new_seq[T,B])
         -> (logits[T,B,A], values[T,B])."""
    apply_eval_w512 = make_apply_eval_w512(network)

    def scan_fn(params, memories, mem_mask, mem_idx, w512_st,
                obs_seq, dones_enter_seq, dones_new_seq):
        def body(carry, inp):
            mem, mask, idx, st = carry
            obs_t, de_t, dn_t = inp
            (mem, mask, idx, st, lg, vl, _mp) = w512_step_forward(
                apply_eval_w512, params, mem, mask, idx, st, obs_t, de_t, dn_t,
                window_mem, num_heads, w512_cfg, segment_len)
            return (mem, mask, idx, st), (lg, vl)
        (_fm, _fmask, _fidx, _fst), (logits, values) = jax.lax.scan(
            body, (memories, mem_mask, mem_idx, w512_st),
            (obs_seq, dones_enter_seq, dones_new_seq))
        return logits, values

    return jax.jit(scan_fn)
