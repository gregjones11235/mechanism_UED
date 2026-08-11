"""RMT16 × P2-Replay — sparse-anchor reconstruction EXTENDED with RMT16 tokens (Phase4A).

EXTENDS (does not modify) the frozen P2-Full-A `memory_anchor.py`. P2 reconstructs only
the GTrXL short memory (memories/mask/idx) from the nearest <=128-step anchor. RMT16 adds
16 persistent memory tokens (+ seg_buf + seg_count) that the actor reads every step and
updates every 128-step segment, so reconstruction must ALSO carry and evolve them — else
the rebuilt pre-action state is not bit-exact (directive §五).

Single source of truth: `rmt_step_forward` is the ONE per-step transition used by BOTH
collection (rmt_collect.py) and reconstruction/loss-region scan here. Bit-exactness of
anchor round-trip (directive gate 6/7) holds BY CONSTRUCTION because collect and rebuild
run the identical transition. The Persistent vs Reset128 difference is the single flag
`carry_mode`: at a 128-step segment boundary,
  * persistent : mem_tokens <- updated tokens (residual cross-attention), carried across;
  * reset128   : mem_tokens <- 0 (cleared at the boundary), single-window read/write only.
This is exactly the bakeoff single-change derivation (config diff carries only carry/reset).

NOTE: anchor steps coincide with RMT segment boundaries (both every 128 from episode
start), so an anchor's seg_count is 0 (segment just completed & reset). A <=128 burn-in
from an anchor therefore accumulates seg_buf but fires no token update unless gap==128;
token updates inside the 512-step LOSS region are handled by the differentiated scan.
"""
import jax
import jax.numpy as jnp
import numpy as np

import memory_anchor as MA          # frozen P2 GTrXL anchor helpers (reused)
import rmt16_memory as rmtm


# ----------------------------- apply (with RMT read) -----------------------------

def make_apply_eval_rmt(network):
    """apply_eval_rmt(params, memories, obs, mask, mem_tokens)
         -> (logits, value, mem_out, h_t).
    Wraps network.model_forward_eval; pads batch 1 -> 2 to dodge the forward_eval squeeze
    bug (per-env state is independent so the slice back is bit-exact)."""
    def _raw(params, memories, obs, mask, mem_tokens):
        pi, value, mem_out, h_t = network.apply(
            {"params": params}, memories, obs, mask, mem_tokens=mem_tokens,
            method=network.model_forward_eval)
        return pi.logits, value, mem_out, h_t

    def apply_eval_rmt(params, memories, obs, mask, mem_tokens):
        # base_gtrxl passes mem_tokens=None -> the network read path is skipped (pure GTrXL
        # backbone). None must propagate untouched (NOT concatenated) so the skip is preserved.
        if memories.shape[0] == 1:
            memories2 = jnp.concatenate([memories, memories], 0)
            obs2 = jnp.concatenate([obs, obs], 0)
            mask2 = jnp.concatenate([mask, mask], 0)
            tok2 = None if mem_tokens is None else jnp.concatenate([mem_tokens, mem_tokens], 0)
            lg, vl, mo, ht = _raw(params, memories2, obs2, mask2, tok2)
            return lg[:1], vl[:1], mo[:1], ht[:1]
        return _raw(params, memories, obs, mask, mem_tokens)

    return apply_eval_rmt


def make_update_fn(network, params):
    """update_fn(tokens (N,16,D), seg_buf (N,128,D)) -> new_tokens (N,16,D)."""
    def _update_fn(tokens, seg_buf):
        return network.apply({"params": params}, tokens, seg_buf,
                             method=network.update_rmt_tokens)
    return _update_fn


# ----------------------------- shared per-step transition -----------------------------

def entering_read_tokens(rmt_st, carry_mode):
    """The tokens the actor READS with on this step's forward.

    persistent / reset128 -> rmt_st["mem_tokens"] (read path executed).
    base_gtrxl            -> None  (read path SKIPPED; policy reduces to the pure GTrXL
                              backbone, RMT params get no gradient and stay frozen at init).

    This is the SINGLE source of truth for the base_gtrxl read-skip: collection (rmt_collect),
    PPO re-forward, replay reconstruction and the loss/ground-truth scan ALL go through
    rmt_step_forward, which calls this helper, so collection old_logp == re-forward new_logp
    by construction (valid PPO ratio).
    """
    if carry_mode == "base_gtrxl":
        return None
    return rmt_st["mem_tokens"]


def rmt_advance_tokens(rmt_st, h_t, done, update_fn, rmt_cfg, carry_mode):
    """Advance RMT state by one env step. SHARED by collect + reconstruction + loss scan.

    1. true-done envs fully reset (tokens/seg_buf/seg_count -> 0).
    2. store h_t into seg_buf; seg_count += 1.
    3. at segment boundary (seg_count >= segment_len):
         persistent          -> mem_tokens <- updated (residual attn), carried across;
         reset128 / base_gtrxl -> mem_tokens <- 0 (cleared at boundary).
       seg_buf/seg_count reset to 0 in ALL modes.

    base_gtrxl shares reset128's zeroing here: its tokens are never READ (entering_read_tokens
    returns None) so the zeroing is semantically inert, but it keeps the carried state bounded
    and bit-identical to the read-skipped forward. The RMT update path still runs (cheap) but
    its output is discarded, so RMT params receive no gradient and stay frozen at init.
    """
    st = rmtm.rmt16_reset_envs(rmt_st, done, rmt_cfg)
    st = rmtm.rmt16_store_h(st, h_t, rmt_cfg)
    is_boundary = (st["seg_count"] >= rmt_cfg.segment_len)          # (N,)
    updated = rmtm.rmt16_update_tokens(st, update_fn, rmt_cfg)     # resets seg_buf/count
    if carry_mode == "persistent":
        new_tokens = updated["mem_tokens"]
    elif carry_mode in ("reset128", "base_gtrxl"):
        new_tokens = jnp.zeros_like(updated["mem_tokens"])
    else:
        raise ValueError(f"unknown carry_mode={carry_mode}")
    return {
        "mem_tokens": jnp.where(is_boundary[:, None, None], new_tokens, st["mem_tokens"]),
        "seg_buf":    jnp.where(is_boundary[:, None, None], updated["seg_buf"], st["seg_buf"]),
        "seg_count":  jnp.where(is_boundary, updated["seg_count"], st["seg_count"]),
    }


def rmt_step_forward(apply_eval_rmt, params, memories, mem_mask, mem_idx,
                     rmt_st, obs, done, window_mem, num_heads, rmt_cfg,
                     carry_mode, update_fn):
    """ONE rollout iteration carrying GTrXL memory + RMT state.

    Mirrors the bakeoff RMT16 _env_step memory dynamics EXACTLY:
      mem_idx decrement (done -> window_mem), mask OR one_hot, read with ENTERING tokens,
      forward_eval -> (logits,value,mem_out,h_t), roll memories, advance RMT state.

    Returns (post_memories, new_mask, new_idx, new_rmt_st, logits, value, mem_pre,
             entering_tokens) where mem_pre / entering_tokens are the PRE-action state
             used for this step's forward (what the loss reads)."""
    entering_tokens = entering_read_tokens(rmt_st, carry_mode)
    mem_pre = memories
    # ---- GTrXL mask/idx advance (done resets idx to window_mem, clears mask) ----
    mem_idx = jnp.where(done, window_mem, jnp.clip(mem_idx - 1, 0, window_mem)).astype(jnp.int32)
    mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
    ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
    ohot = ohot[:, None, None, :].repeat(num_heads, 1)
    mem_mask = jnp.logical_or(mem_mask, ohot)
    # ---- forward (read with ENTERING tokens) ----
    logits, value, mem_out, h_t = apply_eval_rmt(
        params, memories, obs, mem_mask, entering_tokens)
    post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
    # ---- advance RMT state (store h_t, update/clear tokens at boundary) ----
    new_rmt_st = rmt_advance_tokens(rmt_st, h_t, done, update_fn, rmt_cfg, carry_mode)
    return (post_memories, mem_mask, mem_idx, new_rmt_st,
            logits, value, mem_pre, entering_tokens)


# ----------------------------- reconstruction (burn-in) -----------------------------

def reconstruct_rmt_state_with_network(network, apply_eval_rmt, params,
                                       memories, mem_mask, mem_idx, rmt_st,
                                       obs_segment, dones_segment,
                                       window_mem, num_heads, rmt_cfg, carry_mode):
    """Bit-exact burn-in replay from an RMT-extended anchor (<=128 steps, no grad)."""
    update_fn = make_update_fn(network, params)
    gap = int(obs_segment.shape[0])
    for t in range(gap):
        d_t = dones_segment[t] if dones_segment is not None else jnp.zeros((memories.shape[0],), jnp.bool_)
        (memories, mem_mask, mem_idx, rmt_st,
         _lg, _vl, _mp, _et) = rmt_step_forward(
            apply_eval_rmt, params, memories, mem_mask, mem_idx, rmt_st,
            obs_segment[t], d_t, window_mem, num_heads, rmt_cfg, carry_mode, update_fn)
    return memories, mem_mask, mem_idx, rmt_st


# ----------------------------- loss-region / ground-truth scan -----------------------------

def scan_rmt_eval(network, apply_eval_rmt, params, obs_seq, dones_seq,
                  window_mem, num_heads, rmt_cfg, carry_mode,
                  init_memories, init_mask, init_idx, init_rmt_st):
    """Scan rmt_step_forward over obs_seq [L,B,obs] carrying GTrXL + RMT state.

    Returns per-step ENTERING state + outputs:
        pre_memories [L,B,wm,layers,embed], pre_masks [L,B,heads,1,wm+1],
        pre_idxs [L,B], pre_tokens [L,B,num_tokens,embed],
        logits [L,B,A], values [L,B], final (memories,mask,idx,rmt_st).
    Used for the loss-region forward (differentiated) and Gate-1 ground truth."""
    update_fn = make_update_fn(network, params)
    L = int(obs_seq.shape[0])
    pre_mem, pre_mask, pre_idx, pre_tok, logits_l, values_l = [], [], [], [], [], []
    memories, mem_mask, mem_idx, rmt_st = init_memories, init_mask, init_idx, init_rmt_st
    for t in range(L):
        pre_mem.append(memories); pre_mask.append(mem_mask); pre_idx.append(mem_idx)
        pre_tok.append(rmt_st["mem_tokens"])
        d_t = dones_seq[t] if dones_seq is not None else jnp.zeros((memories.shape[0],), jnp.bool_)
        (memories, mem_mask, mem_idx, rmt_st, lg, vl, _mp, _et) = rmt_step_forward(
            apply_eval_rmt, params, memories, mem_mask, mem_idx, rmt_st,
            obs_seq[t], d_t, window_mem, num_heads, rmt_cfg, carry_mode, update_fn)
        logits_l.append(lg); values_l.append(vl)
    return (jnp.stack(pre_mem), jnp.stack(pre_mask), jnp.stack(pre_idx).astype(jnp.int32),
            jnp.stack(pre_tok), jnp.stack(logits_l), jnp.stack(values_l),
            (memories, mem_mask, mem_idx, rmt_st))
