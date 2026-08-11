"""P2-Full-A sparse-anchor -> GTrXL memory reconstruction (frozen design §2 / §11).

Replicates the rollout memory dynamics EXACTLY (see p2_v1_core.collect_rollout) so
that replay can rebuild the pre-action memory at any loss-window start from the
nearest stored anchor (<=128 steps behind) — bit-for-bit — instead of the forbidden
zero-memory mid-episode burn-in.

Rollout iteration being replicated (one env step):
    mem_idx  = clip(mem_idx - 1, 0, window_mem)
    ohot     = one_hot(mem_idx, window_mem + 1)                      # [B, wm+1]
    ohot     = ohot[:, None, None, :].repeat(num_heads, axis=1)      # [B, heads,1,wm+1]
    mem_mask = logical_or(mem_mask, ohot)                            # [B, heads,1,wm+1]
    mem_pre  = memories                                              # pre-action memory
    logits, value, mem_out = forward_eval(params, memories, obs, mem_mask)
    memories = roll(memories, -1, axis=1).at[:, -1].set(mem_out)
    # on done: memories=0, mem_mask=0, mem_idx=window_mem

An anchor at step `a` stores the ENTERING state at the top of iteration a, i.e.
(memories, mem_mask, mem_idx) BEFORE that iteration's mem_idx decrement. Replaying
iterations a..s-1 from that entering state yields the pre-action memory at step s.

NOTE on forward_eval batch==1: Henry's Transformer.forward_eval does `x.squeeze()`,
which collapses the batch axis when batch==1. make_apply_eval pads batch 1 -> 2
(duplicating env 0) and slices back; per-env memory is independent so the slice is
bit-exact. Loss-region scans use K>=2 sequences, so no padding is needed there.
"""
import jax
import jax.numpy as jnp


def fresh_rollout_state(window_mem, num_heads, num_layers, embed, batch,
                        dtype=jnp.float32):
    """Reset state for `batch` envs: zero memory, zero mask, mem_idx=window_mem."""
    memories = jnp.zeros((batch, window_mem, num_layers, embed), dtype)
    mem_mask = jnp.zeros((batch, num_heads, 1, window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.full((batch,), window_mem, dtype=jnp.int32)
    return memories, mem_mask, mem_idx


def _advance_mask(mem_idx, mem_mask, window_mem, num_heads):
    """Replicate the rollout's per-step mem_idx decrement + one_hot OR into the mask."""
    mem_idx = jnp.clip(mem_idx - 1, 0, window_mem).astype(jnp.int32)
    ohot = jax.nn.one_hot(mem_idx, window_mem + 1)               # [B, wm+1]
    ohot = ohot[:, None, None, :].repeat(num_heads, axis=1)      # [B, heads,1,wm+1]
    mem_mask = jnp.logical_or(mem_mask, ohot)
    return mem_idx, mem_mask


def make_apply_eval(network):
    """Return apply_eval(params, memories, obs, mask) -> (logits, value, mem_out).

    Wraps network.model_forward_eval; unwraps the distrax Categorical to logits and
    pads batch 1 -> 2 to dodge the forward_eval squeeze bug (bit-exact slice back).
    """
    def _raw(params, memories, obs, mask):
        pi, value, mem_out = network.apply(
            {"params": params}, memories, obs, mask,
            method=network.model_forward_eval)
        return pi.logits, value, mem_out

    def apply_eval(params, memories, obs, mask):
        if memories.shape[0] == 1:
            memories2 = jnp.concatenate([memories, memories], axis=0)
            obs2 = jnp.concatenate([obs, obs], axis=0)
            mask2 = jnp.concatenate([mask, mask], axis=0)
            logits, value, mem_out = _raw(params, memories2, obs2, mask2)
            return logits[:1], value[:1], mem_out[:1]
        return _raw(params, memories, obs, mask)

    return apply_eval


def step_forward(apply_eval, params, memories, mem_mask, mem_idx, obs,
                 window_mem, num_heads):
    """One rollout iteration. Returns (post_memories, new_mask, new_idx, logits,
    value, mem_pre) where mem_pre is the pre-action memory used for this forward."""
    mem_pre = memories
    mem_idx, mem_mask = _advance_mask(mem_idx, mem_mask, window_mem, num_heads)
    logits, value, mem_out = apply_eval(params, memories, obs, mem_mask)
    post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
    return post_memories, mem_mask, mem_idx, logits, value, mem_pre


@jax.jit
def _noop():  # placeholder to keep imports tidy; real jitted fn built by caller
    return 0


def reconstruct_state(apply_eval, params, memories, mem_mask, mem_idx,
                      obs_segment, window_mem, num_heads):
    """Replay step_forward over obs_segment [gap, B, obs_dim] under stop_gradient,
    returning the ENTERING state (memories, mem_mask, mem_idx) at the loss-window
    start. `memories/mem_mask/mem_idx` are the anchor's entering state.

    Intended to be called inside jax.lax.stop_gradient (burn-in must not backprop).
    """
    gap = int(obs_segment.shape[0])
    for t in range(gap):
        memories, mem_mask, mem_idx, _, _, _ = step_forward(
            apply_eval, params, memories, mem_mask, mem_idx,
            obs_segment[t], window_mem, num_heads)
    return memories, mem_mask, mem_idx


def scan_memory_eval(apply_eval, params, obs_seq, window_mem, num_heads,
                     init_state=None, num_layers=None, embed=None):
    """Simulate a rollout over obs_seq [L, B, obs_dim] from fresh (or init) state.

    Returns per-step ENTERING (pre-action) state and outputs:
        pre_memories [L, B, wm, layers, embed]
        pre_masks    [L, B, heads, 1, wm+1]
        pre_idxs     [L, B]
        logits       [L, B, A]
        values       [L, B]

    Used by Gate 1 to establish the ground-truth memory the anchors must reproduce.
    (Test/collector helper; the production collector snapshots only anchor steps.)
    """
    L, B = obs_seq.shape[0], obs_seq.shape[1]
    if init_state is None:
        memories, mem_mask, mem_idx = fresh_rollout_state(
            window_mem, num_heads, num_layers, embed, B,
            dtype=obs_seq.dtype)
    else:
        memories, mem_mask, mem_idx = init_state

    pre_memories, pre_masks, pre_idxs = [], [], []
    logits_l, values_l = [], []
    for t in range(L):
        pre_memories.append(memories)
        pre_masks.append(mem_mask)
        pre_idxs.append(mem_idx)
        memories, mem_mask, mem_idx, lg, vl, _ = step_forward(
            apply_eval, params, memories, mem_mask, mem_idx,
            obs_seq[t], window_mem, num_heads)
        logits_l.append(lg)
        values_l.append(vl)
    return (jnp.stack(pre_memories), jnp.stack(pre_masks),
            jnp.stack(pre_idxs).astype(jnp.int32),
            jnp.stack(logits_l), jnp.stack(values_l))


def derive_anchor_entering_state(pre_anchor_step, window_mem, num_heads, batch,
                                 dtype=jnp.float32):
    """Derive the anchor's ENTERING (mem_mask, mem_idx) from its absolute step.

    Within an episode (no reset before the anchor) the mask/idx depend ONLY on how
    many steps have elapsed since the episode-start reset, so they are a deterministic
    function of the anchor step:
      * step == 0      -> fresh reset: mask all-zero, idx = window_mem
      * step >= 128    -> steady fill: mask = [True]*window_mem + [False], idx = 0
    (Anchors only ever sit at 0,128,256,... so these two cases are exhaustive.)
    This is bit-exact equal to the state scan_memory_eval records at that step (Gate
    G1.4), so the learner can reconstruct without storing per-anchor mask/idx.

    `pre_anchor_step` may be a scalar or an int array [batch] (per-sequence anchors).
    Returns (mem_mask [batch, heads,1,wm+1], mem_idx [batch]).
    """
    steps = jnp.asarray(pre_anchor_step, dtype=jnp.int32)
    if steps.ndim == 0:
        steps = jnp.full((batch,), int(steps), dtype=jnp.int32)
    steady_mask = jnp.concatenate(
        [jnp.ones(window_mem, dtype=jnp.bool_),
         jnp.zeros(1, dtype=jnp.bool_)])                       # [wm+1]
    steady_mask = jnp.broadcast_to(
        steady_mask[None, None, None, :], (batch, num_heads, 1, window_mem + 1))
    zero_mask = jnp.zeros((batch, num_heads, 1, window_mem + 1), dtype=jnp.bool_)
    is_start = (steps == 0)[:, None, None, None]
    mem_mask = jnp.where(is_start, zero_mask, steady_mask)
    mem_idx = jnp.where(steps == 0, window_mem, 0).astype(jnp.int32)
    return mem_mask, mem_idx


def record_anchors(scan_pre_memories, scan_pre_masks, scan_pre_idxs, anchor_steps):
    """Select the entering-state anchors at `anchor_steps` from a full scan.

    Returns (anchor_memories [N,B,wm,layers,embed],
             anchor_masks    [N,B,heads,1,wm+1],
             anchor_idxs     [N,B]).
    """
    idx = jnp.asarray(anchor_steps, dtype=jnp.int32)
    return (scan_pre_memories[idx], scan_pre_masks[idx], scan_pre_idxs[idx])
