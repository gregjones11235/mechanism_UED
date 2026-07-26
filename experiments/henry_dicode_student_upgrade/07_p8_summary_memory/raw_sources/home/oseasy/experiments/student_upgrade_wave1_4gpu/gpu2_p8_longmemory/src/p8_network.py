"""P8-LONGMEM-SUMMARY network (FROZEN design: ../reports/p8_frozen_design.md).

Keeps the original GTrXL (encoder + short-term windowed attention + actor/value heads)
EXACTLY, inheriting ckpt17500 weights bit-for-bit, and ADDS a long-term summary channel:
  - every K=64 steps, commit summary = summary_proj(mean of last K encoder outputs);
  - keep the last N=16 summaries (effective history 1024 steps);
  - Actor reads them via a single-head masked attention -> context;
  - context enters the ACTOR branch only through `summary_to_actor`, ZERO-initialised,
    so at init the model output is BIT-IDENTICAL to the healthy teacher (clean feature-off).
Long-term state persists across rollouts; reset on true_done only.
"""
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from dicode.transformer.transformerXL import Transformer

K = 64          # summary interval (FROZEN)
N = 16          # number of stored summaries (FROZEN) -> 1024-step coverage
DIM = 256       # summary/embed dim


def init_longstate(num_envs):
    return {
        "summaries": jnp.zeros((num_envs, N, DIM), jnp.float32),
        "valid": jnp.zeros((num_envs, N), jnp.bool_),
        "accum_sum": jnp.zeros((num_envs, DIM), jnp.float32),
        "accum_count": jnp.zeros((num_envs,), jnp.int32),
    }


class ActorCriticLongMem(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = True
    gating_bias: float = 2.0
    use_longmem: bool = True   # feature-off switch for the migration gate

    def setup(self):
        # ---- inherited GTrXL (param paths identical to ActorCriticTransformer) ----
        self.transformer = Transformer(
            encoder_size=self.encoder_size, num_heads=self.num_heads,
            qkv_features=self.qkv_features, num_layers=self.num_layers,
            gating=self.gating, gating_bias=self.gating_bias)
        act = nn.relu if self.activation == "relu" else nn.tanh
        self._act = act
        self.actor_ln1 = nn.Dense(self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_ln2 = nn.Dense(self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_ln1 = nn.Dense(self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_ln2 = nn.Dense(self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        # ---- NEW long-term summary channel (fresh) ----
        self.summary_proj = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.summ_q = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.summ_k = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.summ_v = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.summ_o = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        # ZERO-init so the long-term contribution is exactly 0 at init -> output == teacher
        self.summary_to_actor = nn.Dense(self.hidden_layers, kernel_init=constant(0.0), bias_init=constant(0.0))

    # ----- long-term state update (vectorised over env axis; no cross-env mixing) -----
    def _long_update(self, ls, encoded, reset):
        # reset on true_done BEFORE accumulating this step's obs
        summaries = jnp.where(reset[:, None, None], jnp.zeros_like(ls["summaries"]), ls["summaries"])
        valid = jnp.where(reset[:, None], jnp.zeros_like(ls["valid"]), ls["valid"])
        accum_sum = jnp.where(reset[:, None], jnp.zeros_like(ls["accum_sum"]), ls["accum_sum"])
        accum_count = jnp.where(reset, jnp.zeros_like(ls["accum_count"]), ls["accum_count"])
        # accumulate
        accum_sum = accum_sum + encoded
        accum_count = accum_count + 1
        # commit a summary when count reaches K
        commit = (accum_count >= K)
        mean = accum_sum / jnp.maximum(accum_count, 1)[:, None]
        new_summary = nn.tanh(self.summary_proj(mean))           # (E, DIM)
        # shift ring left, append new summary at the end
        summaries_new = jnp.roll(summaries, -1, axis=1).at[:, -1].set(new_summary)
        valid_new = jnp.roll(valid, -1, axis=1).at[:, -1].set(True)
        summaries = jnp.where(commit[:, None, None], summaries_new, summaries)
        valid = jnp.where(commit[:, None], valid_new, valid)
        accum_sum = jnp.where(commit[:, None], jnp.zeros_like(accum_sum), accum_sum)
        accum_count = jnp.where(commit, jnp.zeros_like(accum_count), accum_count)
        return {"summaries": summaries, "valid": valid, "accum_sum": accum_sum, "accum_count": accum_count}

    def _long_context(self, ls, query):
        # single-head masked attention over the N summary tokens; 0 if none valid
        q = self.summ_q(query)[:, None, :]               # (E,1,DIM)
        k = self.summ_k(ls["summaries"])                 # (E,N,DIM)
        v = self.summ_v(ls["summaries"])                 # (E,N,DIM)
        scores = jnp.sum(q * k, axis=-1) / jnp.sqrt(jnp.float32(DIM))  # (E,N)
        neg_inf = jnp.where(ls["valid"], 0.0, -1e9)
        scores = scores + neg_inf
        any_valid = jnp.any(ls["valid"], axis=-1, keepdims=True)       # (E,1)
        w = jax.nn.softmax(scores, axis=-1)                            # (E,N)
        w = jnp.where(any_valid, w, 0.0)
        ctx = jnp.sum(w[:, :, None] * v, axis=1)                       # (E,DIM)
        ctx = jnp.where(any_valid, ctx, 0.0)
        return self.summ_o(ctx)                                        # (E,DIM)

    def forward_eval(self, memories, obs, mask, longstate, reset, stop_short_grad=False):
        # short-term GTrXL forward (UNCHANGED)
        x, memory_out = self.transformer.forward_eval(memories, obs, mask)
        # long-term channel
        encoded = self.transformer.encoder(obs)
        if stop_short_grad:
            # Distillation-only: differentiate ONLY the long-memory channel. The inherited
            # short-term GTrXL + encoder outputs are treated as constants (their parameter
            # gradients are frozen/masked anyway). This bounds backprop memory to the 256-dim
            # long-mem path. NOTE: stopping a layer's INPUT does not block that layer's OWN
            # parameter gradient, so summary_proj / summary attention / summary_to_actor still
            # train. Forward values are identical either way (stop_gradient is grad-only).
            x = jax.lax.stop_gradient(x)
            memory_out = jax.lax.stop_gradient(memory_out)
            encoded = jax.lax.stop_gradient(encoded)
        ls_new = self._long_update(longstate, encoded, reset)
        if self.use_longmem:
            context = self._long_context(ls_new, encoded)
            actor_in = x + self.summary_to_actor(context)   # zero at init
        else:
            actor_in = x
        pi = distrax.Categorical(logits=self.actor_out(
            self._act(self.actor_ln2(self._act(self.actor_ln1(actor_in))))))
        value = self.critic_out(self._act(self.critic_ln2(self._act(self.critic_ln1(x)))))
        return pi, jnp.squeeze(value, axis=-1), memory_out, ls_new

    def model_forward_train_longmem(self, memories, obs, mask, true_done, longstate_prev):
        """PPO loss re-forward over a window_grad segment batch. Reproduces forward_eval's
        logits — INCLUDING the long-term context — so the importance ratio starts at exactly 1.

        Short-term path: `transformer.forward_train` (UNCHANGED, same as the teacher's
        model_forward_train) yields the per-step `x` over each segment.
        Long-term path: the per-step context is recomputed by scanning the summary-commit +
        readout logic over the FULL rollout, starting from `longstate_prev` (the long-state at
        rollout start, stored once per update exactly like `memories_previous`). This mirrors
        the rollout's forward_eval evolution step-for-step; with unchanged params it is
        bit-identical, so log_prob matches the stored rollout log_prob.

        Layout: obs is segment-batched (B=E*nseg, W, obs_dim) from reshaping the (E, T) rollout
        row-major, so segment b=e*nseg+s covers rollout steps [s*W, (s+1)*W) of env e. The
        full-rollout context (E, T, 256) is reshaped to the SAME (B, W, 256) layout to align
        element-wise with x.

        true_done: (E, T) — the true_done ENTERING each rollout step (reset flag used by
        forward_eval), so the context scan resets exactly when the rollout did.
        longstate_prev: per-env (E, ...) long-state at rollout start.

        With use_longmem=False this is EXACTLY the teacher's model_forward_train (heads on x).
        """
        x = self.transformer.forward_train(memories, obs, mask)   # (B, W, 256)
        if not self.use_longmem:
            pi = distrax.Categorical(logits=self.actor_out(
                self._act(self.actor_ln2(self._act(self.actor_ln1(x))))))
            value = self.critic_out(self._act(self.critic_ln2(self._act(self.critic_ln1(x)))))
            return pi, jnp.squeeze(value, axis=-1)
        E = longstate_prev["summaries"].shape[0]
        B, W, D = x.shape
        T = (B // E) * W
        obs_ET = obs.reshape(E, T, obs.shape[-1])                 # (E, T, obs_dim)
        encoded_ET = self.transformer.encoder(obs_ET)             # (E, T, 256)
        encoded_T = jnp.swapaxes(encoded_ET, 0, 1)                # (T, E, 256)
        td_T = jnp.swapaxes(true_done, 0, 1)                      # (T, E)

        def ctx_step(ls, inp):
            enc_t, td_t = inp
            ls = self._long_update(ls, enc_t, td_t)
            ctx = self._long_context(ls, enc_t)
            return ls, ctx

        _, ctx_T = jax.lax.scan(ctx_step, longstate_prev, (encoded_T, td_T))  # (T, E, 256)
        ctx_ET = jnp.swapaxes(ctx_T, 0, 1)                        # (E, T, 256)
        ctx_seg = ctx_ET.reshape(B, W, D)                         # aligns with x
        actor_in = x + self.summary_to_actor(ctx_seg)
        pi = distrax.Categorical(logits=self.actor_out(
            self._act(self.actor_ln2(self._act(self.actor_ln1(actor_in))))))
        value = self.critic_out(self._act(self.critic_ln2(self._act(self.critic_ln1(x)))))
        return pi, jnp.squeeze(value, axis=-1)
