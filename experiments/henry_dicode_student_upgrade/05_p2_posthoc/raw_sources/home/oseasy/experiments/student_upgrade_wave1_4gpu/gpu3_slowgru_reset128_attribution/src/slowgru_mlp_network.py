"""LC-SLOWGRU-RESET128-MATCHED_MLP network (Phase3 ATTRIBUTION experiment B).
DERIVED VERBATIM from the FROZEN SlowGRU network; the ONLY change is replacing the recurrent GRUCell
with a NON-recurrent 2-layer MLP whose param count MATCHES the GRUCell (256->768->256 = 394240 ==
GRUCell(256) = 394240, +0.000%, within the required +/-2%). h_new = slow_mlp_2(relu(slow_mlp_1(pooled)))
depends ONLY on the current period pooled input, NOT on the previous slow state h -> NO recurrence.
The held h between commits is a sample-and-hold of a MEMORYLESS function (carries no recurrent info).
Pooling window (SLOW_INTERVAL=32), slow_in, slow_read, slow_to_actor (zero-init), reset semantics and
Actor/Value dimensions are UNCHANGED. Public (teacher-inherited) params are bit-identical to Full.
Core comparisons: WITHIN_ROLLOUT_RECURRENCE_EFFECT = Full - MatchedMLP;
                    CAPACITY_REGULARIZATION_EFFECT = MatchedMLP - Control.

--- ORIGINAL FROZEN HEADER BELOW ---
LC-SLOWGRU-PPO network (GPU2 bakeoff candidate B). FROZEN design.

Keeps the original GTrXL (encoder + 128-step windowed FAST state + actor/value heads) EXACTLY,
inheriting ckpt17500 weights bit-for-bit (param paths identical to ActorCriticTransformer), and
ADDS a SLOW recurrent long-term state:

  every SLOW_INTERVAL=32 steps, attention-pool the GTrXL hidden x over that period -> project to
  SLOW_DIM=256 -> update a plain GRU cell (NO S5/SSM, avoids overlap with teammate P0);
  the slow state h persists across rollout boundaries; reset on true_done only;
  read = a projection of h enters the ACTOR branch only through `slow_to_actor`, ZERO-initialised
  -> at init the output is BIT-IDENTICAL to the healthy teacher (clean feature-off). Critic reads x.

No simulator hidden map / stair coords / future info. Per-env independent (no cross-env mixing).
"""
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from dicode.transformer.transformerXL import Transformer

SLOW_INTERVAL = 32   # slow update period in env steps (FROZEN; divides rollout 128 evenly -> 4/rollout)
SLOW_DIM = 256       # slow state dimension (FROZEN)
DIM = 256            # embed dim


def init_longstate(num_envs):
    return {
        "h": jnp.zeros((num_envs, SLOW_DIM), jnp.float32),
        "buf": jnp.zeros((num_envs, SLOW_INTERVAL, DIM), jnp.float32),   # current period's GTrXL hiddens
        "count": jnp.zeros((num_envs,), jnp.int32),
    }


class ActorCriticSlowGRUMLP(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = True
    gating_bias: float = 2.0
    use_longmem: bool = True   # feature-off switch

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
        # ---- NEW slow-GRU long-term channel (fresh) ----
        self.pool_q = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.pool_k = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.pool_v = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.slow_in = nn.Dense(SLOW_DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        # MATCHED_MLP (Phase3): NON-recurrent 2-layer MLP replacing the GRUCell. Param count:
        # 256*768+768 + 768*256+256 = 394240 == GRUCell(256) = 394240 (+0.000%, within +/-2%).
        self.slow_mlp_1 = nn.Dense(768, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.slow_mlp_2 = nn.Dense(SLOW_DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.slow_read = nn.Dense(DIM, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        # ZERO-init so the long-term contribution is exactly 0 at init -> output == teacher
        self.slow_to_actor = nn.Dense(self.hidden_layers, kernel_init=constant(0.0), bias_init=constant(0.0))

    # ----- slow-GRU update (vectorised over env axis; no cross-env mixing) -----
    def _slow_update(self, ls, x, reset):
        h = jnp.where(reset[:, None], jnp.zeros_like(ls["h"]), ls["h"])
        buf = jnp.where(reset[:, None, None], jnp.zeros_like(ls["buf"]), ls["buf"])
        count = jnp.where(reset, jnp.zeros_like(ls["count"]), ls["count"])

        # append current hidden x into the period buffer at position `count`
        oh = jax.nn.one_hot(count, SLOW_INTERVAL)                       # (E, SLOW_INTERVAL)
        buf = buf * (1 - oh[..., None]) + oh[..., None] * x[:, None, :]
        count = count + 1
        commit = (count >= SLOW_INTERVAL)                               # (E,)

        # attention pooling over the period (positions < count valid; at commit count==SLOW_INTERVAL)
        q = self.pool_q(x)[:, None, :]                                  # (E,1,DIM)
        k = self.pool_k(buf)                                            # (E,SLOW_INTERVAL,DIM)
        v = self.pool_v(buf)
        scores = jnp.sum(q * k, axis=-1) / jnp.sqrt(jnp.float32(DIM))  # (E,SLOW_INTERVAL)
        posmask = jnp.arange(SLOW_INTERVAL)[None, :] < count[:, None]
        scores = scores + jnp.where(posmask, 0.0, -1e9)
        w = jax.nn.softmax(scores, axis=-1)
        pooled = jnp.sum(w[:, :, None] * v, axis=1)                     # (E,DIM)
        pooled = nn.tanh(self.slow_in(pooled))                          # (E,SLOW_DIM)

        # MATCHED_MLP (Phase3): h_new is a MEMORYLESS function of the current pooled input only
        # (NO dependence on the previous slow state h) -> within-rollout recurrence removed. The
        # jnp.where(commit,...) below still holds h_new between commits (sample-and-hold), exactly
        # as Full holds the GRU state, so the ONLY isolated change is recurrence vs non-recurrence.
        h_new = self.slow_mlp_2(jax.nn.relu(self.slow_mlp_1(pooled)))   # NON-RECURRENT (no prev h)
        h = jnp.where(commit[:, None], h_new, h)
        buf = jnp.where(commit[:, None, None], jnp.zeros_like(buf), buf)
        count = jnp.where(commit, jnp.zeros_like(count), count)
        return {"h": h, "buf": buf, "count": count}

    def forward_eval(self, memories, obs, mask, longstate, reset):
        x, memory_out = self.transformer.forward_eval(memories, obs, mask)
        ls_new = self._slow_update(longstate, x, reset)
        if self.use_longmem:
            context = self.slow_read(ls_new["h"])
            actor_in = x + self.slow_to_actor(context)         # zero at init
        else:
            actor_in = x
        pi = distrax.Categorical(logits=self.actor_out(
            self._act(self.actor_ln2(self._act(self.actor_ln1(actor_in))))))
        value = self.critic_out(self._act(self.critic_ln2(self._act(self.critic_ln1(x)))))
        return pi, jnp.squeeze(value, axis=-1), memory_out, ls_new

    def model_forward_train_longmem(self, memories, obs, mask, true_done, longstate_prev):
        """PPO loss re-forward over window_grad segments. Reproduces forward_eval logits INCLUDING the
        slow context, so the importance ratio starts at exactly 1. The per-step context is recomputed
        by scanning the slow update over the FULL rollout from longstate_prev (mirrors P8)."""
        x = self.transformer.forward_train(memories, obs, mask)   # (B, W, 256)
        if not self.use_longmem:
            pi = distrax.Categorical(logits=self.actor_out(
                self._act(self.actor_ln2(self._act(self.actor_ln1(x))))))
            value = self.critic_out(self._act(self.critic_ln2(self._act(self.critic_ln1(x)))))
            return pi, jnp.squeeze(value, axis=-1)
        E = longstate_prev["h"].shape[0]
        B, W, D = x.shape
        T = (B // E) * W
        x_ET = x.reshape(E, T, D)
        td_T = jnp.swapaxes(true_done, 0, 1)                      # (T, E)
        x_T = jnp.swapaxes(x_ET, 0, 1)                            # (T, E, 256)

        def slow_step(ls, inp):
            x_t, td_t = inp
            ls = self._slow_update(ls, x_t, td_t)
            ctx = self.slow_read(ls["h"])
            return ls, ctx

        _, ctx_T = jax.lax.scan(slow_step, longstate_prev, (x_T, td_T))   # (T, E, 256)
        ctx_ET = jnp.swapaxes(ctx_T, 0, 1)                        # (E, T, 256)
        ctx_seg = ctx_ET.reshape(B, W, D)
        actor_in = x + self.slow_to_actor(ctx_seg)
        pi = distrax.Categorical(logits=self.actor_out(
            self._act(self.actor_ln2(self._act(self.actor_ln1(actor_in))))))
        value = self.critic_out(self._act(self.critic_ln2(self._act(self.critic_ln1(x)))))
        return pi, jnp.squeeze(value, axis=-1)
