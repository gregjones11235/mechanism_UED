"""[SIL-BC v1] Stage-2: session-level behavioural-cloning phase on the golden buffer.

Grafted after each PPO session (training.py), gated by +training.sil_coef
(>0 enables; absent/0 = block skipped entirely -> bitwise v1). Actor-only
advantage-weighted BC (Oh et al. 2018): w = clip((RTG - V)+, 0, w_max) with
V from the shared critic head under stop-gradient (no value-loss term).
GTrXL burn-in: the first `window_mem` steps of every K-step segment are
excluded from the loss (zero-state + burn-in prefix, design card v1 S5).
Uses a SEPARATE Adam (sil_lr), re-initialised each session: PPO's optimizer
state, LR-anneal count and clip chain are untouched.

Standalone smoke:
  python -m dicode.sil_bc hydra.run.dir=/tmp/silbc_smoke use_wandb=false seed=0 \
    +silbc.ckpt_root=<...>/rl_checkpoints '+silbc.step=0' \
    +training.sil_coef=1.0 [+training.sil_steps=12 +training.sil_batch=32]
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dicode.network import ActorCriticTransformer


def load_golden(buffer_dir, capacity=512):
    """Load every npz in the buffer; cap to `capacity` by highest episode return."""
    files = sorted(f for f in os.listdir(buffer_dir)
                   if f.startswith("seg_") and f.endswith(".npz"))
    obs, act, rtg, ret = [], [], [], []
    for fn in files:
        with np.load(os.path.join(buffer_dir, fn)) as z:
            obs.append(z["obs"].astype(np.float32))
            act.append(z["act"].astype(np.int32))
            rtg.append(z["rtg"].astype(np.float32))
            ret.append(z["ret"].astype(np.float32) if "ret" in z
                       else np.full((z["obs"].shape[0],), np.nan, np.float32))
    if not obs:
        return None
    obs = np.concatenate(obs); act = np.concatenate(act)
    rtg = np.concatenate(rtg); ret = np.concatenate(ret)
    if np.isnan(ret).any():
        raise ValueError("golden buffer has segments without 'ret' meta -- "
                         "re-collect with the meta patch applied (v1.2+)")
    if obs.shape[0] > capacity:
        keep = np.argsort(-ret)[:capacity]
        obs, act, rtg, ret = obs[keep], act[keep], rtg[keep], ret[keep]
    return obs, act, rtg, ret


def _build_network(tcfg, action_dim):
    return ActorCriticTransformer(
        action_dim=action_dim,
        activation=tcfg.activation,
        hidden_layers=tcfg.hidden_layers,
        encoder_size=tcfg.embed_size,
        num_heads=tcfg.num_heads,
        qkv_features=tcfg.qkv_features,
        num_layers=tcfg.num_layers,
        gating=tcfg.gating,
        gating_bias=tcfg.gating_bias,
    )


def run_sil_phase(config, train_state, action_dim=None):
    tcfg = config.training
    coef = float(tcfg.get("sil_coef", 0.0) or 0.0)
    if coef <= 0.0:
        return train_state
    bdir = str(tcfg.get("sil_buffer", "/workspace/golden_buffer"))
    steps = int(tcfg.get("sil_steps", 8))
    B = int(tcfg.get("sil_batch", 64))
    lr = float(tcfg.get("sil_lr", 3e-5))
    wmax = float(tcfg.get("sil_wmax", 5.0))
    seed = int(tcfg.get("sil_seed", 7))

    data = load_golden(bdir)
    if data is None:
        print("[SIL-BC] buffer empty -- phase skipped")
        return train_state
    obs_n, act_n, rtg_n, ret_n = data
    N, K, D = obs_n.shape
    burn = tcfg.get("sil_burn", None)
    if burn is None:
        # window_mem (=128) exceeds segment length K (=64): burn all pre-
        # crossing steps, loss covers the post-crossing tail. Card S5
        # amended 7/21; +training.sil_burn overrides (0 = ablation needle).
        burn = min(int(tcfg.window_mem), K - 16)
    burn = int(burn)
    assert 0 <= burn < K, f"burn-in {burn} vs segment length {K}"
    B = min(B, N)

    if action_dim is None:
        from minicraftax.envs.craftax import CraftaxAugObsTrain
        e = CraftaxAugObsTrain()
        action_dim = e.action_space(e.default_params).n
    network = _build_network(tcfg, action_dim)

    opt = optax.chain(optax.clip_by_global_norm(tcfg.max_grad_norm), optax.adam(lr))
    opt_state = opt.init(train_state.params)

    obs_j = jnp.asarray(obs_n); act_j = jnp.asarray(act_n); rtg_j = jnp.asarray(rtg_n)

    def seg_forward(params, ob, ac):
        Bc = ob.shape[0]
        mem0 = jnp.zeros((Bc, tcfg.window_mem, tcfg.num_layers, tcfg.embed_size))
        mmask0 = jnp.zeros((Bc, tcfg.num_heads, 1, tcfg.window_mem + 1), dtype=jnp.bool_)
        midx0 = jnp.zeros((Bc,), jnp.int32) + (tcfg.window_mem + 1)

        def step(carry, t):
            mem, mmask, midx = carry
            midx = jnp.clip(midx - 1, 0, tcfg.window_mem)
            oh = jax.nn.one_hot(midx, tcfg.window_mem + 1)
            oh = oh[:, None, None, :].repeat(tcfg.num_heads, 1)
            mmask = jnp.logical_or(mmask, oh)
            pi, v, mout = network.apply(params, mem, ob[:, t], mmask,
                                        method=network.model_forward_eval)
            mem = jnp.roll(mem, -1, axis=1).at[:, -1].set(mout)
            return (mem, mmask, midx), (pi.log_prob(ac[:, t]), v)

        _, (logps, vs) = jax.lax.scan(step, (mem0, mmask0, midx0), jnp.arange(K))
        return logps.T, vs.T  # (B, K)

    mask = (jnp.arange(K) >= burn).astype(jnp.float32)[None, :]

    def loss_fn(params, ob, ac, rt):
        logp, v = seg_forward(params, ob, ac)
        w = jnp.clip(jax.nn.relu(rt - jax.lax.stop_gradient(v)), 0.0, wmax)
        denom = jnp.maximum(mask.sum() * ob.shape[0] / K * K, 1.0)  # B*(K-burn)
        loss = -coef * (w * logp * mask).sum() / denom
        return loss, (w, v)

    @jax.jit
    def _upd(params, opt_state, ob, ac, rt):
        (loss, (w, v)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, ob, ac, rt)
        gn = optax.global_norm(g)
        upd, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, gn, w, v

    rng = np.random.default_rng(seed)
    params = train_state.params
    print(f"[SIL-BC] phase start: N={N} K={K} burn={burn} B={B} steps={steps} "
          f"lr={lr} coef={coef} wmax={wmax}")
    for s in range(steps):
        rows = rng.choice(N, size=B, replace=False)
        params, opt_state, loss, gn, w, v = _upd(
            params, opt_state, obs_j[rows], act_j[rows], rtg_j[rows])
        wm = w * mask
        print(f"[SIL-BC] step={s} loss={float(loss):.4f} grad_norm={float(gn):.3f} "
              f"w_mean={float(wm.sum() / jnp.maximum(mask.sum() * B / 1, 1)):.3f} "
              f"w_pos_frac={float(((w > 0) * mask).sum() / (mask.sum() * B) * K / K):.3f}")
    print("[SIL-BC] phase done")
    return train_state.replace(params=params)


# ---------------------------------------------------------------------------
# Standalone smoke: load a checkpoint, run one phase, verify params moved.
# ---------------------------------------------------------------------------
def _smoke_main():
    import hydra
    from omegaconf import DictConfig
    from dicode.utils.general.train_state_utils import load_weights_only
    from dicode.task_utils import EMBEDDING_SIZE
    from minicraftax.envs.craftax import CraftaxAugObsTrain

    @hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
    def main(config: DictConfig) -> None:
        if config.training.conditioning_type == "embedding":
            emb = config.gen_manager.embedding_model.embedding_size
        else:
            emb = EMBEDDING_SIZE
        env = CraftaxAugObsTrain(
            condition_on_task=config.training.condition_on_task,
            conditioning_type=config.training.conditioning_type,
            embedding_size=emb,
            task_embeddings=jnp.zeros((1, emb)),
        )
        env_params = env.default_params
        ckpt = os.path.join(str(config.silbc.ckpt_root), str(config.silbc.step))
        ts = load_weights_only(ckpt, env, env_params, config.training)
        p0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x), ts.params)
        ts2 = run_sil_phase(config, ts,
                            action_dim=env.action_space(env_params).n)
        delta = optax.global_norm(jax.tree_util.tree_map(
            lambda a, b: a - b, ts2.params, p0))
        print(f"[SIL-BC] SMOKE param_delta_norm={float(delta):.6f} "
              f"({'CHANGED' if float(delta) > 0 else 'UNCHANGED -- FAIL'})")
        print("SIL_BC_SMOKE_DONE")

    main()


if __name__ == "__main__":
    _smoke_main()
