"""[SIL-COLLECT v1] Bare-chain golden-segment collector (stage 1: collect only, no training).

Runs a checkpoint's policy on TRAINING-SIDE seeds (never the frozen held-out
protocol seed), captures a fixed-shape K=K_PRE+K_POST state-action window
around each env's FIRST floor-2 entry, filters by episode return (top decile
among finished episodes) and resource health at floor-1 entry (design card
S3), and appends surviving segments to a capacity-capped golden buffer on
disk with (donor:seed:env:step-bucket) dedup keys and lowest-return eviction.

Zero patches to training code; read-only w.r.t. training semantics.

Usage (smoke):
  python experiments/analysis/sil_collect.py hydra.run.dir=/tmp/sil_smoke use_wandb=false seed=0 \
    +sil.ckpt_root=<...>/rl_checkpoints '+sil.step=400' +sil.tag=SMOKE \
    +sil.rollouts=1 +sil.num_envs=128 +sil.num_steps=2048
Full collection: drop num_envs/num_steps overrides (512 x 8192), rollouts=8+.
"""
import json
import os

import hydra
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from dicode.network import ActorCriticTransformer
from dicode.wrappers import BatchEnvWrapper
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.task_utils import EMBEDDING_SIZE, get_achievement_multi_hot
from minicraftax.envs.craftax import CraftaxAugObsTrain

K_PRE, K_POST = 48, 16
K = K_PRE + K_POST


def _state_core(st):
    for _ in range(3):
        if hasattr(st, "player_level"):
            return st
        st = getattr(st, "env_state")
    raise AttributeError("player_level not found")


def make_collect(config, env, env_params, num_envs, num_steps, mode="descend", skill_idx=0):
    tcfg = config.training

    def collect(train_state, rng):
        network = ActorCriticTransformer(
            action_dim=env.action_space(env_params).n,
            activation=tcfg.activation,
            hidden_layers=tcfg.hidden_layers,
            encoder_size=tcfg.embed_size,
            num_heads=tcfg.num_heads,
            qkv_features=tcfg.qkv_features,
            num_layers=tcfg.num_layers,
            gating=tcfg.gating,
            gating_bias=tcfg.gating_bias,
        )
        rng, reset_rng = jax.random.split(rng)
        obsv, env_state = env.reset(reset_rng, env_params)
        obs_dim = obsv.shape[-1]

        c = dict(
            env_state=env_state, obs=obsv,
            done=jnp.zeros((num_envs,), dtype=jnp.bool_),
            mem=jnp.zeros((num_envs, tcfg.window_mem, tcfg.num_layers, tcfg.embed_size)),
            mmask=jnp.zeros((num_envs, tcfg.num_heads, 1, tcfg.window_mem + 1), dtype=jnp.bool_),
            midx=jnp.zeros((num_envs,), dtype=jnp.int32) + (tcfg.window_mem + 1),
            finished=jnp.zeros((num_envs,), dtype=jnp.bool_),
            ret=jnp.zeros((num_envs,), dtype=jnp.float32),
            step=jnp.zeros((), dtype=jnp.int32),
            ring_obs=jnp.zeros((num_envs, K, obs_dim), dtype=jnp.float32),
            ring_act=jnp.zeros((num_envs, K), dtype=jnp.int32),
            ring_cum=jnp.zeros((num_envs, K), dtype=jnp.float32),
            wptr=jnp.zeros((num_envs,), dtype=jnp.int32),
            crossed1=jnp.zeros((num_envs,), dtype=jnp.bool_),
            e1food=jnp.zeros((num_envs,), dtype=jnp.float32),
            e1drink=jnp.zeros((num_envs,), dtype=jnp.float32),
            crossed2=jnp.zeros((num_envs,), dtype=jnp.bool_),
            postcnt=jnp.zeros((num_envs,), dtype=jnp.int32),
            stay_cnt=jnp.zeros((num_envs,), dtype=jnp.int32),
            prev_drink=jnp.full((num_envs,), 9.0),
            skill_prev=jnp.zeros((num_envs,), dtype=jnp.bool_),
            captured=jnp.zeros((num_envs,), dtype=jnp.bool_),
            cap_obs=jnp.zeros((num_envs, K, obs_dim), dtype=jnp.float32),
            cap_act=jnp.zeros((num_envs, K), dtype=jnp.int32),
            cap_cum=jnp.zeros((num_envs, K), dtype=jnp.float32),
            cap_wptr=jnp.zeros((num_envs,), dtype=jnp.int32),
            cross_step=jnp.zeros((num_envs,), dtype=jnp.int32),
            max_floor=jnp.zeros((num_envs,), dtype=jnp.int32),
            rng=rng,
        )

        def _step(c, _):
            done = c["done"]
            midx = jnp.where(done, tcfg.window_mem, jnp.clip(c["midx"] - 1, 0, tcfg.window_mem))
            mmask = jnp.where(
                done[:, None, None, None],
                jnp.zeros((num_envs, tcfg.num_heads, 1, tcfg.window_mem + 1), dtype=jnp.bool_),
                c["mmask"],
            )
            oh = jax.nn.one_hot(midx, tcfg.window_mem + 1)
            oh = oh[:, None, None, :].repeat(tcfg.num_heads, 1)
            mmask = jnp.logical_or(mmask, oh)

            rng, arng = jax.random.split(c["rng"])
            pi, _, mem_out = network.apply(
                train_state.params, c["mem"], c["obs"], mmask,
                method=network.model_forward_eval,
            )
            action = pi.sample(seed=arng)
            mem = jnp.roll(c["mem"], -1, axis=1).at[:, -1].set(mem_out)

            rng, srng = jax.random.split(rng)
            nobs, nstate, reward, ndone, info = env.step(srng, c["env_state"], action, env_params)

            active = jnp.logical_not(c["finished"])
            ret = c["ret"] + reward * active

            # ring write: the (obs, action) pair the policy just executed
            slot = c["wptr"] % K
            ar = jnp.arange(num_envs)
            ring_obs = c["ring_obs"].at[ar, slot].set(
                jnp.where(active[:, None], c["obs"], c["ring_obs"][ar, slot]))
            ring_act = c["ring_act"].at[ar, slot].set(
                jnp.where(active, action, c["ring_act"][ar, slot]))
            ring_cum = c["ring_cum"].at[ar, slot].set(
                jnp.where(active, c["ret"], c["ring_cum"][ar, slot]))
            wptr = c["wptr"] + active.astype(jnp.int32)

            core = _state_core(nstate)
            lvl = core.player_level.astype(jnp.int32)

            x1 = active & (~c["crossed1"]) & (lvl >= 1)
            e1food = jnp.where(x1, core.player_food.astype(jnp.float32), c["e1food"])
            e1drink = jnp.where(x1, core.player_drink.astype(jnp.float32), c["e1drink"])
            crossed1 = c["crossed1"] | x1

            skill_prev = c["skill_prev"]
            if mode == "descend":
                x2 = active & (~c["crossed2"]) & (lvl >= 2)
                crossed2 = c["crossed2"] | x2
                postcnt = jnp.where(x2, K_POST, c["postcnt"])
                counting = crossed2 & (~c["captured"]) & active
                postcnt = jnp.where(counting & (~x2), postcnt - 1, postcnt)
                snap = counting & ((postcnt <= 0) | ndone)
                stay_cnt = c["stay_cnt"]; prev_drink = c["prev_drink"]; mark = x2
            elif mode == "stay":
                # 64-step continuous floor-2 residence window
                on2 = active & (lvl >= 2)
                stay_cnt = jnp.where(on2, c["stay_cnt"] + 1, 0)
                crossed2 = c["crossed2"] | on2
                snap = active & (~c["captured"]) & (stay_cnt >= K)
                postcnt = c["postcnt"]; prev_drink = c["prev_drink"]; mark = snap
            elif mode == "resource":
                # thirst-refill event: drink rose while previously < 3
                drink = core.player_drink.astype(jnp.float32)
                snap = active & (~c["captured"]) & (c["prev_drink"] < 3.0) & (drink > c["prev_drink"] + 0.5)
                prev_drink = jnp.where(active, drink, c["prev_drink"])
                crossed2 = c["crossed2"] | (active & (lvl >= 2))
                postcnt = c["postcnt"]; stay_cnt = c["stay_cnt"]; mark = snap
            elif mode == "skill":
                # snapshot the 64-step window ending at the target
                # achievement's first flip (prep -> craft execution)
                bit = core.achievements[:, skill_idx] > 0.5
                snap = active & (~c["captured"]) & bit & (~c["skill_prev"])
                skill_prev = c["skill_prev"] | (bit & active)
                crossed2 = c["crossed2"] | (active & (lvl >= 2))
                postcnt = c["postcnt"]; stay_cnt = c["stay_cnt"]; prev_drink = c["prev_drink"]; mark = snap
            else:
                raise ValueError(f"unknown sil.mode {mode!r}")
            cap_obs = jnp.where(snap[:, None, None], ring_obs, c["cap_obs"])
            cap_act = jnp.where(snap[:, None], ring_act, c["cap_act"])
            cap_cum = jnp.where(snap[:, None], ring_cum, c["cap_cum"])
            cap_wptr = jnp.where(snap, wptr, c["cap_wptr"])
            cross_step = jnp.where(mark, c["step"], c["cross_step"])
            captured = c["captured"] | snap

            finished = c["finished"] | ndone
            max_floor = jnp.where(c["finished"], c["max_floor"], jnp.maximum(c["max_floor"], lvl))

            c2 = dict(c)
            c2.update(env_state=nstate, obs=nobs, done=ndone, mem=mem, mmask=mmask, midx=midx,
                      finished=finished, ret=ret, step=c["step"] + 1,
                      ring_obs=ring_obs, ring_act=ring_act, ring_cum=ring_cum, wptr=wptr,
                      crossed1=crossed1, e1food=e1food, e1drink=e1drink,
                      crossed2=crossed2, postcnt=postcnt, captured=captured,
                      stay_cnt=stay_cnt, prev_drink=prev_drink, skill_prev=skill_prev,
                      cap_obs=cap_obs, cap_act=cap_act, cap_cum=cap_cum, cap_wptr=cap_wptr,
                      cross_step=cross_step, max_floor=max_floor, rng=rng)
            return c2, None

        c, _ = jax.lax.scan(_step, c, None, num_steps)
        keys = ["cap_obs", "cap_act", "cap_cum", "cap_wptr", "captured", "finished", "ret",
                "e1food", "e1drink", "cross_step", "max_floor", "crossed2"]
        return {k: c[k] for k in keys}

    return collect


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config: DictConfig) -> None:
    sil = config.sil
    seed0 = int(sil.get("seed0", 10007))
    assert seed0 != 0, "seed0 must differ from the frozen held-out protocol seed"
    num_envs = int(sil.get("num_envs", 512))
    num_steps = int(sil.get("num_steps", 8192))
    rollouts = int(sil.get("rollouts", 4))
    out_dir = str(sil.get("out", "/workspace/golden_buffer"))
    tag = str(sil.get("tag", "run"))
    mode = str(sil.get("mode", "descend"))
    skill_name = str(sil.get("skill", "MAKE_IRON_ARMOUR"))
    if mode == "skill":
        from craftax.craftax.constants import Achievement
        skill_idx = int(Achievement[skill_name].value)
        print(f"[SIL-COLLECT] skill target: {skill_name} (idx {skill_idx})")
    else:
        skill_idx = 0
    fmin = float(sil.get("food_min", 5.0))
    dmin = float(sil.get("drink_min", 5.0))
    cap = int(sil.get("capacity", 512))
    os.makedirs(out_dir, exist_ok=True)

    if config.training.conditioning_type == "embedding":
        emb = config.gen_manager.embedding_model.embedding_size
    else:
        emb = EMBEDDING_SIZE  # 67 one-hot, matches training obs dims
    # [v1.2] Condition EXACTLY like the official bare-chain eval (one_hot ->
    # multi-hot of the original task's relevant achievements). v1 fed zeros:
    # OOD conditioning that suppressed descent for BOTH donors (training-seed
    # reached2 2.6-4.4% vs official touched2 15-16%).
    if config.training.conditioning_type == "one_hot":
        emb_vec = jnp.asarray(get_achievement_multi_hot(
            CraftaxAugObsTrain().relevant_achievements))
    else:
        emb_vec = jnp.zeros((emb,))
    print(f"[SIL-COLLECT] conditioning={config.training.conditioning_type} "
          f"vec_sum={float(emb_vec.sum()):.0f}")
    env = CraftaxAugObsTrain(
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=emb,
        task_embeddings=jnp.tile(emb_vec[None, :], (num_envs, 1)),
    )
    env_params = env.default_params.replace(max_timesteps=num_steps)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    ckpt = os.path.join(str(sil.ckpt_root), str(sil.step))
    print(f"[SIL-COLLECT] donor={tag} mode={mode} ckpt={ckpt} envs={num_envs} steps={num_steps} "
          f"rollouts={rollouts} K={K_PRE}+{K_POST}")
    ts = load_weights_only(ckpt, env, env_params, config.training)
    collect_jit = jax.jit(make_collect(config, env, env_params, num_envs, num_steps, mode, skill_idx))

    index_path = os.path.join(out_dir, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"segments": []}

    for i in range(rollouts):
        seed = seed0 + i
        out = jax.device_get(collect_jit(ts, jax.random.PRNGKey(seed)))
        fin = out["finished"].astype(bool)
        capd = out["captured"].astype(bool) & fin & (out["cap_wptr"] >= K)
        rets = out["ret"]
        thr = float(np.percentile(rets[fin], 90)) if fin.sum() else float("inf")
        keep = capd & (rets >= thr)
        if mode == "descend":
            keep = keep & (out["e1food"] >= fmin) & (out["e1drink"] >= dmin)
        existing = {s["key"] for s in index["segments"]}
        segs_o, segs_a, segs_r, metas = [], [], [], []
        for e in np.nonzero(keep)[0]:
            wp = int(out["cap_wptr"][e])
            order = (np.arange(wp - K, wp)) % K
            key = f"{tag}:{seed}:{int(e)}:{int(out['cross_step'][e]) // 256}"
            if key in existing:
                continue
            segs_o.append(out["cap_obs"][e][order].astype(np.float16))
            segs_a.append(out["cap_act"][e][order].astype(np.int16))
            segs_r.append((float(rets[e]) - out["cap_cum"][e][order]).astype(np.float16))
            metas.append(dict(key=key, ret=float(rets[e]), e1food=float(out["e1food"][e]),
                              e1drink=float(out["e1drink"][e]),
                              cross_step=int(out["cross_step"][e]),
                              max_floor=int(out["max_floor"][e]),
                              donor=tag, seed=int(seed), env=int(e)))
        if metas:
            fn = f"seg_{tag}_{seed}.npz"
            np.savez_compressed(os.path.join(out_dir, fn),
                                obs=np.stack(segs_o), act=np.stack(segs_a),
                                rtg=np.stack(segs_r),
                                ret=np.array([m["ret"] for m in metas], dtype=np.float32))
            for m in metas:
                m["file"] = fn
            index["segments"].extend(metas)
        if len(index["segments"]) > cap:  # lowest-return eviction
            index["segments"].sort(key=lambda s: -s["ret"])
            index["segments"] = index["segments"][:cap]
        json.dump(index, open(index_path, "w"), indent=1)
        print(f"[SIL-COLLECT] seed={seed} finished={int(fin.sum())}/{num_envs} "
              f"reached2={int((out['crossed2'] & fin).sum())} captured={int(capd.sum())} "
              f"kept(new)={len(metas)} thr(top10%)={thr:.2f} buffer={len(index['segments'])}")
    print(f"[SIL-COLLECT] DONE tag={tag} buffer_size={len(index['segments'])} -> {out_dir}")
    print("SIL_COLLECT_DONE")


if __name__ == "__main__":
    main()
