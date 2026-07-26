"""v7fix56 P1' SIL designcheck (design doc §2.2/§2.3, S-series).

S.1  config keys present, sil defaults OFF (strict no-op posture)
S.2  episode_success_steps: only steps of in-window SUCCESSFUL episodes are marked
S.3  write_pool: admission gate + pool isolation (S1) + ring cursor + priority score
S.4  sample_pools: staleness window excludes old slots; empty pool contributes nothing
S.5  wiring: ppo_tr cond-gated SIL step / carry threading / return dict; training.py
     revert returns the PRE-session pool; run_dicode entropy-floor + regime-flush hooks
S.6  (--smoke) sil=false vs sil=true-with-forever-empty-pool produce BIT-IDENTICAL params
     after a real 2-update tiny session (S4 no-op discipline, beta gating included)

Run: python v7fix56_sil_designcheck.py [--smoke]
"""

import inspect
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

FAILED = []


def check(tag, ok, detail=""):
    status = "OK " if ok else "FAIL"
    print(f"[{status}] {tag}{(' — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILED.append(tag)


# ---------------------------------------------------------------- S.1 config
cfg = yaml.safe_load(open("conf/training/default.yaml", encoding="utf-8"))
check("S.1a sil defaults OFF", cfg.get("sil") is False)
check("S.1b sil_pools defaults empty", cfg.get("sil_pools") == [])
for k, v in [("sil_beta_max", 0.1), ("sil_vf_coef", 0.01), ("sil_pool_slots", 48),
             ("sil_windows_per_update", 8), ("sil_writes_per_update", 4),
             ("sil_staleness_sessions", 5), ("sil_warmup_sessions", 2),
             ("sil_ramp_sessions", 2), ("sil_entropy_floor", 0.12),
             ("sil_prioritized", True)]:
    check(f"S.1c {k}={v}", cfg.get(k) == v, f"got {cfg.get(k)}")

# ---------------------------------------------------------------- S.2 success mask
from dicode import sil as sil_lib  # noqa: E402

T, N = 8, 3
done = jnp.zeros((T, N), bool)
ach = jnp.zeros((T, N))
# env0: episode ends at t=3 WITH achievement, next episode ends t=6 WITHOUT
done = done.at[3, 0].set(True).at[6, 0].set(True)
ach = ach.at[3, 0].set(1.0)
# env1: episode ends at t=5 WITHOUT achievement
done = done.at[5, 1].set(True)
# env2: no episode ends in-window
sv = np.asarray(sil_lib.episode_success_steps(done, ach))
check("S.2a successful episode steps marked (env0 t0-3)", bool(sv[0:4, 0].all()))
check("S.2b failed follow-up episode unmarked (env0 t4-6)", not sv[4:7, 0].any())
check("S.2c failed episode unmarked (env1)", not sv[:, 1].any())
check("S.2d unfinished episode unmarked (env2)", not sv[:, 2].any())

# ---------------------------------------------------------------- S.3 write_pool
P, S, D, Wm, L, E, H = 2, 4, 5, 4, 1, 3, 2
state = sil_lib.init_sil_state(P, S, T, D, Wm, L, E, H)
obs = jnp.arange(T * N * D, dtype=jnp.float32).reshape(T, N, D)
action = jnp.ones((T, N), jnp.int32)
ret = jnp.full((T, N), 2.0)
mmask = jnp.ones((T, N, H, Wm + 1), bool)
mem = jnp.ones((Wm + T, N, L, E))
prio = jnp.full((T, N), 0.5)

# no admission -> nothing written, cursor unmoved
state1 = sil_lib.write_pool(state, 0, obs, action, ret, mmask, mem,
                            jnp.zeros((T, N), bool), prio, 100, 2)
check("S.3a admission gate: no success -> no write", not bool(state1["written"].any()))
check("S.3b cursor unmoved on empty admission", int(state1["cursor"][0]) == 0)

# env0 admitted into pool 0 only
state2 = sil_lib.write_pool(state, 0, obs, action, ret, mmask, mem,
                            jnp.asarray(sv), prio, 100, 2)
check("S.3c admitted window written to pool 0", bool(state2["written"][0, 0]))
check("S.3d pool isolation: pool 1 untouched", not bool(state2["written"][1].any()))
check("S.3e cursor advanced by n_written", int(state2["cursor"][0]) == 1)
check("S.3f stored obs is env0 column",
      bool(jnp.allclose(state2["obs"][0, 0], obs[:, 0])))
check("S.3g priority score stored", float(state2["score"][0, 0]) > 0)
check("S.3h iupd stamped", int(state2["iupd"][0, 0]) == 100)

# ---------------------------------------------------------------- S.4 sampling
rng = jax.random.PRNGKey(0)
batch, any_valid = sil_lib.sample_pools(state2, rng, 3, 100, 500, True)
check("S.4a fresh slot sampled (any_valid)", bool(any_valid))
check("S.4b sampled svalid nonzero", bool(batch["svalid"].any()))
batch_stale, any_stale = sil_lib.sample_pools(state2, rng, 3, 100 + 501, 500, True)
check("S.4c stale slots excluded", not bool(any_stale))
check("S.4d stale batch contributes nothing", not bool(batch_stale["svalid"].any()))
batch_empty, any_empty = sil_lib.sample_pools(state, rng, 3, 100, 500, True)
check("S.4e empty pool: any_valid False + zero svalid",
      (not bool(any_empty)) and (not bool(batch_empty["svalid"].any())))

# ---------------------------------------------------------------- S.5 wiring
import dicode.ppo_tr as ppo_tr  # noqa: E402

src = inspect.getsource(ppo_tr)
check("S.5a SIL step is lax.cond-gated (beta=0 leaves opt state untouched)",
      "sil_scale > 0.0" in src and "lambda ts: ts.apply_gradients(grads=sil_grads)" in src)
check("S.5b SIL runs AFTER the PPO epochs (Oh et al. ordering)",
      src.index("_update_epoch, update_state, None, config.update_epochs")
      < src.index("sil_lib.write_pool"))
check("S.5c carry threads sil_state", "runner_state, sil_state_c = update_carry" in src)
check("S.5d train returns sil_state", 'out["sil_state"] = sil_state_final' in src)
check("S.5e SIL key fold_in-derived (main rng untouched -> off/on-empty bit-identical)",
      "jax.random.fold_in(update_state[-1], 20260719)" in src
      and "jax.random.split(rng_sil)" not in src)
check("S.5f policy term stops gradient through the weight",
      "jax.lax.stop_gradient(w)" in src)
check("S.5g beta passed as device scalar (no per-value recompiles)",
      "jnp.asarray(sil_beta, jnp.float32)" in src)

import dicode.training as tr_mod  # noqa: E402

tsrc = inspect.getsource(tr_mod.run_session_training)
check("S.5h revert path returns PRE-session pool",
      "pre_session_sil_state" in tsrc
      and tsrc.index("pre_session_sil_state,") > tsrc.index("[guard][SESSION-REVERT]"))
rd = open("experiments/training/run_dicode.py", encoding="utf-8").read()
check("S.5i entropy-floor kill wired", "[sil][ENTROPY-FLOOR]" in rd)
check("S.5j regime flush wired", "[sil][REGIME-FLUSH]" in rd)
check("S.5k beta schedule warmup->ramp wired", "sil_warmup_sessions" in rd and "sil_ramp_sessions" in rd)

# ---------------------------------------------------------------- S.6 smoke
if "--smoke" in sys.argv:
    print("\n[S.6] tiny-session off/on-empty equivalence smoke (CPU, ~minutes)...")
    from types import SimpleNamespace

    from minicraftax.tasks.seed_tasks.original import Env as OriginalTask

    base_kwargs = dict(
        env_name="Craftax-Symbolic-v1", lr=3e-4, min_lr=3e-6, num_envs=4, num_steps=16,
        total_timesteps=10_000, update_epochs=2, num_minibatches=2, gamma=0.99,
        gae_lambda=0.65, clip_eps=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0,
        activation="relu", anneal_lr=True, qkv_features=16, embed_size=16, num_heads=2,
        num_layers=1, hidden_layers=16, window_mem=16, window_grad=8, gating=True,
        gating_bias=2.0, optimistic_reset_ratio=2, debug=False, use_wandb=False,
        mode="achievements", condition_on_task=False, completion_bonus_scale=0.0,
        completion_bonus_min=0.0, bonus_type="static", dynamic_bonus_k=2.0,
        max_updates_per_session=2, scoring_window_updates=2,
        value_target_clip_min=-50.0, value_target_clip_max=300.0,
        sil_beta_max=0.1, sil_vf_coef=0.01, sil_pool_slots=4,
        sil_windows_per_update=2, sil_writes_per_update=2, sil_staleness_sessions=5,
        sil_warmup_sessions=0, sil_ramp_sessions=1, sil_entropy_floor=0.20,
        sil_prioritized=True,
    )

    class Cfg(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    import wandb

    wandb.init(mode="disabled")

    def run_once(sil_flag, pools):
        cfg_ns = Cfg(**base_kwargs, sil=sil_flag, sil_pools=pools)
        train_fn = ppo_tr.make_train(cfg_ns, [OriginalTask], 2, None, None, 0)
        rng0 = jax.random.PRNGKey(7)
        if sil_flag:
            return jax.jit(train_fn)(rng0, None, 0.0, None, jnp.asarray(0.1, jnp.float32))
        return jax.jit(train_fn)(rng0, None, 0.0)

    out_off = run_once(False, [])
    # defeat_kobold can never fire in 16 random steps -> pool provably stays empty,
    # so the cond gate must keep params bit-identical even with beta=0.1.
    out_on = run_once(True, ["defeat_kobold"])
    leaves_off = jax.tree_util.tree_leaves(out_off["train_state"].params)
    leaves_on = jax.tree_util.tree_leaves(out_on["train_state"].params)
    same = all(bool(jnp.array_equal(a, b)) for a, b in zip(leaves_off, leaves_on))
    check("S.6a off vs on-with-empty-pool params bit-identical", same)
    check("S.6b sil_state returned & shaped",
          "sil_state" in out_on and out_on["sil_state"]["written"].shape == (1, 4))
    check("S.6c pool stayed empty (no phantom admission)",
          not bool(out_on["sil_state"]["written"].any()))
else:
    print("\n[S.6] SKIPPED (pass --smoke to run the tiny-session equivalence check)")

print()
if FAILED:
    print(f"DESIGNCHECK FAILED: {len(FAILED)} point(s): {FAILED}")
    sys.exit(1)
print("v7fix56 SIL designcheck: ALL GREEN")
