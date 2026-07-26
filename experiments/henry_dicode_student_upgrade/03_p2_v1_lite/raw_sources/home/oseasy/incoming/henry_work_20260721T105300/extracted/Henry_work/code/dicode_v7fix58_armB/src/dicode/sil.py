"""v7fix56 P1' — Self-Imitation Learning buffer (device-side, per-wall pools).

Design (fable_research_reports/v7fix56设计.md §2.2, revised 2026-07-19 after reading the
real train loop): the whole session is ONE jax.jit around nested lax.scans, so the
originally-sketched host-side numpy ring buffer is architecturally impossible (no host
interaction inside the update scan, and shipping obs to host every update would move
~4 GB/update). The buffer therefore lives ON DEVICE as fixed-shape ring arrays threaded
through the update-scan carry, and persists across sessions by being returned from
train() and passed back in by the host loop.

Storage unit = one env's full rollout window (num_steps steps) together with the
transformer context needed to replay it through model_forward_train exactly like a PPO
minibatch row: obs / action / bootstrapped return-to-go (the GAE targets at insertion) /
per-step episode-success mask / the env's memory column / the per-step memory masks.
The (R - V)_+ weight is recomputed against the CURRENT value head inside the loss, which
gives SIL its self-annealing property (Oh et al. 2018): once V catches up, old windows'
weights go to zero on their own.

Admission (spec: "准入=episode 达成该 wall 成就位"): a window is admissible for pool p iff
some episode ENDING inside the window achieved pool p's wall achievement; the per-step
mask marks exactly the steps belonging to successful episodes (fragments of failed or
unfinished episodes in the same window carry zero SIL weight). Pools are per-wall and
strictly isolated (spec S1). Staleness (spec "政权标+5 session 淘汰"): slots older than
sil_staleness_sessions are excluded at sample time; regime flushes are done host-side by
zeroing `written` when the notebook's relay regime changes (run_dicode).

Pure shape-level functions, unit-testable on CPU with tiny dims (no network needed here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_sil_state(num_pools, slots, num_steps, obs_dim, window_mem, num_layers,
                   embed_size, num_heads):
    """Fixed-shape ring buffers for all pools. ~6 MB/slot at production dims."""
    P, S, T = num_pools, slots, num_steps
    return {
        "obs": jnp.zeros((P, S, T, obs_dim), dtype=jnp.float32),
        "action": jnp.zeros((P, S, T), dtype=jnp.int32),
        "ret": jnp.zeros((P, S, T), dtype=jnp.float32),
        "svalid": jnp.zeros((P, S, T), dtype=jnp.bool_),
        "mem": jnp.zeros((P, S, window_mem + T, num_layers, embed_size), dtype=jnp.float32),
        "mmask": jnp.zeros((P, S, T, num_heads, window_mem + 1), dtype=jnp.bool_),
        "score": jnp.zeros((P, S), dtype=jnp.float32),
        "written": jnp.zeros((P, S), dtype=jnp.bool_),
        "iupd": jnp.zeros((P, S), dtype=jnp.int32),
        "cursor": jnp.zeros((P,), dtype=jnp.int32),
    }


def resolve_achievement_key(info_keys, wall_name):
    """Trace-time lookup of the info dict key carrying `wall_name`'s achievement flag.

    craftax's log_achievements_to_info writes keys like "Achievements/<name>" whose value
    is achievements[i] * done — nonzero exactly at the final step of an episode that
    achieved it. Case-tolerant so a craftax rename fails loudly here, not silently.
    """
    want = wall_name.lower()
    for k in info_keys:
        kl = str(k).lower()
        if kl == f"achievements/{want}" or kl.endswith("/" + want):
            return k
    raise KeyError(
        f"[sil] no achievement info key for wall '{wall_name}'; "
        f"available keys: {sorted(str(k) for k in info_keys)[:80]}"
    )


def episode_success_steps(done, ach_flag):
    """(T, N) mask of steps belonging to episodes that ended IN-WINDOW with the achievement.

    done: (T, N) episode-end flags. ach_flag: (T, N), nonzero at the done step of a
    successful episode (craftax semantics: achievements * done). Steps of the pre-window
    episode fragment count as episode id 0 and are validated iff that episode's in-window
    done step was successful; fragments whose episode never ends in-window stay False.
    """
    done_i = done.astype(jnp.int32)
    cum = jnp.cumsum(done_i, axis=0)
    epi = cum - done_i  # episode id of each step (the done step belongs to its own episode)
    succ_at_done = (ach_flag > 0) & (done_i > 0)
    T, N = done_i.shape
    cols = jnp.broadcast_to(jnp.arange(N)[None, :], (T, N))
    table = jnp.zeros((T + 1, N), dtype=jnp.bool_).at[epi, cols].max(succ_at_done)
    return jnp.take_along_axis(table, epi, axis=0)


def write_pool(state, pool_idx, obs, action, ret, mmask, mem_cat, step_valid,
               prio_steps, global_update, writes_per_update):
    """Ring-write up to K admitted env-windows into pool `pool_idx`.

    obs (T,N,D) / action (T,N) / ret (T,N) / mmask (T,N,H,Wm+1) / mem_cat (Wm+T,N,L,E)
    step_valid (T,N) from episode_success_steps / prio_steps (T,N) = relu(targets - value).
    Selection among admitted envs is by mean positive-advantage score (top-K).
    """
    S = state["written"].shape[1]
    cnt = step_valid.sum(axis=0)
    score_env = (prio_steps * step_valid).sum(axis=0) / jnp.maximum(cnt, 1)
    admitted = step_valid.any(axis=0)
    rank = jnp.where(admitted, 1.0 + score_env, -jnp.inf)
    top_scores, top_envs = jax.lax.top_k(rank, writes_per_update)

    cursor = state["cursor"][pool_idx]
    new_state = dict(state)
    for j in range(writes_per_update):  # static unroll, K is small
        ok = top_scores[j] > -jnp.inf
        slot = (cursor + j) % S
        e = top_envs[j]

        def put(name, col):
            old = jax.lax.dynamic_index_in_dim(new_state[name][pool_idx], slot,
                                               axis=0, keepdims=False)
            new_state[name] = new_state[name].at[pool_idx, slot].set(
                jnp.where(ok, col, old)
            )

        put("obs", jnp.take(obs, e, axis=1))
        put("action", jnp.take(action, e, axis=1))
        put("ret", jnp.take(ret, e, axis=1))
        put("svalid", jnp.take(step_valid, e, axis=1))
        put("mem", jnp.take(mem_cat, e, axis=1))
        put("mmask", jnp.take(mmask, e, axis=1))
        put("score", score_env[e])
        put("written", ok)  # where(ok, True, old) == old when not ok
        put("iupd", jnp.asarray(global_update, dtype=jnp.int32))
    n_written = (top_scores > -jnp.inf).sum().astype(jnp.int32)
    new_state["cursor"] = state["cursor"].at[pool_idx].set((cursor + n_written) % S)
    return new_state


def sample_pools(state, rng, windows_per_pool, global_update, staleness_updates,
                 prioritized):
    """Sample M windows from EACH pool (concatenated); stale/unwritten slots contribute
    nothing because their svalid rows are zeroed. Returns (batch dict, any_valid scalar)."""
    P, S = state["written"].shape
    fresh = state["written"] & (state["iupd"] >= global_update - staleness_updates)
    batches = []
    any_valid = jnp.zeros((), dtype=jnp.bool_)
    for p in range(P):  # static unroll, P is small
        rng, k = jax.random.split(rng)
        valid = fresh[p]
        base = state["score"][p] if prioritized else jnp.ones((S,), jnp.float32)
        w = jnp.where(valid, base + 1e-6, 0.0)
        has = valid.any()
        w_safe = jnp.where(has, w, jnp.ones_like(w))
        idx = jax.random.categorical(k, jnp.log(w_safe), shape=(windows_per_pool,))
        row_ok = valid[idx]  # (M,) — kills stale/unwritten/empty-pool draws
        b = {name: state[name][p][idx]
             for name in ("obs", "action", "ret", "svalid", "mem", "mmask")}
        b["svalid"] = b["svalid"] & row_ok[:, None]
        batches.append(b)
        any_valid = any_valid | has
    batch = {name: jnp.concatenate([b[name] for b in batches], axis=0)
             for name in batches[0]}
    return batch, any_valid


def pool_fill_fraction(state):
    """(scalar) mean written fraction across pools — for wandb telemetry."""
    return state["written"].mean()
