"""v7fix56 P1' SIL buffer unit tests (mirror of designcheck S.2-S.4).

jax-dependent: skipped automatically on machines without jax (local Windows runs);
the Oscar launcher gate runs them for real. Wiring-level asserts (no jax needed)
always run.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from dicode import sil as sil_lib  # noqa: E402


def _mk_case():
    T, N = 8, 3
    done = jnp.zeros((T, N), bool).at[3, 0].set(True).at[6, 0].set(True).at[5, 1].set(True)
    ach = jnp.zeros((T, N)).at[3, 0].set(1.0)
    return T, N, done, ach


def test_episode_success_steps_marks_only_successful_episode():
    T, N, done, ach = _mk_case()
    sv = sil_lib.episode_success_steps(done, ach)
    assert bool(sv[0:4, 0].all())          # winning episode's steps
    assert not bool(sv[4:7, 0].any())      # the failed follow-up episode
    assert not bool(sv[:, 1].any())        # failed episode
    assert not bool(sv[:, 2].any())        # never-ending fragment


def test_write_pool_admission_isolation_and_ring():
    T, N, done, ach = _mk_case()
    sv = sil_lib.episode_success_steps(done, ach)
    P, S, D, Wm, L, E, H = 2, 4, 5, 4, 1, 3, 2
    state = sil_lib.init_sil_state(P, S, T, D, Wm, L, E, H)
    obs = jnp.arange(T * N * D, dtype=jnp.float32).reshape(T, N, D)
    args = (obs, jnp.ones((T, N), jnp.int32), jnp.full((T, N), 2.0),
            jnp.ones((T, N, H, Wm + 1), bool), jnp.ones((Wm + T, N, L, E)))
    prio = jnp.full((T, N), 0.5)

    empty = sil_lib.write_pool(state, 0, *args, jnp.zeros((T, N), bool), prio, 100, 2)
    assert not bool(empty["written"].any()) and int(empty["cursor"][0]) == 0

    st = sil_lib.write_pool(state, 0, *args, sv, prio, 100, 2)
    assert bool(st["written"][0, 0])
    assert not bool(st["written"][1].any())          # S1 pool isolation
    assert int(st["cursor"][0]) == 1
    assert bool(jnp.allclose(st["obs"][0, 0], obs[:, 0]))
    assert int(st["iupd"][0, 0]) == 100


def test_sample_pools_staleness_and_empty():
    T, N, done, ach = _mk_case()
    sv = sil_lib.episode_success_steps(done, ach)
    P, S, D, Wm, L, E, H = 2, 4, 5, 4, 1, 3, 2
    state = sil_lib.init_sil_state(P, S, T, D, Wm, L, E, H)
    obs = jnp.zeros((T, N, D))
    args = (obs, jnp.ones((T, N), jnp.int32), jnp.full((T, N), 2.0),
            jnp.ones((T, N, H, Wm + 1), bool), jnp.ones((Wm + T, N, L, E)))
    st = sil_lib.write_pool(state, 0, *args, sv, jnp.full((T, N), 0.5), 100, 2)
    rng = jax.random.PRNGKey(0)

    _, fresh_ok = sil_lib.sample_pools(st, rng, 3, 100, 500, True)
    assert bool(fresh_ok)
    stale_batch, stale_ok = sil_lib.sample_pools(st, rng, 3, 601, 500, True)
    assert not bool(stale_ok) and not bool(stale_batch["svalid"].any())
    empty_batch, empty_ok = sil_lib.sample_pools(state, rng, 3, 100, 500, True)
    assert not bool(empty_ok) and not bool(empty_batch["svalid"].any())


def test_wiring_cond_gate_and_revert_path():
    import dicode.ppo_tr as ppo_tr
    import dicode.training as tr_mod

    src = inspect.getsource(ppo_tr)
    assert "sil_scale > 0.0" in src                     # beta/empty gate
    assert "jax.lax.stop_gradient(w)" in src            # BC weight detached
    tsrc = inspect.getsource(tr_mod.run_session_training)
    assert "pre_session_sil_state" in tsrc              # guard revert covers the pool
