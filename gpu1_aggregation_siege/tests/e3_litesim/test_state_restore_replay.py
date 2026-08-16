import jax
import numpy as np

from dicode.e3_litesim.data import lightweight_rollout as lr
from dicode.e3_litesim.measurement.failure_capsule import (
    capture_failure_capsule, exact_replay_check, restore_capsule)
from helpers import make_setup


def _capsule(s):
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 1)
    _o, state0 = lr.batched_reset(s["env"], s["env_params"], keys)
    mem = {k: np.asarray(v) for k, v in
           s["backend"].init_runner_memory(1).items()}
    batch = lr.collect_rollouts(env=s["env"], env_params=s["env_params"],
                                backend=s["backend"], params=s["params"],
                                start_states=[state0], start_memories=[mem],
                                horizon=8, rng=rng, collect_trace=True,
                                collect_memory_trace=True)
    cap = capture_failure_capsule(env_state=batch.trace[4],
                                  memory=batch.memory_trace[4],
                                  params_hash="h", rng_seed=3,
                                  tier_id="tier1_survive", probe_id="p",
                                  episode_timestep=4)
    return cap


def test_exact_replay(s=None):
    s = make_setup()
    cap = _capsule(s)
    res = exact_replay_check(cap, s["backend"], s["params"], env=s["env"],
                             env_params=s["env_params"], horizon=4)
    assert res["ok"], res


def test_restore_roundtrip():
    s = make_setup()
    cap = _capsule(s)
    state, mem = restore_capsule(cap)
    assert cap.base_state_hash
    assert set(mem.keys())
    obs = lr.batched_get_obs(s["env"], state)
    assert np.isfinite(np.asarray(obs)).all()