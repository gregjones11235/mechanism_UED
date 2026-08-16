import jax
import numpy as np

from dicode.e3_litesim.data import lightweight_rollout as lr
from dicode.e3_litesim.measurement import behavior_metrics
from helpers import make_setup


def test_metrics_panel():
    s = make_setup()
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 2)
    _o, state0 = lr.batched_reset(s["env"], s["env_params"], keys)
    mem = {k: np.asarray(v) for k, v in
           s["backend"].init_runner_memory(2).items()}
    batch = lr.collect_rollouts(env=s["env"], env_params=s["env_params"],
                                backend=s["backend"], params=s["params"],
                                start_states=[state0], start_memories=[mem],
                                horizon=8, rng=rng, collect_trace=True)
    success = np.ones(2)
    m = behavior_metrics.trace_metrics(batch.trace, batch.actions,
                                       batch.rewards, batch.dones, success)
    for key in ("success", "health_delta", "floor_reached",
                "oscillation_rate", "stall_rate", "max_no_progress_steps",
                "threat_damage_events", "return_sum"):
        assert key in m
    assert (m["oscillation_rate"] >= 0).all() and (m["oscillation_rate"] <= 1).all()
    agg = behavior_metrics.aggregate(m)
    assert agg["success"] == 1.0