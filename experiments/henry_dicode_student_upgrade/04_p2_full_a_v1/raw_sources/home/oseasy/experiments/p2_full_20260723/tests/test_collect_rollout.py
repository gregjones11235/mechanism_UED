"""collect_rollout correctness on a FAKE deterministic env (CPU, real network).

  CR.align   obs[t] is the obs used to pick action[t] (no off-by-one); marker==step
  CR.anchor  anchors captured at episode steps 0,128,... ; anchor_steps == expected;
             anchor_masks/idxs present; validate_anchors passes
  CR.initmem initial_memory is the ENTERING memory at episode step 0 (zero after reset)
  CR.persist episodes persist ACROSS rollouts (a 150-step episode spanning two 100-step
             rollouts completes with anchors [0,128] both preserved); no spurious done
             written at the rollout boundary
  CR.reset   auto-reset isolation: after a done the slot gets a NEW episode id and a
             fresh step-0 anchor; the new episode does not inherit the old memory
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax
import jax.numpy as jnp

import fputil
import rng_utils as RU
import memory_anchor as MA
import full_p2_core as CORE
from replay_buffer import anchor_steps_for_length
from pending_episodes import PendingEpisodeBuffers

CFG = fputil.CFG
N_ACH = 4
NUM_ENVS = 4
EP_LEN = 150


class FakeEnvState:
    def __init__(self, counters):
        self.counters = counters  # numpy [E]: steps taken in current episode


class FakeEnv:
    """Deterministic env: obs[:,0] == episode step counter; done at EP_LEN; +1 terminal."""
    def __init__(self, num_envs, obs_dim, ep_len):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.ep_len = ep_len

    def reset_obs(self, counters):
        obs = np.zeros((self.num_envs, self.obs_dim), np.float32)
        obs[:, 0] = counters.astype(np.float32)
        return obs

    def step(self, rng, env_state, actions):
        counters = env_state.counters.copy() + 1
        done = counters >= self.ep_len
        reward = done.astype(np.float32)
        next_counters = np.where(done, 0, counters)   # auto-reset
        next_obs = self.reset_obs(next_counters)
        return next_obs, FakeEnvState(next_counters), reward, done, {}


def _fresh():
    net, params, a_rec, a_raw, scan_fn = fputil.build_net()
    jit_fwd = CORE.make_jit_forward(net)
    env = FakeEnv(NUM_ENVS, CFG.obs_dim, EP_LEN)
    env_state = FakeEnvState(np.zeros(NUM_ENVS, np.int64))
    obsv = jnp.asarray(env.reset_obs(np.zeros(NUM_ENVS, np.int64)))
    mem, mask, idx = MA.fresh_rollout_state(
        CFG.window_mem, CFG.num_heads, CFG.num_layers, CFG.embed, NUM_ENVS)
    rng = jax.random.PRNGKey(0)
    action_rng = RU.make_action_rng(0)
    pending = PendingEpisodeBuffers(num_envs=NUM_ENVS)
    target = np.zeros(N_ACH, np.float32); target[0] = 1.0
    return dict(net=net, params=params, jit_fwd=jit_fwd, env=env, env_state=env_state,
                obsv=obsv, mem=mem, mask=mask, idx=idx, rng=rng, action_rng=action_rng,
                pending=pending, target=target)


def _roll(s, steps, uc=0):
    trajs, carry, stats = CORE.collect_rollout(
        s["env"], s["env_state"], s["net"], s["params"], s["obsv"],
        s["mem"], s["mask"], s["idx"], s["rng"], s["action_rng"],
        s["pending"], s["target"], rollout_steps=steps,
        window_mem=CFG.window_mem, num_heads=CFG.num_heads,
        collected_update_count=uc, jit_forward=s["jit_fwd"])
    s.update(env_state=carry["env_state"], obsv=carry["obsv"], mem=carry["memories"],
             mask=carry["mem_mask"], idx=carry["mem_idx"], rng=carry["rng"])
    return trajs, stats


def test_align_anchor_initmem_single_rollout():
    s = _fresh()
    trajs, stats = _roll(s, EP_LEN)            # exactly one episode per env
    assert stats["completed_episodes"] == NUM_ENVS, stats
    assert len(trajs) == NUM_ENVS
    t = trajs[0]
    # CR.align: marker == episode step
    assert t.length == EP_LEN
    assert np.array_equal(np.asarray(t.observations)[:, 0].astype(int),
                          np.arange(EP_LEN)), "obs/step off-by-one"
    assert t.dones[-1] and not t.dones[:-1].any()
    # CR.anchor
    assert list(t.anchor_steps) == anchor_steps_for_length(EP_LEN) == [0, 128]
    assert t.n_anchors == 2
    assert t.memory_anchors.shape == (2, CFG.window_mem, CFG.num_layers, CFG.embed)
    assert t.anchor_masks.shape == (2, CFG.num_heads, 1, CFG.window_mem + 1)
    assert t.anchor_idxs.shape == (2,)
    t.validate_anchors()
    # CR.initmem: episode began from a fresh reset -> zero entering memory
    assert np.allclose(np.asarray(t.initial_memory), 0.0), "init_mem not zero at reset"
    print("PASS CR.align + CR.anchor + CR.initmem (single rollout, anchors [0,128])")


def test_persist_across_rollouts_and_reset():
    s = _fresh()
    # rollout 1: 100 steps, no done (EP_LEN=150)
    ep_ids_before = list(s["pending"].episode_id)
    trajs1, st1 = _roll(s, 100, uc=0)
    assert st1["completed_episodes"] == 0, "no episode should finish in 100 steps"
    assert len(trajs1) == 0
    assert s["pending"].slot_lengths() == [100] * NUM_ENVS
    assert all(not d for d in s["pending"].slots[0]["don"]), "spurious done at boundary"
    assert s["pending"].episode_id == ep_ids_before, "episode id must persist mid-episode"
    # step-0 anchor captured in rollout 1
    assert s["pending"].slot_anchor_steps(0) == [0]

    # rollout 2: 100 more steps -> episode completes at step 150, new episode starts
    trajs2, st2 = _roll(s, 100, uc=1)
    assert st2["completed_episodes"] == NUM_ENVS, st2
    assert len(trajs2) == NUM_ENVS
    t = trajs2[0]
    assert t.length == EP_LEN
    # cross-rollout continuity: marker == episode step across the boundary
    assert np.array_equal(np.asarray(t.observations)[:, 0].astype(int),
                          np.arange(EP_LEN)), "episode continuity broken across rollouts"
    # BOTH anchors preserved across the rollout boundary
    assert list(t.anchor_steps) == [0, 128]
    t.validate_anchors()
    assert t.dones[-1] and not t.dones[:-1].any()

    # CR.reset: after the done, each slot got a NEW episode id + fresh step-0 anchor
    assert all(new != old for new, old in zip(s["pending"].episode_id, ep_ids_before)), \
        "auto-reset must assign new episode ids"
    # new episode ran 50 steps (200 total - 150); anchor at its step 0 only
    assert s["pending"].slot_lengths() == [50] * NUM_ENVS
    assert s["pending"].slot_anchor_steps(0) == [0]
    # new episode started from zeroed memory -> its step-0 anchor memory is ~0
    assert np.allclose(np.asarray(s["pending"].slots[0]["anchor_mem"][0]), 0.0, atol=1e-6), \
        "new episode inherited stale memory"
    print("PASS CR.persist (150-step ep across two 100-step rollouts, anchors [0,128]) "
          "+ CR.reset (new id, fresh zero-memory step-0 anchor)")


if __name__ == "__main__":
    test_align_anchor_initmem_single_rollout()
    test_persist_across_rollouts_and_reset()
    print("ALL_COLLECT_ROLLOUT_TESTS_PASS")
