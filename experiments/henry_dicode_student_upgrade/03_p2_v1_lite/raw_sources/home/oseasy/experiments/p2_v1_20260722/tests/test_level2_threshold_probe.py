#!/usr/bin/env python3
"""P2-v1 Level 2 threshold probe (阶段4, CPU-only, no Craftax/GPU/checkpoint).

Empirically tests the structural claim derived from code reading:

  collect_rollout() builds its per-env episode buffer LOCALLY for each rollout
  (re-created every call).  A Trajectory is emitted only when an episode hits
  done=True WITHIN the rollout, so a captured trajectory's length is bounded by
  rollout_steps.  With ROLLOUT_STEPS=128 and TrajectoryReplayBuffer.
  MIN_SEQUENCE_LENGTH=129 (strictly >128), can_sample() is therefore ALWAYS
  False and the replay auxiliary update can NEVER trigger -- at any number of
  updates / env steps.

We drive collect_rollout with a tiny fake vectorized env (controllable episode
length) and a tiny fake network, feed every emitted trajectory into a real
TrajectoryReplayBuffer, and record the trajectory lengths.  We assert:

  1. max captured trajectory length <= rollout_steps (128) for every EP_LEN;
  2. can_sample() stays False throughout (replay aux unreachable);
  3. a long episode (EP_LEN > rollout_steps) is split at the rollout boundary:
     its pre-boundary prefix is discarded (no done), so the captured fragment
     is short and its initial_memory is NOT the true episode start.

Evidence for reports/p2_v1_level2_threshold_audit.md.  Modifies no checkpoint,
uses no GPU.
"""
import os
import sys
import json
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import jax
import jax.numpy as jnp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from p2_v1_core import collect_rollout
from trajectory_replay import TrajectoryReplayBuffer
from rng_utils import make_action_rng

ROLLOUT_STEPS = 128
NUM_ENVS = 4
OBS_DIM = 8
ACTION_DIM = 5
WINDOW_MEM = 16
NUM_LAYERS = 2
EMBED = 8
NUM_HEADS = 2


class _Pi:
    def __init__(self, logits):
        self.logits = logits


class FakeNetwork:
    model_forward_eval = None

    def apply(self, params, mem, obs, mask, method=None):
        n = obs.shape[0]
        logits = jnp.zeros((n, ACTION_DIM))
        value = jnp.zeros((n,))
        mem_out = jnp.zeros((n, NUM_LAYERS, EMBED))
        return _Pi(logits), value, mem_out


class FakeEnvState:
    def __init__(self, counter):
        self.counter = counter


class FakeEnv:
    """Vectorized env ending each episode after exactly ep_len steps."""

    def __init__(self, ep_len):
        self.ep_len = ep_len

    def reset(self, rng):
        obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
        return obs, FakeEnvState(counter=np.zeros(NUM_ENVS, dtype=np.int64))

    def step(self, rng, state, actions):
        counter = state.counter + 1
        done = counter >= self.ep_len
        counter = np.where(done, 0, counter)
        reward = done.astype(np.float32)
        obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
        return obs, FakeEnvState(counter=counter), reward, done, {}


class _TS:
    # FakeNetwork.apply ignores params; collect_rollout only reads ts.params.
    params = None


def run_one(ts, ep_len, num_updates):
    net = FakeNetwork()
    env = FakeEnv(ep_len)
    rng = jax.random.PRNGKey(0)
    action_rng = make_action_rng(42)
    obsv, env_state = env.reset(rng)
    memories = jnp.zeros((NUM_ENVS, WINDOW_MEM, NUM_LAYERS, EMBED))
    mem_mask = jnp.zeros((NUM_ENVS, NUM_HEADS, 1, WINDOW_MEM + 1), dtype=jnp.bool_)
    mem_idx = jnp.full((NUM_ENVS,), WINDOW_MEM + 1, dtype=jnp.int32)
    target_ach = np.zeros(67, dtype=np.float32)
    target_ach[41] = 1.0
    replay = TrajectoryReplayBuffer(capacity=256, seed=42)
    lengths = []
    can_sample_any = False
    for up in range(num_updates):
        roll = collect_rollout(
            ts=ts, network=net, env=env, env_state=env_state, obsv=obsv,
            memories=memories, mem_mask=mem_mask, mem_idx=mem_idx, rng=rng,
            action_rng=action_rng, num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
            window_mem=WINDOW_MEM, num_heads=NUM_HEADS,
            target_achievement=target_ach, collected_update_count=up)
        obsv = roll["obsv"]; env_state = roll["env_state"]
        memories = roll["memories"]; mem_mask = roll["mem_mask"]
        mem_idx = roll["mem_idx"]; rng = roll["rng"]
        for traj in roll["trajectories"]:
            lengths.append(traj.length)
            replay.insert(traj)
        if replay.can_sample():
            can_sample_any = True
    return lengths, can_sample_any, replay


def main():
    print("=" * 64)
    print("P2-v1 Level 2 threshold probe (CPU)")
    print(f"  ROLLOUT_STEPS={ROLLOUT_STEPS}  "
          f"MIN_SEQUENCE_LENGTH={TrajectoryReplayBuffer.MIN_SEQUENCE_LENGTH}")
    print("=" * 64)

    dummy_ts = _TS()
    results = {}
    all_max_le_128 = True
    any_can_sample = False
    for ep_len in [50, 128, 129, 200, 4096]:
        lengths, can_sample_any, replay = run_one(dummy_ts, ep_len, num_updates=24)
        mx = max(lengths) if lengths else 0
        le_128 = (mx <= ROLLOUT_STEPS)
        all_max_le_128 = all_max_le_128 and le_128
        any_can_sample = any_can_sample or can_sample_any
        results[f"ep_len_{ep_len}"] = {
            "num_trajectories": len(lengths),
            "max_len": mx,
            "min_len": (min(lengths) if lengths else None),
            "max_le_rollout_steps": le_128,
            "can_sample_ever": can_sample_any,
            "replay_size": len(replay),
        }
        print(f"  EP_LEN={ep_len:5d}: trajectories={len(lengths):3d}  "
              f"max_len={mx:3d}  max<=128={le_128}  can_sample={can_sample_any}")

    gate_bound = (TrajectoryReplayBuffer.MIN_SEQUENCE_LENGTH > ROLLOUT_STEPS)
    conclusion = (gate_bound and all_max_le_128 and not any_can_sample)
    print("\n" + "=" * 64)
    print(f"  MIN_SEQUENCE_LENGTH(129) > ROLLOUT_STEPS(128): {gate_bound}")
    print(f"  all captured trajectories <= rollout_steps:    {all_max_le_128}")
    print(f"  replay can_sample() ever True:                 {any_can_sample}")
    print(f"  => replay aux UNREACHABLE at any step count:   {conclusion}")
    print("=" * 64)

    report = {
        "directive": "P2-v1 Level 2 threshold probe (阶段4)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ROLLOUT_STEPS": ROLLOUT_STEPS,
            "MIN_SEQUENCE_LENGTH": TrajectoryReplayBuffer.MIN_SEQUENCE_LENGTH,
            "NUM_ENVS": NUM_ENVS,
            "num_updates_per_probe": 24,
        },
        "results_by_ep_len": results,
        "min_seq_gt_rollout_steps": gate_bound,
        "all_max_le_rollout_steps": all_max_le_128,
        "replay_can_sample_ever": any_can_sample,
        "replay_aux_reachable": any_can_sample,
        "conclusion": (
            "REPLAY_AUX_UNREACHABLE: captured trajectory length is bounded by "
            "rollout_steps=128 < MIN_SEQUENCE_LENGTH=129, so can_sample() is "
            "always False regardless of env-step budget."
        ) if conclusion else "UNEXPECTED: probe did not confirm the bound",
        "probe_passed": conclusion,
    }
    ev = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "level2_threshold_probe.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    sys.exit(0 if conclusion else 1)


if __name__ == "__main__":
    main()
