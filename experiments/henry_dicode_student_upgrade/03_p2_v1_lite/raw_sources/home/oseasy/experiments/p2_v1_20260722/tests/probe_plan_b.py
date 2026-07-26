"""方案B (cross-rollout persistent episode buffers) CPU mechanism probe.

Verifies the CORE mechanism that makes P2-v1 Level 2 replay/hindsight reachable,
independently of Craftax and of the PPO update, using the shape-correct test
network + deterministic FakeEnv from test_p2_v1.py:

  PROBE 1  cross-rollout reconstruction + replay reachability
           An episode longer than rollout_steps (here 200 > 128) spans two
           128-step rollouts; it is reconstructed with no loss / no duplicate
           (total transitions == completed-replay + pending), its completed
           length is >= MIN_SEQUENCE_LENGTH (129), and inserting it into a real
           TrajectoryReplayBuffer makes can_sample() True — the exact event that
           was structurally unreachable before 方案B.

  PROBE 2  pending-buffer checkpoint round-trip
           PendingEpisodeBuffers.state_dict -> pickle -> from_state_dict is
           bit-exact (per-slot transitions + episode ids + counters).

  PROBE 3  interrupted-vs-uninterrupted bit-exact resume (test 7 core)
           Running 3 rollouts straight yields the SAME trajectories (count,
           lengths, and per-transition obs/actions/rewards bit-exact) as running
           1 rollout, checkpointing the collector+pending+RNG state, restoring,
           and running the remaining 2 rollouts WITHOUT an env reset.

Runs on CPU only (JAX_PLATFORM_NAME=cpu, set by importing test_p2_v1).
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp

from test_p2_v1 import (  # harness: network, fake env, train-state builder
    P2V1TestNet, FakeEnv, make_net, make_train_state, ACTION_DIM, OBS_DIM,
)
from p2_v1_core import collect_rollout
from pending_episodes import PendingEpisodeBuffers
from trajectory_replay import TrajectoryReplayBuffer
from rng_utils import make_action_rng, action_rng_state, restore_action_rng

NUM_ENVS = 2
ROLLOUT_STEPS = 128          # the real P2-v1 rollout length
WINDOW_MEM = 16
NUM_HEADS = 4
NUM_LAYERS = 2
EMBED = 32
DONE_EVERY = 200             # episode length 200 > 128 -> spans two rollouts
TARGET = np.zeros(67, dtype=np.float32)


def _fresh_collector(num_envs):
    return {
        "memories": jnp.zeros((num_envs, WINDOW_MEM, NUM_LAYERS, EMBED)),
        "mem_mask": jnp.zeros((num_envs, NUM_HEADS, 1, WINDOW_MEM + 1),
                              dtype=jnp.bool_),
        "mem_idx": jnp.full((num_envs,), WINDOW_MEM + 1, dtype=jnp.int32),
    }


def run_rollouts(net, ts, env, n_rollouts,
                 pending=None, collector=None, action_rng=None, rng=None,
                 action_seed=0, collected_update_count=0):
    """Drive collect_rollout for n_rollouts, threading 方案B state across calls.

    Mirrors run_session's inner collection loop (no PPO update) so the
    episode-buffer / collector / RNG mechanics are exercised in isolation.
    Returns (trajectories, pending, collector, action_rng_state, rng).
    """
    if rng is None:
        rng = jax.random.PRNGKey(123)
    if action_rng is None:
        action_rng = make_action_rng(action_seed)

    if collector is not None:
        obsv = collector["obsv"]
        env_state = collector["env_state"]
        memories = collector["memories"]
        mem_mask = collector["mem_mask"]
        mem_idx = collector["mem_idx"]
    else:
        # True fresh start: reset the env (consumes one reset rng split).
        rng, reset_rng = jax.random.split(rng)
        obsv, env_state = env.reset(reset_rng)
        c = _fresh_collector(NUM_ENVS)
        memories, mem_mask, mem_idx = c["memories"], c["mem_mask"], c["mem_idx"]

    all_trajs = []
    for _ in range(n_rollouts):
        roll = collect_rollout(
            ts=ts, network=net, env=env, env_state=env_state, obsv=obsv,
            memories=memories, mem_mask=mem_mask, mem_idx=mem_idx, rng=rng,
            action_rng=action_rng, num_envs=NUM_ENVS,
            rollout_steps=ROLLOUT_STEPS, window_mem=WINDOW_MEM,
            num_heads=NUM_HEADS, target_achievement=TARGET,
            collected_update_count=collected_update_count, pending=pending,
        )
        all_trajs.extend(roll["trajectories"])
        pending = roll["pending"]
        obsv = roll["obsv"]
        env_state = roll["env_state"]
        memories = roll["memories"]
        mem_mask = roll["mem_mask"]
        mem_idx = roll["mem_idx"]
        rng = roll["rng"]

    collector = {
        "obsv": obsv, "env_state": env_state, "memories": memories,
        "mem_mask": mem_mask, "mem_idx": mem_idx,
    }
    return all_trajs, pending, collector, action_rng_state(action_rng), rng


def _traj_signature(trajs):
    """A comparable, order-stable signature of a list of trajectories."""
    sig = []
    for t in trajs:
        sig.append((
            int(t.length),
            np.asarray(t.observations).tobytes(),
            np.asarray(t.actions).tobytes(),
            np.asarray(t.rewards).tobytes(),
            np.asarray(t.dones).tobytes(),
        ))
    return sig


# ---------------------------------------------------------------------------
# PROBE 1 — cross-rollout reconstruction + replay reachability
# ---------------------------------------------------------------------------

def probe_1_cross_rollout():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=DONE_EVERY)

    # Rollout 1: no episode can finish (200 > 128) -> no trajectories,
    # pending holds the full 128-step prefix on every slot.
    trajs1, pending1, coll1, arng1, rng1 = run_rollouts(
        net, ts, env, n_rollouts=1, action_seed=1)
    assert len(trajs1) == 0, f"rollout1 emitted {len(trajs1)} trajs, expected 0"
    lens1 = pending1.slot_lengths()
    assert all(L == ROLLOUT_STEPS for L in lens1), \
        f"rollout1 pending lengths {lens1} != {ROLLOUT_STEPS}"

    # Rollout 2: continue the SAME pending buffers -> the 200-step episode
    # completes mid-rollout.
    # NOTE: probe_1 checks STRUCTURE only (lengths / no-loss-no-dup / replay
    # reachability), which does not depend on the action stream, so a fresh
    # action_rng here is fine.  Action-stream continuity is verified bit-exact
    # in probe_3 (which resumes action_rng + jax rng explicitly).
    trajs2, pending2, coll2, arng2, rng2 = run_rollouts(
        net, ts, env, n_rollouts=1, pending=pending1, collector=coll1,
        action_seed=1)
    assert len(trajs2) >= 1, "rollout2 emitted no trajectory (episode not carried)"
    lengths2 = sorted(t.length for t in trajs2)
    assert max(lengths2) == DONE_EVERY, \
        f"expected a completed episode of length {DONE_EVERY}, got {lengths2}"
    assert max(lengths2) >= 129, "completed trajectory < MIN_SEQUENCE_LENGTH(129)"

    # No-loss / no-dup invariant over the two rollouts:
    #   total transitions collected == completed-replay + pending.
    total = 2 * NUM_ENVS * ROLLOUT_STEPS  # 2 rollouts
    completed = sum(t.length for t in trajs2)
    pend = pending2.total_pending_transitions()
    assert completed + pend == total, \
        f"no-loss/no-dup FAIL: completed({completed})+pending({pend}) != total({total})"

    # Replay reachability: the whole point of 方案B.  Inserting the 200-step
    # trajectory makes can_sample() True (was ALWAYS False before the fix).
    replay = TrajectoryReplayBuffer(capacity=16, seed=7)
    assert replay.can_sample() is False, "empty replay should not be sampleable"
    for t in trajs2:
        if t.length >= 129:
            replay.insert(t)
    assert replay.can_sample() is True, \
        "replay STILL not sampleable after inserting a 129+ trajectory"
    assert replay.longest_trajectory_length >= 129

    print(f"  PROBE 1 PASS: 200-step episode reconstructed across two 128-rollouts"
          f" (no loss/dup: {completed}+{pend}=={total}); replay.can_sample()==True,"
          f" longest={replay.longest_trajectory_length}")


# ---------------------------------------------------------------------------
# PROBE 2 — pending-buffer checkpoint round-trip (bit-exact)
# ---------------------------------------------------------------------------

def probe_2_pending_roundtrip():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=DONE_EVERY)

    _, pending, _, _, _ = run_rollouts(net, ts, env, n_rollouts=1, action_seed=2)
    assert pending.total_pending_transitions() > 0

    state = pending.state_dict()
    blob = pickle.dumps(state)              # checkpoint serialization
    restored = PendingEpisodeBuffers.from_state_dict(pickle.loads(blob))

    assert restored.num_envs == pending.num_envs
    assert restored.next_episode_id == pending.next_episode_id
    assert restored.episode_id == pending.episode_id
    assert restored.policy_version == pending.policy_version
    assert restored.slot_lengths() == pending.slot_lengths()
    for e in range(pending.num_envs):
        a, b = pending.slots[e], restored.slots[e]
        for key in ("obs", "act", "rew", "don", "val", "lp",
                    "next_obs", "mem_pre", "mask_pre", "ach"):
            for xa, xb in zip(a[key], b[key]):
                assert np.array_equal(np.asarray(xa), np.asarray(xb)), \
                    f"slot{e}.{key} not bit-exact after round-trip"
        assert np.array_equal(np.asarray(a["init_mem"]),
                              np.asarray(b["init_mem"])), \
            f"slot{e}.init_mem not bit-exact after round-trip"

    print(f"  PROBE 2 PASS: pending state_dict round-trip bit-exact "
          f"({pending.total_pending_transitions()} pending transitions, "
          f"next_episode_id={restored.next_episode_id})")


# ---------------------------------------------------------------------------
# PROBE 3 — interrupted-vs-uninterrupted bit-exact resume
# ---------------------------------------------------------------------------

def probe_3_exact_resume():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=DONE_EVERY)

    # (A) Uninterrupted: 3 rollouts straight.
    trajs_full, _, _, _, _ = run_rollouts(net, ts, env, n_rollouts=3, action_seed=3)

    # (B) Interrupted: 1 rollout, checkpoint, restore, 2 more rollouts (no reset).
    trajs_a, pending_a, coll_a, arng_a, rng_a = run_rollouts(
        net, ts, env, n_rollouts=1, action_seed=3)

    # --- checkpoint: pending + collector + action_rng + jax rng ---
    ckpt = pickle.dumps({
        "pending_state": pending_a.state_dict(),
        "collector_state": coll_a,
        "action_rng_state": arng_a,
        "rng": rng_a,
    })

    # --- restore ---
    r = pickle.loads(ckpt)
    pending_r = PendingEpisodeBuffers.from_state_dict(r["pending_state"])
    action_rng_r = restore_action_rng(r["action_rng_state"], seed=0)

    trajs_b, _, _, _, _ = run_rollouts(
        net, ts, env, n_rollouts=2,
        pending=pending_r, collector=r["collector_state"],
        action_rng=action_rng_r, rng=r["rng"])

    trajs_resumed = trajs_a + trajs_b

    # Same number of completed trajectories ...
    assert len(trajs_full) == len(trajs_resumed), \
        f"trajectory count mismatch: full={len(trajs_full)} resumed={len(trajs_resumed)}"
    # ... and identical per-transition content, in order.
    sig_full = _traj_signature(trajs_full)
    sig_resumed = _traj_signature(trajs_resumed)
    assert sig_full == sig_resumed, \
        "resumed trajectories differ bit-wise from the uninterrupted run"

    lens = sorted(t.length for t in trajs_full)
    print(f"  PROBE 3 PASS: interrupted(1+2) == uninterrupted(3) bit-exact "
          f"({len(trajs_full)} trajectories, lengths={lens})")


def main():
    print("=" * 64)
    print("方案B CPU mechanism probe (cross-rollout persistent episode buffers)")
    print("=" * 64)
    probe_1_cross_rollout()
    probe_2_pending_roundtrip()
    probe_3_exact_resume()
    print("=" * 64)
    print("方案B PROBE: ALL PASS")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
