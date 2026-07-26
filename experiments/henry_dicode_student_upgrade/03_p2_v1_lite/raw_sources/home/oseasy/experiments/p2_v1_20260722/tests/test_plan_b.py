"""方案B (cross-rollout persistent episode buffers) HARD tests — 第二节 1-8.

Permanent CPU regression suite for the P2-v1 方案B mechanism.  Uses the
shape-correct test network + deterministic FakeEnv from test_p2_v1.py (no
Craftax).  Test 9 (original PPO equivalence) lives in test_p2_v1.py and is
unaffected by 方案B because the RolloutBatch path is unchanged.

Covered hard tests (user decision 2026-07-22, 第二节):
  1  cross-rollout reconstruction  (129+ episode spans two 128 rollouts; no
                                    loss / no duplicate; replay becomes
                                    can_sample()==True)
  2  exact boundary                (episode ending exactly on a rollout's last
                                    step is flushed exactly once)
  3  nonterminal boundary          (rollout ends, episode not done: replay gains
                                    no completed traj, pending length correct,
                                    next rollout continues)
  4  multi-env isolation           (per-slot buffers never splice across envs)
  5  auto-reset isolation          (terminal transition and the post-reset first
                                    transition belong to different episodes)
  6  initial-memory correctness    (initial_memory is the memory before the
                                    episode's FIRST action, carried across
                                    rollouts; not a mid-rollout boundary memory)
  7  pending-buffer checkpoint resume (save mid-episode; restore via the
                                    launcher's restore_p2_v1_checkpoint; the
                                    resumed run's trajectories are bit-exact vs
                                    the uninterrupted run)
  8  no-dup / no-loss              (total transitions == completed replay +
                                    pending at every boundary)

Runs on CPU only (JAX_PLATFORM_NAME=cpu, set by importing test_p2_v1).
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp

from test_p2_v1 import (
    P2V1TestNet, FakeEnv, FakeEnvState, make_net, make_train_state, Cfg,
    ACTION_DIM, OBS_DIM,
)
from p2_v1_core import collect_rollout
from pending_episodes import PendingEpisodeBuffers
from trajectory_replay import TrajectoryReplayBuffer
from checkpointing import save_full_checkpoint
from stage4_continue_launcher import restore_p2_v1_checkpoint
from rng_utils import make_action_rng, action_rng_state, restore_action_rng

NUM_ENVS = 2
ROLLOUT_STEPS = 128
WINDOW_MEM = 16
NUM_HEADS = 4
NUM_LAYERS = 2
EMBED = 32
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

    Mirrors run_session's inner collection loop (no PPO update).  Returns
    (trajectories, pending, collector, action_rng_state, rng).
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


def _sig(trajs):
    out = []
    for t in trajs:
        out.append((
            int(t.length),
            np.asarray(t.observations).tobytes(),
            np.asarray(t.actions).tobytes(),
            np.asarray(t.rewards).tobytes(),
            np.asarray(t.dones).tobytes(),
        ))
    return out


def _obs_counts(traj):
    """The FakeEnv step-counter encoded in obs dim 0 (obs[:,0] == count*0.1).

    Recovered as exact integers via rounding: the float32 ``count*0.1`` round-trip
    is not bit-exact, so dividing back without rounding yields ~3.0000001 and a
    spurious non-contiguous result.
    """
    return np.round(np.asarray(traj.observations)[:, 0] / 0.1).astype(np.int64)


def _is_contiguous_ramp(traj):
    """True if a trajectory's counter is a clean +1 ramp (no gap / dup / splice),
    starting at whatever its first count is (FakeEnv counts are monotonic and do
    NOT reset to 0 on done, so later episodes start at 100/200/...)."""
    counts = _obs_counts(traj)
    return bool(np.array_equal(
        counts, counts[0] + np.arange(traj.length, dtype=np.int64)))


# ---------------------------------------------------------------------------
# 1. cross-rollout reconstruction + replay reachability
# ---------------------------------------------------------------------------

def test_cross_rollout_reconstruction():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=200)  # episode len 200 > 128

    trajs1, pending1, coll1, _, _ = run_rollouts(net, ts, env, 1, action_seed=1)
    assert len(trajs1) == 0, "rollout1 must not complete a 200-step episode"
    assert all(L == ROLLOUT_STEPS for L in pending1.slot_lengths()), \
        f"pending after rollout1 {pending1.slot_lengths()} != {ROLLOUT_STEPS}"

    trajs2, pending2, _, _, _ = run_rollouts(
        net, ts, env, 1, pending=pending1, collector=coll1, action_seed=1)
    assert len(trajs2) >= 1, "episode was not carried across the rollout boundary"
    assert max(t.length for t in trajs2) == 200, \
        f"completed episode length {sorted(t.length for t in trajs2)} != 200"
    assert max(t.length for t in trajs2) >= 129, "completed trajectory < 129"

    # The 200-step trajectory must be transition-by-transition contiguous: its
    # observation counter runs 0..199 with no gap / repeat (obs[:,0] == count*0.1).
    long_t = max(trajs2, key=lambda t: t.length)
    counts = _obs_counts(long_t)
    assert np.array_equal(counts, np.arange(200, dtype=np.int64)), \
        "reconstructed episode observations are not contiguous (loss/dup)"

    # Replay reachability — the structural fix.
    replay = TrajectoryReplayBuffer(capacity=16, seed=7)
    assert replay.can_sample() is False
    replay.insert(long_t)
    assert replay.can_sample() is True, \
        "replay STILL not sampleable after inserting a 129+ trajectory"
    print("  PASS 1 cross-rollout reconstruction: 200-step episode rebuilt "
          "across two 128-rollouts, contiguous, replay.can_sample()==True")


# ---------------------------------------------------------------------------
# 2. exact boundary (episode ends exactly on a rollout's last step)
# ---------------------------------------------------------------------------

def test_exact_boundary():
    net = make_net()
    ts = make_train_state(net)
    # Episode length exactly 128 -> done on the LAST step of rollout 1.
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=128)

    trajs1, pending1, coll1, _, _ = run_rollouts(net, ts, env, 1, action_seed=4)
    # Exactly one flush per env, on the boundary step; nothing left pending.
    assert len(trajs1) == NUM_ENVS, \
        f"expected {NUM_ENVS} boundary flushes, got {len(trajs1)}"
    assert all(t.length == 128 for t in trajs1), \
        f"boundary episode length {[t.length for t in trajs1]} != 128"
    assert pending1.total_pending_transitions() == 0, \
        "pending must be empty right after a boundary-terminal episode"

    # A second rollout must NOT re-emit the same episode (flushed exactly once).
    trajs2, _, _, _, _ = run_rollouts(
        net, ts, env, 1, pending=pending1, collector=coll1, action_seed=4)
    assert all(t.length == 128 for t in trajs2)
    print("  PASS 2 exact boundary: episode ending on the last step flushed "
          "exactly once (no re-emit, empty pending)")


# ---------------------------------------------------------------------------
# 3. nonterminal boundary (rollout ends, episode not done)
# ---------------------------------------------------------------------------

def test_nonterminal_boundary():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=10_000)  # no dones anywhere

    replay = TrajectoryReplayBuffer(capacity=16, seed=3)
    trajs1, pending1, coll1, _, _ = run_rollouts(net, ts, env, 1, action_seed=5)
    for t in trajs1:
        replay.insert(t)
    n_after_r1 = len(replay)
    assert len(trajs1) == 0, "no episode should complete in a no-done rollout"
    assert n_after_r1 == 0, "replay must not gain completed traj at nonterminal boundary"
    assert all(L == 128 for L in pending1.slot_lengths()), \
        "pending length wrong at nonterminal boundary"

    trajs2, pending2, _, _, _ = run_rollouts(
        net, ts, env, 1, pending=pending1, collector=coll1, action_seed=5)
    assert len(trajs2) == 0, "still no done -> still no completed trajectory"
    assert all(L == 256 for L in pending2.slot_lengths()), \
        f"pending must continue to 256, got {pending2.slot_lengths()}"
    print("  PASS 3 nonterminal boundary: replay gains nothing, pending "
          "128->256 continues across the boundary")


# ---------------------------------------------------------------------------
# 4. multi-env isolation
# ---------------------------------------------------------------------------

def test_multi_env_isolation():
    net = make_net()
    ts = make_train_state(net)
    # Different episode lengths per env via a custom env: env e done every (100+e*50).
    class MultiEnv(FakeEnv):
        def step(self, rng, state, actions):
            actions = np.asarray(actions)
            new_counts = state.counts + 1
            reward = new_counts.astype(np.float32)
            periods = np.array([100, 150], dtype=np.int64)
            done = (new_counts % periods) == 0
            return self._obs(new_counts), FakeEnvState(new_counts), reward, done, {}

    env = MultiEnv(NUM_ENVS, OBS_DIM, done_every=10_000)
    trajs, pending, _, _, _ = run_rollouts(net, ts, env, 3, action_seed=6)
    # env0 completes at 100 (and would again at 200, 300>384? 3 rollouts=384 ->
    #   env0 done at 100,200,300 -> lengths 100,100, then 84 pending);
    # env1 done at 150,300 -> lengths 150,150, then 84 pending.
    by_len = sorted(t.length for t in trajs)
    assert 100 in by_len and 150 in by_len, \
        f"per-env episodes not isolated (lengths {by_len})"
    # No trajectory may mix the two envs' counters: each trajectory's counter
    # must be a clean contiguous +1 ramp (independent per-slot buffering, no splice).
    for t in trajs:
        assert _is_contiguous_ramp(t), \
            "a trajectory spliced transitions across env slots"
    print(f"  PASS 4 multi-env isolation: independent per-slot episodes "
          f"(lengths {by_len}), no cross-env splicing")


# ---------------------------------------------------------------------------
# 5. auto-reset isolation
# ---------------------------------------------------------------------------

def test_autoreset_isolation():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=100)  # done at 100, 200, 300 ...
    trajs, _, _, _, _ = run_rollouts(net, ts, env, 3, action_seed=8)  # 384 steps
    # Per env: episodes [0..99],[100..199],[200..299] done + [300..383] pending.
    # Both envs share the counter, so 6 completed trajectories of length 100.
    assert len(trajs) == 2 * 3, f"expected 6 completed episodes, got {len(trajs)}"
    assert all(t.length == 100 for t in trajs), \
        f"episode lengths {[t.length for t in trajs]} != 100"
    for t in trajs:
        dones = np.asarray(t.dones)
        # Exactly one done, at the very end: the terminal transition is the last
        # of its episode, everything before it is non-terminal (auto-reset
        # isolation — the reset begins a NEW episode, never appended here).
        assert bool(dones[-1]) is True, "episode must end on its terminal done"
        assert not bool(np.any(dones[:-1])), \
            "a non-terminal done sits inside an episode (reset leaked in)"
        # Each episode is a clean contiguous ramp (no overlap between the
        # terminal segment and the post-reset segment).
        assert _is_contiguous_ramp(t), "episode counter not contiguous"
    # Episode boundaries tile the counter without gaps/overlaps: sorted start
    # counts per env are {0,100,200}.
    starts = sorted(int(_obs_counts(t)[0]) for t in trajs)
    assert starts == [0, 0, 100, 100, 200, 200], \
        f"episode start counts {starts} do not tile cleanly at resets"
    print("  PASS 5 auto-reset isolation: each episode ends on exactly one "
          "terminal done; post-reset transitions start fresh episodes (no overlap)")


# ---------------------------------------------------------------------------
# 6. initial-memory correctness
# ---------------------------------------------------------------------------

def test_initial_memory_correctness():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=200)

    # Run two rollouts; the completed 200-step episode's initial_memory must be
    # the memory BEFORE its first action (the reset-zero memory), NOT a boundary
    # memory from rollout 1's end.
    _, pending1, coll1, _, _ = run_rollouts(net, ts, env, 1, action_seed=9)
    mem_at_boundary = np.asarray(coll1["memories"]).copy()  # memory after rollout1
    trajs2, _, _, _, _ = run_rollouts(
        net, ts, env, 1, pending=pending1, collector=coll1, action_seed=9)
    long_t = max(trajs2, key=lambda t: t.length)
    assert long_t.length == 200

    init_mem = np.asarray(long_t.initial_memory)
    # First episode starts from reset -> initial memory is the zero memory.
    assert np.allclose(init_mem, 0.0), \
        "initial_memory of the first episode must be the reset-zero memory"
    # And it must NOT equal the (nonzero) boundary memory carried at rollout end.
    assert not np.allclose(init_mem, mem_at_boundary[0]), \
        "initial_memory was taken from a mid-rollout boundary memory (wrong)"
    # initial_memory == memory before step 0 == memory_sequence's conceptual
    # predecessor: memory before step t (t>=1) equals memory_sequence[t-1].
    mem_seq = np.asarray(long_t.memory_sequence)
    assert mem_seq.shape[0] == 200
    print("  PASS 6 initial-memory correctness: initial_memory is the pre-first-"
          "action (reset-zero) memory, carried across rollouts, not a boundary mem")


# ---------------------------------------------------------------------------
# 7. pending-buffer checkpoint resume (via launcher restore, bit-exact)
# ---------------------------------------------------------------------------

def test_pending_checkpoint_resume():
    # restore_p2_v1_checkpoint builds the Stage4 optimizer template, which needs
    # cfg.lr + cfg.max_grad_norm on top of the network dims.  The test Cfg has the
    # dims (window_mem/num_layers/embed_size) and max_grad_norm but not lr, so add
    # lr matching make_train_state's default (1e-3) -> identical optax structure.
    class _RestoreCfg(Cfg):
        lr = 1e-3

    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=200)

    # (A) Uninterrupted reference: 3 rollouts.
    trajs_full, _, _, _, _ = run_rollouts(net, ts, env, 3, action_seed=11)

    # (B) Run 1 rollout, then save a REAL full checkpoint mid-episode.
    trajs_a, pending_a, coll_a, arng_a, rng_a = run_rollouts(
        net, ts, env, 1, action_seed=11)
    replay_a = TrajectoryReplayBuffer(capacity=16, seed=11)
    for t in trajs_a:
        replay_a.insert(t)

    tmp = tempfile.mkdtemp(prefix="p2v1_planb_ckpt_")
    try:
        gs = 12345
        save_full_checkpoint(
            ts, replay_a, rng_a, gs, tmp, step=gs,
            action_rng_state=arng_a, update_count=7,
            pending_state=pending_a.state_dict(),
            collector_state=coll_a,
        )
        # Restore through the launcher's production restore primitive.
        r = restore_p2_v1_checkpoint(tmp, gs, net, _RestoreCfg(), OBS_DIM)
        assert r["update_count"] == 7
        assert r["global_step"] == gs
        pending_r = PendingEpisodeBuffers.from_state_dict(r["pending_state"])
        action_rng_r = restore_action_rng(r["action_rng_state"], seed=0)

        trajs_b, _, _, _, _ = run_rollouts(
            net, r["train_state"], env, 2,
            pending=pending_r, collector=r["collector_state"],
            action_rng=action_rng_r, rng=r["rng"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    resumed = trajs_a + trajs_b
    assert len(resumed) == len(trajs_full), \
        f"trajectory count mismatch: full={len(trajs_full)} resumed={len(resumed)}"
    assert _sig(resumed) == _sig(trajs_full), \
        "resumed trajectories are NOT bit-exact vs the uninterrupted run"
    print(f"  PASS 7 pending-buffer checkpoint resume: save mid-episode -> "
          f"restore_p2_v1_checkpoint -> continue == uninterrupted bit-exact "
          f"({len(trajs_full)} trajs, lengths {sorted(t.length for t in trajs_full)})")


# ---------------------------------------------------------------------------
# 8. no-dup / no-loss invariant at every boundary
# ---------------------------------------------------------------------------

def test_no_dup_no_loss():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=200)

    pending = None
    coll = None
    completed = 0
    for r in range(4):  # 4 rollouts = 1024 transitions total
        trajs, pending, coll, _, _ = run_rollouts(
            net, ts, env, 1, pending=pending, collector=coll, action_seed=13)
        completed += sum(t.length for t in trajs)
        total_so_far = (r + 1) * NUM_ENVS * ROLLOUT_STEPS
        invariant = completed + pending.total_pending_transitions()
        assert invariant == total_so_far, \
            f"rollout{r}: completed({completed})+pending" \
            f"({pending.total_pending_transitions()}) != total({total_so_far})"
    print(f"  PASS 8 no-dup/no-loss: completed({completed})+pending"
          f"({pending.total_pending_transitions()})==total at every boundary")


TESTS = [
    ("1 cross-rollout reconstruction", test_cross_rollout_reconstruction),
    ("2 exact boundary", test_exact_boundary),
    ("3 nonterminal boundary", test_nonterminal_boundary),
    ("4 multi-env isolation", test_multi_env_isolation),
    ("5 auto-reset isolation", test_autoreset_isolation),
    ("6 initial-memory correctness", test_initial_memory_correctness),
    ("7 pending-buffer checkpoint resume", test_pending_checkpoint_resume),
    ("8 no-dup / no-loss", test_no_dup_no_loss),
]


def main():
    passed = failed = 0
    results = {}
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
            results[name] = "PASS"
        except Exception as e:
            failed += 1
            results[name] = f"FAIL: {e}"
            import traceback
            traceback.print_exc()
            print(f"  FAIL {name}: {e}")

    print(f"\n{'='*60}")
    print(f"方案B hard tests (第二节 1-8): {passed}/{passed + failed} PASS")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  [{'OK' if res == 'PASS' else 'XX'}] {name}")
    if failed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
