"""Gate 1.7 / replay conservation — sparse anchor storage & sampling (CPU).

No network needed (anchors are synthetic distinct arrays). Verifies:
  * anchor count == ceil(L/128), episode-start anchor present, steps == [0,128,...]
  * insert REJECTS wrong/missing anchors and non-terminal episodes
  * sample picks nearest anchor <= start; burn_in_obs = obs[anchor:start]; gap<=128
  * state_dict/from_state_dict round-trip preserves anchors (hash equal)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

import replay_buffer as RB
from replay_buffer import Trajectory, ReplayBuffer, anchor_steps_for_length

WM, LAY, EMB, OBS = 4, 2, 3, 10


def make_episode(length, done_last=True, seed=0):
    rng = np.random.RandomState(seed)
    steps = anchor_steps_for_length(length)
    anchors = np.stack([np.full((WM, LAY, EMB), float(s)) for s in steps])  # distinct per anchor
    dones = np.zeros(length, dtype=bool)
    if done_last:
        dones[-1] = True
    return Trajectory(
        observations=rng.standard_normal((length, OBS)).astype(np.float32),
        actions=rng.randint(0, 5, length),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=dones,
        values=rng.standard_normal(length).astype(np.float32),
        log_probs=rng.standard_normal(length).astype(np.float32),
        initial_memory=np.zeros((WM, LAY, EMB), np.float32),
        achievements=np.zeros((length, 8), np.float32),
        target_achievements=np.zeros(8, np.float32),
        next_observations=rng.standard_normal((length, OBS)).astype(np.float32),
        memory_anchors=anchors,
        anchor_steps=np.array(steps, dtype=np.int64),
        collected_update_count=0,
    )


def test_anchor_steps_formula():
    assert anchor_steps_for_length(400) == [0, 128, 256, 384]
    assert anchor_steps_for_length(128) == [0]
    assert anchor_steps_for_length(129) == [0, 128]
    assert anchor_steps_for_length(512) == [0, 128, 256, 384]
    print("PASS test_anchor_steps_formula")


def test_validate_anchors_ok():
    t = make_episode(400)
    assert t.n_anchors == 4
    assert t.validate_anchors() is True
    assert int(t.anchor_steps[0]) == 0
    print("PASS test_validate_anchors_ok")


def test_insert_rejects_bad_anchors():
    t = make_episode(400)
    t.anchor_steps = np.array([0, 128, 256], dtype=np.int64)  # missing 384
    buf = ReplayBuffer(capacity=4)
    try:
        buf.insert(t)
        raise AssertionError("expected anchor conservation failure")
    except ValueError:
        pass
    # missing episode-start anchor
    t2 = make_episode(400)
    t2.anchor_steps = np.array([128, 256, 384], dtype=np.int64)
    t2.memory_anchors = t2.memory_anchors[1:]
    try:
        buf.insert(t2)
        raise AssertionError("expected episode-start anchor failure")
    except ValueError:
        pass
    print("PASS test_insert_rejects_bad_anchors")


def test_insert_rejects_nonterminal():
    t = make_episode(400, done_last=False)
    buf = ReplayBuffer(capacity=4)
    try:
        buf.insert(t)
        raise AssertionError("expected non-terminal rejection")
    except ValueError:
        pass
    print("PASS test_insert_rejects_nonterminal")


def test_sample_nearest_anchor_and_burnin():
    t = make_episode(400, seed=1)
    buf = ReplayBuffer(capacity=4)
    buf.insert(t)
    s = buf.sample(sequence_length=129, start_step=200)
    # nearest anchor <= 200 is 128
    assert s.pre_anchor_step == 128, s.pre_anchor_step
    assert s.burn_in_length == 200 - 128 == 72
    assert np.allclose(s.burn_in_obs, t.observations[128:200])
    assert np.allclose(s.pre_anchor_memory, t.anchor_at(128))
    assert s.length == 129
    assert np.allclose(s.observations, t.observations[200:329])
    assert s.start_step == 200
    print("PASS test_sample_nearest_anchor_and_burnin")


def test_sample_start_on_anchor_zero_burnin():
    t = make_episode(400, seed=2)
    buf = ReplayBuffer(capacity=4)
    buf.insert(t)
    s = buf.sample(sequence_length=129, start_step=128)
    assert s.pre_anchor_step == 128
    assert s.burn_in_length == 0
    assert np.allclose(s.observations, t.observations[128:257])
    print("PASS test_sample_start_on_anchor_zero_burnin")


def test_sample_start_zero_uses_episode_anchor():
    t = make_episode(400, seed=3)
    buf = ReplayBuffer(capacity=4)
    buf.insert(t)
    s = buf.sample(sequence_length=200, start_step=0)
    assert s.pre_anchor_step == 0
    assert s.burn_in_length == 0
    assert np.allclose(s.pre_anchor_memory, t.anchor_at(0))
    print("PASS test_sample_start_zero_uses_episode_anchor")


def test_burnin_gap_bounded_128():
    t = make_episode(1000, seed=4)
    buf = ReplayBuffer(capacity=4)
    buf.insert(t)
    for start in [0, 100, 127, 128, 255, 256, 500, 871]:
        s = buf.sample(sequence_length=129, start_step=start)
        assert s.burn_in_length <= 128, (start, s.burn_in_length)
        assert s.pre_anchor_step <= start
        assert start - s.pre_anchor_step == s.burn_in_length
    print("PASS test_burnin_gap_bounded_128")


def test_no_trajectory_over_128_rejects():
    buf = ReplayBuffer(capacity=4)
    buf.insert(make_episode(128))  # exactly 128, NOT > 128
    assert not buf.can_sample()
    try:
        buf.sample()
        raise AssertionError("expected Gate4 rejection")
    except RuntimeError:
        pass
    print("PASS test_no_trajectory_over_128_rejects")


def test_state_roundtrip_preserves_anchors():
    t = make_episode(400, seed=5)
    buf = ReplayBuffer(capacity=4, seed=7)
    buf.insert(t)
    buf.sample(sequence_length=150)  # advance rng
    st = buf.state_dict()
    buf2 = ReplayBuffer.from_state_dict(st)
    assert buf.hash_digest() == buf2.hash_digest()
    assert buf2._buffer[0].n_anchors == 4
    assert np.allclose(buf2._buffer[0].memory_anchors, t.memory_anchors)
    # same rng state -> same next sample
    sa = buf.sample(sequence_length=129)
    sb = buf2.sample(sequence_length=129)
    assert sa.start_step == sb.start_step
    print("PASS test_state_roundtrip_preserves_anchors")


def test_capacity_eviction():
    buf = ReplayBuffer(capacity=2)
    buf.insert(make_episode(200, seed=10))
    buf.insert(make_episode(210, seed=11))
    buf.insert(make_episode(220, seed=12))
    assert len(buf) == 2
    print("PASS test_capacity_eviction")


if __name__ == "__main__":
    test_anchor_steps_formula()
    test_validate_anchors_ok()
    test_insert_rejects_bad_anchors()
    test_insert_rejects_nonterminal()
    test_sample_nearest_anchor_and_burnin()
    test_sample_start_on_anchor_zero_burnin()
    test_sample_start_zero_uses_episode_anchor()
    test_burnin_gap_bounded_128()
    test_no_trajectory_over_128_rejects()
    test_state_roundtrip_preserves_anchors()
    test_capacity_eviction()
    print("ALL_REPLAY_ANCHOR_TESTS_PASS")
