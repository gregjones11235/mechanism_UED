"""Gate 4 (data layer) — hindsight relabel for the sparse-anchor schema (CPU).

Verifies: goal conditioning replaces trailing dims (obs, next_obs, burn_in_obs);
reward recomputed; anchor memory + behavior log_probs carried through; Gate 5/6
enforced (fabricated / unreached goals REJECTED).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

import hindsight as H
from replay_buffer import ReplaySample, Trajectory, ReplayBuffer, anchor_steps_for_length

OBS, EMB, NAch = 12, 5, 8
WM, LAY, EMBED = 4, 2, 3


def make_sample(goal_achieved_idx=3, ach_step=10, length=160, seed=0):
    rng = np.random.RandomState(seed)
    achievements = np.zeros((length, NAch), np.float32)
    achievements[ach_step:, goal_achieved_idx] = 1.0  # achieved from ach_step onward
    target = np.zeros(NAch, np.float32); target[0] = 1.0  # original target = goal 0
    steps = anchor_steps_for_length(length)
    return ReplaySample(
        observations=rng.standard_normal((length, OBS)).astype(np.float32),
        actions=rng.randint(0, 5, length),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.concatenate([np.zeros(length - 1, bool), [True]]),
        values=rng.standard_normal(length).astype(np.float32),
        log_probs=rng.standard_normal(length).astype(np.float32),
        achievements=achievements,
        target_achievements=target,
        pre_anchor_memory=np.full((WM, LAY, EMBED), 7.0, np.float32),
        pre_anchor_step=128,
        burn_in_obs=rng.standard_normal((32, OBS)).astype(np.float32),
        next_observations=rng.standard_normal((length, OBS)).astype(np.float32),
        source_trajectory_id=0, start_step=160, length=length,
        next_value=0.0, episode_done=True, collected_update_count=3,
    )


def test_relabel_changes_goal_reward_and_obs():
    s = make_sample(goal_achieved_idx=3, ach_step=10)
    r = H.relabel_sample(s, goal_index=3, embedding_size=EMB)
    # target changed to goal 3
    assert r.target_achievements[3] == 1.0 and r.target_achievements[0] == 0.0
    # obs trailing EMB dims == goal_embedding(3)
    ge = H.goal_embedding(3, EMB)
    assert np.allclose(r.observations[:, -EMB:], ge)
    assert np.allclose(r.next_observations[:, -EMB:], ge)
    assert np.allclose(r.burn_in_obs[:, -EMB:], ge)  # burn-in re-contextualized
    # leading (non-goal) dims UNCHANGED
    assert np.allclose(r.observations[:, :-EMB], s.observations[:, :-EMB])
    assert np.allclose(r.burn_in_obs[:, :-EMB], s.burn_in_obs[:, :-EMB])
    # reward recomputed: +1 at ach_step (first achievement) under goal 3
    assert r.rewards[10] == 1.0
    assert r.rewards[11] == 0.0  # already achieved -> no further +1
    assert not np.allclose(r.rewards, s.rewards)
    print("PASS test_relabel_changes_goal_reward_and_obs")


def test_relabel_carries_anchor_and_logprobs():
    s = make_sample()
    r = H.relabel_sample(s, goal_index=3, embedding_size=EMB)
    assert np.allclose(r.pre_anchor_memory, s.pre_anchor_memory)  # anchor carried
    assert r.pre_anchor_step == s.pre_anchor_step
    assert np.allclose(r.log_probs, s.log_probs)  # behavior mu carried (V-trace/diag)
    assert r.collected_update_count == s.collected_update_count
    assert np.allclose(r.achievements, s.achievements)
    print("PASS test_relabel_carries_anchor_and_logprobs")


def test_gate6_fabricated_goal_rejected():
    s = make_sample(goal_achieved_idx=3)  # only goal 3 achieved
    try:
        H.relabel_sample(s, goal_index=5, embedding_size=EMB)  # 5 NOT achieved
        raise AssertionError("Gate6 should reject fabricated goal")
    except ValueError:
        pass
    print("PASS test_gate6_fabricated_goal_rejected")


def test_gate6_no_achievement_rejected():
    s = make_sample()
    s.achievements = np.zeros_like(s.achievements)  # nothing achieved
    try:
        H.relabel_sample(s, embedding_size=EMB)
        raise AssertionError("Gate6 should reject when nothing achieved")
    except ValueError:
        pass
    print("PASS test_gate6_no_achievement_rejected")


def test_gate5_default_picks_achieved():
    s = make_sample(goal_achieved_idx=3)
    r = H.relabel_sample(s, embedding_size=EMB)  # no explicit goal -> min achieved
    assert r.target_achievements[3] == 1.0
    print("PASS test_gate5_default_picks_achieved")


def test_relabel_trajectory_carries_anchors():
    rng = np.random.RandomState(1)
    length = 300
    achievements = np.zeros((length, NAch), np.float32)
    achievements[50:, 2] = 1.0
    steps = anchor_steps_for_length(length)
    anchors = np.stack([np.full((WM, LAY, EMBED), float(x)) for x in steps])
    t = Trajectory(
        observations=rng.standard_normal((length, OBS)).astype(np.float32),
        actions=rng.randint(0, 5, length),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.concatenate([np.zeros(length - 1, bool), [True]]),
        values=rng.standard_normal(length).astype(np.float32),
        log_probs=rng.standard_normal(length).astype(np.float32),
        initial_memory=np.zeros((WM, LAY, EMBED), np.float32),
        achievements=achievements,
        target_achievements=np.zeros(NAch, np.float32),
        next_observations=rng.standard_normal((length, OBS)).astype(np.float32),
        memory_anchors=anchors, anchor_steps=np.array(steps, np.int64),
    )
    r = H.relabel_trajectory(t, goal_index=2, embedding_size=EMB)
    assert r.n_anchors == len(steps)
    assert np.allclose(r.memory_anchors, anchors)
    assert np.allclose(r.anchor_steps, np.array(steps))
    assert r.target_achievements[2] == 1.0
    assert np.allclose(r.observations[:, -EMB:], H.goal_embedding(2, EMB))
    print("PASS test_relabel_trajectory_carries_anchors")


if __name__ == "__main__":
    test_relabel_changes_goal_reward_and_obs()
    test_relabel_carries_anchor_and_logprobs()
    test_gate6_fabricated_goal_rejected()
    test_gate6_no_achievement_rejected()
    test_gate5_default_picks_achieved()
    test_relabel_trajectory_carries_anchors()
    print("ALL_HINDSIGHT_DATA_TESTS_PASS")
